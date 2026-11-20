//! The three MotionBricks networks, each the arithmetic of its PyTorch class
//! in `tools/onnx/models/{layers,soma/networks,g1/networks}.py` and loaded
//! from the same state-dict keys (after the `mmm_net.` / `backbone_net.` /
//! `pose_net.` prefix is stripped).
//!
//! Tensors cross between host and device freely: the per-frame gathers,
//! `where`s and concatenations are host code on a few hundred rows, the
//! linears, transformers and convolutions run on the backend.

use crate::config::{PoseNetConfig, RootNetConfig, VqConfig};
use motion_core::backend::{Act, Backend, Conv1dShape};
use motion_core::tensor;
use motion_core::weights::WeightSource;
use motion_core::Result;
use motion_nn::{Encoder, Linear, Mlp};

/// Host convenience: `act(x·Wᵀ + b)` on `rows` host rows.
pub fn lin_host<B: Backend>(be: &B, l: &Linear<B>, x: &[f32], act: Act) -> Vec<f32> {
    let rows = x.len() / l.in_dim;
    assert_eq!(rows * l.in_dim, x.len(), "lin_host: input width");
    let xb = be.upload(x);
    be.download(&l.apply(be, &xb, rows, act))
}

/// `FCBlock(num_layers, layer_width, size_in, size_out)`: LeakyReLU after every
/// `fc_layers.i`, then `forward_projection` without activation.
pub fn load_fc_block<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    num_layers: usize,
    width: usize,
    size_in: usize,
    size_out: usize,
) -> Result<Mlp<B>> {
    let mut layers = Vec::new();
    for i in 0..num_layers {
        let din = if i == 0 { size_in } else { width };
        layers.push(Linear::load(be, src, &format!("{key}.fc_layers.{i}"), din, width, true)?);
    }
    layers.push(Linear::load(be, src, &format!("{key}.forward_projection"), width, size_out, true)?);
    Ok(Mlp::new(layers, Act::LeakyRelu))
}

pub fn mlp_host<B: Backend>(be: &B, m: &Mlp<B>, x: &[f32]) -> Vec<f32> {
    let rows = x.len() / m.layers[0].in_dim;
    let xb = be.upload(x);
    be.download(&m.forward(be, &xb, rows))
}

fn load_vec(src: &dyn WeightSource, key: &str, n: usize) -> Result<Vec<f32>> {
    src.get(key, &[n])
}

struct Conv<B: Backend> {
    w: B::Weight,
    b: B::Buf,
    cin: usize,
    cout: usize,
    k: usize,
}

impl<B: Backend> Conv<B> {
    fn load(be: &B, src: &dyn WeightSource, key: &str, cin: usize, cout: usize, k: usize) -> Result<Self> {
        let w = src.get(&format!("{key}.weight"), &[cout, cin, k])?;
        let b = src.get(&format!("{key}.bias"), &[cout])?;
        Ok(Self { w: be.weight(&w, cout, cin * k), b: be.upload(&b), cin, cout, k })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        be: &B,
        x: &B::Buf,
        batch: usize,
        len: usize,
        dilation: usize,
        upsample: bool,
        act: Act,
        residual: Option<&B::Buf>,
    ) -> (B::Buf, usize) {
        let s = Conv1dShape {
            batch,
            cin: self.cin,
            cout: self.cout,
            len,
            k: self.k,
            stride: 1,
            pad: dilation * (self.k / 2),
            dilation,
            upsample,
        };
        let out = be.alloc(s.out_elems());
        be.conv1d(x, s, &self.w, Some(&self.b), act, residual, &out);
        (out, s.out_len())
    }
}

/// `ResConv1DBlock(n, n, dilation, activation="relu", norm=None)`.
struct ResBlock<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    dilation: usize,
}

/// `Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True)`:
/// `model.{d}` has dilation `rate^(depth−1−d)`.
struct Resnet<B: Backend> {
    blocks: Vec<ResBlock<B>>,
    width: usize,
}

impl<B: Backend> Resnet<B> {
    fn load(be: &B, src: &dyn WeightSource, key: &str, width: usize, depth: usize, rate: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        for d in 0..depth {
            let dilation = rate.pow((depth - 1 - d) as u32);
            blocks.push(ResBlock {
                conv1: Conv::load(be, src, &format!("{key}.model.{d}.conv1"), width, width, 3)?,
                conv2: Conv::load(be, src, &format!("{key}.model.{d}.conv2"), width, width, 1)?,
                dilation,
            });
        }
        Ok(Self { blocks, width })
    }

    /// In place semantics: returns the new activation `[batch, width, len]`.
    fn forward(&self, be: &B, x: &B::Buf, batch: usize, len: usize, mask: &B::Buf) -> B::Buf {
        let mut x = x.clone();
        for b in &self.blocks {
            be.mask_time(&x, batch, self.width, len, mask);
            let orig = x.clone();
            let h = be.alloc(batch * self.width * len);
            be.unary(&x, 0, batch * self.width * len, Act::Relu, &h, 0);
            be.mask_time(&h, batch, self.width, len, mask);
            let (h, _) = b.conv1.run(be, &h, batch, len, b.dilation, false, Act::Relu, None);
            be.mask_time(&h, batch, self.width, len, mask);
            let (h, _) = b.conv2.run(be, &h, batch, len, 1, false, Act::None, Some(&orig));
            be.mask_time(&h, batch, self.width, len, mask);
            x = h;
        }
        x
    }
}

/// `DoubleCondDecoder` (layers.py) with `cond_fusion_last_layer=False` and
/// `norm=None`: tokens `[batch, in_ch, positions]` are upsampled `down_t`
/// times by 2; before each stage the per-frame target conditions replace the
/// matching channel groups where `has_target` and the external per-frame
/// features are concatenated and projected.
pub struct Decoder<B: Backend> {
    conv_in: Conv<B>,
    stages: Vec<(Resnet<B>, Conv<B>)>,
    conv_mid: Conv<B>,
    conv_out: Conv<B>,
    target_blocks: Vec<Linear<B>>,
    external_blocks: Vec<Linear<B>>,
    pub width: usize,
    pub down_t: usize,
    pub out_ch: usize,
    pub target_dim: usize,
    pub external_dim: usize,
}

impl<B: Backend> Decoder<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        be: &B,
        src: &dyn WeightSource,
        key: &str,
        out_ch: usize,
        in_ch: usize,
        down_t: usize,
        width: usize,
        depth: usize,
        rate: usize,
        target_dim: usize,
        external_dim: usize,
    ) -> Result<Self> {
        let conv_in = Conv::load(be, src, &format!("{key}.model.0"), in_ch, width, 3)?;
        let mut stages = Vec::new();
        for i in 0..down_t {
            let m = format!("{key}.model.{}", i + 2);
            stages.push((
                Resnet::load(be, src, &format!("{m}.0"), width, depth, rate)?,
                Conv::load(be, src, &format!("{m}.2"), width, width, 3)?,
            ));
        }
        let conv_mid = Conv::load(be, src, &format!("{key}.model.{}", 2 + down_t), width, width, 3)?;
        let conv_out = Conv::load(be, src, &format!("{key}.model.{}", 2 + down_t + 2), width, out_ch, 3)?;
        let mut target_blocks = Vec::new();
        let mut external_blocks = Vec::new();
        for i in 0..down_t {
            let nfpp = 1usize << (down_t - i);
            target_blocks.push(Linear::load(
                be,
                src,
                &format!("{key}.target_cond_blocks.{}", 2 * i),
                target_dim,
                width / nfpp,
                true,
            )?);
            external_blocks.push(Linear::load(
                be,
                src,
                &format!("{key}.external_cond_blocks.{}", 2 * i),
                nfpp * external_dim + width,
                width,
                true,
            )?);
        }
        Ok(Self {
            conv_in,
            stages,
            conv_mid,
            conv_out,
            target_blocks,
            external_blocks,
            width,
            down_t,
            out_ch,
            target_dim,
            external_dim,
        })
    }

    /// `x` `[batch, in_ch, positions]` (host, channels first); `external`
    /// `[batch, frames, external_dim]`, `target` `[batch, frames, target_dim]`,
    /// `has_target` `[batch, frames]`, `token_mask` `[batch, positions]`, with
    /// `frames = positions · 2^down_t`. Returns `[batch, frames, out_ch]` (host, channels last).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        be: &B,
        x: &[f32],
        batch: usize,
        positions: usize,
        external: &[f32],
        target: &[f32],
        has_target: &[bool],
        token_mask: &[f32],
    ) -> Vec<f32> {
        let w = self.width;
        let frames = positions << self.down_t;
        let mut mask = token_mask.to_vec();
        let mut len = positions;
        let xb = be.upload(x);
        be.mask_time(&xb, batch, self.conv_in.cin, len, &be.upload(&mask));
        let (mut h, _) = self.conv_in.run(be, &xb, batch, len, 1, false, Act::Relu, None);
        for i in 0..self.down_t {
            let nfpp = 1usize << (self.down_t - i);
            // Target conditions: h [b, w, pos] → [b, pos, w] → per frame [b, pos·nfpp, w/nfpp].
            let ht = lin_host(be, &self.target_blocks[i], target, Act::Relu); // [b·frames, w/nfpp]
            let hc = be.download(&h);
            let mut hl = tensor::to_channels_last(&hc, batch, w, len); // [b, pos, w]
            let g = w / nfpp;
            for b in 0..batch {
                for f in 0..frames {
                    if has_target[b * frames + f] {
                        let dst = b * len * w + f * g; // frame f is chunk f of the flattened [pos·w] row
                        hl[dst..dst + g].copy_from_slice(&ht[(b * frames + f) * g..(b * frames + f + 1) * g]);
                    }
                }
            }
            // External conditions: [b, pos, nfpp·E] concatenated after the hidden channels.
            let e = self.external_dim;
            let mut cat = Vec::with_capacity(batch * len * (w + nfpp * e));
            for b in 0..batch {
                for p in 0..len {
                    cat.extend_from_slice(&hl[(b * len + p) * w..(b * len + p + 1) * w]);
                    let f0 = b * frames + p * nfpp;
                    cat.extend_from_slice(&external[f0 * e..(f0 + nfpp) * e]);
                }
            }
            let proj = lin_host(be, &self.external_blocks[i], &cat, Act::Relu); // [b·pos, w]
            h = be.upload(&tensor::to_channels_first(&proj, batch, len, w));
            let mb = be.upload(&mask);
            let (res, conv) = &self.stages[i];
            h = res.forward(be, &h, batch, len, &mb);
            be.mask_time(&h, batch, w, len, &mb);
            // nn.Upsample(×2) + mask(repeat_interleave) + conv, the upsample fused into the conv.
            let (o, l2) = conv.run(be, &h, batch, len, 1, true, Act::None, None);
            mask = mask.iter().flat_map(|&m| [m, m]).collect();
            len = l2;
            h = o;
        }
        let mb = be.upload(&mask);
        be.mask_time(&h, batch, w, len, &mb);
        let (h2, _) = self.conv_mid.run(be, &h, batch, len, 1, false, Act::Relu, None);
        be.mask_time(&h2, batch, w, len, &mb);
        let (o, _) = self.conv_out.run(be, &h2, batch, len, 1, false, Act::None, None);
        be.mask_time(&o, batch, self.out_ch, len, &mb);
        tensor::to_channels_last(&be.download(&o), batch, self.out_ch, len)
    }
}

/// The root module (`root_mmm_network` / `root_backbone_network`): predicts the
/// token count class and the normalised global root path `[64, 5]`.
pub struct RootNet<B: Backend> {
    cfg: RootNetConfig,
    proj_local_pose: Linear<B>,
    proj_local_root: Linear<B>,
    proj_global_root: Linear<B>,
    no_local_pose: Vec<f32>,
    no_local_root: Vec<f32>,
    no_global_root: Vec<f32>,
    conv_no_frame: Vec<f32>,
    proj_start: Mlp<B>,
    proj_end: Mlp<B>,
    input_position: Vec<f32>,
    num_tokens_emb: Vec<f32>,
    position: Vec<f32>,
    logit: Linear<B>,
    middle_token: Option<Vec<f32>>,
    shared: Encoder<B>,
    root_token: Option<Encoder<B>>,
    decoder: Decoder<B>,
}

pub struct RootOut {
    /// `[batch, classes]` after the out-of-reach and allowed-count masking.
    pub logits: Vec<f32>,
    pub pred_num_tokens: Vec<usize>,
    /// Normalised global root values `[batch, frames, 5]`.
    pub global_root: Vec<f32>,
}

/// Count of `nn.TransformerEncoder` layers stored under `prefix`.
pub fn count_layers(src: &dyn WeightSource, prefix: &str) -> usize {
    (0..).take_while(|i| src.contains(&format!("{prefix}.layers.{i}.linear1.weight"))).count()
}

fn ff_dim(src: &dyn WeightSource, prefix: &str, d: usize) -> Result<usize> {
    let s = src.shape(&format!("{prefix}.layers.0.linear1.weight"));
    match s {
        Ok(s) => Ok(s[0]),
        // Synthetic sources have no shapes: PyTorch's default dim_feedforward.
        Err(_) => {
            let _ = d;
            Ok(2048)
        }
    }
}

impl<B: Backend> RootNet<B> {
    pub fn load(be: &B, src: &dyn WeightSource, cfg: &RootNetConfig) -> Result<Self> {
        let e = cfg.n_embd;
        let classes = cfg.max_tokens - cfg.min_tokens + 2;
        let enc = |p: &str, n: usize| -> Result<Encoder<B>> {
            let ff = ff_dim(src, p, e)?;
            Encoder::load(be, src, p, n, e, cfg.n_head, ff, Act::Relu, 1e-5)
        };
        let proj_in = cfg.pose_feat_dim + cfg.local_root_feat_dim + cfg.global_root_feat_dim;
        let root_token = if cfg.n_layers_root_token > 0 {
            Some(enc("_root_token_transformer_model", cfg.n_layers_root_token)?)
        } else {
            None
        };
        Ok(Self {
            proj_local_pose: Linear::load(be, src, "_proj_local_pose", cfg.local_pose_dim, cfg.pose_feat_dim, true)?,
            proj_local_root: Linear::load(be, src, "_proj_local_root_value", 4, cfg.local_root_feat_dim, true)?,
            proj_global_root: Linear::load(be, src, "_proj_global_root_value", 5, cfg.global_root_feat_dim, true)?,
            no_local_pose: load_vec(src, "_no_local_pose_emb", cfg.pose_feat_dim)?,
            no_local_root: load_vec(src, "_no_local_root_emb", cfg.local_root_feat_dim)?,
            no_global_root: load_vec(src, "_no_global_root_emb", cfg.global_root_feat_dim)?,
            conv_no_frame: load_vec(src, "_conv_no_frame_emb", e)?,
            proj_start: load_fc_block(be, src, "_proj_start_input", cfg.mlp_layers, e, proj_in, e)?,
            proj_end: load_fc_block(be, src, "_proj_end_input", cfg.mlp_layers, e, proj_in, e)?,
            input_position: src.get("_input_position_emb.weight", &[2 * cfg.frames_per_token, e])?,
            num_tokens_emb: src.get("_proj_input_num_tokens.weight", &[classes + 1, e])?,
            position: src.get("_position_emb.embed", &[cfg.max_tokens, 1, e])?,
            logit: Linear::load(be, src, "_proj_num_token_output_logit", e, classes, true)?,
            middle_token: if root_token.is_some() {
                Some(src.get("_middle_token_emb.weight", &[classes, e])?)
            } else {
                None
            },
            shared: enc("_shared_transformer_model", cfg.n_layers_shared)?,
            root_token,
            decoder: Decoder::load(
                be,
                src,
                "_conv_output",
                5,
                e,
                cfg.down_t,
                cfg.width,
                cfg.depth,
                cfg.dilation_growth_rate,
                5,
                e,
            )?,
            cfg: cfg.clone(),
        })
    }

    /// Inputs are normalised, `[batch, 8, ·]`; masks `[batch, 8]`; `allowed`
    /// `[batch, max − min + 1]` (1 = allowed). The token count input is the
    /// masked class (`MASKED_NUM_TOKENS`), as in the release graph.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        be: &B,
        batch: usize,
        global_root: &[f32],
        has_global: &[bool],
        local_root: &[f32],
        has_local: &[bool],
        poses: &[f32],
        has_poses: &[bool],
        allowed: Option<&[i32]>,
        allow_out_of_reach: bool,
    ) -> RootOut {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let fpt = cfg.frames_per_token;
        let slots = 2 * fpt;
        let blend = |emb: Vec<f32>, has: &[bool], none: &[f32]| -> Vec<f32> {
            let w = none.len();
            let mut out = emb;
            for (r, h) in has.iter().enumerate() {
                if !h {
                    out[r * w..(r + 1) * w].copy_from_slice(none);
                }
            }
            out
        };
        let lp = blend(lin_host(be, &self.proj_local_pose, poses, Act::None), has_poses, &self.no_local_pose);
        let lr = blend(lin_host(be, &self.proj_local_root, local_root, Act::None), has_local, &self.no_local_root);
        let gr = blend(lin_host(be, &self.proj_global_root, global_root, Act::None), has_global, &self.no_global_root);
        let rows = batch * slots;
        let cat = tensor::cat_cols(
            &[(&lp, cfg.pose_feat_dim), (&lr, cfg.local_root_feat_dim), (&gr, cfg.global_root_feat_dim)],
            rows,
        );
        let w = cat.len() / rows;
        let (mut start_in, mut end_in) = (Vec::new(), Vec::new());
        for b in 0..batch {
            start_in.extend_from_slice(&cat[(b * slots) * w..(b * slots + fpt) * w]);
            end_in.extend_from_slice(&cat[(b * slots + fpt) * w..(b + 1) * slots * w]);
        }
        let start = mlp_host(be, &self.proj_start, &start_in);
        let end = mlp_host(be, &self.proj_end, &end_in);
        let mut frame = Vec::with_capacity(rows * e); // [b, 8, e]
        for b in 0..batch {
            frame.extend_from_slice(&start[b * fpt * e..(b + 1) * fpt * e]);
            frame.extend_from_slice(&end[b * fpt * e..(b + 1) * fpt * e]);
        }
        let mut positioned = frame.clone();
        for r in 0..rows {
            for c in 0..e {
                positioned[r * e + c] += self.input_position[(r % slots) * e + c];
            }
        }
        // Stage 1: [time | 8 frames] → count logits.
        let masked_class = cfg.max_tokens + 2 - cfg.min_tokens;
        let s1 = 1 + slots;
        let mut seq = Vec::with_capacity(batch * s1 * e);
        for b in 0..batch {
            seq.extend_from_slice(&self.num_tokens_emb[masked_class * e..(masked_class + 1) * e]);
            seq.extend_from_slice(&positioned[b * slots * e..(b + 1) * slots * e]);
        }
        let xb = be.upload(&seq);
        self.shared.forward(be, &xb, batch, s1, None);
        let out1 = be.download(&xb);
        let first: Vec<f32> = (0..batch).flat_map(|b| out1[b * s1 * e..b * s1 * e + e].to_vec()).collect();
        let mut logits = lin_host(be, &self.logit, &first, Act::None);
        let classes = cfg.max_tokens - cfg.min_tokens + 2;
        let counts = cfg.max_tokens - cfg.min_tokens + 1;
        let mut chosen = Vec::with_capacity(batch);
        for b in 0..batch {
            let l = &mut logits[b * classes..(b + 1) * classes];
            if !allow_out_of_reach {
                l[counts] = f32::NEG_INFINITY; // OUT_OF_REACH_NUM_TOKENS − min
            }
            if let Some(a) = allowed {
                for i in 0..counts {
                    if a[b * counts + i] != 1 {
                        l[i] = f32::NEG_INFINITY;
                    }
                }
            }
            chosen.push(tensor::argmax(l));
        }
        let pred: Vec<usize> = chosen.iter().map(|c| c + cfg.min_tokens).collect();
        // Stage 2: [count token | 8 positioned frames | 16 positions] → path features.
        let mt = cfg.max_tokens;
        let second: Vec<f32> = match (&self.root_token, &self.middle_token) {
            (Some(enc), Some(mid)) => {
                let s2 = 1 + slots + mt;
                let mut seq = Vec::with_capacity(batch * s2 * e);
                let mut km = Vec::with_capacity(batch * s2);
                for b in 0..batch {
                    seq.extend_from_slice(&mid[chosen[b] * e..(chosen[b] + 1) * e]);
                    seq.extend_from_slice(&positioned[b * slots * e..(b + 1) * slots * e]);
                    seq.extend_from_slice(&self.position[..mt * e]);
                    km.extend(std::iter::repeat_n(1.0, 1 + slots));
                    km.extend((0..mt).map(|p| if p < pred[b] { 1.0 } else { 0.0 }));
                }
                let xb = be.upload(&seq);
                enc.forward(be, &xb, batch, s2, Some(&be.upload(&km)));
                let o = be.download(&xb);
                (0..batch).flat_map(|b| o[(b * s2 + 1 + slots) * e..(b + 1) * s2 * e].to_vec()).collect()
            }
            _ => (0..batch).flat_map(|_| self.position[..mt * e].to_vec()).collect(),
        };
        // Dense per-frame conditions for the conv decoder.
        let frames = mt * fpt;
        let mut ext = Vec::with_capacity(batch * frames * e);
        let mut tgt = vec![0f32; batch * frames * 5];
        let mut has_t = vec![false; batch * frames];
        for b in 0..batch {
            let mut fr: Vec<Vec<f32>> = (0..frames).map(|_| self.conv_no_frame.clone()).collect();
            let has_frame = |s: usize| has_poses[b * slots + s] || has_global[b * slots + s] || has_local[b * slots + s];
            let pick = |s: usize| -> Vec<f32> {
                if has_frame(s) { frame[(b * slots + s) * e..(b * slots + s + 1) * e].to_vec() } else { self.conv_no_frame.clone() }
            };
            for i in 0..fpt {
                fr[i] = pick(i);
                tgt[(b * frames + i) * 5..(b * frames + i + 1) * 5]
                    .copy_from_slice(&global_root[(b * slots + i) * 5..(b * slots + i + 1) * 5]);
                has_t[b * frames + i] = has_global[b * slots + i];
            }
            for i in 0..fpt {
                let off = pred[b] * fpt - fpt + i;
                let s = fpt + i;
                fr[off] = pick(s);
                tgt[(b * frames + off) * 5..(b * frames + off + 1) * 5]
                    .copy_from_slice(&global_root[(b * slots + s) * 5..(b * slots + s + 1) * 5]);
                has_t[b * frames + off] = has_global[b * slots + s];
            }
            for f in fr {
                ext.extend_from_slice(&f);
            }
        }
        let x = tensor::to_channels_first(&second, batch, mt, e);
        let token_mask: Vec<f32> = (0..batch)
            .flat_map(|b| {
                let n = pred[b];
                (0..mt).map(move |p| if p < n { 1.0 } else { 0.0 })
            })
            .collect();
        let global_root = self.decoder.forward(be, &x, batch, mt, &ext, &tgt, &has_t, &token_mask);
        RootOut { logits, pred_num_tokens: pred, global_root }
    }
}

/// The pose module (`pose_mmm_network` / `pose_backbone_network`): one pass
/// over the 16 token slots gives per-head code logits `[batch, 16, heads, codes]`.
pub struct PoseNet<B: Backend> {
    cfg: PoseNetConfig,
    token_emb: Vec<f32>,
    proj_token: Mlp<B>,
    proj_root: Linear<B>,
    proj_pose: Linear<B>,
    num_valid: Vec<f32>,
    proj_input: Linear<B>,
    position: Vec<f32>,
    transformer: Encoder<B>,
    logit: Linear<B>,
}

impl<B: Backend> PoseNet<B> {
    pub fn load(be: &B, src: &dyn WeightSource, cfg: &PoseNetConfig) -> Result<Self> {
        let h = cfg.heads;
        let td = cfg.code_dim / h;
        let fpt = cfg.frames_per_token;
        let ff = ff_dim(src, "_transformer_model", cfg.n_embd)?;
        Ok(Self {
            token_emb: src.get("_pose_token_emb.weight", &[h * (cfg.codes + 1), td])?,
            proj_token: load_fc_block(be, src, "_proj_pose_token_emb", cfg.mlp_layers, cfg.pose_feat_width, h * td, cfg.pose_feat_width)?,
            proj_root: Linear::load(be, src, "_proj_local_root_values", cfg.root_cond_dim * fpt, cfg.root_feat_width, true)?,
            proj_pose: Linear::load(be, src, "_proj_local_pose", cfg.local_pose_dim, cfg.pose_feat_width / fpt, true)?,
            num_valid: src.get("_proj_num_valid_positions.weight", &[cfg.max_tokens - cfg.min_tokens + 1, cfg.token_length_feat_width])?,
            proj_input: Linear::load(
                be,
                src,
                "_proj_input.0",
                cfg.pose_feat_width + cfg.root_feat_width + cfg.token_length_feat_width,
                cfg.n_embd,
                true,
            )?,
            position: src.get("_position_emb.embed", &[cfg.max_tokens, 1, cfg.n_embd])?,
            transformer: Encoder::load(be, src, "_transformer_model", cfg.n_layers, cfg.n_embd, cfg.n_head, ff, Act::Relu, 1e-5)?,
            logit: Linear::load(be, src, "_proj_pose_output_logit", cfg.n_embd, h * cfg.codes, true)?,
            cfg: cfg.clone(),
        })
    }

    pub fn mask_id(&self) -> usize {
        self.cfg.codes
    }

    /// `tokens` `[batch, 16, heads]`, `root_cond` `[batch, 64, root_cond_dim]`,
    /// `pose_cond` `[batch, 64, local_pose_dim]`, `has_pose` `[batch, 64]`.
    pub fn forward(
        &self,
        be: &B,
        batch: usize,
        tokens: &[usize],
        root_cond: &[f32],
        pose_cond: &[f32],
        has_pose: &[bool],
        num_tokens: &[usize],
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (mt, fpt, h) = (c.max_tokens, c.frames_per_token, c.heads);
        let td = c.code_dim / h;
        let pfw = c.pose_feat_width;
        let root_emb = lin_host(be, &self.proj_root, root_cond, Act::None); // [b·16, rfw]
        let cond_emb = lin_host(be, &self.proj_pose, pose_cond, Act::None); // [b·64, pfw/4]
        let mut tok = Vec::with_capacity(batch * mt * h * td);
        for (i, &t) in tokens.iter().enumerate() {
            let row = (i % h) * (c.codes + 1) + t;
            tok.extend_from_slice(&self.token_emb[row * td..(row + 1) * td]);
        }
        let tok_emb = mlp_host(be, &self.proj_token, &tok); // [b·16, pfw] == [b·64, pfw/4]
        let g = pfw / fpt;
        let mut pose_emb = tok_emb;
        for f in 0..batch * mt * fpt {
            if has_pose[f] {
                pose_emb[f * g..(f + 1) * g].copy_from_slice(&cond_emb[f * g..(f + 1) * g]);
            }
        }
        let tl = c.token_length_feat_width;
        let mut nt = Vec::with_capacity(batch * mt * tl);
        for b in 0..batch {
            let r = num_tokens[b] - c.min_tokens;
            for _ in 0..mt {
                nt.extend_from_slice(&self.num_valid[r * tl..(r + 1) * tl]);
            }
        }
        let rows = batch * mt;
        let cat = tensor::cat_cols(&[(&pose_emb, pfw), (&root_emb, c.root_feat_width), (&nt, tl)], rows);
        let mut x = lin_host(be, &self.proj_input, &cat, Act::Relu);
        let e = c.n_embd;
        for r in 0..rows {
            for k in 0..e {
                x[r * e + k] += self.position[(r % mt) * e + k];
            }
        }
        let km: Vec<f32> = (0..batch)
            .flat_map(|b| {
                let n = num_tokens[b];
                (0..mt).map(move |p| if p < n { 1.0 } else { 0.0 })
            })
            .collect();
        let xb = be.upload(&x);
        self.transformer.forward(be, &xb, batch, mt, Some(&be.upload(&km)));
        be.download(&self.logit.apply(be, &xb, rows, Act::None))
    }
}

/// The VQ-VAE's decoding half: multi-head codebook lookup and the conditioned decoder.
pub struct VqDecoder<B: Backend> {
    pub codebook: Vec<f32>,
    pub cfg: VqConfig,
    pub decoder: Decoder<B>,
}

impl<B: Backend> VqDecoder<B> {
    pub fn load(be: &B, src: &dyn WeightSource, cfg: &VqConfig) -> Result<Self> {
        let dh = cfg.code_dim / cfg.heads;
        Ok(Self {
            codebook: src.get("quantizer.vq._codebook.embed", &[cfg.heads, cfg.codes, dh])?,
            decoder: Decoder::load(
                be,
                src,
                "decoder",
                cfg.out_dim,
                cfg.code_dim,
                cfg.down_t,
                cfg.width,
                cfg.depth,
                cfg.dilation_growth_rate,
                cfg.target_dim,
                cfg.external_dim,
            )?,
            cfg: cfg.clone(),
        })
    }

    /// `forward_decoder(use_overall_indices=False)`: tokens `[batch, 16, heads]` →
    /// normalised local features `[batch, 64, out_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        be: &B,
        batch: usize,
        positions: usize,
        tokens: &[usize],
        external: &[f32],
        target: &[f32],
        has_target: &[bool],
        token_mask: &[f32],
    ) -> Vec<f32> {
        let h = self.cfg.heads;
        let dh = self.cfg.code_dim / h;
        let mut codes = Vec::with_capacity(batch * positions * h * dh);
        for (i, &t) in tokens.iter().enumerate() {
            let head = i % h;
            let o = (head * self.cfg.codes + t) * dh;
            codes.extend_from_slice(&self.codebook[o..o + dh]);
        }
        let x = tensor::to_channels_first(&codes, batch, positions, self.cfg.code_dim);
        self.decoder.forward(be, &x, batch, positions, external, target, has_target, token_mask)
    }
}
