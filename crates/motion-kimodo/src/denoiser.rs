//! `kimodo/model/{backbone,twostage_denoiser}.py`: the two-stage transformer
//! denoiser. Each stage is a `TransformerEncoderBlock`:
//! `[50 text tokens | time token | heading token | T motion frames]` with the
//! sinusoidal table added to the whole sequence, a post-norm
//! `nn.TransformerEncoder`, and the motion rows projected out. The root stage
//! predicts the 5-d global root; its local form (heading rate, ground
//! velocity, height) replaces the global root in the body stage's input.

use crate::config::KimodoConfig;
use crate::rep::KimodoMotionRep;
use motion_core::backend::{Act, Backend};
use motion_core::weights::WeightSource;
use motion_core::Result;
use motion_nn::{Encoder, Linear};

/// `PositionalEncoding` exactly as PyTorch computes it in float32:
/// `div = 10000^(−2i/d)` (f32 exponent, correctly rounded power), `arg = f32(pos)·div`,
/// `sin` / `cos` of the f32 argument.
pub fn kimodo_pe(max_len: usize, d: usize) -> Vec<f32> {
    let mut pe = vec![0f32; max_len * d];
    // The exponent `-arange(0, d, 2).float() / d` is an f32 division. `torch.pow` is a
    // vectorised f32 power within one ulp of the exact value; the correctly rounded
    // power differs from it on 9 of the 512 divisors at d = 1024 (up to 3e-5 in the
    // table at position 999, invisible after the time MLP).
    let divs: Vec<f32> = (0..d)
        .step_by(2)
        .map(|i| 10000f64.powf((-(i as f32) / d as f32) as f64) as f32)
        .collect();
    for pos in 0..max_len {
        for (k, &div) in divs.iter().enumerate() {
            let a = (pos as f32 * div) as f64;
            pe[pos * d + 2 * k] = a.sin() as f32;
            if 2 * k + 1 < d {
                pe[pos * d + 2 * k + 1] = a.cos() as f32;
            }
        }
    }
    pe
}

pub const PE_LEN: usize = 5000;

/// One stage: `TransformerEncoderBlock`.
pub struct Block<B: Backend> {
    embed_text: Linear<B>,
    time1: Linear<B>,
    time2: Linear<B>,
    input_linear: Linear<B>,
    output_linear: Linear<B>,
    heading: Option<Linear<B>>,
    encoder: Encoder<B>,
    d: usize,
    text_tokens: usize,
    llm_dim: usize,
    use_text_mask: bool,
}

/// Per-sample conditioning shared by both stages.
pub struct Cond<'a> {
    /// `[batch, text_tokens_in, llm_dim]` (Kimodo passes one pooled token).
    pub text: &'a [f32],
    pub text_tokens_in: usize,
    /// `[batch, text_tokens_in]`, 1 = valid.
    pub text_mask: &'a [f32],
    /// Base-schedule timestep per sample (`map_tensor[t]`).
    pub timesteps: &'a [usize],
    pub first_heading: &'a [f32],
    /// `[batch, frames]`, 1 = valid frame.
    pub pad_mask: &'a [f32],
}

impl<B: Backend> Block<B> {
    pub fn load(
        be: &B,
        src: &dyn WeightSource,
        prefix: &str,
        cfg: &KimodoConfig,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let d = cfg.latent_dim;
        let k = |s: &str| format!("{prefix}.{s}");
        Ok(Self {
            embed_text: Linear::load(be, src, &k("embed_text"), cfg.llm_dim(), d, true)?,
            time1: Linear::load(be, src, &k("embed_timestep.time_embed.0"), d, d, true)?,
            time2: Linear::load(be, src, &k("embed_timestep.time_embed.2"), d, d, true)?,
            input_linear: Linear::load(be, src, &k("input_linear"), input_dim, d, true)?,
            output_linear: Linear::load(be, src, &k("output_linear"), d, output_dim, true)?,
            heading: if cfg.input_first_heading_angle {
                Some(Linear::load(be, src, &k("linear_first_heading_angle"), 2, d, true)?)
            } else {
                None
            },
            encoder: Encoder::load(
                be,
                src,
                &k("seqTransEncoder"),
                cfg.num_layers,
                d,
                cfg.num_heads,
                cfg.ff_size,
                cfg.activation,
                1e-5,
            )?,
            d,
            text_tokens: cfg.num_text_tokens,
            llm_dim: cfg.llm_dim(),
            use_text_mask: cfg.use_text_mask,
        })
    }

    /// `x` `[batch, frames, input_dim]` (host) → `[batch, frames, output_dim]`.
    pub fn forward(&self, be: &B, x: &[f32], batch: usize, frames: usize, c: &Cond<'_>, pe: &[f32]) -> Vec<f32> {
        let d = self.d;
        let xb = be.upload(x);
        let xe = be.download(&self.input_linear.apply(be, &xb, batch * frames, Act::None));
        // Text tokens padded (or cropped) to `text_tokens` with zero features.
        let (nt, li) = (self.text_tokens, c.text_tokens_in);
        let mut text = vec![0f32; batch * nt * self.llm_dim];
        let mut tmask = vec![0f32; batch * nt];
        for b in 0..batch {
            for t in 0..li.min(nt) {
                let src = &c.text[(b * li + t) * self.llm_dim..(b * li + t + 1) * self.llm_dim];
                text[(b * nt + t) * self.llm_dim..(b * nt + t + 1) * self.llm_dim].copy_from_slice(src);
                tmask[b * nt + t] = c.text_mask[b * li + t];
            }
        }
        if !self.use_text_mask {
            tmask.fill(1.0);
        }
        let te = be.download(&self.embed_text.apply(be, &be.upload(&text), batch * nt, Act::None));
        let trows: Vec<f32> = c.timesteps.iter().flat_map(|&t| pe[t * d..(t + 1) * d].to_vec()).collect();
        let h = self.time1.apply(be, &be.upload(&trows), batch, Act::Silu);
        let tm = be.download(&self.time2.apply(be, &h, batch, Act::None));
        let hd = self.heading.as_ref().map(|l| {
            let cs: Vec<f32> = c.first_heading.iter().flat_map(|a| [a.cos(), a.sin()]).collect();
            be.download(&l.apply(be, &be.upload(&cs), batch, Act::None))
        });
        let prefix = nt + 1 + hd.is_some() as usize;
        let seq = prefix + frames;
        let mut s = Vec::with_capacity(batch * seq * d);
        let mut km = Vec::with_capacity(batch * seq);
        for b in 0..batch {
            s.extend_from_slice(&te[b * nt * d..(b + 1) * nt * d]);
            s.extend_from_slice(&tm[b * d..(b + 1) * d]);
            km.extend_from_slice(&tmask[b * nt..(b + 1) * nt]);
            km.push(1.0);
            if let Some(hd) = &hd {
                s.extend_from_slice(&hd[b * d..(b + 1) * d]);
                km.push(1.0);
            }
            s.extend_from_slice(&xe[b * frames * d..(b + 1) * frames * d]);
            km.extend_from_slice(&c.pad_mask[b * frames..(b + 1) * frames]);
        }
        for b in 0..batch {
            for p in 0..seq {
                for k in 0..d {
                    s[(b * seq + p) * d + k] += pe[p * d + k];
                }
            }
        }
        let sb = be.upload(&s);
        self.encoder.forward(be, &sb, batch, seq, Some(&be.upload(&km)));
        let o = be.download(&sb);
        let motion: Vec<f32> = (0..batch).flat_map(|b| o[(b * seq + prefix) * d..(b + 1) * seq * d].to_vec()).collect();
        be.download(&self.output_linear.apply(be, &be.upload(&motion), batch * frames, Act::None))
    }
}

/// `TwostageDenoiser` (motion_mask_mode `concat`).
pub struct TwoStage<B: Backend> {
    pub root: Block<B>,
    pub body: Block<B>,
    pub pe: Vec<f32>,
    dim: usize,
    root_dim: usize,
}

impl<B: Backend> TwoStage<B> {
    pub fn load(be: &B, src: &dyn WeightSource, cfg: &KimodoConfig, rep: &KimodoMotionRep) -> Result<Self> {
        let dim = rep.dim();
        let root_dim = rep.global_root_dim();
        let local_dim = dim - root_dim + rep.local_root_dim();
        Ok(Self {
            root: Block::load(be, src, "root_model", cfg, 2 * dim, root_dim)?,
            body: Block::load(be, src, "body_model", cfg, local_dim + dim, dim - root_dim)?,
            pe: kimodo_pe(PE_LEN, cfg.latent_dim),
            dim,
            root_dim,
        })
    }

    /// The clean prediction for `x` `[batch, frames, dim]` (normalised).
    /// `mask` / `observed` are `[batch, frames, dim]` (zeros when unconstrained).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        be: &B,
        rep: &KimodoMotionRep,
        x: &[f32],
        batch: usize,
        frames: usize,
        mask: &[f32],
        observed: &[f32],
        c: &Cond<'_>,
    ) -> Vec<f32> {
        let (dim, rd) = (self.dim, self.root_dim);
        let xm: Vec<f32> = x
            .iter()
            .zip(mask.iter().zip(observed))
            .map(|(&v, (&m, &o))| v * (1.0 - m) + o * m)
            .collect();
        let rows = batch * frames;
        let ext = motion_core::tensor::cat_cols(&[(&xm, dim), (mask, dim)], rows);
        let root = self.root.forward(be, &ext, batch, frames, c, &self.pe);
        let mut local = Vec::with_capacity(rows * rep.local_root_dim());
        for b in 0..batch {
            let len = c.pad_mask[b * frames..(b + 1) * frames].iter().filter(|&&m| m > 0.5).count();
            local.extend(rep.global_root_to_local_root(&root[b * frames * rd..(b + 1) * frames * rd], true, len));
        }
        let body_x = motion_core::tensor::slice_cols(&xm, dim, rd, dim);
        let ext2 = motion_core::tensor::cat_cols(
            &[(&local, rep.local_root_dim()), (&body_x, dim - rd), (mask, dim)],
            rows,
        );
        let body = self.body.forward(be, &ext2, batch, frames, c, &self.pe);
        motion_core::tensor::cat_cols(&[(&root, rd), (&body, dim - rd)], rows)
    }
}
