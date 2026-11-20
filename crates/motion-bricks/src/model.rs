//! The batched inference graph of the release ONNX models
//! (`mmm_inference_class.predict` / `motion_inference.predict` with the
//! exporter's config: one masked-token pass, argmax tokens, the dynamic
//! root conversion, `from_height_constraint_violation` for SOMA and
//! `from_pose_module` for G1), and the rig wrappers around it
//! (`BatchedSOMA`'s T-pose change, `BatchedG1`'s qpos encode / decode).

use crate::config::{BricksConfig, Rig, FRAMES_PER_TOKEN, MAX_FRAMES, MAX_TOKENS, MIN_TOKENS};
use crate::nets::{PoseNet, RootNet, VqDecoder};
use crate::rep::{self, Layout, Stats};
use crate::rig_g1::G1Rig;
use motion_core::assets;
use motion_core::backend::Backend;
use motion_core::geometry::{self as geo, Mat3};
use motion_core::npy::Npy;
use motion_core::tensor;
use motion_core::weights::{Prefixed, SafetensorsDir, WeightSource};
use motion_core::{Error, Result};
use std::path::{Path, PathBuf};

pub const SLOTS: usize = 2 * FRAMES_PER_TOKEN;
pub const COUNT_CLASSES: usize = MAX_TOKENS - MIN_TOKENS + 1;

/// One request in the ONNX tensor layout of the SOMA model (`soma_n1.json`).
#[derive(Debug, Clone)]
pub struct SomaInputs {
    /// `[8, 5]`: root position, cos / sin heading.
    pub global_root_values: Vec<f32>,
    pub has_global_root_values: [bool; SLOTS],
    /// `[8, 4]`: heading rate, ground velocity x / z, root height.
    pub local_root_values: Vec<f32>,
    pub has_local_root_values: [bool; SLOTS],
    /// `[8, 267]`: 29 child positions relative to the root's ground point, 30 6-D rotations.
    pub local_poses: Vec<f32>,
    pub has_local_poses: [bool; SLOTS],
    /// Allowed token counts 6..=16.
    pub allowed: [bool; COUNT_CLASSES],
}

#[derive(Debug, Clone)]
pub struct SomaOutputs {
    /// `[64, 366]` in SOMA model space (global layout); the first `4·tokens` frames are valid.
    pub pred_global_motions: Vec<f32>,
    pub pred_num_tokens: usize,
}

/// One request of the G1 model (`unitree_g1_n1.json`).
#[derive(Debug, Clone)]
pub struct G1Inputs {
    /// `[4, 36]` each: root xyz, root quaternion xyzw, 29 joint angles.
    pub context_qpos: Vec<f32>,
    pub target_qpos: Vec<f32>,
    pub has_global_root_values: [bool; SLOTS],
    pub has_local_root_values: [bool; SLOTS],
    pub has_local_poses: [bool; SLOTS],
    pub allowed: [bool; COUNT_CLASSES],
}

#[derive(Debug, Clone)]
pub struct G1Outputs {
    /// `[64, 36]`.
    pub pred_qpos: Vec<f32>,
    /// `[64, 4]`: left heel, left toe, right heel, right toe (unclamped).
    pub pred_foot_contacts: Vec<f32>,
    pub pred_num_tokens: usize,
}

/// Every intermediate of one request, for parity tests and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub global_root_norm: Vec<f32>,
    pub local_root_norm: Vec<f32>,
    pub local_poses_norm: Vec<f32>,
    pub count_logits: Vec<f32>,
    pub pred_num_tokens: usize,
    pub root_global: Vec<f32>,
    pub pred_global_root: Vec<f32>,
    pub pred_local_root: Vec<f32>,
    pub pose_logits: Vec<f32>,
    pub pose_tokens: Vec<usize>,
    pub pose_cond: Vec<f32>,
    pub has_pose_cond: Vec<bool>,
    pub pred_local_poses: Vec<f32>,
    pub pred_global_pre_reapply: Vec<f32>,
    pub used_root_module: bool,
}

/// The joint-space core of either rig: raw root / pose inputs in model space →
/// global features `[64, global]`.
pub struct MotionBricks<B: Backend> {
    pub cfg: BricksConfig,
    pub stats: Stats,
    pub root: RootNet<B>,
    pub pose: PoseNet<B>,
    pub vq: VqDecoder<B>,
    pub g1: Option<G1Rig>,
}

/// The component files of one HF tree (`nvidia/MotionBricks/{soma,g1}`).
pub struct TreeFiles {
    pub pose: PathBuf,
    pub root: PathBuf,
    pub vqvae: PathBuf,
    pub mean: PathBuf,
    pub std: PathBuf,
}

/// Find the three checkpoints under a `soma/` or `g1/` download (any depth).
pub fn find_tree(dir: &Path) -> Result<(Rig, TreeFiles)> {
    let mut all = Vec::new();
    walk(dir, &mut all)?;
    let pick = |pred: &dyn Fn(&str) -> bool| -> Result<PathBuf> {
        all.iter()
            .find(|p| pred(&p.to_string_lossy()))
            .cloned()
            .ok_or_else(|| Error::Weights(format!("{}: incomplete MotionBricks tree", dir.display())))
    };
    let st = |s: &str, part: &str| s.ends_with(".safetensors") && s.contains(part);
    let pose = pick(&|s| st(s, "pose"))?;
    let root = pick(&|s| st(s, "root"))?;
    let vqvae = pick(&|s| st(s, "vqvae"))?;
    let comp = pose.parent().and_then(|p| p.parent()).ok_or_else(|| Error::Weights("bad tree".into()))?;
    let mean = comp.join("stats/motion/mean.npy");
    let std = comp.join("stats/motion/std.npy");
    let rig = if pose.to_string_lossy().contains("soma") { Rig::Soma } else { Rig::G1 };
    Ok((rig, TreeFiles { pose, root, vqvae, mean, std }))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            walk(&p, out)?;
        } else {
            out.push(p);
        }
    }
    out.sort();
    Ok(())
}

impl<B: Backend> MotionBricks<B> {
    /// From the three component sources (prefixes already stripped) and the dual statistics.
    pub fn new(
        be: &B,
        cfg: BricksConfig,
        pose: &dyn WeightSource,
        root: &dyn WeightSource,
        vqvae: &dyn WeightSource,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Self> {
        if mean.len() != cfg.layout.dual() {
            return Err(Error::Weights(format!("stats have {} columns, expected {}", mean.len(), cfg.layout.dual())));
        }
        let g1 = if cfg.rig == Rig::G1 { Some(G1Rig::new()) } else { None };
        Ok(Self {
            stats: Stats::new(mean, &std),
            root: RootNet::load(be, root, &cfg.root)?,
            pose: PoseNet::load(be, pose, &cfg.pose)?,
            vq: VqDecoder::load(be, vqvae, &cfg.vq)?,
            g1,
            cfg,
        })
    }

    /// From a downloaded `nvidia/MotionBricks` `soma/` or `g1/` directory.
    pub fn load(be: &B, dir: &Path) -> Result<Self> {
        let (rig, f) = find_tree(dir)?;
        let cfg = match rig {
            Rig::Soma => BricksConfig::soma(),
            Rig::G1 => BricksConfig::g1(),
        };
        let prefix = match rig {
            Rig::Soma => "mmm_net.",
            Rig::G1 => "backbone_net.",
        };
        let pose = SafetensorsDir::open(&f.pose)?;
        let root = SafetensorsDir::open(&f.root)?;
        let vq = SafetensorsDir::open(&f.vqvae)?;
        let mean = Npy::read(&f.mean)?.to_f32();
        let std = Npy::read(&f.std)?.to_f32();
        Self::new(
            be,
            cfg,
            &Prefixed { inner: &pose, prefix: prefix.into() },
            &Prefixed { inner: &root, prefix: prefix.into() },
            &Prefixed { inner: &vq, prefix: "pose_net.".into() },
            mean,
            std,
        )
    }

    pub fn layout(&self) -> Layout {
        self.cfg.layout
    }

    /// The joint-space graph for one request. `local_poses` is `[8, ric + rot]`
    /// (without the hip height). Returns the unnormalised global features
    /// `[64, global]` and the token count.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_features(
        &self,
        be: &B,
        global_root: &[f32],
        has_g: &[bool; SLOTS],
        local_root: &[f32],
        has_l: &[bool; SLOTS],
        local_poses: &[f32],
        has_p: &[bool; SLOTS],
        allowed: &[bool; COUNT_CLASSES],
        trace: Option<&mut Trace>,
    ) -> (Vec<f32>, usize) {
        let lay = self.cfg.layout;
        let st = &self.stats;
        let fpt = FRAMES_PER_TOKEN;
        let fps = self.cfg.fps;
        // _extract_initial_root_info: recentre x / z on frame 0, remember its heading.
        let off = [global_root[0], global_root[2]];
        let heading0 = global_root[4].atan2(global_root[3]);
        let mut g = global_root.to_vec();
        for t in 0..SLOTS {
            g[t * 5] -= off[0];
            g[t * 5 + 2] -= off[1];
        }
        let heights: Vec<f32> = (0..SLOTS).map(|t| g[t * 5 + 1]).collect();
        st.normalize(&mut g, &rep::range(0, 5));
        let mut l = local_root.to_vec();
        st.normalize(&mut l, &rep::range(5, 9));
        let lpw = lay.local_pose_feature();
        let pose_cols: Vec<usize> = (3..3 + lpw).map(|c| lay.dual_of_local(c)).collect();
        let mut p = Vec::with_capacity(SLOTS * lpw);
        for t in 0..SLOTS {
            p.push(heights[t]);
            p.extend_from_slice(&local_poses[t * (lpw - 1)..(t + 1) * (lpw - 1)]);
        }
        st.normalize(&mut p, &pose_cols);
        let allowed_i: Vec<i32> = allowed.iter().map(|&a| a as i32).collect();

        // Root module.
        let r = self.root.forward(be, 1, &g, has_g, &l, has_l, &p, has_p, Some(&allowed_i), false);
        let n = r.pred_num_tokens[0];
        let frames = n * fpt;
        let mut local = rep::global_root_to_local(st, &r.global_root, frames, fps);
        let last = frames - 1;
        for c in 0..4 {
            if has_l[SLOTS - 1] {
                local[last * 4 + c] = l[(SLOTS - 1) * 4 + c];
            }
        }

        // Pose module: all tokens masked (one pass, cosine schedule at step 0).
        let mt = MAX_TOKENS;
        let heads = self.cfg.pose.heads;
        let mut pose_cond = vec![0f32; MAX_FRAMES * lpw];
        let mut has_pose = vec![false; MAX_FRAMES];
        pose_cond[..fpt * lpw].copy_from_slice(&p[..fpt * lpw]);
        has_pose[..fpt].copy_from_slice(&has_p[..fpt]);
        for i in 0..fpt {
            let f = n * fpt - fpt + i;
            for c in 0..lpw {
                pose_cond[f * lpw + c] += p[(fpt + i) * lpw + c];
            }
            has_pose[f] = has_pose[f] || has_p[fpt + i];
        }
        let root_cond = tensor::gather_cols(&r.global_root, 5, &[0, 2, 3, 4]);
        let masked = vec![self.pose.mask_id(); mt * heads];
        let logits = self.pose.forward(be, 1, &masked, &root_cond, &pose_cond, &has_pose, &[n]);
        let codes = self.cfg.pose.codes;
        let tokens: Vec<usize> = logits.chunks(codes).map(tensor::argmax).collect();

        // Decoder.
        let external = tensor::gather_cols(&local, 4, &[1, 2]);
        let token_mask: Vec<f32> = (0..mt).map(|q| if q < n { 1.0 } else { 0.0 }).collect();
        let recon = self.vq.decode(be, 1, mt, &tokens, &external, &pose_cond, &has_pose, &token_mask);
        let mut global = rep::local_to_global(&lay, st, &recon, fps);
        let mut pred_local = recon.clone();
        st.unnormalize(&mut pred_local, &(0..lay.local()).map(|c| lay.dual_of_local(c)).collect::<Vec<_>>());

        let mut used_root = false;
        if self.cfg.rig == Rig::Soma {
            // from_height_constraint_violation.
            let mut groot = r.global_root.clone();
            st.unnormalize(&mut groot, &rep::range(0, 5));
            let (mut e_pose, mut e_root, mut cnt) = (0f32, 0f32, 0f32);
            for i in 0..fpt {
                let f = (n - 1) * fpt + i;
                let hm = if has_g[fpt + i] { 1.0 } else { 0.0 };
                let target_h = heights[fpt + i];
                e_pose += (global[f * lay.global() + 1] - target_h).abs() * hm;
                e_root += (groot[f * 5 + 1] - target_h).abs() * hm;
                cnt += hm;
            }
            let (e_pose, e_root) = (e_pose / cnt, e_root / cnt);
            if e_root < e_pose && e_pose > 0.05 {
                used_root = true;
                let mut lroot = local.clone();
                st.unnormalize(&mut lroot, &rep::range(5, 9));
                for f in 0..MAX_FRAMES {
                    global[f * lay.global()..f * lay.global() + 5].copy_from_slice(&groot[f * 5..f * 5 + 5]);
                    pred_local[f * lay.local()..f * lay.local() + 4].copy_from_slice(&lroot[f * 4..f * 4 + 4]);
                }
            }
        }
        let pre = global.clone();
        // _reapply_initial_root_info.
        let gw = lay.global();
        for f in 0..MAX_FRAMES {
            let o = &mut global[f * gw..(f + 1) * gw];
            let a = o[4].atan2(o[3]) + heading0;
            o[3] = a.cos();
            o[4] = a.sin();
            o[0] += off[0];
            o[2] += off[1];
        }
        if let Some(tr) = trace {
            tr.global_root_norm = g;
            tr.local_root_norm = l;
            tr.local_poses_norm = p;
            tr.count_logits = r.logits;
            tr.pred_num_tokens = n;
            tr.root_global = r.global_root.clone();
            tr.pred_global_root = r.global_root;
            tr.pred_local_root = local;
            tr.pose_logits = logits;
            tr.pose_tokens = tokens;
            tr.pose_cond = pose_cond;
            tr.has_pose_cond = has_pose;
            tr.pred_local_poses = pred_local;
            tr.pred_global_pre_reapply = pre;
            tr.used_root_module = used_root;
        }
        (global, n)
    }

    /// `BatchedSOMA.forward` for one request.
    pub fn predict_soma(&self, be: &B, x: &SomaInputs, trace: Option<&mut Trace>) -> SomaOutputs {
        assert_eq!(self.cfg.rig, Rig::Soma, "predict_soma on a G1 model");
        let lay = self.cfg.layout;
        let j = lay.joints;
        let ric = lay.ric();
        let w = ric + lay.rot();
        let mut lp = x.local_poses.clone();
        for t in 0..SLOTS {
            for jj in 0..j {
                let o = t * w + ric + jj * 6;
                let m = soma_change_tpose(&lp[o..o + 6], jj, true);
                lp[o..o + 6].copy_from_slice(&geo::matrix_to_cont6d(&m));
            }
        }
        let (mut out, n) = self.predict_features(
            be,
            &x.global_root_values,
            &x.has_global_root_values,
            &x.local_root_values,
            &x.has_local_root_values,
            &lp,
            &x.has_local_poses,
            &x.allowed,
            trace,
        );
        let gw = lay.global();
        let rs = lay.global_rot_start();
        for f in 0..MAX_FRAMES {
            for jj in 0..j {
                let o = f * gw + rs + jj * 6;
                let m = soma_change_tpose(&out[o..o + 6], jj, false);
                out[o..o + 6].copy_from_slice(&geo::matrix_to_cont6d(&m));
            }
        }
        SomaOutputs { pred_global_motions: out, pred_num_tokens: n }
    }

    /// `BatchedG1.forward` for one request.
    pub fn predict_g1(&self, be: &B, x: &G1Inputs, trace: Option<&mut Trace>) -> G1Outputs {
        let rig = self.g1.as_ref().expect("predict_g1 on a SOMA model");
        let (g, l, p) = rig.preprocess(&x.context_qpos, &x.target_qpos);
        let (feat, n) = self.predict_features(
            be,
            &g,
            &x.has_global_root_values,
            &l,
            &x.has_local_root_values,
            &p,
            &x.has_local_poses,
            &x.allowed,
            trace,
        );
        let lay = self.cfg.layout;
        let (qpos, contacts) = rig.postprocess(&feat, &lay);
        G1Outputs { pred_qpos: qpos, pred_foot_contacts: contacts, pred_num_tokens: n }
    }
}

/// `change_t_pose_global_mats_onnx_compatible`: `R · O_from · O_toᵀ`
/// (capture → standard on the way in, standard → capture on the way out).
pub fn soma_change_tpose(c6: &[f32], joint: usize, into_model: bool) -> Mat3 {
    let m = geo::cont6d_to_matrix(c6);
    let cap = geo::mat_at(&assets::MB_SOMA_CAPTURE_OFFSETS, joint);
    let std = geo::mat_at(&assets::MB_SOMA_STANDARD_OFFSETS, joint);
    let (from, to) = if into_model { (cap, std) } else { (std, cap) };
    geo::matmul(&geo::matmul(&m, &from), &geo::transpose(&to))
}
