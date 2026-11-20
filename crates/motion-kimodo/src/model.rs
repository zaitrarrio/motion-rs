//! `kimodo/model/kimodo_model.py`: the sampling loop (`_generate`), single
//! and batched generation (`__call__`) and multi-prompt sequences with
//! blended transitions (`_multiprompt`). The motion-correction post-process
//! (a C++ extension upstream) is not ported.

use crate::cfg::CfgType;
use crate::config::KimodoConfig;
use crate::constraints::{build_condition_data, Constraint};
use crate::denoiser::{Cond, TwoStage};
use crate::diffusion::Schedule;
use crate::output::MotionOutput;
use crate::rep::{Decoded, KimodoMotionRep};
use motion_core::backend::Backend;
use motion_core::rng::Rng;
use motion_core::weights::{SafetensorsDir, WeightSource};
use motion_core::{Error, Result};
use std::path::Path;

/// One prompt's conditioning: a pooled LLM2Vec embedding, or the empty prompt
/// (zero features, as upstream encodes `""`).
#[derive(Debug, Clone)]
pub enum TextCond {
    Embedding(Vec<f32>),
    Empty,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// One prompt per segment (`multi_prompt`) or one prompt for the whole clip.
    pub prompts: Vec<TextCond>,
    /// Frames per prompt.
    pub num_frames: Vec<usize>,
    pub steps: usize,
    pub cfg: CfgType,
    pub seed: u64,
    /// Initial noise `[frames, dim]` per segment, overriding the seeded draw (parity tests).
    pub noise: Option<Vec<Vec<f32>>>,
    pub constraints: Vec<Constraint>,
    pub multi_prompt: bool,
    pub num_transition_frames: usize,
    /// Heading of frame 0 in radians (0 faces +Z).
    pub first_heading_angle: f32,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompts: vec![TextCond::Empty],
            num_frames: vec![150],
            steps: 100,
            cfg: CfgType::Separated(2.0, 2.0),
            seed: 0,
            noise: None,
            constraints: Vec::new(),
            multi_prompt: false,
            num_transition_frames: 5,
            first_heading_angle: 0.0,
        }
    }
}

pub struct KimodoModel<B: Backend> {
    pub cfg: KimodoConfig,
    pub rep: KimodoMotionRep,
    pub denoiser: TwoStage<B>,
}

/// Progress callback: `(segment, step, steps)`.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, usize);

impl<B: Backend> KimodoModel<B> {
    /// A checkpoint directory (`config.yaml`, `model.safetensors`, `stats/motion/…`).
    pub fn load(be: &B, dir: &Path) -> Result<Self> {
        let cfg = KimodoConfig::load(dir)?;
        let rep = KimodoMotionRep::load(cfg.skeleton.clone(), cfg.fps, &dir.join("stats/motion"))?;
        let src = SafetensorsDir::open(&dir.join("model.safetensors"))?;
        Self::from_parts(be, cfg, rep, &src)
    }

    pub fn from_parts(be: &B, cfg: KimodoConfig, rep: KimodoMotionRep, src: &dyn WeightSource) -> Result<Self> {
        let denoiser = TwoStage::load(be, src, &cfg, &rep)?;
        Ok(Self { cfg, rep, denoiser })
    }

    pub fn fps(&self) -> f32 {
        self.cfg.fps
    }

    /// `_generate` for one sample: `steps` DDIM steps from `noise` (`[frames, dim]`).
    /// `cond` is `(observed, mask)` normalised, or `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        be: &B,
        text: &TextCond,
        frames: usize,
        steps: usize,
        cfg: CfgType,
        cond: Option<(&[f32], &[f32])>,
        first_heading: f32,
        noise: Vec<f32>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Vec<f32> {
        let d = self.rep.dim();
        let llm = self.cfg.llm_dim();
        let (feat, len) = match text {
            TextCond::Embedding(v) => (v.clone(), 1.0),
            TextCond::Empty => (vec![0.0; llm], 0.0),
        };
        let zeros = vec![0f32; frames * d];
        let copies = cfg.conditions();
        let nb = copies.len();
        let mut text_b = Vec::with_capacity(nb * llm);
        let mut tmask = Vec::with_capacity(nb);
        let mut mask_b = Vec::with_capacity(nb * frames * d);
        let mut obs_b = Vec::with_capacity(nb * frames * d);
        for &(t_on, c_on) in copies {
            if t_on {
                text_b.extend_from_slice(&feat);
                tmask.push(len);
            } else {
                text_b.extend(std::iter::repeat_n(0.0, llm));
                tmask.push(0.0);
            }
            match cond {
                Some((obs, mask)) => {
                    obs_b.extend_from_slice(obs);
                    if c_on { mask_b.extend_from_slice(mask) } else { mask_b.extend_from_slice(&zeros) }
                }
                None => {
                    obs_b.extend_from_slice(&zeros);
                    mask_b.extend_from_slice(&zeros);
                }
            }
        }
        let pad = vec![1f32; nb * frames];
        let headings = vec![first_heading; nb];
        let sched = Schedule::new(self.cfg.num_base_steps, steps);
        let mut x = noise;
        for (k, i) in (0..steps).rev().enumerate() {
            let ts = vec![sched.timesteps[i]; nb];
            let c = Cond {
                text: &text_b,
                text_tokens_in: 1,
                text_mask: &tmask,
                timesteps: &ts,
                first_heading: &headings,
                pad_mask: &pad,
            };
            let xb: Vec<f32> = (0..nb).flat_map(|_| x.iter().copied()).collect();
            let out = self.denoiser.forward(be, &self.rep, &xb, nb, frames, &mask_b, &obs_b, &c);
            let per: Vec<Vec<f32>> = out.chunks(frames * d).map(|c| c.to_vec()).collect();
            let x0 = cfg.combine(&per);
            x = sched.ddim(i, &x, &x0);
            progress(k + 1, steps);
        }
        x
    }

    fn noise_for(&self, req: &GenerateRequest, seg: usize, frames: usize, rng: &mut Rng) -> Result<Vec<f32>> {
        let d = self.rep.dim();
        match &req.noise {
            Some(n) => {
                let v = n.get(seg).ok_or_else(|| Error::Config(format!("no noise for segment {seg}")))?;
                if v.len() != frames * d {
                    return Err(Error::Config(format!("segment {seg}: noise has {} values, need {}", v.len(), frames * d)));
                }
                Ok(v.clone())
            }
            None => Ok(rng.normals(frames * d)),
        }
    }

    /// `Kimodo.__call__(…, num_samples=None)` for one sample: the whole request
    /// (one prompt, or `multi_prompt` segments) → motion on the output skeleton.
    pub fn generate(&self, be: &B, req: &GenerateRequest, progress: Progress<'_>) -> Result<MotionOutput> {
        if req.prompts.is_empty() || req.prompts.len() != req.num_frames.len() {
            return Err(Error::Config("one frame count per prompt".into()));
        }
        let mut rng = Rng::new(req.seed);
        let rep = &self.rep;
        let sk = &rep.skeleton;
        if !req.multi_prompt || req.prompts.len() == 1 {
            let frames = req.num_frames[0];
            let cond = if req.constraints.is_empty() {
                None
            } else {
                let data = build_condition_data(sk, &req.constraints)?;
                let (mut obs, mask) = rep.create_conditions(&data, frames)?;
                rep.normalize(&mut obs);
                Some((obs, mask))
            };
            let noise = self.noise_for(req, 0, frames, &mut rng)?;
            let x = self.sample(
                be,
                &req.prompts[0],
                frames,
                req.steps,
                req.cfg,
                cond.as_ref().map(|(o, m)| (o.as_slice(), m.as_slice())),
                req.first_heading_angle,
                noise,
                &mut |s, n| progress(0, s, n),
            );
            return Ok(MotionOutput::from_decoded(rep.inverse(&x, true), sk, self.fps()));
        }
        // _multiprompt.
        let d = rep.dim();
        let k = req.num_transition_frames.max(1);
        let mut generated: Vec<Vec<f32>> = Vec::new(); // unnormalised features
        let mut current = 0usize;
        for (seg, (text, &n)) in req.prompts.iter().zip(&req.num_frames).enumerate() {
            let base: Vec<Constraint> = req.constraints.iter().map(|c| c.crop_move(current, current + n)).collect();
            let data = build_condition_data(sk, &base)?;
            let (mut obs, mut mask) = rep.create_conditions(&data, n)?;
            let mut frames = n;
            let mut heading = req.first_heading_angle;
            let mut carry: Option<(Vec<f32>, [f32; 2])> = None;
            if let Some(latest) = generated.pop() {
                let lt = latest.len() / d;
                let kk = k.min(lt);
                generated.push(latest[..(lt - kk) * d].to_vec());
                let latest_frames = latest[(lt - kk) * d..].to_vec();
                let last: Decoded = rep.inverse(&latest_frames, false);
                let smooth: Vec<[f32; 2]> =
                    (0..kk).map(|t| [last.smooth_root_pos[t * 3], last.smooth_root_pos[t * 3 + 2]]).collect();
                let tf: Vec<usize> = (0..kk).collect();
                let fb = Constraint::full_body(
                    sk,
                    tf.clone(),
                    last.posed_joints.clone(),
                    last.global_rot_mats.clone(),
                    Some(smooth.clone()),
                );
                let ee = Constraint::end_effector(
                    sk,
                    "end-effector",
                    ["LeftHand", "RightHand", "LeftFoot", "RightFoot"].iter().map(|s| s.to_string()).collect(),
                    tf,
                    last.posed_joints.clone(),
                    last.global_rot_mats.clone(),
                    Some(smooth.clone()),
                );
                let td = build_condition_data(sk, &[fb, ee])?;
                let (ot, mt) = rep.create_conditions(&td, kk)?;
                obs = [ot, obs].concat();
                mask = [mt, mask].concat();
                let off = smooth[0];
                rep.translate_2d(&mut obs, -off[0], -off[1]);
                for (o, m) in obs.iter_mut().zip(&mask) {
                    *o *= m;
                }
                frames += kk;
                heading = sk.heading_angle_kimodo(&last.posed_joints)[0];
                carry = Some((latest_frames, off));
            }
            rep.normalize(&mut obs);
            let noise = self.noise_for(req, seg, frames, &mut rng)?;
            let x = self.sample(
                be,
                text,
                frames,
                req.steps,
                req.cfg,
                Some((&obs, &mask)),
                heading,
                noise,
                &mut |s, nn| progress(seg, s, nn),
            );
            let mut motion = rep.stats.unnormalized(&x);
            if let Some((latest_frames, off)) = carry {
                rep.translate_2d(&mut motion, off[0], off[1]);
                let kk = latest_frames.len() / d;
                let (trans, rest) = motion.split_at(kk * d);
                let mut blended = vec![0f32; kk * d];
                for t in 0..kk {
                    let a = if kk == 1 { 1.0 } else { 1.0 - t as f32 / (kk - 1) as f32 };
                    for c in 0..d {
                        blended[t * d + c] = latest_frames[t * d + c] * a + (1.0 - a) * trans[t * d + c];
                    }
                }
                generated.push(blended);
                motion = rest.to_vec();
            }
            generated.push(motion);
            current += n;
        }
        let all: Vec<f32> = generated.concat();
        Ok(MotionOutput::from_decoded(rep.inverse(&all, false), sk, self.fps()))
    }
}
