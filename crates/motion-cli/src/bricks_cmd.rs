//! `motion inbetween`: MotionBricks transitions between poses taken from clips
//! (BVH for SOMA, CSV for G1), chained the way the Inbetweener demo does.

use anyhow::{bail, Context, Result};
use clap::Args;
use motion_bricks::engine::{Chain, GenerateConfig, SomaEngine, G1Engine};
use motion_bricks::io::{every_nth, read_bvh, read_g1_csv, write_bvh, write_mujoco_csv};
use motion_bricks::{MotionBricks, Rig};
use motion_core::backend::Backend;
use std::path::PathBuf;

#[derive(Args, Debug, Clone)]
pub struct InbetweenArgs {
    /// A MotionBricks `soma/` or `g1/` checkpoint directory (`motion download motionbricks`).
    #[arg(long)]
    pub model_dir: PathBuf,
    /// The clip the poses come from: a SOMA BVH or a G1 CSV.
    #[arg(long)]
    pub clip: PathBuf,
    /// Source frame rate of a G1 CSV.
    #[arg(long, default_value_t = 120.0)]
    pub csv_fps: f32,
    /// Context: the four 30 Hz frames starting here (after resampling to 30 Hz).
    #[arg(long, default_value_t = 0)]
    pub context_frame: usize,
    /// Targets in order: each the first of up to four 30 Hz frames.
    #[arg(long, value_delimiter = ',', required = true)]
    pub target_frames: Vec<usize>,
    /// Target poses per target, 1 to 4.
    #[arg(long, default_value_t = 4)]
    pub target_poses: usize,
    /// Frames before the end of each generation at which the next one starts.
    #[arg(long, default_value_t = 0)]
    pub transition_early: usize,
    /// Allowed token counts, e.g. 8-14.
    #[arg(long, default_value = "8-14")]
    pub tokens: String,
    #[arg(long, default_value = "inbetween.bvh")]
    pub out: PathBuf,
}

fn token_range(s: &str) -> Result<Vec<usize>> {
    let (a, b) = s.split_once('-').map_or((s, s), |x| x);
    let (a, b): (usize, usize) = (a.trim().parse()?, b.trim().parse()?);
    if a > b {
        bail!("bad token range {s}");
    }
    Ok((a..=b).collect())
}

pub fn run<B: Backend>(be: &B, a: &InbetweenArgs) -> Result<()> {
    let model = MotionBricks::load(be, &a.model_dir).with_context(|| format!("loading {}", a.model_dir.display()))?;
    let cfg = GenerateConfig { allowed_token_counts: token_range(&a.tokens)?, ..Default::default() };
    let tp = a.target_poses.clamp(1, 4);
    match model.cfg.rig {
        Rig::Soma => {
            let clip = read_bvh(&a.clip)?.rebase("Hips")?;
            let fps = 1.0 / clip.frame_time;
            let frames = every_nth(&clip.poses, fps);
            let eng = SomaEngine::new(&model, clip.skeleton.clone())?;
            let get = |f: usize, n: usize| -> Result<Vec<_>> {
                if f + n > frames.len() {
                    bail!("frame {f}+{n} beyond the clip's {} frames at 30 Hz", frames.len());
                }
                Ok(frames[f..f + n].to_vec())
            };
            let mut chain = Chain { played: get(a.context_frame, 4)?, contacts: vec![[1.0; 4]; 4] };
            for &t in &a.target_frames {
                let n = chain.step(be, &eng, &get(t, tp)?, a.transition_early, &cfg)?;
                eprintln!("target at frame {t}: {n} frames");
            }
            std::fs::write(&a.out, write_bvh(&clip.skeleton, &chain.played, 30.0))?;
        }
        Rig::G1 => {
            let clip = read_g1_csv(&a.clip, a.csv_fps)?;
            let frames = every_nth(&clip.qpos, a.csv_fps);
            let eng = G1Engine::new(&model)?;
            let mut played: Vec<Vec<f32>> = frames[a.context_frame..a.context_frame + 4].to_vec();
            for &t in &a.target_frames {
                let ctx = played[played.len() - 4..].to_vec();
                let m = eng.generate(be, &ctx, &frames[t..t + tp], &cfg)?;
                let keep = m.qpos.len().saturating_sub(a.transition_early).max(1);
                played.extend(m.qpos[1..keep].iter().cloned());
                eprintln!("target at frame {t}: {} tokens, {} frames", m.tokens, keep - 1);
            }
            std::fs::write(&a.out, write_mujoco_csv(&played))?;
        }
    }
    eprintln!("wrote {}", a.out.display());
    Ok(())
}
