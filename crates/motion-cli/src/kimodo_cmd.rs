//! `motion generate`: Kimodo text (and constraint) to motion, the CLI of
//! `kimodo/scripts/generate.py`.

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use motion_core::backend::Backend;
use motion_kimodo::{CfgType, GenerateRequest, KimodoModel, TextCond};
use motion_text::Embeddings;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args, Debug, Clone)]
pub struct GenerateArgs {
    /// One prompt, or several separated by periods (a multi-prompt sequence).
    /// Empty: constraint-only motion.
    #[arg(default_value = "")]
    pub prompt: String,
    /// A Kimodo checkpoint directory (`motion download kimodo-soma-rp`).
    #[arg(long)]
    pub model_dir: PathBuf,
    /// Precomputed prompt embeddings (`motion encode`).
    #[arg(long)]
    pub embeddings: Option<PathBuf>,
    /// Seconds per prompt: one value for all, or one per prompt ("3 4.5").
    #[arg(long, default_value = "5.0")]
    pub duration: String,
    #[arg(long, default_value_t = 100)]
    pub steps: usize,
    /// nocfg, regular or separated.
    #[arg(long, default_value = "separated")]
    pub cfg_type: String,
    /// One weight (regular) or text and constraint weights (separated).
    #[arg(long, num_args = 0..=2, default_values_t = [2.0, 2.0])]
    pub cfg_weight: Vec<f32>,
    /// A constraints.json saved by the Kimodo demo or the studio.
    #[arg(long)]
    pub constraints: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 5)]
    pub num_transition_frames: usize,
    /// Heading of the first frame, degrees (0 faces +Z).
    #[arg(long, default_value_t = 0.0)]
    pub heading: f32,
    /// Output stem: writes <out>.npz and, with --bvh, <out>.bvh.
    #[arg(long, default_value = "output")]
    pub out: PathBuf,
    #[arg(long)]
    pub bvh: bool,
    /// Export the BVH with the standard T-pose as rest pose.
    #[arg(long)]
    pub bvh_standard_tpose: bool,
}

/// Split a prompt into segments the way `generate.py` does.
pub fn split_prompts(prompt: &str) -> Vec<String> {
    let v: Vec<String> =
        prompt.split('.').map(str::trim).filter(|s| !s.is_empty()).map(|s| format!("{s}.")).collect();
    if v.is_empty() { vec![String::new()] } else { v }
}

pub fn durations(s: &str, n: usize, fps: f32) -> Result<Vec<usize>> {
    let parts: Vec<f32> = s.split_whitespace().map(|v| v.parse::<f32>()).collect::<std::result::Result<_, _>>()?;
    let secs = match parts.len() {
        1 => vec![parts[0]; n],
        k if k == n => parts,
        k => bail!("{k} durations for {n} prompts"),
    };
    Ok(secs.iter().map(|d| (d * fps) as usize).collect())
}

pub fn resolve_text(prompt: &str, emb: Option<&Embeddings>, dim: usize) -> Result<TextCond> {
    if prompt.trim().is_empty() {
        return Ok(TextCond::Empty);
    }
    let e = emb.ok_or_else(|| {
        anyhow!("prompt `{prompt}` needs a text embedding: pass --embeddings (motion encode), or an empty prompt")
    })?;
    let v = e.get(prompt).ok_or_else(|| {
        anyhow!("`{}` is not in the embeddings file; it holds: {}", motion_text::sanitize_text(prompt), e.prompts.join(" | "))
    })?;
    if v.len() != dim {
        bail!("embedding has {} values, the model expects {dim}", v.len());
    }
    Ok(TextCond::Embedding(v.to_vec()))
}

pub fn run<B: Backend>(be: &B, a: &GenerateArgs) -> Result<()> {
    let t0 = Instant::now();
    let model = KimodoModel::load(be, &a.model_dir).with_context(|| format!("loading {}", a.model_dir.display()))?;
    eprintln!("loaded {} in {:.1} s", a.model_dir.display(), t0.elapsed().as_secs_f32());
    let emb = a.embeddings.as_ref().map(|p| Embeddings::load(p)).transpose()?;
    let prompts = split_prompts(&a.prompt);
    let frames = durations(&a.duration, prompts.len(), model.fps())?;
    let texts = prompts.iter().map(|p| resolve_text(p, emb.as_ref(), model.cfg.llm_dim())).collect::<Result<Vec<_>>>()?;
    let constraints = match &a.constraints {
        Some(p) => motion_kimodo::load_constraints_json(p, &model.rep.skeleton)?,
        None => Vec::new(),
    };
    let cfg = CfgType::parse(&a.cfg_type, &a.cfg_weight).map_err(|e| anyhow!(e))?;
    let req = GenerateRequest {
        multi_prompt: texts.len() > 1,
        prompts: texts,
        num_frames: frames.clone(),
        steps: a.steps,
        cfg,
        seed: a.seed,
        noise: None,
        constraints,
        num_transition_frames: a.num_transition_frames,
        first_heading_angle: a.heading.to_radians(),
    };
    let t1 = Instant::now();
    let mut last = Instant::now();
    let out = model.generate(be, &req, &mut |seg, s, n| {
        if last.elapsed().as_secs_f32() > 2.0 || s == n {
            eprintln!("  segment {} step {s}/{n} ({:.1} s)", seg + 1, t1.elapsed().as_secs_f32());
            last = Instant::now();
        }
    })?;
    eprintln!("generated {} frames ({:.1} s at {} fps) in {:.1} s", out.frames, out.frames as f32 / out.fps, out.fps, t1.elapsed().as_secs_f32());
    let npz = a.out.with_extension("npz");
    out.write_npz(&npz)?;
    eprintln!("wrote {}", npz.display());
    if a.bvh {
        let p = a.out.with_extension("bvh");
        std::fs::write(&p, out.to_bvh(a.bvh_standard_tpose))?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
