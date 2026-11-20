//! `motion encode`: precompute LLM2Vec prompt embeddings for Kimodo.

use anyhow::{Context, Result};
use clap::Args;
use motion_core::backend::Backend;
use motion_text::{Embeddings, TextEncoder};
use std::path::PathBuf;

#[derive(Args, Debug, Clone)]
pub struct EncodeArgs {
    /// Prompts, one per argument; or use --prompts-file.
    pub prompts: Vec<String>,
    /// A text file with one prompt per line.
    #[arg(long)]
    pub prompts_file: Option<PathBuf>,
    /// Meta-Llama-3-8B-Instruct (gated: accept the licence on Hugging Face, then `motion download llama-3-8b-instruct`).
    #[arg(long)]
    pub llama_dir: PathBuf,
    /// The LLM2Vec adapters in order: -mntp, then -mntp-supervised.
    #[arg(long, num_args = 1.., required = true)]
    pub adapters: Vec<PathBuf>,
    /// Embeddings file to create or extend.
    #[arg(long, default_value = "prompts.safetensors")]
    pub out: PathBuf,
}

pub fn run<B: Backend>(be: &B, a: &EncodeArgs) -> Result<()> {
    let mut prompts = a.prompts.clone();
    if let Some(f) = &a.prompts_file {
        prompts.extend(std::fs::read_to_string(f)?.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from));
    }
    let enc = TextEncoder::load(&a.llama_dir, &a.adapters).context("loading the text encoder")?;
    let mut emb = if a.out.exists() { Embeddings::load(&a.out)? } else { Embeddings::default() };
    emb.source = format!("LLM2Vec {:?} over {}", a.adapters, a.llama_dir.display());
    for p in &prompts {
        let t = std::time::Instant::now();
        let v = enc.encode(be, &[motion_text::sanitize_text(p)])?.remove(0);
        emb.push(p, v);
        eprintln!("{:.1} s  {}", t.elapsed().as_secs_f32(), motion_text::sanitize_text(p));
    }
    emb.save(&a.out)?;
    eprintln!("wrote {} ({} prompts)", a.out.display(), emb.prompts.len());
    Ok(())
}
