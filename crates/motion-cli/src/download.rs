//! `motion download`: a Hugging Face snapshot of one of the known model
//! trees into a local directory (the same layout the loaders expect), with
//! `hf download` as the documented alternative. Honors `HF_TOKEN` for gated
//! repositories.

use anyhow::{Context, Result, anyhow};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The model trees this binary knows how to load.
pub const KNOWN: &[(&str, &str, &str)] = &[
    ("kimodo-soma-rp", "nvidia/Kimodo-SOMA-RP-v1.2", "Kimodo on SOMA, Bones Rigplay 1 (default Kimodo model)"),
    ("kimodo-soma-seed", "nvidia/Kimodo-SOMA-SEED-v1.1", "Kimodo on SOMA, BONES-SEED"),
    ("kimodo-g1-rp", "nvidia/Kimodo-G1-RP-v1", "Kimodo on the Unitree G1 skeleton"),
    ("motionbricks", "nvidia/MotionBricks", "MotionBricks SOMA (soma/) and G1 (g1/) checkpoints"),
    ("llm2vec-mntp", "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "LLM2Vec adapter 1 + tokenizer"),
    ("llm2vec-supervised", "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "LLM2Vec adapter 2"),
    ("llama-3-8b-instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "the gated LLM2Vec base (needs HF_TOKEN)"),
];

pub fn resolve_repo(name: &str) -> String {
    KNOWN.iter().find(|k| k.0 == name).map(|k| k.1.to_string()).unwrap_or_else(|| name.to_string())
}

fn agent() -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(30));
    if let Ok(p) = ureq::Proxy::try_from_env() {
        b = b.proxy(p);
    }
    b.build()
}

fn with_token(req: ureq::Request) -> ureq::Request {
    match std::env::var("HF_TOKEN") {
        Ok(t) if !t.is_empty() => req.set("Authorization", &format!("Bearer {t}")),
        _ => req,
    }
}

#[derive(serde::Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(serde::Deserialize)]
struct ModelInfo {
    siblings: Vec<Sibling>,
}

/// Download every file of `repo` whose path starts with one of `include`
/// (all when empty) into `out`, skipping files already present with the
/// right size.
pub fn snapshot(repo: &str, out: &Path, include: &[String]) -> Result<Vec<PathBuf>> {
    let agent = agent();
    let info: ModelInfo = with_token(agent.get(&format!("https://huggingface.co/api/models/{repo}?blobs=true")))
        .call()
        .with_context(|| format!("listing {repo} (gated repositories need HF_TOKEN)"))?
        .into_json()?;
    let mut done = Vec::new();
    for s in info.siblings {
        if !include.is_empty() && !include.iter().any(|p| s.rfilename.starts_with(p.as_str())) {
            continue;
        }
        let dst = out.join(&s.rfilename);
        if let (Ok(meta), Some(size)) = (std::fs::metadata(&dst), s.size) {
            if meta.len() == size {
                eprintln!("  have {}", s.rfilename);
                done.push(dst);
                continue;
            }
        }
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        eprintln!("  get  {} ({})", s.rfilename, s.size.map(human).unwrap_or_default());
        let url = format!("https://huggingface.co/{repo}/resolve/main/{}", s.rfilename);
        let resp = with_token(agent.get(&url)).call().with_context(|| format!("downloading {url}"))?;
        let tmp = dst.with_extension("part");
        {
            let mut f = std::fs::File::create(&tmp)?;
            let mut r = resp.into_reader();
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = r.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                f.write_all(&buf[..n])?;
            }
        }
        if let Some(size) = s.size {
            let got = std::fs::metadata(&tmp)?.len();
            if got != size {
                return Err(anyhow!("{}: got {got} bytes, expected {size}", s.rfilename));
            }
        }
        std::fs::rename(&tmp, &dst)?;
        done.push(dst);
    }
    Ok(done)
}

fn human(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{b} B")
    }
}
