//! Kimodo's text conditioner in Rust: LLM2Vec (McGill-NLP) over
//! Meta-Llama-3-8B-Instruct, as `kimodo/model/llm2vec` runs it — the
//! `-mntp` adapter merged into the base, the `-mntp-supervised` adapter on
//! top, bidirectional attention, the Llama-3-Instruct user header around the
//! prompt, and the mean over the prompt's own tokens (plus `<|eot_id|>`).
//! One 4096-d vector per prompt.
//!
//! [`Embeddings`] is the precomputed form: Kimodo runs from it without the
//! 16 GB encoder (`motion encode` writes one, the studio's gallery reads it).

pub mod embeddings;
pub mod llama;
pub mod lora;
pub mod sanitize;

pub use embeddings::Embeddings;
pub use sanitize::sanitize_text;

use motion_core::backend::Backend;
use motion_core::{Error, Result};
use std::path::{Path, PathBuf};

/// LLM2Vec's separator between the instruction and the text (`_convert_to_str`).
pub const SEPARATOR: &str = "!@#$%^&*()";
pub const LLAMA3_USER_HEADER: &str = "<|start_header_id|>user<|end_header_id|>\n\n";
pub const LLAMA3_EOT: &str = "<|eot_id|>";

pub struct TextEncoder {
    pub llama: llama::Llama,
    tokenizer: tokenizers::Tokenizer,
    pub max_length: usize,
}

/// The tokens of one prompt and how many of the last ones are pooled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokens {
    pub ids: Vec<u32>,
    pub pooled: usize,
}

impl TextEncoder {
    /// `base_dir`: Meta-Llama-3-8B-Instruct (`model-*.safetensors`); `adapters`:
    /// the LLM2Vec `-mntp` then `-mntp-supervised` directories. The tokenizer and
    /// config come from the first adapter directory when it has them.
    pub fn load(base_dir: &Path, adapters: &[PathBuf]) -> Result<Self> {
        let cfg_path = adapters
            .iter()
            .map(|a| a.join("config.json"))
            .find(|p| p.exists())
            .unwrap_or_else(|| base_dir.join("config.json"));
        let cfg = llama::LlamaConfig::load(&cfg_path)?;
        let tok_path = adapters
            .iter()
            .map(|a| a.join("tokenizer.json"))
            .chain(std::iter::once(base_dir.join("tokenizer.json")))
            .find(|p| p.exists())
            .ok_or_else(|| Error::Config("no tokenizer.json in the adapter or base directories".into()))?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Config(format!("{}: {e}", tok_path.display())))?;
        let ads = adapters.iter().map(|a| lora::Adapter::load(a)).collect::<Result<Vec<_>>>()?;
        let mut shards: Vec<PathBuf> = std::fs::read_dir(base_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        let llama = llama::Llama::open(&shards, cfg, ads)?;
        Ok(Self { llama, tokenizer, max_length: 512 })
    }

    pub fn from_parts(llama: llama::Llama, tokenizer: tokenizers::Tokenizer) -> Self {
        Self { llama, tokenizer, max_length: 512 }
    }

    /// `LLM2Vec.encode([text])` tokenisation: `prepare_for_tokenization` adds the
    /// Llama-3 user header when the model is Meta-Llama-3-8B-Instruct, `tokenize`
    /// encodes the whole string with BOS and pools the tokens of the part after
    /// the separator (the text and `<|eot_id|>`), encoded without special tokens.
    pub fn tokenize(&self, text: &str) -> Result<Tokens> {
        let inner = format!("{SEPARATOR}{text}");
        let prepared = if self.llama.cfg.name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct" {
            format!("{LLAMA3_USER_HEADER}{}{LLAMA3_EOT}", inner.trim())
        } else {
            inner
        };
        let parts: Vec<&str> = prepared.split(SEPARATOR).collect();
        let original = parts.concat();
        let embed = parts.get(1).copied().unwrap_or("");
        let enc = |s: &str, special: bool| -> Result<Vec<u32>> {
            Ok(self.tokenizer.encode(s, special).map_err(|e| Error::Config(format!("tokenizer: {e}")))?.get_ids().to_vec())
        };
        let mut ids = enc(&original, true)?;
        ids.truncate(self.max_length);
        let pooled = enc(embed, false)?.len().min(ids.len());
        Ok(Tokens { ids, pooled })
    }

    /// One pooled vector per prompt. Empty prompts give zeros, as Kimodo does.
    pub fn encode<B: Backend>(&self, be: &B, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        let h = self.llama.cfg.hidden_size;
        prompts
            .iter()
            .map(|p| {
                if p.trim().is_empty() {
                    return Ok(vec![0.0; h]);
                }
                let t = self.tokenize(p)?;
                let hidden = self.llama.forward(be, &t.ids)?;
                Ok(mean_last(&hidden, h, t.pooled))
            })
            .collect()
    }
}

/// Mean of the last `n` rows of `[S, h]` (LLM2Vec `pooling_mode="mean"` with left padding).
pub fn mean_last(hidden: &[f32], h: usize, n: usize) -> Vec<f32> {
    let s = hidden.len() / h;
    let mut out = vec![0f64; h];
    for r in s - n..s {
        for k in 0..h {
            out[k] += hidden[r * h + k] as f64;
        }
    }
    out.iter().map(|v| (v / n as f64) as f32).collect()
}
