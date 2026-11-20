//! Precomputed prompt embeddings: one safetensors file holding `embeddings`
//! `[n, dim]` (f32) and the sanitised prompts in its metadata (`prompts`, a
//! JSON list), so Kimodo can run without the text encoder.

use crate::sanitize::sanitize_text;
use motion_core::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct Embeddings {
    pub prompts: Vec<String>,
    pub vectors: Vec<Vec<f32>>,
    /// Free-form provenance (encoder, adapters).
    pub source: String,
}

impl Embeddings {
    pub fn dim(&self) -> usize {
        self.vectors.first().map_or(0, |v| v.len())
    }

    pub fn push(&mut self, prompt: &str, v: Vec<f32>) {
        let p = sanitize_text(prompt);
        if let Some(i) = self.prompts.iter().position(|x| *x == p) {
            self.vectors[i] = v;
        } else {
            self.prompts.push(p);
            self.vectors.push(v);
        }
    }

    /// The embedding of `prompt` after sanitising, if present.
    pub fn get(&self, prompt: &str) -> Option<&[f32]> {
        let p = sanitize_text(prompt);
        self.prompts.iter().position(|x| *x == p).map(|i| self.vectors[i].as_slice())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let d = self.dim();
        let flat: Vec<u8> = self.vectors.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![self.prompts.len(), d], &flat)
            .map_err(|e| Error::Weights(e.to_string()))?;
        let mut meta = HashMap::new();
        meta.insert("prompts".to_string(), serde_json::to_string(&self.prompts)?);
        meta.insert("source".to_string(), self.source.clone());
        let bytes = safetensors::serialize(vec![("embeddings".to_string(), view)], Some(meta))
            .map_err(|e| Error::Weights(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let m = meta.metadata().clone().unwrap_or_default();
        let prompts: Vec<String> = serde_json::from_str(m.get("prompts").map_or("[]", |s| s.as_str()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor("embeddings").map_err(|e| Error::Weights(e.to_string()))?;
        let shape = t.shape().to_vec();
        if shape.len() != 2 || shape[0] != prompts.len() {
            return Err(Error::Weights(format!("{}: embeddings {shape:?} for {} prompts", path.display(), prompts.len())));
        }
        let all = motion_core::weights::bytes_to_f32(t.dtype(), t.data())?;
        let vectors = all.chunks(shape[1]).map(|c| c.to_vec()).collect();
        Ok(Self { prompts, vectors, source: m.get("source").cloned().unwrap_or_default() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut e = Embeddings { source: "test".into(), ..Default::default() };
        e.push("a person walks", vec![1.0, 2.0]);
        e.push("Jump", vec![3.0, 4.0]);
        let p = std::env::temp_dir().join(format!("motion-emb-{}.safetensors", std::process::id()));
        e.save(&p).unwrap();
        let back = Embeddings::load(&p).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(back.prompts, vec!["A person walks.", "Jump."]);
        assert_eq!(back.get(" a person   walks.").unwrap(), &[1.0, 2.0]);
        assert_eq!(back.source, "test");
    }
}
