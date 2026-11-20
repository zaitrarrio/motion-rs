//! The bidirectional Llama of LLM2Vec (`kimodo/model/llm2vec/models/bidirectional_llama.py`):
//! Llama decoder layers with the causal mask removed, run one layer at a time
//! from the mmapped shards (an 8 B model never has to be resident), with the
//! LoRA adapters merged into each layer's weights as it is loaded.

use crate::lora::Adapter;
use motion_core::backend::{Act, Backend};
use motion_core::weights::{SafetensorsDir, WeightSource};
use motion_core::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    #[serde(default, rename = "_name_or_path")]
    pub name_or_path: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl LlamaConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if c.rope_scaling.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(Error::Config("rope_scaling is not supported (Meta-Llama-3-8B-Instruct has none)".into()));
        }
        Ok(c)
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

pub struct Llama {
    pub cfg: LlamaConfig,
    src: SafetensorsDir,
    prefix: String,
    adapters: Vec<Adapter>,
}

impl Llama {
    /// `dir` holds `config.json` and the base shards (`model*.safetensors`, HF key
    /// names under `model.`); `config` may come from elsewhere (the LLM2Vec repo's).
    pub fn open(weights: &[PathBuf], cfg: LlamaConfig, adapters: Vec<Adapter>) -> Result<Self> {
        let src = SafetensorsDir::open_many(weights)?;
        let prefix = if src.contains("model.embed_tokens.weight") { "model." } else { "" }.to_string();
        Ok(Self { cfg, src, prefix, adapters })
    }

    fn key(&self, s: &str) -> String {
        format!("{}{s}", self.prefix)
    }

    /// A linear of layer `i` with every adapter merged: `W + Σ scale·B·A`.
    fn merged(&self, i: usize, module: &str, out: usize, inp: usize) -> Result<Vec<f32>> {
        let mut w = self.src.get(&self.key(&format!("layers.{i}.{module}.weight")), &[out, inp])?;
        for a in &self.adapters {
            a.merge_into(&mut w, &format!("layers.{i}.{module}"), out, inp)?;
        }
        Ok(w)
    }

    /// `[S, hidden]` last hidden states (after the final RMSNorm) for one sequence.
    pub fn forward<B: Backend>(&self, be: &B, ids: &[u32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (h, s) = (c.hidden_size, ids.len());
        let (nh, nkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim());
        let ff = c.intermediate_size;
        let rows: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
        let x = be.upload(&self.src.get_rows(&self.key("embed_tokens.weight"), &rows)?);
        // RoPE tables: inv_freq = 1 / θ^(2i/d) in f32, freqs = pos · inv_freq (transformers' default rope).
        let half = hd / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / c.rope_theta.powf((2 * i) as f32 / hd as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(s * half), Vec::with_capacity(s * half));
        for p in 0..s {
            for f in &inv {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let (cos, sin) = (be.upload(&cos), be.upload(&sin));
        let qkv_w = (nh + 2 * nkv) * hd;
        let normed = be.alloc(s * h);
        let qkv = be.alloc(s * qkv_w);
        let attn = be.alloc(s * nh * hd);
        let tmp = be.alloc(s * h);
        let gu = be.alloc(s * 2 * ff);
        let act = be.alloc(s * ff);
        for i in 0..c.num_hidden_layers {
            let wq = self.merged(i, "self_attn.q_proj", nh * hd, h)?;
            let wk = self.merged(i, "self_attn.k_proj", nkv * hd, h)?;
            let wv = self.merged(i, "self_attn.v_proj", nkv * hd, h)?;
            let fused: Vec<f32> = [wq, wk, wv].concat();
            let w_qkv = be.weight(&fused, qkv_w, h);
            drop(fused);
            let w_o = be.weight(&self.merged(i, "self_attn.o_proj", h, nh * hd)?, h, nh * hd);
            let gate_up: Vec<f32> = [self.merged(i, "mlp.gate_proj", ff, h)?, self.merged(i, "mlp.up_proj", ff, h)?].concat();
            let w_gu = be.weight(&gate_up, 2 * ff, h);
            drop(gate_up);
            let w_down = be.weight(&self.merged(i, "mlp.down_proj", h, ff)?, h, ff);
            let ln1 = be.upload(&self.src.get(&self.key(&format!("layers.{i}.input_layernorm.weight")), &[h])?);
            let ln2 = be.upload(&self.src.get(&self.key(&format!("layers.{i}.post_attention_layernorm.weight")), &[h])?);

            be.rms_norm(&x, 0, s, h, &ln1, c.rms_norm_eps, &normed, 0);
            be.linear(&normed, 0, s, &w_qkv, None, Act::None, &qkv, 0);
            be.rope(&qkv, s, nh, nkv, hd, &cos, &sin);
            be.attention(&qkv, 1, s, nh, nkv, hd, None, &attn);
            be.linear(&attn, 0, s, &w_o, None, Act::None, &tmp, 0);
            be.axpby(&x, 0, 1.0, &tmp, 0, 1.0, s * h);
            be.rms_norm(&x, 0, s, h, &ln2, c.rms_norm_eps, &normed, 0);
            be.linear(&normed, 0, s, &w_gu, None, Act::None, &gu, 0);
            be.swiglu(&gu, s, ff, &act);
            be.linear(&act, 0, s, &w_down, None, Act::None, &tmp, 0);
            be.axpby(&x, 0, 1.0, &tmp, 0, 1.0, s * h);
        }
        let fin = be.upload(&self.src.get(&self.key("norm.weight"), &[h])?);
        be.rms_norm(&x, 0, s, h, &fin, c.rms_norm_eps, &normed, 0);
        Ok(be.download(&normed))
    }
}
