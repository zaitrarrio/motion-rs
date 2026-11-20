//! PEFT LoRA adapters (`adapter_config.json` + `adapter_model.safetensors`),
//! merged into the base weights at load: `W += scale · B · A` with
//! `scale = lora_alpha / r`, or `lora_alpha / √r` with `use_rslora`
//! (`peft.tuners.lora.LoraLayer.update_layer`). Dropout is inactive at inference.

use motion_core::weights::{SafetensorsDir, WeightSource};
use motion_core::{Error, Result};
use std::path::Path;

pub struct Adapter {
    src: SafetensorsDir,
    pub r: usize,
    pub scale: f32,
    pub targets: Vec<String>,
    prefix: String,
}

impl Adapter {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json"))?)?;
        if cfg["peft_type"].as_str() != Some("LORA") {
            return Err(Error::Config(format!("{}: not a LoRA adapter", dir.display())));
        }
        let r = cfg["r"].as_u64().ok_or_else(|| Error::Config("adapter without r".into()))? as usize;
        let alpha = cfg["lora_alpha"].as_f64().unwrap_or(r as f64) as f32;
        let rs = cfg["use_rslora"].as_bool().unwrap_or(false);
        let scale = if rs { alpha / (r as f32).sqrt() } else { alpha / r as f32 };
        let targets = cfg["target_modules"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        let src = SafetensorsDir::open(&dir.join("adapter_model.safetensors"))?;
        Ok(Self { src, r, scale, targets, prefix: "base_model.model.".into() })
    }

    /// Add this adapter's delta for `module` (e.g. `layers.3.self_attn.q_proj`) to
    /// `w` `[out, inp]`; modules the adapter does not target are left alone.
    pub fn merge_into(&self, w: &mut [f32], module: &str, out: usize, inp: usize) -> Result<()> {
        let name = module.rsplit('.').next().unwrap_or(module);
        if !self.targets.iter().any(|t| t == name) {
            return Ok(());
        }
        let ka = format!("{}{module}.lora_A.weight", self.prefix);
        let kb = format!("{}{module}.lora_B.weight", self.prefix);
        if !self.src.contains(&ka) {
            return Err(Error::Weights(format!("adapter is missing `{ka}`")));
        }
        let a = self.src.get(&ka, &[self.r, inp])?;
        let b = self.src.get(&kb, &[out, self.r])?;
        merge(w, &a, &b, out, inp, self.r, self.scale);
        Ok(())
    }
}

/// `w[o, i] += scale · Σ_k b[o, k] · a[k, i]`.
pub fn merge(w: &mut [f32], a: &[f32], b: &[f32], out: usize, inp: usize, r: usize, scale: f32) {
    for o in 0..out {
        let row = &mut w[o * inp..(o + 1) * inp];
        for k in 0..r {
            let c = scale * b[o * r + k];
            if c == 0.0 {
                continue;
            }
            let ar = &a[k * inp..(k + 1) * inp];
            for (wv, av) in row.iter_mut().zip(ar) {
                *wv += c * av;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn merge_is_scaled_outer_product() {
        let mut w = vec![1.0, 0.0, 0.0, 1.0];
        super::merge(&mut w, &[1.0, 2.0], &[3.0, 4.0], 2, 2, 1, 0.5);
        assert_eq!(w, vec![2.5, 3.0, 2.0, 5.0]);
    }
}
