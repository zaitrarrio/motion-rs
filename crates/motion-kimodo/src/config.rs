//! `config.yaml` of a Kimodo checkpoint (the Hydra tree instantiated by
//! `kimodo.model.load_model`): only the keys inference needs.

use motion_core::backend::Act;
use motion_core::skeleton::Skeleton;
use motion_core::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct Root {
    #[serde(default = "default_base_steps")]
    num_base_steps: usize,
    #[serde(default)]
    cfg_type: Option<String>,
    denoiser: DenoiserYaml,
}

fn default_base_steps() -> usize {
    1000
}

#[derive(Debug, Clone, Deserialize)]
struct DenoiserYaml {
    #[serde(rename = "_target_", default)]
    target: String,
    motion_rep: MotionRepYaml,
    motion_mask_mode: String,
    llm_shape: Vec<usize>,
    use_text_mask: bool,
    latent_dim: usize,
    ff_size: usize,
    num_layers: usize,
    num_heads: usize,
    activation: String,
    #[serde(default)]
    norm_first: bool,
    #[serde(default)]
    input_first_heading_angle: bool,
    #[serde(default)]
    num_text_tokens_override: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct MotionRepYaml {
    #[serde(rename = "_target_", default)]
    target: String,
    fps: f32,
    skeleton: SkeletonYaml,
}

#[derive(Debug, Clone, Deserialize)]
struct SkeletonYaml {
    #[serde(rename = "_target_")]
    target: String,
}

/// The resolved configuration of a Kimodo model.
#[derive(Debug, Clone)]
pub struct KimodoConfig {
    pub num_base_steps: usize,
    /// `nocfg`, `regular` or `separated` (the checkpoint's default).
    pub cfg_type: String,
    pub fps: f32,
    pub skeleton: Skeleton,
    /// Always `concat` for the released models: the constraint mask is
    /// concatenated to the noisy motion.
    pub motion_mask_mode: String,
    /// `[tokens, dim]` of the text features as the encoder emits them (`[1, 4096]`).
    pub llm_shape: Vec<usize>,
    pub use_text_mask: bool,
    pub latent_dim: usize,
    pub ff_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub activation: Act,
    pub norm_first: bool,
    pub input_first_heading_angle: bool,
    /// Text tokens the prefix is padded to (50 for the released models).
    pub num_text_tokens: usize,
}

impl KimodoConfig {
    pub fn from_yaml(text: &str) -> Result<Self> {
        let root: Root = serde_yaml::from_str(text).map_err(|e| Error::Config(format!("config.yaml: {e}")))?;
        let d = root.denoiser;
        if !d.target.is_empty() && !d.target.ends_with("TwostageDenoiser") {
            return Err(Error::Config(format!("unsupported denoiser `{}`", d.target)));
        }
        if !d.motion_rep.target.is_empty() && !d.motion_rep.target.ends_with("KimodoMotionRep") {
            return Err(Error::Config(format!(
                "unsupported motion representation `{}` (only KimodoMotionRep is ported)",
                d.motion_rep.target
            )));
        }
        if d.motion_mask_mode != "concat" {
            return Err(Error::Config(format!("unsupported motion_mask_mode `{}`", d.motion_mask_mode)));
        }
        if d.norm_first {
            return Err(Error::Config("norm_first transformer layers are not ported".into()));
        }
        let skeleton = match d.motion_rep.skeleton.target.rsplit('.').next().unwrap_or("") {
            "SOMASkeleton30" => Skeleton::soma30(),
            "SOMASkeleton77" => Skeleton::soma77(),
            "G1Skeleton34" => Skeleton::g1_34(),
            "SMPLXSkeleton22" => Skeleton::smplx22(),
            other => return Err(Error::Config(format!("unknown skeleton `{other}`"))),
        };
        let activation = match d.activation.as_str() {
            "gelu" => Act::GeluErf,
            "relu" => Act::Relu,
            other => return Err(Error::Config(format!("unknown activation `{other}`"))),
        };
        if d.llm_shape.len() != 2 {
            return Err(Error::Config(format!("llm_shape must be [tokens, dim], got {:?}", d.llm_shape)));
        }
        Ok(Self {
            num_base_steps: root.num_base_steps,
            cfg_type: root.cfg_type.unwrap_or_else(|| "nocfg".into()),
            fps: d.motion_rep.fps,
            skeleton,
            motion_mask_mode: d.motion_mask_mode,
            num_text_tokens: d.num_text_tokens_override.unwrap_or(d.llm_shape[0]),
            llm_shape: d.llm_shape,
            use_text_mask: d.use_text_mask,
            latent_dim: d.latent_dim,
            ff_size: d.ff_size,
            num_layers: d.num_layers,
            num_heads: d.num_heads,
            activation,
            norm_first: d.norm_first,
            input_first_heading_angle: d.input_first_heading_angle,
        })
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.yaml");
        let text = std::fs::read_to_string(&p).map_err(|e| Error::Config(format!("{}: {e}", p.display())))?;
        Self::from_yaml(&text)
    }

    pub fn llm_dim(&self) -> usize {
        self.llm_shape[1]
    }

    /// A shrunk configuration for tests (same layout, tiny widths).
    pub fn tiny(skeleton: Skeleton) -> Self {
        Self {
            num_base_steps: 1000,
            cfg_type: "separated".into(),
            fps: 30.0,
            skeleton,
            motion_mask_mode: "concat".into(),
            llm_shape: vec![1, 16],
            use_text_mask: false,
            latent_dim: 32,
            ff_size: 64,
            num_layers: 2,
            num_heads: 2,
            activation: Act::GeluErf,
            norm_first: false,
            input_first_heading_angle: true,
            num_text_tokens: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = r#"
_target_: kimodo.model.Kimodo
num_base_steps: 1000
cfg_type: separated
denoiser:
  _target_: kimodo.model.twostage_denoiser.TwostageDenoiser
  motion_rep:
    _target_: kimodo.motion_rep.KimodoMotionRep
    fps: 30
    stats_path: ${oc.select:checkpoint_dir}/stats/motion/
    skeleton:
      _target_: kimodo.skeleton.SOMASkeleton30
  motion_mask_mode: concat
  llm_shape:
  - 1
  - 4096
  use_text_mask: false
  latent_dim: 1024
  ff_size: 2048
  num_layers: 16
  num_heads: 8
  activation: gelu
  dropout: 0.0
  pe_dropout: 0.0
  norm_first: false
  input_first_heading_angle: true
  num_text_tokens_override: 50
  ckpt_path: ${oc.select:checkpoint_dir}/model.safetensors
"#;

    #[test]
    fn parses_released_config() {
        let c = KimodoConfig::from_yaml(YAML).unwrap();
        assert_eq!(c.latent_dim, 1024);
        assert_eq!(c.num_text_tokens, 50);
        assert_eq!(c.skeleton.nbjoints(), 30);
        assert_eq!(c.activation, Act::GeluErf);
        assert_eq!(c.cfg_type, "separated");
    }
}
