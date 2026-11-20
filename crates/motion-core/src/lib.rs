//! motion-rs core: the op vocabulary every backend implements, checkpoint
//! access (safetensors, `.npy`, synthetic), rotation geometry, skeletons and
//! forward kinematics. No model lives here; `motion-kimodo` and
//! `motion-bricks` build on it, `motion-cpu` / `motion-wgpu` implement it.

pub mod assets;
pub mod backend;
pub mod geometry;
pub mod npy;
pub mod rng;
pub mod skeleton;
pub mod tensor;
pub mod weights;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("weights: {0}")]
    Weights(String),
    #[error("shape: {0}")]
    Shape(String),
    #[error("config: {0}")]
    Config(String),
    #[error("backend: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
