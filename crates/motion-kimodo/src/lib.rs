//! Kimodo in Rust: the two-stage transformer denoiser, DDIM sampling with
//! classifier-free guidance, the smooth-root motion representation with its
//! kinematic constraints, and the NPZ / BVH writers. Every module names the
//! upstream Python (`nv-tlabs/kimodo`) it mirrors.
//!
//! The text encoder (LLM2Vec over Llama-3-8B-Instruct) is not here: prompts
//! arrive as 4096-d embeddings ([`TextCond::Embedding`]) or as the empty
//! prompt, which upstream encodes as an all-zero feature.

pub mod cfg;
pub mod config;
pub mod constraints;
pub mod denoiser;
pub mod diffusion;
pub mod model;
pub mod output;
pub mod rep;
pub mod smooth_root;
pub mod stats;

pub use cfg::CfgType;
pub use config::KimodoConfig;
pub use constraints::{load_constraints_json, Constraint};
pub use model::{GenerateRequest, KimodoModel, TextCond};
pub use output::MotionOutput;
pub use rep::KimodoMotionRep;
