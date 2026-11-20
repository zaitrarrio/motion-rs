//! MotionBricks (NVlabs, SIGGRAPH 2026) in Rust: the root module, the masked
//! pose transformer and the VQ-VAE decoder for the SOMA (30 joints) and
//! Unitree G1 (29 DoF) rigs, the release ONNX graph's tensor interface
//! ([`model`]), and a runtime that turns placed context and target poses into
//! continuous motion ([`engine`]).
//!
//! The reference is `tools/onnx/models` of NVlabs/motionbricks: every network
//! reads the checkpoint's own state-dict keys, and `tests/` hold the
//! PyTorch fixtures of `scripts/reference/motionbricks_fixtures.py`.

pub mod config;
pub mod engine;
pub mod io;
pub mod model;
pub mod nets;
pub mod rep;
pub mod rig_g1;

pub use config::{BricksConfig, Rig};
pub use model::{G1Inputs, G1Outputs, MotionBricks, SomaInputs, SomaOutputs, Trace};
