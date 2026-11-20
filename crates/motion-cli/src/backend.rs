//! Backend selection for the commands.

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// The f32 reference backend (rayon).
    #[default]
    Cpu,
}
