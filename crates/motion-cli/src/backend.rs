//! Backend selection for the commands.

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// The f32 reference backend (rayon).
    #[default]
    Cpu,
    /// WebGPU (Vulkan / Metal / DX12). Run `motion check --backend wgpu` on a
    /// card before trusting it there. A CPU rasterizer is refused unless
    /// MOTION_ALLOW_SOFTWARE=1.
    Wgpu,
}
