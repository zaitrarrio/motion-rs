//! Network dimensions. The release values are those `build_soma` / `build_g1`
//! in `tools/onnx/models/{soma,g1}/model.py` hard-code; `tiny_*` are the shrunk
//! networks of the committed random-weight fixtures.

use crate::rep::Layout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rig {
    Soma,
    G1,
}

#[derive(Debug, Clone)]
pub struct RootNetConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layers_shared: usize,
    pub n_layers_root_token: usize,
    pub pose_feat_dim: usize,
    pub local_root_feat_dim: usize,
    pub global_root_feat_dim: usize,
    pub mlp_layers: usize,
    pub width: usize,
    pub depth: usize,
    pub dilation_growth_rate: usize,
    pub down_t: usize,
    pub local_pose_dim: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub frames_per_token: usize,
}

#[derive(Debug, Clone)]
pub struct PoseNetConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layers: usize,
    /// Codebook heads and codes per head.
    pub heads: usize,
    pub codes: usize,
    pub code_dim: usize,
    pub root_feat_width: usize,
    pub pose_feat_width: usize,
    pub token_length_feat_width: usize,
    pub mlp_layers: usize,
    /// `root_without_hip_height` of the global layout: x, z, cos, sin.
    pub root_cond_dim: usize,
    pub local_pose_dim: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub frames_per_token: usize,
}

#[derive(Debug, Clone)]
pub struct VqConfig {
    pub heads: usize,
    pub codes: usize,
    pub code_dim: usize,
    pub width: usize,
    pub depth: usize,
    pub dilation_growth_rate: usize,
    pub down_t: usize,
    /// The local layout width the decoder reconstructs.
    pub out_dim: usize,
    pub target_dim: usize,
    /// `root_without_hip_height_without_heading` of the local layout: vel_x, vel_z.
    pub external_dim: usize,
}

#[derive(Debug, Clone)]
pub struct BricksConfig {
    pub rig: Rig,
    pub layout: Layout,
    pub root: RootNetConfig,
    pub pose: PoseNetConfig,
    pub vq: VqConfig,
    pub fps: f32,
}

pub const MIN_TOKENS: usize = 6;
pub const MAX_TOKENS: usize = 16;
pub const FRAMES_PER_TOKEN: usize = 4;
pub const MAX_FRAMES: usize = MAX_TOKENS * FRAMES_PER_TOKEN;
pub const BOUNDARY_FRAMES: usize = FRAMES_PER_TOKEN - 1;
pub const FPS: f32 = 30.0;

#[allow(clippy::too_many_arguments)]
fn build(
    rig: Rig,
    joints: usize,
    heads: usize,
    codes: usize,
    code_dim: usize,
    pose: (usize, usize, usize, usize, usize, usize),
    root: (usize, usize, usize, usize, usize, usize, usize, usize, usize),
    vq: (usize, usize),
) -> BricksConfig {
    let layout = Layout::new(joints);
    let lpd = layout.local_pose_feature();
    let (pe, ph, pl, rfw, pfw, tlw) = pose;
    let (re, rh, rs, rt, pfd, lrf, grf, rw, rd) = root;
    BricksConfig {
        rig,
        layout,
        root: RootNetConfig {
            n_embd: re,
            n_head: rh,
            n_layers_shared: rs,
            n_layers_root_token: rt,
            pose_feat_dim: pfd,
            local_root_feat_dim: lrf,
            global_root_feat_dim: grf,
            mlp_layers: 2,
            width: rw,
            depth: rd,
            dilation_growth_rate: 3,
            down_t: 2,
            local_pose_dim: lpd,
            min_tokens: MIN_TOKENS,
            max_tokens: MAX_TOKENS,
            frames_per_token: FRAMES_PER_TOKEN,
        },
        pose: PoseNetConfig {
            n_embd: pe,
            n_head: ph,
            n_layers: pl,
            heads,
            codes,
            code_dim,
            root_feat_width: rfw,
            pose_feat_width: pfw,
            token_length_feat_width: tlw,
            mlp_layers: 2,
            root_cond_dim: 4,
            local_pose_dim: lpd,
            min_tokens: MIN_TOKENS,
            max_tokens: MAX_TOKENS,
            frames_per_token: FRAMES_PER_TOKEN,
        },
        vq: VqConfig {
            heads,
            codes,
            code_dim,
            width: vq.0,
            depth: vq.1,
            dilation_growth_rate: 3,
            down_t: 2,
            out_dim: layout.local(),
            target_dim: lpd,
            external_dim: 2,
        },
        fps: FPS,
    }
}

impl BricksConfig {
    /// MotionBricks-SOMA-RP-v1: 5 heads × 16 codes (2^20), root 6 + 2 layers.
    pub fn soma() -> Self {
        build(Rig::Soma, 30, 5, 16, 160, (1024, 16, 16, 256, 640, 128), (512, 16, 6, 2, 256, 64, 64, 512, 4), (512, 4))
    }
    /// MotionBricks-G1-RP-v1: 8 heads × 10 codes (10^8), root 3 + 3 layers.
    pub fn g1() -> Self {
        build(Rig::G1, 34, 8, 10, 256, (1024, 16, 16, 256, 640, 128), (512, 16, 3, 3, 256, 64, 64, 512, 4), (512, 4))
    }
    /// The networks of `soma_tiny.safetensors` (`scripts/reference/motionbricks_fixtures.py`).
    pub fn tiny_soma() -> Self {
        build(Rig::Soma, 30, 2, 4, 8, (32, 2, 2, 16, 64, 16), (32, 2, 2, 1, 16, 8, 8, 32, 2), (32, 2))
    }
    pub fn tiny_g1() -> Self {
        build(Rig::G1, 34, 2, 10, 8, (32, 2, 2, 16, 64, 16), (32, 2, 2, 1, 16, 8, 8, 32, 2), (32, 2))
    }
}
