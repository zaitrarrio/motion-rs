//! The MotionBricks feature layout (`DualRootGlobalJoints` of
//! `tools/onnx/models/representation.py`) and its normalisation.
//!
//! The *dual* vector the statistics cover is
//! `[global_root_pos 3 | global_root_heading 2 | local_root_rot_vel 1 | local_root_vel 2 | global_root_y 1 | ric (J−1)·3 | global_rot 6J | local_vel 3J | foot_contacts 4]`.
//! The *global* representation drops the four local-root columns, the *local*
//! one drops the five global-root columns; both share the body block.
//! Normalisation is `(x − mean) / √(std² + 1e-5)` in f32 (`Stats(legacy=False)`).

/// Column ranges of the three layouts for a skeleton of `j` joints.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub joints: usize,
}

impl Layout {
    pub fn new(joints: usize) -> Self {
        Self { joints }
    }
    /// Body block width: ric + rotations + velocities + contacts.
    pub fn body(&self) -> usize {
        (self.joints - 1) * 3 + self.joints * 6 + self.joints * 3 + 4
    }
    pub fn dual(&self) -> usize {
        9 + self.body()
    }
    pub fn global(&self) -> usize {
        5 + self.body()
    }
    pub fn local(&self) -> usize {
        4 + self.body()
    }
    /// Offset of the body block's ric data inside each layout.
    pub fn ric(&self) -> usize {
        (self.joints - 1) * 3
    }
    pub fn rot(&self) -> usize {
        self.joints * 6
    }
    /// Global layout: rotation columns start here.
    pub fn global_rot_start(&self) -> usize {
        5 + self.ric()
    }
    /// Global layout: foot contacts start here.
    pub fn global_contacts_start(&self) -> usize {
        self.global() - 4
    }
    /// Local layout: `joint_positions_and_rotations_and_hip_height` =
    /// `[global_root_y, ric, global_rot]` — `1 + ric + rot` columns starting at 3.
    pub fn local_pose_feature(&self) -> usize {
        1 + self.ric() + self.rot()
    }
    /// Dual column of local-layout column `c`.
    pub fn dual_of_local(&self, c: usize) -> usize {
        if c < 4 { 5 + c } else { 9 + (c - 4) }
    }
    /// Dual column of global-layout column `c`.
    pub fn dual_of_global(&self, c: usize) -> usize {
        if c < 5 { c } else { 9 + (c - 5) }
    }
}

/// Normalisation statistics over the dual vector.
#[derive(Debug, Clone)]
pub struct Stats {
    pub mean: Vec<f32>,
    /// `√(std² + 1e-5)` in f32.
    pub std_eps: Vec<f32>,
}

impl Stats {
    pub fn new(mean: Vec<f32>, std: &[f32]) -> Self {
        let std_eps = std.iter().map(|s| (s * s + 1e-5f32).sqrt()).collect();
        Self { mean, std_eps }
    }

    /// Normalise `rows × cols.len()` values whose column `k` is dual column `cols[k]`.
    pub fn normalize(&self, x: &mut [f32], cols: &[usize]) {
        let w = cols.len();
        for row in x.chunks_mut(w) {
            for (k, v) in row.iter_mut().enumerate() {
                *v = (*v - self.mean[cols[k]]) / self.std_eps[cols[k]];
            }
        }
    }

    pub fn unnormalize(&self, x: &mut [f32], cols: &[usize]) {
        let w = cols.len();
        for row in x.chunks_mut(w) {
            for (k, v) in row.iter_mut().enumerate() {
                *v = *v * self.std_eps[cols[k]] + self.mean[cols[k]];
            }
        }
    }
}

/// Dual columns of a contiguous range.
pub fn range(a: usize, b: usize) -> Vec<usize> {
    (a..b).collect()
}

/// `diff_angles` of representation.py on one sequence, padded with a zero
/// at the end (`[angles.len()]`).
fn angular_velocity(angles: &[f32], fps: f32) -> Vec<f32> {
    let mut out: Vec<f32> = angles
        .windows(2)
        .map(|w| {
            let (c0, s0) = (w[0].cos(), w[0].sin());
            let (c1, s1) = (w[1].cos(), w[1].sin());
            let cd = c1 * c0 + s1 * s0;
            let sd = s1 * c0 - c1 * s0;
            fps * sd.atan2(cd)
        })
        .collect();
    out.push(0.0);
    out
}

/// `global_root_to_local_onnx_compatible`: normalised global root `[n, 5]`
/// (one sequence of `n` frames) → normalised local root `[n, 4]`
/// (`rot_vel, vel_x, vel_z, y`), the velocities at `length − 1` repeating the
/// frame before. `removing_heading` is false in the release (no rotation of
/// the velocities into the heading frame).
pub fn global_root_to_local(stats: &Stats, g_norm: &[f32], length: usize, fps: f32) -> Vec<f32> {
    let n = g_norm.len() / 5;
    let mut g = g_norm.to_vec();
    stats.unnormalize(&mut g, &range(0, 5));
    // compute_root_pos_and_rot → quaternion → angle again (atan2(q_y, q_w)·2).
    let angles: Vec<f32> = (0..n)
        .map(|t| {
            let a = g[t * 5 + 4].atan2(g[t * 5 + 3]);
            (a / 2.0).sin().atan2((a / 2.0).cos()) * 2.0
        })
        .collect();
    let mut av = angular_velocity(&angles, fps);
    let mut vel = vec![[0f32; 3]; n];
    for t in 0..n.saturating_sub(1) {
        for c in 0..3 {
            vel[t][c] = fps * (g[(t + 1) * 5 + c] - g[t * 5 + c]);
        }
    }
    let last = length - 1;
    av[last] = av[last - 1];
    vel[last] = vel[last - 1];
    let mut out = vec![0f32; n * 4];
    for t in 0..n {
        out[t * 4] = av[t];
        out[t * 4 + 1] = vel[t][0];
        out[t * 4 + 2] = vel[t][2];
        out[t * 4 + 3] = g[t * 5 + 1];
    }
    stats.normalize(&mut out, &range(5, 9));
    out
}

/// `DualRootLocalBody.local_to_global(is_normalized=True, to_normalize=False)` on one
/// sequence: normalised local features `[n, local]` → unnormalised global `[n, global]`.
/// The root is integrated from frame 0 (`LocalRootLocalBody.compute_root_pos_and_rot`,
/// `removing_heading` false).
pub fn local_to_global(lay: &Layout, stats: &Stats, local_norm: &[f32], fps: f32) -> Vec<f32> {
    let lw = lay.local();
    let n = local_norm.len() / lw;
    let mut root = Vec::with_capacity(n * 4);
    let mut body = Vec::with_capacity(n * lay.body());
    for t in 0..n {
        root.extend_from_slice(&local_norm[t * lw..t * lw + 4]);
        body.extend_from_slice(&local_norm[t * lw + 4..(t + 1) * lw]);
    }
    stats.unnormalize(&mut root, &range(5, 9));
    stats.unnormalize(&mut body, &range(9, lay.dual()));
    let mut ang = vec![0f32; n];
    let mut pos = vec![[0f32; 3]; n];
    for t in 1..n {
        ang[t] = ang[t - 1] + root[(t - 1) * 4] / fps;
        pos[t][0] = pos[t - 1][0] + root[(t - 1) * 4 + 1] / fps;
        pos[t][2] = pos[t - 1][2] + root[(t - 1) * 4 + 2] / fps;
    }
    let gw = lay.global();
    let mut out = vec![0f32; n * gw];
    for t in 0..n {
        let o = &mut out[t * gw..(t + 1) * gw];
        // quat [cos(a/2), 0, sin(a/2), 0] → angle atan2(q_y, q_w)·2.
        let a = (ang[t] / 2.0).sin().atan2((ang[t] / 2.0).cos()) * 2.0;
        o[0] = pos[t][0];
        o[1] = root[t * 4 + 3];
        o[2] = pos[t][2];
        o[3] = a.cos();
        o[4] = a.sin();
        o[5..].copy_from_slice(&body[t * lay.body()..(t + 1) * lay.body()]);
    }
    out
}
