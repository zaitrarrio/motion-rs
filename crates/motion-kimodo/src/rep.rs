//! `KimodoMotionRep` (`kimodo/motion_rep/reps/kimodo_motionrep.py`): per frame
//! `[smooth_root_pos 3 | global_root_heading 2 | local_joints_positions 3J |
//! global_rot_data 6J | velocities 3J | foot_contacts 4]`, relative to the
//! smoothed root path, normalised by the split `global_root` / `body`
//! statistics; the `local_root` statistics normalise the denoiser's
//! intermediate local root (`global_root_to_local_root`).

use crate::constraints::ConditionData;
use crate::smooth_root::get_smooth_root_pos;
use crate::stats::Stats;
use motion_core::geometry::{self as geo};
use motion_core::skeleton::Skeleton;
use motion_core::Result;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct KimodoMotionRep {
    pub skeleton: Skeleton,
    pub fps: f32,
    pub global_root_stats: Stats,
    pub local_root_stats: Stats,
    pub body_stats: Stats,
    /// `[global_root | body]`.
    pub stats: Stats,
}

/// `inverse()` output for one sequence (`[T, …]`, flat).
#[derive(Debug, Clone)]
pub struct Decoded {
    pub frames: usize,
    pub local_rot_mats: Vec<f32>,
    pub global_rot_mats: Vec<f32>,
    pub posed_joints: Vec<f32>,
    pub root_positions: Vec<f32>,
    pub smooth_root_pos: Vec<f32>,
    pub foot_contacts: Vec<f32>,
    pub global_root_heading: Vec<f32>,
}

/// `compute_vel_xyz` on one sequence of `n` points of `w` = 3·J values: forward
/// differences × fps, a zero at the end, the last valid frame repeating the one before.
pub fn vel_xyz(p: &[f32], w: usize, fps: f32, len: usize) -> Vec<f32> {
    let n = p.len() / w;
    let mut v = vec![0f32; n * w];
    for t in 0..n.saturating_sub(1) {
        for k in 0..w {
            v[t * w + k] = fps * (p[(t + 1) * w + k] - p[t * w + k]);
        }
    }
    if len >= 1 && len - 1 < n {
        let last = len - 1;
        for k in 0..w {
            v[last * w + k] = if last == 0 { 0.0 } else { v[(last - 1) * w + k] };
        }
    }
    v
}

/// `compute_vel_angle`: `diff_angles` × fps, a zero at the end, the last valid frame repeating the one before.
pub fn vel_angle(a: &[f32], fps: f32, len: usize) -> Vec<f32> {
    let mut v = geo::diff_angles(a, fps);
    v.push(0.0);
    if len >= 1 && len - 1 < v.len() {
        let last = len - 1;
        v[last] = if last == 0 { 0.0 } else { v[last - 1] };
    }
    v
}

impl KimodoMotionRep {
    pub fn load(skeleton: Skeleton, fps: f32, stats_dir: &Path) -> Result<Self> {
        let g = Stats::load(&stats_dir.join("global_root"))?;
        let l = Stats::load(&stats_dir.join("local_root"))?;
        let b = Stats::load(&stats_dir.join("body"))?;
        Ok(Self::new(skeleton, fps, g, l, b))
    }

    pub fn new(skeleton: Skeleton, fps: f32, global_root: Stats, local_root: Stats, body: Stats) -> Self {
        let stats = Stats::concat(&[&global_root, &body]);
        Self { skeleton, fps, global_root_stats: global_root, local_root_stats: local_root, body_stats: body, stats }
    }

    pub fn nbjoints(&self) -> usize {
        self.skeleton.nbjoints()
    }
    pub fn dim(&self) -> usize {
        5 + self.nbjoints() * 12 + 4
    }
    pub fn global_root_dim(&self) -> usize {
        5
    }
    pub fn local_root_dim(&self) -> usize {
        4
    }
    pub fn s_local_pos(&self) -> usize {
        5
    }
    pub fn s_rot(&self) -> usize {
        5 + 3 * self.nbjoints()
    }
    pub fn s_vel(&self) -> usize {
        5 + 9 * self.nbjoints()
    }
    pub fn s_contacts(&self) -> usize {
        5 + 12 * self.nbjoints()
    }

    pub fn normalize(&self, x: &mut [f32]) {
        self.stats.normalize(x)
    }
    pub fn unnormalize(&self, x: &mut [f32]) {
        self.stats.unnormalize(x)
    }

    /// `global_root_to_local_root` on one sequence `[T, 5]` → `[T, 4]`.
    pub fn global_root_to_local_root(&self, root: &[f32], normalized: bool, len: usize) -> Vec<f32> {
        let n = root.len() / 5;
        let r = if normalized { self.global_root_stats.unnormalized(root) } else { root.to_vec() };
        let ang: Vec<f32> = (0..n).map(|t| r[t * 5 + 4].atan2(r[t * 5 + 3])).collect();
        let rv = vel_angle(&ang, self.fps, len);
        let pos: Vec<f32> = (0..n).flat_map(|t| [r[t * 5], r[t * 5 + 1], r[t * 5 + 2]]).collect();
        let v = vel_xyz(&pos, 3, self.fps, len);
        let mut out = Vec::with_capacity(n * 4);
        for t in 0..n {
            out.extend_from_slice(&[rv[t], v[t * 3], v[t * 3 + 2], r[t * 5 + 1]]);
        }
        if normalized {
            self.local_root_stats.normalize(&mut out);
        }
        out
    }

    /// `inverse(features, is_normalized)` for one sequence, posed joints from the rotations.
    pub fn inverse(&self, features: &[f32], is_normalized: bool) -> Decoded {
        let d = self.dim();
        let j = self.nbjoints();
        let n = features.len() / d;
        let f = if is_normalized { self.stats.unnormalized(features) } else { features.to_vec() };
        let mut g = vec![0f32; n * j * 9];
        let mut smooth = Vec::with_capacity(n * 3);
        let mut heading = Vec::with_capacity(n * 2);
        let mut root = Vec::with_capacity(n * 3);
        let mut contacts = Vec::with_capacity(n * 4);
        for t in 0..n {
            let row = &f[t * d..(t + 1) * d];
            smooth.extend_from_slice(&row[0..3]);
            heading.extend_from_slice(&row[3..5]);
            for jj in 0..j {
                let o = self.s_rot() + jj * 6;
                geo::set_mat(&mut g, t * j + jj, &geo::cont6d_to_matrix(&row[o..o + 6]));
            }
            let lp = &row[self.s_local_pos()..self.s_local_pos() + 3];
            root.extend_from_slice(&[lp[0] + row[0], lp[1], lp[2] + row[2]]);
            contacts.extend(row[self.s_contacts()..self.s_contacts() + 4].iter().map(|&c| (c > 0.5) as u8 as f32));
        }
        let local = self.skeleton.global_rots_to_local(&g);
        let (posed, _) = self.skeleton.joints_from_global_rots(&g, &root);
        Decoded {
            frames: n,
            local_rot_mats: local,
            global_rot_mats: g,
            posed_joints: posed,
            root_positions: root,
            smooth_root_pos: smooth,
            foot_contacts: contacts,
            global_root_heading: heading,
        }
    }

    /// `__call__(local_joint_rots, root_positions, to_normalize=False)` for one sequence.
    pub fn encode(&self, local_rots: &[f32], root_pos: &[f32]) -> Vec<f32> {
        let j = self.nbjoints();
        let n = root_pos.len() / 3;
        let (g, posed, p0) = self.skeleton.fk(local_rots, root_pos);
        let head = self.skeleton.heading_angle_kimodo(&posed);
        let smooth = get_smooth_root_pos(root_pos);
        let vel = vel_xyz(&posed, 3 * j, self.fps, n);
        let contacts = self.foot_contacts(&posed, &vel);
        let d = self.dim();
        let mut out = vec![0f32; n * d];
        for t in 0..n {
            let row = &mut out[t * d..(t + 1) * d];
            row[0..3].copy_from_slice(&smooth[t * 3..t * 3 + 3]);
            row[3] = head[t].cos();
            row[4] = head[t].sin();
            let off = [root_pos[t * 3] - smooth[t * 3], root_pos[t * 3 + 1], root_pos[t * 3 + 2] - smooth[t * 3 + 2]];
            for jj in 0..j {
                let p = geo::vec_at(&p0, t * j + jj);
                let o = self.s_local_pos() + jj * 3;
                row[o..o + 3].copy_from_slice(&geo::add(&p, &off));
                let o = self.s_rot() + jj * 6;
                row[o..o + 6].copy_from_slice(&geo::matrix_to_cont6d(&geo::mat_at(&g, t * j + jj)));
                let o = self.s_vel() + jj * 3;
                row[o..o + 3].copy_from_slice(&vel[(t * j + jj) * 3..(t * j + jj) * 3 + 3]);
            }
            row[self.s_contacts()..self.s_contacts() + 4].copy_from_slice(&contacts[t * 4..t * 4 + 4]);
        }
        out
    }

    /// `foot_detect_from_pos_and_vel(…, 0.15, 0.10)`: `[T, 4]` (left ankle/toe, right ankle/toe).
    pub fn foot_contacts(&self, posed: &[f32], vel: &[f32]) -> Vec<f32> {
        let j = self.nbjoints();
        let n = posed.len() / (3 * j);
        let ids: Vec<usize> = self.skeleton.left_foot[..2].iter().chain(&self.skeleton.right_foot[..2]).copied().collect();
        let mut out = Vec::with_capacity(n * 4);
        for t in 0..n {
            for &i in &ids {
                let v = geo::norm(&geo::vec_at(vel, t * j + i));
                let h = posed[(t * j + i) * 3 + 1];
                out.push((v < 0.15 && h < 0.10) as u8 as f32);
            }
        }
        out
    }

    /// `translate_2d` on unnormalised features: shift the smooth root x / z.
    pub fn translate_2d(&self, f: &mut [f32], dx: f32, dz: f32) {
        let d = self.dim();
        for row in f.chunks_mut(d) {
            row[0] += dx;
            row[2] += dz;
        }
    }

    /// `create_conditions`: unnormalised observed features and the boolean mask (as 0/1), `[len, dim]`.
    pub fn create_conditions(&self, data: &ConditionData, len: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.dim();
        let mut obs = vec![0f32; len * d];
        let mut mask = vec![0f32; len * d];
        let put = |obs: &mut [f32], mask: &mut [f32], t: usize, c: usize, v: f32| {
            if t < len {
                obs[t * d + c] = v;
                mask[t * d + c] = 1.0;
            }
        };
        for (&t, v) in &data.smooth_root_2d {
            put(&mut obs, &mut mask, t, 0, v[0]);
            put(&mut obs, &mut mask, t, 2, v[1]);
        }
        for (&t, &y) in &data.root_y {
            put(&mut obs, &mut mask, t, 1, y);
        }
        for (&t, h) in &data.heading {
            put(&mut obs, &mut mask, t, 3, h[0]);
            put(&mut obs, &mut mask, t, 4, h[1]);
        }
        for (&(t, jj), m) in &data.rotations {
            let c6 = geo::matrix_to_cont6d(m);
            for k in 0..6 {
                put(&mut obs, &mut mask, t, self.s_rot() + jj * 6 + k, c6[k]);
            }
        }
        for (&(t, jj), p) in &data.positions {
            if t >= len {
                continue;
            }
            if mask[t * d] == 0.0 || mask[t * d + 2] == 0.0 {
                return Err(motion_core::Error::Config(
                    "for constraining global positions, the smooth root should also be constrained".into(),
                ));
            }
            let (sx, sz) = (obs[t * d], obs[t * d + 2]);
            let local = [p[0] - sx, p[1], p[2] - sz];
            for k in 0..3 {
                put(&mut obs, &mut mask, t, self.s_local_pos() + jj * 3 + k, local[k]);
            }
        }
        Ok((obs, mask))
    }
}
