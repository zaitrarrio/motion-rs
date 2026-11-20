//! The Unitree G1 rig of the release graph (`tools/onnx/models/g1/rig.py`):
//! forward kinematics over MuJoCo joint coordinates into the 34-joint motion
//! skeleton (`G1PlacedPosePreprocess`), and the decode of motion features back
//! to coordinates (`G1PlacedPosePostprocess`). Arrays come from
//! `tools/onnx/assets/g1_model_rig.npz` (embedded in `motion_core::assets`).

use crate::config::MAX_FRAMES;
use crate::model::SLOTS;
use crate::rep::Layout;
use motion_core::assets as a;
use motion_core::geometry::{self as geo, Mat3, Vec3, IDENTITY};

pub const QPOS: usize = 36;
pub const DOFS: usize = 29;

pub struct G1Rig {
    basis: Mat3,
    axes: Vec<Vec3>,
    model_axes: Vec<Vec3>,
    q2t: Vec<Mat3>,
    f2q: Vec<Mat3>,
    rest: Vec<Vec3>,
    neutral: Vec<Vec3>,
    parents: Vec<i32>,
    model_parents: Vec<i32>,
    reverse: Vec<i32>,
    mapping: Vec<usize>,
}

fn mats(v: &[f32]) -> Vec<Mat3> {
    (0..v.len() / 9).map(|i| geo::mat_at(v, i)).collect()
}
fn vecs(v: &[f32]) -> Vec<Vec3> {
    (0..v.len() / 3).map(|i| geo::vec_at(v, i)).collect()
}

fn wrap(v: f32) -> f32 {
    let p = std::f32::consts::PI;
    (v + p).rem_euclid(2.0 * p) - p
}

impl Default for G1Rig {
    fn default() -> Self {
        Self::new()
    }
}

impl G1Rig {
    pub fn new() -> Self {
        Self {
            basis: geo::mat_at(&a::MB_G1_MUJOCO_TO_MOTION, 0),
            axes: vecs(&a::MB_G1_AXES_MUJOCO),
            model_axes: vecs(&a::MB_G1_AXES_MOTION),
            q2t: mats(&a::MB_G1_ROT_OFFSETS_Q2T),
            f2q: mats(&a::MB_G1_ROT_OFFSETS_F2Q),
            rest: vecs(&a::MB_G1_CAPTURE_NEUTRAL_JOINTS_MUJOCO),
            neutral: vecs(&a::MB_G1_NEUTRAL_JOINTS),
            parents: a::MB_G1_MUJOCO_PARENTS.to_vec(),
            model_parents: a::MB_G1_JOINT_PARENTS.to_vec(),
            reverse: a::MB_G1_MOTION_TO_MUJOCO.to_vec(),
            mapping: a::MB_G1_MUJOCO_TO_MOTION_INDICES.iter().map(|&i| i as usize).collect(),
        }
    }

    /// `G1Rig.fk` for one pose with a wxyz root quaternion: model-space positions
    /// and rotations of the 34 motion joints.
    pub fn fk(&self, q: &[f32]) -> (Vec<Vec3>, Vec<Mat3>) {
        let quat = [q[3], q[4], q[5], q[6]];
        let mut rot = vec![geo::quat_to_matrix(&quat)];
        let mut pos = vec![geo::add(&[q[0], q[1], q[2]], &self.rest[0])];
        let local: Vec<Mat3> = (0..DOFS)
            .map(|d| {
                let ang = q[7 + d];
                let (s, c) = (ang / 2.0).sin_cos();
                let ax = self.axes[d];
                let qm = geo::quat_to_matrix(&[c, s * ax[0], s * ax[1], s * ax[2]]);
                geo::matmul(&self.q2t[self.mapping[d]], &qm)
            })
            .collect();
        for joint in 1..=DOFS {
            let par = self.parents[joint] as usize;
            let off = geo::sub(&self.rest[joint], &self.rest[par]);
            pos.push(geo::add(&pos[par], &geo::apply(&rot[par], &off)));
            rot.push(geo::matmul(&rot[par], &local[joint - 1]));
        }
        let bt = geo::transpose(&self.basis);
        let p: Vec<Vec3> = pos.iter().map(|v| geo::apply(&self.basis, v)).collect();
        let r: Vec<Mat3> = rot.iter().map(|m| geo::matmul(&geo::matmul(&self.basis, m), &bt)).collect();
        let mut mp: Vec<Vec3> = Vec::with_capacity(self.reverse.len());
        let mut mr: Vec<Mat3> = Vec::with_capacity(self.reverse.len());
        for (joint, &ri) in self.reverse.iter().enumerate() {
            if ri >= 0 {
                mp.push(p[ri as usize]);
                mr.push(r[ri as usize]);
            } else {
                let par = self.model_parents[joint] as usize;
                let off = geo::sub(&self.neutral[joint], &self.neutral[par]);
                let v = geo::add(&mp[par], &geo::apply(&mr[par], &off));
                mp.push(v);
                mr.push(IDENTITY);
            }
        }
        (mp, mr)
    }

    /// `G1PlacedPosePreprocess`: context + target qpos (xyzw root quaternions)
    /// → global root `[8, 5]`, local root `[8, 4]`, local poses `[8, 303]`.
    pub fn preprocess(&self, ctx: &[f32], tgt: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut root = Vec::new();
        let mut heading = Vec::new();
        let mut lp = Vec::new();
        for f in 0..SLOTS {
            let src = if f < 4 { &ctx[f * QPOS..(f + 1) * QPOS] } else { &tgt[(f - 4) * QPOS..(f - 3) * QPOS] };
            let mut q = src.to_vec();
            // xyzw → wxyz.
            q[3] = src[6];
            q[4] = src[3];
            q[5] = src[4];
            q[6] = src[5];
            let (p, r) = self.fk(&q);
            let r0 = p[0];
            heading.push(r[0][2].atan2(r[0][8]));
            root.push(r0);
            for pj in &p[1..] {
                lp.extend_from_slice(&[pj[0] - r0[0], pj[1], pj[2] - r0[2]]);
            }
            for rj in &r {
                lp.extend_from_slice(&[rj[0], rj[3], rj[6], rj[1], rj[4], rj[7]]);
            }
        }
        // ric and rotations are interleaved per frame above as [ric..., rot...]; keep that order.
        let mut g = Vec::with_capacity(SLOTS * 5);
        for f in 0..SLOTS {
            g.extend_from_slice(&[root[f][0], root[f][1], root[f][2], heading[f].cos(), heading[f].sin()]);
        }
        let mut l = vec![0f32; SLOTS * 4];
        for s in [0, 4] {
            for i in 0..3 {
                let f = s + i;
                l[f * 4] = wrap(heading[f + 1] - heading[f]) * 30.0;
                l[f * 4 + 1] = (root[f + 1][0] - root[f][0]) * 30.0;
                l[f * 4 + 2] = (root[f + 1][2] - root[f][2]) * 30.0;
                l[f * 4 + 3] = root[f][1];
            }
            l[(s + 3) * 4 + 3] = root[s + 3][1];
        }
        (g, l, lp)
    }

    /// `G1PlacedPosePostprocess`: global features `[64, 414]` → qpos `[64, 36]`
    /// (xyzw root quaternion) and the foot-contact columns `[64, 4]`.
    pub fn postprocess(&self, feat: &[f32], lay: &Layout) -> (Vec<f32>, Vec<f32>) {
        let gw = lay.global();
        let j = lay.joints;
        let rs = lay.global_rot_start();
        let cs = lay.global_contacts_start();
        let bt = geo::transpose(&self.basis);
        let mut qpos = Vec::with_capacity(MAX_FRAMES * QPOS);
        let mut contacts = Vec::with_capacity(MAX_FRAMES * 4);
        for f in 0..MAX_FRAMES {
            let o = &feat[f * gw..(f + 1) * gw];
            let r: Vec<Mat3> = (0..j).map(|jj| geo::cont6d_to_matrix(&o[rs + jj * 6..rs + jj * 6 + 6])).collect();
            let local: Vec<Mat3> = (0..j)
                .map(|jj| {
                    let par = if jj == 0 { IDENTITY } else { r[self.model_parents[jj] as usize] };
                    geo::matmul(&self.f2q[jj], &geo::matmul(&geo::transpose(&par), &r[jj]))
                })
                .collect();
            let rp = geo::apply(&bt, &[o[0], o[1], o[2]]);
            let rr = geo::matmul(&geo::matmul(&bt, &local[0]), &self.basis);
            let qq = geo::matrix_to_quat(&rr);
            qpos.extend_from_slice(&[rp[0], rp[1], rp[2], qq[1], qq[2], qq[3], qq[0]]);
            for d in 0..DOFS {
                let h = local[self.mapping[d]];
                let ang = [h[7].atan2(h[8]), h[2].atan2(h[0]), h[3].atan2(h[4])];
                let ax = self.model_axes[d];
                qpos.push(ang[0] * ax[0] + ang[1] * ax[1] + ang[2] * ax[2]);
            }
            contacts.extend_from_slice(&o[cs..cs + 4]);
        }
        (qpos, contacts)
    }
}
