//! The MotionBricks runtime engines: place context and target poses in the
//! scene, generate the transition, and chain generations into continuous
//! motion. A port of `motionbricks/model/{generation,engine}.py` and the
//! `soma/` and `unitree_g1/` kernels, in SOMA / G1 model space (Y up for
//! SOMA, Z up for G1; apply your own world transform outside).
//!
//! Transforms follow Warp's algebra: `a · b = (a.p + a.q·b.p, a.q·b.q)`.

use crate::config::{BricksConfig, Rig, BOUNDARY_FRAMES, FRAMES_PER_TOKEN, MAX_TOKENS, MIN_TOKENS};
use crate::model::{G1Inputs, MotionBricks, SomaInputs, COUNT_CLASSES, SLOTS};
use crate::rig_g1::QPOS;
use motion_core::backend::Backend;
use motion_core::geometry::{self as geo, Quat, Vec3};
use motion_core::{Error, Result};

/// A rigid transform, quaternion `(w, x, y, z)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub p: Vec3,
    pub q: Quat,
}

impl Transform {
    pub const IDENTITY: Self = Self { p: [0.0; 3], q: [1.0, 0.0, 0.0, 0.0] };

    pub fn new(p: Vec3, q: Quat) -> Self {
        Self { p, q }
    }
    pub fn mul(&self, o: &Self) -> Self {
        Self { p: geo::add(&self.p, &geo::quat_apply(&self.q, &o.p)), q: geo::quat_mul(&self.q, &o.q) }
    }
    pub fn inverse(&self) -> Self {
        let qi = geo::quat_conjugate(&normalize(&self.q));
        Self { p: geo::scale(&geo::quat_apply(&qi, &self.p), -1.0), q: qi }
    }
}

pub fn normalize(q: &Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Shortest-arc spherical interpolation (`wp.quat_slerp`).
pub fn slerp(a: &Quat, b: &Quat, t: f32) -> Quat {
    let mut b = *b;
    let mut d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if d < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        d = -d;
    }
    if d > 0.9995 {
        let q = [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
            a[3] + t * (b[3] - a[3]),
        ];
        return normalize(&q);
    }
    let th = d.clamp(-1.0, 1.0).acos();
    let s = th.sin();
    let (wa, wb) = (((1.0 - t) * th).sin() / s, (t * th).sin() / s);
    [wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2], wa * a[3] + wb * b[3]]
}

pub fn blend(a: &Transform, b: &Transform, t: f32) -> Transform {
    Transform {
        p: [a.p[0] + t * (b.p[0] - a.p[0]), a.p[1] + t * (b.p[1] - a.p[1]), a.p[2] + t * (b.p[2] - a.p[2])],
        q: slerp(&a.q, &b.q, t),
    }
}

fn quat_y(angle: f32) -> Quat {
    let (s, c) = (angle / 2.0).sin_cos();
    [c, 0.0, s, 0.0]
}

fn quat_z(angle: f32) -> Quat {
    let (s, c) = (angle / 2.0).sin_cos();
    [c, 0.0, 0.0, s]
}

/// `compute_delta_transform`: `src⁻¹ · dst`.
fn delta(src: &Transform, dst: &Transform) -> Transform {
    src.inverse().mul(dst)
}

/// `apply_delta_transform`.
fn apply_delta(t: &Transform, d: &Transform, w: f32) -> Transform {
    t.mul(&blend(&Transform::IDENTITY, d, w))
}

/// `compute_root_endpoint_delta`: world-space offset, rotation applied after the generated one.
fn root_delta(generated: &Transform, guide: &Transform) -> Transform {
    let r = geo::quat_mul(&geo::quat_conjugate(&normalize(&generated.q)), &normalize(&guide.q));
    Transform { p: geo::sub(&guide.p, &generated.p), q: normalize(&r) }
}

/// `apply_root_endpoint_correction`.
fn apply_root(root: &Transform, s: &Transform, e: &Transform, we: f32) -> Transform {
    let ws = 1.0 - we;
    let id = Transform::IDENTITY.q;
    let q = geo::quat_mul(&geo::quat_mul(&root.q, &slerp(&id, &s.q, ws)), &slerp(&id, &e.q, we));
    Transform { p: geo::add(&geo::add(&root.p, &geo::scale(&s.p, ws)), &geo::scale(&e.p, we)), q: normalize(&q) }
}

/// Generation settings (`MotionBricksInferenceConfig`).
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Token counts the model may choose, each 6..=16 (default 8..=14).
    pub allowed_token_counts: Vec<usize>,
    /// Endpoint correction onto the last context pose and the first target pose.
    pub correct_generated_diff: bool,
    /// Ground direction the last context pose is turned to face in model space.
    pub heading_alignment: [f32; 2],
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self { allowed_token_counts: (8..=14).collect(), correct_generated_diff: true, heading_alignment: [1.0, 0.0] }
    }
}

impl GenerateConfig {
    fn allowed(&self) -> Result<[bool; COUNT_CLASSES]> {
        let mut a = [false; COUNT_CLASSES];
        for &c in &self.allowed_token_counts {
            if !(MIN_TOKENS..=MAX_TOKENS).contains(&c) {
                return Err(Error::Config(format!("token count {c} outside {MIN_TOKENS}..={MAX_TOKENS}")));
            }
            a[c - MIN_TOKENS] = true;
        }
        if !a.iter().any(|&x| x) {
            return Err(Error::Config("no allowed token count".into()));
        }
        Ok(a)
    }
}

/// `write_motionbricks_conditions_kernel`: which of the eight slots are given.
pub fn conditions(num_targets: usize) -> ([bool; SLOTS], [bool; SLOTS], [bool; SLOTS]) {
    let mut g = [false; SLOTS];
    let mut l = [false; SLOTS];
    for i in 0..SLOTS {
        let v = i < FRAMES_PER_TOKEN + num_targets;
        g[i] = v;
        l[i] = v && !(i == FRAMES_PER_TOKEN - 1 || i == FRAMES_PER_TOKEN + num_targets - 1);
    }
    (g, l, g)
}

/// Number of valid frames once the boundary frames are removed.
pub fn valid_frames(tokens: usize) -> usize {
    tokens * FRAMES_PER_TOKEN - 2 * BOUNDARY_FRAMES
}

/// A runtime skeleton: names, parents, and the reference local transforms
/// (bone offsets; the rest rotations of BVH clips are the identity).
#[derive(Debug, Clone)]
pub struct RigSkeleton {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    pub reference: Vec<Transform>,
}

impl RigSkeleton {
    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// World transforms of a pose of local transforms (`compute_world_poses`).
    pub fn world(&self, local: &[Transform]) -> Vec<Transform> {
        let mut w: Vec<Transform> = Vec::with_capacity(local.len());
        for (j, t) in local.iter().enumerate() {
            let p = self.parents[j];
            w.push(if p < 0 { *t } else { w[p as usize].mul(t) });
        }
        w
    }

    /// The 30-joint SOMA skeleton of the MotionBricks sample clips (the capture
    /// rest pose of BONES-SEED: Kimodo's `somaskel77/bvh_joints.p`, subset).
    pub fn soma_capture() -> Self {
        let s30 = motion_core::skeleton::Skeleton::soma30();
        let s77 = motion_core::skeleton::Skeleton::soma77();
        let bvh = motion_core::skeleton::soma77_bvh_neutral_joints();
        let map = s30.slice_into(&s77).expect("soma30 ⊂ soma77");
        let pos = |j30: usize| geo::vec_at(bvh, map[j30]);
        let reference = (0..30)
            .map(|j| {
                let p = match s30.parents[j] {
                    Some(par) => geo::sub(&pos(j), &pos(par)),
                    None => pos(j),
                };
                Transform::new(p, [1.0, 0.0, 0.0, 0.0])
            })
            .collect();
        Self {
            names: s30.joint_names.iter().map(|s| s.to_string()).collect(),
            parents: s30.parents.iter().map(|p| p.map_or(-1, |x| x as i32)).collect(),
            reference,
        }
    }
}

/// Generated motion of one generation.
#[derive(Debug, Clone)]
pub struct SomaMotion {
    /// `[frames][joints]` local transforms on the runtime skeleton.
    pub poses: Vec<Vec<Transform>>,
    /// `[frames][4]`: left heel, left toe, right heel, right toe, clamped to 0..1.
    pub foot_contacts: Vec<[f32; 4]>,
    pub tokens: usize,
}

/// The SOMA engine (`MotionBricksSOMAEngine`).
pub struct SomaEngine<'a, B: Backend> {
    pub model: &'a MotionBricks<B>,
    pub skeleton: RigSkeleton,
    model_to_pose: Vec<usize>,
    pose_to_model: Vec<i32>,
    right_leg: usize,
    left_leg: usize,
}

fn heading_from_hips(right: &Vec3, left: &Vec3) -> f32 {
    let across = [right[0] - left[0], 0.0, right[2] - left[2]];
    let n = geo::norm(&across);
    if n <= 1e-6 {
        return 0.0;
    }
    let a = geo::scale(&across, 1.0 / n);
    let f = geo::cross(&[0.0, 1.0, 0.0], &a);
    f[0].atan2(f[2])
}

impl<'a, B: Backend> SomaEngine<'a, B> {
    pub const MODEL_JOINTS: [&'static str; 30] = [
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    ];

    pub fn new(model: &'a MotionBricks<B>, skeleton: RigSkeleton) -> Result<Self> {
        if model.cfg.rig != Rig::Soma {
            return Err(Error::Config("SomaEngine needs the SOMA model".into()));
        }
        let mut model_to_pose = Vec::new();
        for n in Self::MODEL_JOINTS {
            model_to_pose.push(
                skeleton.index(n).ok_or_else(|| Error::Config(format!("skeleton is missing model joint {n}")))?,
            );
        }
        if model_to_pose[0] != 0 {
            return Err(Error::Config("the skeleton must store the root at index zero".into()));
        }
        let mut pose_to_model = vec![-1i32; skeleton.names.len()];
        for (mj, &pj) in model_to_pose.iter().enumerate() {
            let par = skeleton.parents[pj];
            if par < 0 || pose_to_model[par as usize] >= 0 {
                pose_to_model[pj] = mj as i32;
            }
        }
        let right_leg = skeleton.index("RightLeg").expect("checked");
        let left_leg = skeleton.index("LeftLeg").expect("checked");
        Ok(Self { model, skeleton, model_to_pose, pose_to_model, right_leg, left_leg })
    }

    /// Align the guide poses and write the model inputs (`compute_guide_alignment_kernel`,
    /// `encode_motionbricks_inputs_kernel`, `write_motionbricks_conditions_kernel`).
    /// Returns the inputs and the alignment that maps model output back to the scene.
    pub fn encode(
        &self,
        context: &[Vec<Transform>],
        target: &[Vec<Transform>],
        cfg: &GenerateConfig,
    ) -> Result<(SomaInputs, Transform)> {
        if context.len() != FRAMES_PER_TOKEN || target.is_empty() || target.len() > FRAMES_PER_TOKEN {
            return Err(Error::Config("need 4 context poses and 1 to 4 target poses".into()));
        }
        let nt = target.len();
        let mut guide: Vec<Vec<Transform>> = context.iter().map(|p| self.skeleton.world(p)).collect();
        for i in 0..FRAMES_PER_TOKEN {
            guide.push(self.skeleton.world(&target[i.min(nt - 1)]));
        }
        let last = &guide[FRAMES_PER_TOKEN - 1];
        let hips = last[0].p;
        let cur = heading_from_hips(&last[self.right_leg].p, &last[self.left_leg].p);
        let des = cfg.heading_alignment[0].atan2(cfg.heading_alignment[1]);
        let align = Transform::new([hips[0], 0.0, hips[2]], quat_y(cur - des));
        let ainv_q = geo::quat_conjugate(&align.q);
        let apos = |p: &Vec3| geo::quat_apply(&ainv_q, &geo::sub(p, &align.p));
        let j = 30;
        let mut g = vec![0f32; SLOTS * 5];
        let mut l = vec![0f32; SLOTS * 4];
        let mut lp = vec![0f32; SLOTS * 267];
        for f in 0..SLOTS {
            let nxt = if f == FRAMES_PER_TOKEN - 1 { f } else { (f + 1).min(SLOTS - 1) };
            let ch = apos(&guide[f][0].p);
            let nh = apos(&guide[nxt][0].p);
            let c_head = heading_from_hips(&apos(&guide[f][self.right_leg].p), &apos(&guide[f][self.left_leg].p));
            let n_head = heading_from_hips(&apos(&guide[nxt][self.right_leg].p), &apos(&guide[nxt][self.left_leg].p));
            let d = (n_head - c_head).sin().atan2((n_head - c_head).cos());
            g[f * 5..f * 5 + 5].copy_from_slice(&[ch[0], ch[1], ch[2], c_head.cos(), c_head.sin()]);
            l[f * 4..f * 4 + 4].copy_from_slice(&[d * 30.0, (nh[0] - ch[0]) * 30.0, (nh[2] - ch[2]) * 30.0, ch[1]]);
            let root = [ch[0], 0.0, ch[2]];
            for mj in 0..j {
                let pj = self.model_to_pose[mj];
                let jp = apos(&guide[f][pj].p);
                let jr = geo::quat_mul(&ainv_q, &guide[f][pj].q);
                if mj > 0 {
                    let o = f * 267 + (mj - 1) * 3;
                    lp[o..o + 3].copy_from_slice(&geo::sub(&jp, &root));
                }
                let x = geo::quat_apply(&jr, &[1.0, 0.0, 0.0]);
                let y = geo::quat_apply(&jr, &[0.0, 1.0, 0.0]);
                let o = f * 267 + 29 * 3 + mj * 6;
                lp[o..o + 6].copy_from_slice(&[x[0], x[1], x[2], y[0], y[1], y[2]]);
            }
        }
        let (has_g, has_l, has_p) = conditions(nt);
        let inputs = SomaInputs {
            global_root_values: g,
            has_global_root_values: has_g,
            local_root_values: l,
            has_local_root_values: has_l,
            local_poses: lp,
            has_local_poses: has_p,
            allowed: cfg.allowed()?,
        };
        Ok((inputs, align))
    }

    /// Generate from four context poses and one to four target poses (local
    /// transforms, roots in model space).
    pub fn generate(
        &self,
        be: &B,
        context: &[Vec<Transform>],
        target: &[Vec<Transform>],
        cfg: &GenerateConfig,
    ) -> Result<SomaMotion> {
        let (inputs, align) = self.encode(context, target, cfg)?;
        let out = self.model.predict_soma(be, &inputs, None);
        let tokens = out.pred_num_tokens;
        if !cfg.allowed_token_counts.contains(&tokens) {
            return Err(Error::Backend(format!("model chose {tokens} tokens, outside the allowed set")));
        }
        // decode_motionbricks_output_kernel.
        let gw = self.model.layout().global();
        let rs = self.model.layout().global_rot_start();
        let cs = self.model.layout().global_contacts_start();
        let count = valid_frames(tokens);
        let read_rot = |raw: usize, mj: usize| -> Quat {
            let o = raw * gw + rs + mj * 6;
            geo::matrix_to_quat(&geo::cont6d_to_matrix(&out.pred_global_motions[o..o + 6]))
        };
        let mut poses = Vec::with_capacity(count);
        let mut contacts = Vec::with_capacity(count);
        for fr in 0..count {
            let raw = BOUNDARY_FRAMES + fr;
            let row = &out.pred_global_motions[raw * gw..(raw + 1) * gw];
            contacts.push(std::array::from_fn(|c| row[cs + c].clamp(0.0, 1.0)));
            let hips = align.mul(&Transform::new([row[0], row[1], row[2]], read_rot(raw, 0)));
            let mut pose = vec![Transform::IDENTITY; self.skeleton.names.len()];
            pose[0] = hips;
            for pj in 1..pose.len() {
                let mj = self.pose_to_model[pj];
                if mj < 0 {
                    pose[pj] = self.skeleton.reference[pj];
                    continue;
                }
                let pr = read_rot(raw, self.pose_to_model[self.skeleton.parents[pj] as usize] as usize);
                let cr = read_rot(raw, mj as usize);
                let lq = normalize(&geo::quat_mul(&geo::quat_conjugate(&pr), &cr));
                pose[pj] = Transform::new(self.skeleton.reference[pj].p, lq);
            }
            poses.push(pose);
        }
        if cfg.correct_generated_diff && count >= 2 {
            let lc = &context[FRAMES_PER_TOKEN - 1];
            let t0 = &target[0];
            let lastf = count - 1;
            let nj = poses[0].len();
            let starts: Vec<Transform> = (0..nj)
                .map(|jj| if jj == 0 { root_delta(&poses[0][0], &lc[0]) } else { delta(&poses[0][jj], &lc[jj]) })
                .collect();
            let ends: Vec<Transform> = (0..nj)
                .map(|jj| if jj == 0 { root_delta(&poses[lastf][0], &t0[0]) } else { delta(&poses[lastf][jj], &t0[jj]) })
                .collect();
            for (fr, pose) in poses.iter_mut().enumerate() {
                let we = fr as f32 / lastf as f32;
                let ws = 1.0 - we;
                pose[0] = apply_root(&pose[0], &starts[0], &ends[0], we);
                for jj in 1..nj {
                    let c = apply_delta(&pose[jj], &starts[jj], ws);
                    pose[jj] = apply_delta(&c, &ends[jj], we);
                }
            }
        }
        Ok(SomaMotion { poses, foot_contacts: contacts, tokens })
    }
}

/// Chain generations the way the demos do: play each generated segment up to
/// its transition point, then take the last four played poses (at 30 Hz) as
/// the next context.
pub struct Chain {
    pub played: Vec<Vec<Transform>>,
    pub contacts: Vec<[f32; 4]>,
}

impl Chain {
    /// Start from one pose held still (`seed_history_from_pose`).
    pub fn new(start: Vec<Transform>) -> Self {
        Self { played: vec![start; FRAMES_PER_TOKEN], contacts: vec![[1.0; 4]; FRAMES_PER_TOKEN] }
    }

    pub fn context(&self) -> Vec<Vec<Transform>> {
        self.played[self.played.len() - FRAMES_PER_TOKEN..].to_vec()
    }

    /// Generate toward `target` and play it, ending `transition_early` frames
    /// before the generated end (0 plays the whole segment).
    pub fn step<B: Backend>(
        &mut self,
        be: &B,
        engine: &SomaEngine<'_, B>,
        target: &[Vec<Transform>],
        transition_early: usize,
        cfg: &GenerateConfig,
    ) -> Result<usize> {
        let m = engine.generate(be, &self.context(), target, cfg)?;
        let keep = m.poses.len().saturating_sub(transition_early).max(1);
        // The first generated frame repeats the last context pose; skip it.
        self.played.extend(m.poses[1..keep].iter().cloned());
        self.contacts.extend(m.foot_contacts[1..keep].iter().copied());
        Ok(keep - 1)
    }
}

/// The G1 engine (`MotionBricksUnitreeG1Engine`): poses are 36 joint
/// coordinates (root xyz, root quaternion xyzw, 29 angles) in G1 model space (Z up).
pub struct G1Engine<'a, B: Backend> {
    pub model: &'a MotionBricks<B>,
}

/// `ROOT_POSITION_FILTER` of `unitree_g1/kernels.py`.
pub const G1_ROOT_FILTER: [f32; 7] = [-0.0465, 0.0639, 0.2827, 0.3998, 0.2827, 0.0639, -0.0465];
pub const G1_ANGLE_FADE_SECONDS: f32 = 0.3;

fn read_root(q: &[f32]) -> Transform {
    Transform::new([q[0], q[1], q[2]], normalize(&[q[6], q[3], q[4], q[5]]))
}

fn write_root(t: &Transform, q: &mut [f32]) {
    let r = normalize(&t.q);
    q[..3].copy_from_slice(&t.p);
    q[3..7].copy_from_slice(&[r[1], r[2], r[3], r[0]]);
}

fn quadratic_fade(v: f32) -> f32 {
    let t = v.clamp(0.0, 1.0);
    let r = 1.0 - t;
    if t < 0.5 { 1.0 - 2.0 * t * t } else { 2.0 * r * r }
}

#[derive(Debug, Clone)]
pub struct G1Motion {
    /// `[frames][36]`.
    pub qpos: Vec<Vec<f32>>,
    pub foot_contacts: Vec<[f32; 4]>,
    pub tokens: usize,
}

impl<'a, B: Backend> G1Engine<'a, B> {
    pub fn new(model: &'a MotionBricks<B>) -> Result<Self> {
        if model.cfg.rig != Rig::G1 {
            return Err(Error::Config("G1Engine needs the G1 model".into()));
        }
        Ok(Self { model })
    }

    pub fn generate(&self, be: &B, context: &[Vec<f32>], target: &[Vec<f32>], cfg: &GenerateConfig) -> Result<G1Motion> {
        if context.len() != FRAMES_PER_TOKEN || target.is_empty() || target.len() > FRAMES_PER_TOKEN {
            return Err(Error::Config("need 4 context poses and 1 to 4 target poses".into()));
        }
        let nt = target.len();
        let root = read_root(&context[FRAMES_PER_TOKEN - 1]);
        let fwd = geo::quat_apply(&root.q, &[1.0, 0.0, 0.0]);
        let cur = fwd[1].atan2(fwd[0]);
        let des = cfg.heading_alignment[1].atan2(cfg.heading_alignment[0]);
        let align = Transform::new([root.p[0], root.p[1], 0.0], quat_z(cur - des));
        let aq = geo::quat_conjugate(&align.q);
        let enc = |q: &[f32]| -> Vec<f32> {
            let r = read_root(q);
            let p = geo::quat_apply(&aq, &geo::sub(&r.p, &align.p));
            let rot = geo::quat_mul(&aq, &r.q);
            let mut o = q.to_vec();
            write_root(&Transform::new([p[0], p[1], r.p[2]], rot), &mut o);
            o
        };
        let ctx: Vec<f32> = context.iter().flat_map(|q| enc(q)).collect();
        let tgt: Vec<f32> = (0..FRAMES_PER_TOKEN).flat_map(|i| enc(&target[i.min(nt - 1)])).collect();
        let (has_g, has_l, has_p) = conditions(nt);
        let out = self.model.predict_g1(
            be,
            &G1Inputs {
                context_qpos: ctx,
                target_qpos: tgt,
                has_global_root_values: has_g,
                has_local_root_values: has_l,
                has_local_poses: has_p,
                allowed: cfg.allowed()?,
            },
            None,
        );
        let tokens = out.pred_num_tokens;
        let raw_count = tokens * FRAMES_PER_TOKEN;
        let count = valid_frames(tokens);
        let pq = |f: usize| &out.pred_qpos[f * QPOS..(f + 1) * QPOS];
        let mut qpos = Vec::with_capacity(count);
        let mut contacts = Vec::with_capacity(count);
        let radius = G1_ROOT_FILTER.len() / 2;
        for fr in 0..count {
            let raw = fr + BOUNDARY_FRAMES;
            contacts.push(std::array::from_fn(|c| out.pred_foot_contacts[raw * 4 + c].clamp(0.0, 1.0)));
            let mut pos = [0f32; 3];
            for (tap, w) in G1_ROOT_FILTER.iter().enumerate() {
                let src = (raw + tap).saturating_sub(radius).min(raw_count - 1);
                let p = read_root(pq(src)).p;
                for c in 0..3 {
                    pos[c] += w * p[c];
                }
            }
            let r = align.mul(&Transform::new(pos, read_root(pq(raw)).q));
            let mut o = pq(raw).to_vec();
            write_root(&Transform::new([r.p[0], r.p[1], pos[2]], r.q), &mut o);
            qpos.push(o);
        }
        if cfg.correct_generated_diff && count >= 2 {
            let lc = &context[FRAMES_PER_TOKEN - 1];
            let t0 = &target[0];
            let lastf = count - 1;
            let ds = root_delta(&read_root(&qpos[0]), &read_root(lc));
            let de = root_delta(&read_root(&qpos[lastf]), &read_root(t0));
            let a_s: Vec<f32> = (7..QPOS).map(|i| lc[i] - qpos[0][i]).collect();
            let a_e: Vec<f32> = (7..QPOS).map(|i| t0[i] - qpos[lastf][i]).collect();
            let fade = (G1_ANGLE_FADE_SECONDS * 30.0).min(0.5 * lastf as f32);
            for (fr, q) in qpos.iter_mut().enumerate() {
                let we = fr as f32 / lastf as f32;
                let r = apply_root(&read_root(q), &ds, &de, we);
                write_root(&r, q);
                let ws_a = quadratic_fade(fr as f32 / fade);
                let we_a = quadratic_fade((lastf - fr) as f32 / fade);
                for i in 7..QPOS {
                    q[i] += a_s[i - 7] * ws_a + a_e[i - 7] * we_a;
                }
            }
        }
        Ok(G1Motion { qpos, foot_contacts: contacts, tokens })
    }
}

/// Re-exported for callers that build configs.
pub fn default_config(_cfg: &BricksConfig) -> GenerateConfig {
    GenerateConfig::default()
}
