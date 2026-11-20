//! Skeletons (joint names, hierarchy, neutral pose) and forward kinematics,
//! mirroring `kimodo/skeleton/{base,definitions,kinematics,transforms}.py`.
//! Rotations are flat `[frames, joints, 9]` row-major matrices
//! ([`crate::geometry`]), positions `[frames, joints, 3]`.
//!
//! The neutral joints and rest poses come from the embedded upstream assets
//! ([`crate::assets`]).

use crate::assets;
use crate::geometry::{self, Mat3, IDENTITY};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub name: &'static str,
    pub joint_names: Vec<&'static str>,
    /// Parent index per joint, `None` for the root (always joint 0).
    pub parents: Vec<Option<usize>>,
    /// Neutral (rest) joint positions `[J, 3]`, root at the origin.
    pub neutral_joints: Vec<f32>,
    /// Baked rest local rotations `[J, 9]` (G1: identity local = XML rest pose).
    pub rest_local_rots: Option<Vec<f32>>,
    /// Hierarchy levels: joints grouped by depth, each level after its parents.
    pub levels: Vec<Vec<usize>>,
    pub left_foot: Vec<usize>,
    pub right_foot: Vec<usize>,
    pub left_hand: Vec<usize>,
    pub right_hand: Vec<usize>,
    /// `[right, left]` hip joints, the heading reference.
    pub hips: [usize; 2],
}

impl Skeleton {
    pub fn nbjoints(&self) -> usize {
        self.joint_names.len()
    }

    pub fn root_idx(&self) -> usize {
        0
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.joint_names.iter().position(|n| *n == name)
    }

    pub fn index_of(&self, name: &str) -> Result<usize> {
        self.index(name).ok_or_else(|| Error::Config(format!("{}: no joint `{name}`", self.name)))
    }

    fn build(
        name: &'static str,
        bones: &[(&'static str, Option<&'static str>)],
        neutral: &[f32],
        rest_local: Option<&[f32]>,
        left_foot: &[&str],
        right_foot: &[&str],
        left_hand: &[&str],
        right_hand: &[&str],
        hips: [&str; 2],
    ) -> Self {
        let joint_names: Vec<&'static str> = bones.iter().map(|b| b.0).collect();
        let idx = |n: &str| joint_names.iter().position(|x| *x == n).expect("joint name");
        let parents: Vec<Option<usize>> = bones.iter().map(|b| b.1.map(idx)).collect();
        assert_eq!(parents[0], None, "root must be joint 0");
        assert_eq!(neutral.len(), joint_names.len() * 3);
        let mut depth = vec![0usize; joint_names.len()];
        let mut levels: Vec<Vec<usize>> = Vec::new();
        for j in 1..joint_names.len() {
            let p = parents[j].expect("non-root joint has a parent");
            assert!(p < j, "parents must precede children");
            depth[j] = depth[p] + 1;
            let l = depth[j] - 1;
            if levels.len() <= l {
                levels.push(Vec::new());
            }
            levels[l].push(j);
        }
        let to_idx = |names: &[&str]| names.iter().map(|n| idx(n)).collect::<Vec<_>>();
        Self {
            name,
            left_foot: to_idx(left_foot),
            right_foot: to_idx(right_foot),
            left_hand: to_idx(left_hand),
            right_hand: to_idx(right_hand),
            hips: [idx(hips[0]), idx(hips[1])],
            joint_names,
            parents,
            neutral_joints: neutral.to_vec(),
            rest_local_rots: rest_local.map(|r| r.to_vec()),
            levels,
        }
    }

    /// The 30-joint SOMA skeleton the Kimodo-SOMA and MotionBricks-SOMA models predict.
    pub fn soma30() -> Self {
        Self::build(
            "somaskel30",
            SOMA30_BONES,
            &assets::SOMASKEL30_JOINTS,
            None,
            &["LeftFoot", "LeftToeBase"],
            &["RightFoot", "RightToeBase"],
            &["LeftHand", "LeftHandMiddleEnd"],
            &["RightHand", "RightHandMiddleEnd"],
            ["RightLeg", "LeftLeg"],
        )
    }

    /// The 77-joint SOMA skeleton (fingers, toes) Kimodo outputs for export.
    pub fn soma77() -> Self {
        Self::build(
            "somaskel77",
            SOMA77_BONES,
            &assets::SOMASKEL77_JOINTS,
            None,
            &["LeftFoot", "LeftToeBase", "LeftToeEnd"],
            &["RightFoot", "RightToeBase", "RightToeEnd"],
            &["LeftHand"],
            &["RightHand"],
            ["RightLeg", "LeftLeg"],
        )
    }

    /// The 34-joint Unitree G1 skeleton (32 articulated plus two toe ends).
    pub fn g1_34() -> Self {
        Self::build(
            "g1skel34",
            G1_BONES,
            &assets::G1SKEL34_JOINTS,
            // Kimodo stores `rest_pose_local_rot.p` but its FK reads `rest_local_rots`,
            // which no skeleton defines: the rest rotations are never applied.
            None,
            &["left_ankle_roll_skel", "left_toe_base"],
            &["right_ankle_roll_skel", "right_toe_base"],
            &["left_wrist_yaw_skel", "left_hand_roll_skel"],
            &["right_wrist_yaw_skel", "right_hand_roll_skel"],
            ["right_hip_pitch_skel", "left_hip_pitch_skel"],
        )
    }

    /// SMPL-X body (22 joints).
    pub fn smplx22() -> Self {
        Self::build(
            "smplx22",
            SMPLX22_BONES,
            &assets::SMPLX22_JOINTS,
            None,
            &["left_ankle", "left_foot"],
            &["right_ankle", "right_foot"],
            &["left_wrist"],
            &["right_wrist"],
            ["right_hip", "left_hip"],
        )
    }

    pub fn by_joint_count(n: usize) -> Result<Self> {
        Ok(match n {
            30 => Self::soma30(),
            77 => Self::soma77(),
            34 => Self::g1_34(),
            22 => Self::smplx22(),
            _ => return Err(Error::Config(format!("no skeleton with {n} joints"))),
        })
    }

    /// Bone offset of each joint from its parent in the neutral pose (root: its own position).
    pub fn rel_joints(&self) -> Vec<f32> {
        let j = self.nbjoints();
        let mut rel = vec![0.0; j * 3];
        for i in 0..j {
            let p = geometry::vec_at(&self.neutral_joints, i);
            let v = match self.parents[i] {
                Some(par) => geometry::sub(&p, &geometry::vec_at(&self.neutral_joints, par)),
                None => p,
            };
            geometry::set_vec(&mut rel, i, &v);
        }
        rel
    }

    /// Forward kinematics (`kimodo.skeleton.kinematics.fk` with
    /// `root_positions_is_global=True`): `local_rots` `[T, J, 9]`, `root_pos`
    /// `[T, 3]`. Returns `(global_rots [T, J, 9], posed [T, J, 3], posed_norootpos [T, J, 3])`.
    pub fn fk(&self, local_rots: &[f32], root_pos: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let j = self.nbjoints();
        let frames = root_pos.len() / 3;
        assert_eq!(local_rots.len(), frames * j * 9, "fk: local rotation count");
        // Neutral joints recentred on the root.
        let root0 = geometry::vec_at(&self.neutral_joints, 0);
        let mut rel = self.rel_joints();
        geometry::set_vec(&mut rel, 0, &geometry::sub(&root0, &root0));
        let mut g = vec![0.0; frames * j * 9];
        let mut p = vec![0.0; frames * j * 3];
        let mut p0 = vec![0.0; frames * j * 3];
        for t in 0..frames {
            let base = t * j;
            let local = |i: usize| -> Mat3 {
                let m = geometry::mat_at(local_rots, base + i);
                match &self.rest_local_rots {
                    Some(r) => geometry::matmul(&geometry::mat_at(r, i), &m),
                    None => m,
                }
            };
            geometry::set_mat(&mut g, base, &local(0));
            geometry::set_vec(&mut p0, base, &[0.0, 0.0, 0.0]);
            for level in &self.levels {
                for &i in level {
                    let par = self.parents[i].expect("child");
                    let gp = geometry::mat_at(&g, base + par);
                    let gi = geometry::matmul(&gp, &local(i));
                    geometry::set_mat(&mut g, base + i, &gi);
                    let pos = geometry::add(
                        &geometry::vec_at(&p0, base + par),
                        &geometry::apply(&gp, &geometry::vec_at(&rel, i)),
                    );
                    geometry::set_vec(&mut p0, base + i, &pos);
                }
            }
            let rp = geometry::vec_at(root_pos, t);
            for i in 0..j {
                geometry::set_vec(&mut p, base + i, &geometry::add(&geometry::vec_at(&p0, base + i), &rp));
            }
        }
        (g, p, p0)
    }

    /// Joint positions straight from global rotations
    /// (`joints_from_global_rots`): `p_j = p_parent + R_g[parent]·rel_j`.
    /// Returns `(posed [T, J, 3], posed_norootpos)`.
    pub fn joints_from_global_rots(&self, global_rots: &[f32], root_pos: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let j = self.nbjoints();
        let frames = root_pos.len() / 3;
        assert_eq!(global_rots.len(), frames * j * 9);
        let rel = self.rel_joints();
        let mut p = vec![0.0; frames * j * 3];
        let mut p0 = vec![0.0; frames * j * 3];
        for t in 0..frames {
            let base = t * j;
            for level in &self.levels {
                for &i in level {
                    let par = self.parents[i].expect("child");
                    let gp = geometry::mat_at(global_rots, base + par);
                    let pos = geometry::add(
                        &geometry::vec_at(&p0, base + par),
                        &geometry::apply(&gp, &geometry::vec_at(&rel, i)),
                    );
                    geometry::set_vec(&mut p0, base + i, &pos);
                }
            }
            let rp = geometry::vec_at(root_pos, t);
            for i in 0..j {
                geometry::set_vec(&mut p, base + i, &geometry::add(&geometry::vec_at(&p0, base + i), &rp));
            }
        }
        (p, p0)
    }

    /// `local_j = R_g[parent]ᵀ · R_g[j]` (root: parent identity).
    pub fn global_rots_to_local(&self, global_rots: &[f32]) -> Vec<f32> {
        let j = self.nbjoints();
        let frames = global_rots.len() / (j * 9);
        let mut out = vec![0.0; global_rots.len()];
        for t in 0..frames {
            for i in 0..j {
                let g = geometry::mat_at(global_rots, t * j + i);
                let pt = match self.parents[i] {
                    Some(p) => geometry::transpose(&geometry::mat_at(global_rots, t * j + p)),
                    None => IDENTITY,
                };
                geometry::set_mat(&mut out, t * j + i, &geometry::matmul(&pt, &g));
            }
        }
        out
    }

    /// Index of each of this skeleton's joints inside `other` (by name).
    pub fn slice_into(&self, other: &Skeleton) -> Result<Vec<usize>> {
        self.joint_names.iter().map(|n| other.index_of(n)).collect()
    }

    /// Heading angle per frame from the hip joints as Kimodo defines it:
    /// `atan2(dz, −dx)` of `right_hip − left_hip`.
    pub fn heading_angle_kimodo(&self, posed: &[f32]) -> Vec<f32> {
        let j = self.nbjoints();
        let frames = posed.len() / (j * 3);
        (0..frames)
            .map(|t| {
                let r = geometry::vec_at(posed, t * j + self.hips[0]);
                let l = geometry::vec_at(posed, t * j + self.hips[1]);
                let d = geometry::sub(&r, &l);
                d[2].atan2(-d[0])
            })
            .collect()
    }

    /// Lowest neutral-joint height (toe), used to seat a rig on the ground.
    pub fn min_height(&self) -> f32 {
        (0..self.nbjoints()).map(|i| self.neutral_joints[i * 3 + 1]).fold(f32::INFINITY, f32::min)
    }
}

/// SOMA 30 → 77: the relaxed-hand rest pose with the 30 predicted joints
/// written in (`SOMASkeleton30.to_SOMASkeleton77`). `local_rots` `[T, 30, 9]` → `[T, 77, 9]`.
pub fn soma30_to_77(local_rots: &[f32], skel30: &Skeleton, skel77: &Skeleton) -> Result<Vec<f32>> {
    let slice = skel30.slice_into(skel77)?;
    let frames = local_rots.len() / (30 * 9);
    let rest = &assets::SOMASKEL77_RELAXED_HANDS_REST_POSE;
    let mut out = Vec::with_capacity(frames * 77 * 9);
    for t in 0..frames {
        let mut frame = rest.to_vec();
        for (i30, &i77) in slice.iter().enumerate() {
            frame[i77 * 9..i77 * 9 + 9].copy_from_slice(&local_rots[(t * 30 + i30) * 9..(t * 30 + i30 + 1) * 9]);
        }
        out.extend_from_slice(&frame);
    }
    Ok(out)
}

/// SOMA 77 → 30: pick the subset.
pub fn soma77_to_30(local_rots: &[f32], skel30: &Skeleton, skel77: &Skeleton) -> Result<Vec<f32>> {
    let slice = skel30.slice_into(skel77)?;
    let frames = local_rots.len() / (77 * 9);
    let mut out = Vec::with_capacity(frames * 30 * 9);
    for t in 0..frames {
        for &i77 in &slice {
            out.extend_from_slice(&local_rots[(t * 77 + i77) * 9..(t * 77 + i77 + 1) * 9]);
        }
    }
    Ok(out)
}

/// Re-express local rotations in another T-pose given global rotation offsets
/// (`change_tpose`): its einsum `"T N m n, N o n -> T N m o"` is `R'_g = R_g · O_jᵀ`,
/// back to locals.
pub fn change_tpose(local_rots: &[f32], offsets: &[f32], skel: &Skeleton) -> (Vec<f32>, Vec<f32>) {
    let j = skel.nbjoints();
    let frames = local_rots.len() / (j * 9);
    let dummy_root = vec![0.0; frames * 3];
    let (g, _, _) = skel.fk(local_rots, &dummy_root);
    let mut ng = vec![0.0; g.len()];
    for t in 0..frames {
        for i in 0..j {
            let m = geometry::matmul(&geometry::mat_at(&g, t * j + i), &geometry::transpose(&geometry::mat_at(offsets, i)));
            geometry::set_mat(&mut ng, t * j + i, &m);
        }
    }
    let nl = skel.global_rots_to_local(&ng);
    (nl, ng)
}

/// `to_standard_tpose` for SOMA 77 (the offsets as stored).
pub fn soma77_to_standard_tpose(local_rots: &[f32], skel77: &Skeleton) -> (Vec<f32>, Vec<f32>) {
    change_tpose(local_rots, &assets::SOMASKEL77_STANDARD_TPOSE_GLOBAL_OFFSETS, skel77)
}

/// `from_standard_tpose` for SOMA 77 (transposed offsets).
pub fn soma77_from_standard_tpose(local_rots: &[f32], skel77: &Skeleton) -> (Vec<f32>, Vec<f32>) {
    let mut t = vec![0.0; 77 * 9];
    for i in 0..77 {
        geometry::set_mat(
            &mut t,
            i,
            &geometry::transpose(&geometry::mat_at(&assets::SOMASKEL77_STANDARD_TPOSE_GLOBAL_OFFSETS, i)),
        );
    }
    change_tpose(local_rots, &t, skel77)
}

/// The G1 rest-pose local rotations `[34, 9]` (`g1skel34/rest_pose_local_rot.p`),
/// for callers that want the MuJoCo XML rest orientation.
pub fn g1_rest_pose_local_rot() -> &'static [f32] {
    &assets::G1SKEL34_REST_POSE_LOCAL_ROT
}

/// The BVH rest-pose joints of SOMA 77 (`bvh_joints.p`), `[77, 3]`.
pub fn soma77_bvh_neutral_joints() -> &'static [f32] {
    &assets::SOMASKEL77_BVH_JOINTS
}

pub const SOMA30_BONES: &[(&str, Option<&str>)] = &[
    ("Hips", None),
    ("Spine1", Some("Hips")),
    ("Spine2", Some("Spine1")),
    ("Chest", Some("Spine2")),
    ("Neck1", Some("Chest")),
    ("Neck2", Some("Neck1")),
    ("Head", Some("Neck2")),
    ("Jaw", Some("Head")),
    ("LeftEye", Some("Head")),
    ("RightEye", Some("Head")),
    ("LeftShoulder", Some("Chest")),
    ("LeftArm", Some("LeftShoulder")),
    ("LeftForeArm", Some("LeftArm")),
    ("LeftHand", Some("LeftForeArm")),
    ("LeftHandThumbEnd", Some("LeftHand")),
    ("LeftHandMiddleEnd", Some("LeftHand")),
    ("RightShoulder", Some("Chest")),
    ("RightArm", Some("RightShoulder")),
    ("RightForeArm", Some("RightArm")),
    ("RightHand", Some("RightForeArm")),
    ("RightHandThumbEnd", Some("RightHand")),
    ("RightHandMiddleEnd", Some("RightHand")),
    ("LeftLeg", Some("Hips")),
    ("LeftShin", Some("LeftLeg")),
    ("LeftFoot", Some("LeftShin")),
    ("LeftToeBase", Some("LeftFoot")),
    ("RightLeg", Some("Hips")),
    ("RightShin", Some("RightLeg")),
    ("RightFoot", Some("RightShin")),
    ("RightToeBase", Some("RightFoot")),
];

pub const SOMA77_BONES: &[(&str, Option<&str>)] = &[
    ("Hips", None),
    ("Spine1", Some("Hips")),
    ("Spine2", Some("Spine1")),
    ("Chest", Some("Spine2")),
    ("Neck1", Some("Chest")),
    ("Neck2", Some("Neck1")),
    ("Head", Some("Neck2")),
    ("HeadEnd", Some("Head")),
    ("Jaw", Some("Head")),
    ("LeftEye", Some("Head")),
    ("RightEye", Some("Head")),
    ("LeftShoulder", Some("Chest")),
    ("LeftArm", Some("LeftShoulder")),
    ("LeftForeArm", Some("LeftArm")),
    ("LeftHand", Some("LeftForeArm")),
    ("LeftHandThumb1", Some("LeftHand")),
    ("LeftHandThumb2", Some("LeftHandThumb1")),
    ("LeftHandThumb3", Some("LeftHandThumb2")),
    ("LeftHandThumbEnd", Some("LeftHandThumb3")),
    ("LeftHandIndex1", Some("LeftHand")),
    ("LeftHandIndex2", Some("LeftHandIndex1")),
    ("LeftHandIndex3", Some("LeftHandIndex2")),
    ("LeftHandIndex4", Some("LeftHandIndex3")),
    ("LeftHandIndexEnd", Some("LeftHandIndex4")),
    ("LeftHandMiddle1", Some("LeftHand")),
    ("LeftHandMiddle2", Some("LeftHandMiddle1")),
    ("LeftHandMiddle3", Some("LeftHandMiddle2")),
    ("LeftHandMiddle4", Some("LeftHandMiddle3")),
    ("LeftHandMiddleEnd", Some("LeftHandMiddle4")),
    ("LeftHandRing1", Some("LeftHand")),
    ("LeftHandRing2", Some("LeftHandRing1")),
    ("LeftHandRing3", Some("LeftHandRing2")),
    ("LeftHandRing4", Some("LeftHandRing3")),
    ("LeftHandRingEnd", Some("LeftHandRing4")),
    ("LeftHandPinky1", Some("LeftHand")),
    ("LeftHandPinky2", Some("LeftHandPinky1")),
    ("LeftHandPinky3", Some("LeftHandPinky2")),
    ("LeftHandPinky4", Some("LeftHandPinky3")),
    ("LeftHandPinkyEnd", Some("LeftHandPinky4")),
    ("RightShoulder", Some("Chest")),
    ("RightArm", Some("RightShoulder")),
    ("RightForeArm", Some("RightArm")),
    ("RightHand", Some("RightForeArm")),
    ("RightHandThumb1", Some("RightHand")),
    ("RightHandThumb2", Some("RightHandThumb1")),
    ("RightHandThumb3", Some("RightHandThumb2")),
    ("RightHandThumbEnd", Some("RightHandThumb3")),
    ("RightHandIndex1", Some("RightHand")),
    ("RightHandIndex2", Some("RightHandIndex1")),
    ("RightHandIndex3", Some("RightHandIndex2")),
    ("RightHandIndex4", Some("RightHandIndex3")),
    ("RightHandIndexEnd", Some("RightHandIndex4")),
    ("RightHandMiddle1", Some("RightHand")),
    ("RightHandMiddle2", Some("RightHandMiddle1")),
    ("RightHandMiddle3", Some("RightHandMiddle2")),
    ("RightHandMiddle4", Some("RightHandMiddle3")),
    ("RightHandMiddleEnd", Some("RightHandMiddle4")),
    ("RightHandRing1", Some("RightHand")),
    ("RightHandRing2", Some("RightHandRing1")),
    ("RightHandRing3", Some("RightHandRing2")),
    ("RightHandRing4", Some("RightHandRing3")),
    ("RightHandRingEnd", Some("RightHandRing4")),
    ("RightHandPinky1", Some("RightHand")),
    ("RightHandPinky2", Some("RightHandPinky1")),
    ("RightHandPinky3", Some("RightHandPinky2")),
    ("RightHandPinky4", Some("RightHandPinky3")),
    ("RightHandPinkyEnd", Some("RightHandPinky4")),
    ("LeftLeg", Some("Hips")),
    ("LeftShin", Some("LeftLeg")),
    ("LeftFoot", Some("LeftShin")),
    ("LeftToeBase", Some("LeftFoot")),
    ("LeftToeEnd", Some("LeftToeBase")),
    ("RightLeg", Some("Hips")),
    ("RightShin", Some("RightLeg")),
    ("RightFoot", Some("RightShin")),
    ("RightToeBase", Some("RightFoot")),
    ("RightToeEnd", Some("RightToeBase")),
];

pub const G1_BONES: &[(&str, Option<&str>)] = &[
    ("pelvis_skel", None),
    ("left_hip_pitch_skel", Some("pelvis_skel")),
    ("left_hip_roll_skel", Some("left_hip_pitch_skel")),
    ("left_hip_yaw_skel", Some("left_hip_roll_skel")),
    ("left_knee_skel", Some("left_hip_yaw_skel")),
    ("left_ankle_pitch_skel", Some("left_knee_skel")),
    ("left_ankle_roll_skel", Some("left_ankle_pitch_skel")),
    ("left_toe_base", Some("left_ankle_roll_skel")),
    ("right_hip_pitch_skel", Some("pelvis_skel")),
    ("right_hip_roll_skel", Some("right_hip_pitch_skel")),
    ("right_hip_yaw_skel", Some("right_hip_roll_skel")),
    ("right_knee_skel", Some("right_hip_yaw_skel")),
    ("right_ankle_pitch_skel", Some("right_knee_skel")),
    ("right_ankle_roll_skel", Some("right_ankle_pitch_skel")),
    ("right_toe_base", Some("right_ankle_roll_skel")),
    ("waist_yaw_skel", Some("pelvis_skel")),
    ("waist_roll_skel", Some("waist_yaw_skel")),
    ("waist_pitch_skel", Some("waist_roll_skel")),
    ("left_shoulder_pitch_skel", Some("waist_pitch_skel")),
    ("left_shoulder_roll_skel", Some("left_shoulder_pitch_skel")),
    ("left_shoulder_yaw_skel", Some("left_shoulder_roll_skel")),
    ("left_elbow_skel", Some("left_shoulder_yaw_skel")),
    ("left_wrist_roll_skel", Some("left_elbow_skel")),
    ("left_wrist_pitch_skel", Some("left_wrist_roll_skel")),
    ("left_wrist_yaw_skel", Some("left_wrist_pitch_skel")),
    ("left_hand_roll_skel", Some("left_wrist_yaw_skel")),
    ("right_shoulder_pitch_skel", Some("waist_pitch_skel")),
    ("right_shoulder_roll_skel", Some("right_shoulder_pitch_skel")),
    ("right_shoulder_yaw_skel", Some("right_shoulder_roll_skel")),
    ("right_elbow_skel", Some("right_shoulder_yaw_skel")),
    ("right_wrist_roll_skel", Some("right_elbow_skel")),
    ("right_wrist_pitch_skel", Some("right_wrist_roll_skel")),
    ("right_wrist_yaw_skel", Some("right_wrist_pitch_skel")),
    ("right_hand_roll_skel", Some("right_wrist_yaw_skel")),
];

pub const SMPLX22_BONES: &[(&str, Option<&str>)] = &[
    ("pelvis", None),
    ("left_hip", Some("pelvis")),
    ("right_hip", Some("pelvis")),
    ("spine1", Some("pelvis")),
    ("left_knee", Some("left_hip")),
    ("right_knee", Some("right_hip")),
    ("spine2", Some("spine1")),
    ("left_ankle", Some("left_knee")),
    ("right_ankle", Some("right_knee")),
    ("spine3", Some("spine2")),
    ("left_foot", Some("left_ankle")),
    ("right_foot", Some("right_ankle")),
    ("neck", Some("spine3")),
    ("left_collar", Some("spine3")),
    ("right_collar", Some("spine3")),
    ("head", Some("neck")),
    ("left_shoulder", Some("left_collar")),
    ("right_shoulder", Some("right_collar")),
    ("left_elbow", Some("left_shoulder")),
    ("right_elbow", Some("right_shoulder")),
    ("left_wrist", Some("left_elbow")),
    ("right_wrist", Some("right_elbow")),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeletons_build() {
        for s in [Skeleton::soma30(), Skeleton::soma77(), Skeleton::g1_34(), Skeleton::smplx22()] {
            assert_eq!(s.neutral_joints.len(), s.nbjoints() * 3, "{}", s.name);
            assert_eq!(s.parents[0], None);
        }
        assert_eq!(Skeleton::soma30().nbjoints(), 30);
        assert_eq!(Skeleton::soma77().nbjoints(), 77);
    }

    #[test]
    fn identity_fk_reproduces_neutral_pose() {
        let s = Skeleton::soma30();
        let j = s.nbjoints();
        let local: Vec<f32> = (0..j).flat_map(|_| IDENTITY).collect();
        let (g, p, p0) = s.fk(&local, &[1.0, 2.0, 3.0]);
        for i in 0..j {
            let n = geometry::vec_at(&s.neutral_joints, i);
            let got = geometry::vec_at(&p0, i);
            for k in 0..3 {
                assert!((n[k] - got[k]).abs() < 1e-5);
            }
            let w = geometry::vec_at(&p, i);
            assert!((w[0] - n[0] - 1.0).abs() < 1e-5);
        }
        let (p2, _) = s.joints_from_global_rots(&g, &[1.0, 2.0, 3.0]);
        assert_eq!(p, p2);
        let l2 = s.global_rots_to_local(&g);
        assert_eq!(l2, local);
    }

    #[test]
    fn fk_global_local_round_trip() {
        let s = Skeleton::soma30();
        let j = s.nbjoints();
        let mut rng = crate::rng::Rng::new(3);
        let local: Vec<f32> = (0..j)
            .flat_map(|_| geometry::axis_angle_to_matrix(&[rng.normal() * 0.5, rng.normal() * 0.5, rng.normal() * 0.5]))
            .collect();
        let (g, p, _) = s.fk(&local, &[0.1, 0.9, -0.3]);
        let back = s.global_rots_to_local(&g);
        for (a, b) in local.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5);
        }
        let (p2, _) = s.joints_from_global_rots(&g, &[0.1, 0.9, -0.3]);
        for (a, b) in p.iter().zip(&p2) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn soma_30_77_round_trip() {
        let s30 = Skeleton::soma30();
        let s77 = Skeleton::soma77();
        let local: Vec<f32> = (0..30).flat_map(|i| geometry::rot_y(i as f32 * 0.1)).collect();
        let full = soma30_to_77(&local, &s30, &s77).unwrap();
        assert_eq!(full.len(), 77 * 9);
        let back = soma77_to_30(&full, &s30, &s77).unwrap();
        assert_eq!(back, local);
    }
}
