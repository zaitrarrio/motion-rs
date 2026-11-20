//! Between the two models' pose formats. Kimodo emits SOMA-77 local rotations
//! in its standard T-pose; MotionBricks reads poses on the BONES capture rest
//! pose (the sample clips' BVH skeleton). Kimodo's BVH export already converts
//! to that rest pose (`from_standard_tpose` + `bvh_joints.p`), and its bone
//! offsets equal the clips' to the micrometre, so the conversion is exact.

use motion_bricks::engine::{RigSkeleton, Transform};
use motion_core::geometry as geo;
use motion_core::skeleton::{self, Skeleton};
use motion_kimodo::MotionOutput;

/// The SOMA-77 capture skeleton (`bvh_joints.p`), root at the origin.
pub fn soma77_capture() -> RigSkeleton {
    let s = Skeleton::soma77();
    let n = skeleton::soma77_bvh_neutral_joints();
    let reference = (0..77)
        .map(|j| {
            let p = match s.parents[j] {
                Some(par) => geo::sub(&geo::vec_at(n, j), &geo::vec_at(n, par)),
                None => geo::vec_at(n, j),
            };
            Transform::new(p, [1.0, 0.0, 0.0, 0.0])
        })
        .collect();
    RigSkeleton {
        names: s.joint_names.iter().map(|x| x.to_string()).collect(),
        parents: s.parents.iter().map(|p| p.map_or(-1, |x| x as i32)).collect(),
        reference,
    }
}

/// Kimodo output (SOMA-77) → per-frame local transforms on [`soma77_capture`].
pub fn kimodo_to_capture(out: &MotionOutput) -> Vec<Vec<Transform>> {
    assert_eq!(out.skeleton.name, "somaskel77", "Kimodo output must be on SOMA-77");
    let (local, _) = skeleton::soma77_from_standard_tpose(&out.local_rot_mats, &out.skeleton);
    let rig = soma77_capture();
    (0..out.frames)
        .map(|t| {
            (0..77)
                .map(|j| {
                    let q = geo::matrix_to_quat(&geo::mat_at(&local, t * 77 + j));
                    let p = if j == 0 { geo::vec_at(&out.root_positions, t) } else { rig.reference[j].p };
                    Transform::new(p, q)
                })
                .collect()
        })
        .collect()
}

/// World joint positions `[T][J·3]` for the viewer.
pub fn world_positions(rig: &RigSkeleton, poses: &[Vec<Transform>]) -> Vec<Vec<f32>> {
    poses.iter().map(|p| rig.world(p).iter().flat_map(|t| t.p).collect()).collect()
}
