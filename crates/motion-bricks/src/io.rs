//! Clip I/O for the engines.
//!
//! * BVH (the SOMA sample clips and the format MotionBricks / Kimodo export):
//!   `OFFSET`s and root positions in centimetres, Euler channels in degrees,
//!   each joint's rotation the product of its rotation channels in file order
//!   (`R = R_{c1}·R_{c2}·R_{c3}`). Read into meters and quaternions on a
//!   [`RigSkeleton`] whose reference transforms are the offsets.
//! * G1 CSV (the robot sample clips): a header row, then
//!   `frame, root x y z [cm], root roll pitch yaw [deg, extrinsic xyz], 29 joint angles [deg]`.

use crate::engine::{normalize, RigSkeleton, Transform};
use motion_core::geometry::{self as geo, Mat3};
use motion_core::{Error, Result};
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Clip {
    pub skeleton: RigSkeleton,
    /// `[frames][joints]` local transforms (root: position and rotation in the file's space).
    pub poses: Vec<Vec<Transform>>,
    pub frame_time: f32,
}

fn rot_axis(axis: usize, deg: f32) -> Mat3 {
    let (s, c) = deg.to_radians().sin_cos();
    match axis {
        0 => [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c],
        1 => [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c],
        _ => [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0],
    }
}

fn axis_of(ch: &str) -> Result<(bool, usize)> {
    let lower = ch.to_ascii_lowercase();
    let axis = match lower.as_bytes().first() {
        Some(b'x') => 0,
        Some(b'y') => 1,
        Some(b'z') => 2,
        _ => return Err(Error::Config(format!("bad BVH channel {ch}"))),
    };
    Ok((lower.ends_with("position"), axis))
}

/// Read a BVH file. End sites are skipped.
pub fn read_bvh(path: &Path) -> Result<Clip> {
    parse_bvh(&std::fs::read_to_string(path)?)
}

pub fn parse_bvh(text: &str) -> Result<Clip> {
    let mut names = Vec::new();
    let mut parents: Vec<i32> = Vec::new();
    let mut offsets: Vec<[f32; 3]> = Vec::new();
    let mut channels: Vec<Vec<String>> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut in_end = false;
    let mut lines = text.lines();
    let bad = |m: &str| Error::Config(format!("BVH: {m}"));
    for line in lines.by_ref() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "MOTION" => break,
            "ROOT" | "JOINT" => {
                names.push(t.get(1).ok_or_else(|| bad("joint without a name"))?.to_string());
                parents.push(stack.last().map_or(-1, |&p| p as i32));
                stack.push(names.len() - 1);
                offsets.push([0.0; 3]);
                channels.push(Vec::new());
            }
            "End" => in_end = true,
            "OFFSET" if !in_end => {
                let j = *stack.last().ok_or_else(|| bad("OFFSET outside a joint"))?;
                for k in 0..3 {
                    offsets[j][k] = t[1 + k].parse::<f32>().map_err(|_| bad("bad OFFSET"))? * 0.01;
                }
            }
            "CHANNELS" => {
                let j = *stack.last().ok_or_else(|| bad("CHANNELS outside a joint"))?;
                channels[j] = t[2..].iter().map(|s| s.to_string()).collect();
            }
            "}" => {
                if in_end {
                    in_end = false;
                } else {
                    stack.pop();
                }
            }
            _ => {}
        }
    }
    let mut frames = 0usize;
    let mut frame_time = 1.0 / 30.0;
    let mut data: Vec<Vec<f32>> = Vec::new();
    for line in lines {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(v) = l.strip_prefix("Frames:") {
            frames = v.trim().parse().map_err(|_| bad("bad Frames"))?;
        } else if let Some(v) = l.strip_prefix("Frame Time:") {
            frame_time = v.trim().parse().map_err(|_| bad("bad Frame Time"))?;
        } else {
            data.push(l.split_whitespace().map(|v| v.parse::<f32>().unwrap_or(0.0)).collect());
        }
    }
    data.truncate(frames.max(data.len().min(frames)));
    let mut poses = Vec::with_capacity(data.len());
    for row in &data {
        let mut col = 0;
        let mut pose = Vec::with_capacity(names.len());
        for (j, ch) in channels.iter().enumerate() {
            let mut p = offsets[j];
            let mut r = geo::IDENTITY;
            for c in ch {
                let v = *row.get(col).ok_or_else(|| bad("short frame row"))?;
                col += 1;
                let (is_pos, axis) = axis_of(c)?;
                if is_pos {
                    p[axis] = v * 0.01;
                } else {
                    r = geo::matmul(&r, &rot_axis(axis, v));
                }
            }
            pose.push(Transform::new(p, normalize(&geo::matrix_to_quat(&r))));
        }
        poses.push(pose);
    }
    let reference = offsets.iter().map(|o| Transform::new(*o, [1.0, 0.0, 0.0, 0.0])).collect();
    Ok(Clip { skeleton: RigSkeleton { names, parents, reference }, poses, frame_time })
}

impl Clip {
    /// Drop every ancestor of `root` (BVH exports wrap `Hips` in a `Root`
    /// joint), composing their transforms into it, so `root` becomes joint 0.
    pub fn rebase(mut self, root: &str) -> Result<Self> {
        let r = self.skeleton.index(root).ok_or_else(|| Error::Config(format!("no joint {root}")))?;
        let mut chain = vec![r];
        while self.skeleton.parents[*chain.last().expect("non-empty")] >= 0 {
            let p = self.skeleton.parents[*chain.last().expect("non-empty")] as usize;
            chain.push(p);
        }
        let ancestors: Vec<usize> = chain[1..].to_vec();
        if ancestors.is_empty() {
            return Ok(self);
        }
        // Keep `root` and its descendants, in file order.
        let keep: Vec<usize> = (0..self.skeleton.names.len())
            .filter(|&j| {
                let mut k = j as i32;
                while k >= 0 {
                    if k as usize == r {
                        return true;
                    }
                    k = self.skeleton.parents[k as usize];
                }
                false
            })
            .collect();
        let new_index = |old: usize| keep.iter().position(|&k| k == old);
        for pose in &mut self.poses {
            let mut t = pose[r];
            for &a in &ancestors {
                t = pose[a].mul(&t);
            }
            pose[r] = t;
            *pose = keep.iter().map(|&k| pose[k]).collect();
        }
        let mut reference: Vec<Transform> = keep.iter().map(|&k| self.skeleton.reference[k]).collect();
        let mut t0 = self.skeleton.reference[r];
        for &a in &ancestors {
            t0 = self.skeleton.reference[a].mul(&t0);
        }
        reference[0] = t0;
        self.skeleton = RigSkeleton {
            names: keep.iter().map(|&k| self.skeleton.names[k].clone()).collect(),
            parents: keep
                .iter()
                .map(|&k| {
                    let p = self.skeleton.parents[k];
                    if k == r { -1 } else { new_index(p as usize).map_or(-1, |i| i as i32) }
                })
                .collect(),
            reference,
        };
        Ok(self)
    }
}

/// `R = Rz(a)·Ry(b)·Rx(c)` → `(a, b, c)` in degrees.
fn euler_zyx(m: &Mat3) -> [f32; 3] {
    let b = (-m[6]).atan2((m[0] * m[0] + m[3] * m[3]).sqrt());
    let a = m[3].atan2(m[0]);
    let c = m[7].atan2(m[8]);
    [a.to_degrees(), b.to_degrees(), c.to_degrees()]
}

/// Write a BVH with `Xposition Yposition Zposition Zrotation Yrotation Xrotation`
/// on the root and `Zrotation Yrotation Xrotation` on every other joint.
pub fn write_bvh(skel: &RigSkeleton, poses: &[Vec<Transform>], fps: f32) -> String {
    let mut s = String::from("HIERARCHY\n");
    let children: Vec<Vec<usize>> =
        (0..skel.names.len()).map(|j| (0..skel.names.len()).filter(|&c| skel.parents[c] == j as i32).collect()).collect();
    fn emit(s: &mut String, skel: &RigSkeleton, children: &[Vec<usize>], j: usize, depth: usize) {
        let ind = "  ".repeat(depth);
        let kind = if skel.parents[j] < 0 { "ROOT" } else { "JOINT" };
        let o = skel.reference[j].p;
        let _ = writeln!(s, "{ind}{kind} {}\n{ind}{{", skel.names[j]);
        let _ = writeln!(s, "{ind}  OFFSET {:.6} {:.6} {:.6}", o[0] * 100.0, o[1] * 100.0, o[2] * 100.0);
        if skel.parents[j] < 0 {
            let _ = writeln!(s, "{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation");
        } else {
            let _ = writeln!(s, "{ind}  CHANNELS 3 Zrotation Yrotation Xrotation");
        }
        if children[j].is_empty() {
            let _ = writeln!(s, "{ind}  End Site\n{ind}  {{\n{ind}    OFFSET 0.0 0.0 0.0\n{ind}  }}");
        }
        for &c in &children[j] {
            emit(s, skel, children, c, depth + 1);
        }
        let _ = writeln!(s, "{ind}}}");
    }
    for j in 0..skel.names.len() {
        if skel.parents[j] < 0 {
            emit(&mut s, skel, &children, j, 0);
        }
    }
    // Channel order follows the hierarchy's depth-first order.
    let mut order = Vec::new();
    fn dfs(children: &[Vec<usize>], j: usize, out: &mut Vec<usize>) {
        out.push(j);
        for &c in &children[j] {
            dfs(children, c, out);
        }
    }
    for j in 0..skel.names.len() {
        if skel.parents[j] < 0 {
            dfs(&children, j, &mut order);
        }
    }
    let _ = writeln!(s, "MOTION\nFrames: {}\nFrame Time: {:.8}", poses.len(), 1.0 / fps);
    for pose in poses {
        let mut row = Vec::new();
        for &j in &order {
            if skel.parents[j] < 0 {
                for c in 0..3 {
                    row.push(format!("{:.6}", pose[j].p[c] * 100.0));
                }
            }
            let e = euler_zyx(&geo::quat_to_matrix(&pose[j].q));
            for v in e {
                row.push(format!("{v:.6}"));
            }
        }
        let _ = writeln!(s, "{}", row.join(" "));
    }
    s
}

/// A G1 joint-coordinate clip: `[frames][36]` (root xyz m, quaternion xyzw, 29 angles rad).
#[derive(Debug, Clone)]
pub struct DofClip {
    pub qpos: Vec<Vec<f32>>,
    pub fps: f32,
}

/// Extrinsic `xyz` Euler (SciPy `from_euler("xyz")`): `R = Rz(γ)·Ry(β)·Rx(α)`.
fn extrinsic_xyz(deg: [f32; 3]) -> Mat3 {
    geo::matmul(&rot_axis(2, deg[2]), &geo::matmul(&rot_axis(1, deg[1]), &rot_axis(0, deg[0])))
}

pub fn read_g1_csv(path: &Path, fps: f32) -> Result<DofClip> {
    let text = std::fs::read_to_string(path)?;
    let mut qpos = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let v: Vec<f32> = line.split(',').map(|x| x.trim().parse::<f32>().unwrap_or(0.0)).collect();
        if v.len() < 36 {
            return Err(Error::Config(format!("{}: row {i} has {} columns", path.display(), v.len())));
        }
        let q = normalize(&geo::matrix_to_quat(&extrinsic_xyz([v[4], v[5], v[6]])));
        let mut row = vec![v[1] * 0.01, v[2] * 0.01, v[3] * 0.01, q[1], q[2], q[3], q[0]];
        row.extend(v[7..36].iter().map(|d| d.to_radians()));
        qpos.push(row);
    }
    Ok(DofClip { qpos, fps })
}

/// Resample a clip to 30 Hz by taking every `round(fps/30)`-th frame.
pub fn every_nth<T: Clone>(frames: &[T], src_fps: f32) -> Vec<T> {
    let step = (src_fps / 30.0).round().max(1.0) as usize;
    frames.iter().step_by(step).cloned().collect()
}

/// The MuJoCo `qpos` CSV Kimodo and ProtoMotions read: one row per frame,
/// `x y z qw qx qy qz` then the joint angles, no header.
pub fn write_mujoco_csv(qpos: &[Vec<f32>]) -> String {
    let mut s = String::new();
    for q in qpos {
        let row: Vec<String> = [q[0], q[1], q[2], q[6], q[3], q[4], q[5]]
            .iter()
            .chain(q[7..].iter())
            .map(|v| format!("{v:.6}"))
            .collect();
        let _ = writeln!(s, "{}", row.join(","));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvh_round_trip() {
        let skel = RigSkeleton::soma_capture();
        let mut rng = motion_core::rng::Rng::new(5);
        let poses: Vec<Vec<Transform>> = (0..3)
            .map(|_| {
                (0..30)
                    .map(|j| {
                        let r = geo::axis_angle_to_matrix(&[rng.normal() * 0.6, rng.normal() * 0.6, rng.normal() * 0.6]);
                        let p = if j == 0 { [rng.normal(), 0.9, rng.normal()] } else { skel.reference[j].p };
                        Transform::new(p, geo::matrix_to_quat(&r))
                    })
                    .collect()
            })
            .collect();
        let text = write_bvh(&skel, &poses, 30.0);
        let clip = parse_bvh(&text).unwrap();
        assert_eq!(clip.skeleton.names, skel.names);
        for (a, b) in poses.iter().zip(&clip.poses) {
            for (ta, tb) in a.iter().zip(b) {
                let (ma, mb) = (geo::quat_to_matrix(&ta.q), geo::quat_to_matrix(&tb.q));
                for k in 0..9 {
                    assert!((ma[k] - mb[k]).abs() < 1e-4);
                }
                for k in 0..3 {
                    assert!((ta.p[k] - tb.p[k]).abs() < 1e-5);
                }
            }
        }
        assert!((clip.frame_time - 1.0 / 30.0).abs() < 1e-6);
    }
}
