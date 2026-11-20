//! Kinematic constraints (`kimodo/constraints.py`): root 2-D waypoints / paths,
//! full-body keyframes and end-effector (hands, feet, hips) keyframes, loaded
//! from the demo's `constraints.json` and written into the denoiser's
//! observed features and mask ([`crate::rep::KimodoMotionRep::create_conditions`]).

use motion_core::geometry::{self as geo, Mat3, Vec3};
use motion_core::skeleton::{self, Skeleton};
use motion_core::{Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Kind {
    Root2d { heading: Option<Vec<[f32; 2]>> },
    FullBody,
    /// Base end-effector names (`LeftFoot`, `RightFoot`, `LeftHand`, `RightHand`, `Hips`).
    EndEffector { name: String, joints: Vec<String> },
}

/// One constraint set over `frames`.
#[derive(Debug, Clone)]
pub struct Constraint {
    pub kind: Kind,
    pub frames: Vec<usize>,
    pub smooth_root_2d: Vec<[f32; 2]>,
    /// `[K, J, 3]` / `[K, J, 9]` global positions and rotations (empty for root-only).
    pub positions: Vec<f32>,
    pub rotations: Vec<f32>,
}

/// The per-feature dictionaries `build_condition_dicts` fills, deduplicated by index.
#[derive(Debug, Default, Clone)]
pub struct ConditionData {
    pub smooth_root_2d: BTreeMap<usize, [f32; 2]>,
    pub root_y: BTreeMap<usize, f32>,
    pub heading: BTreeMap<usize, [f32; 2]>,
    pub positions: BTreeMap<(usize, usize), Vec3>,
    pub rotations: BTreeMap<(usize, usize), Mat3>,
}

fn heading_cs(sk: &Skeleton, pos: &[f32], k: usize) -> [f32; 2] {
    let j = sk.nbjoints();
    let a = sk.heading_angle_kimodo(&pos[k * j * 3..(k + 1) * j * 3])[0];
    [a.cos(), a.sin()]
}

impl Constraint {
    pub fn name(&self) -> &str {
        match &self.kind {
            Kind::Root2d { .. } => "root2d",
            Kind::FullBody => "fullbody",
            Kind::EndEffector { name, .. } => name,
        }
    }

    pub fn root2d(frames: Vec<usize>, path: Vec<[f32; 2]>, heading: Option<Vec<[f32; 2]>>) -> Self {
        Self { kind: Kind::Root2d { heading }, frames, smooth_root_2d: path, positions: vec![], rotations: vec![] }
    }

    /// A full-body keyframe set from global positions / rotations (`FullBodyConstraintSet`);
    /// `smooth_root_2d` defaults to the pelvis x / z.
    pub fn full_body(sk: &Skeleton, frames: Vec<usize>, pos: Vec<f32>, rot: Vec<f32>, smooth: Option<Vec<[f32; 2]>>) -> Self {
        let j = sk.nbjoints();
        let smooth = smooth.unwrap_or_else(|| (0..frames.len()).map(|k| [pos[k * j * 3], pos[k * j * 3 + 2]]).collect());
        Self { kind: Kind::FullBody, frames, smooth_root_2d: smooth, positions: pos, rotations: rot }
    }

    pub fn end_effector(
        sk: &Skeleton,
        name: &str,
        joints: Vec<String>,
        frames: Vec<usize>,
        pos: Vec<f32>,
        rot: Vec<f32>,
        smooth: Option<Vec<[f32; 2]>>,
    ) -> Self {
        let j = sk.nbjoints();
        let smooth = smooth.unwrap_or_else(|| (0..frames.len()).map(|k| [pos[k * j * 3], pos[k * j * 3 + 2]]).collect());
        Self {
            kind: Kind::EndEffector { name: name.to_string(), joints },
            frames,
            smooth_root_2d: smooth,
            positions: pos,
            rotations: rot,
        }
    }

    /// `crop_move(start, end)`: keep frames in `[start, end)`, shifted to start at 0.
    pub fn crop_move(&self, start: usize, end: usize) -> Self {
        let keep: Vec<usize> = (0..self.frames.len()).filter(|&k| self.frames[k] >= start && self.frames[k] < end).collect();
        let per = |v: &[f32], w: usize| -> Vec<f32> {
            if v.is_empty() { vec![] } else { keep.iter().flat_map(|&k| v[k * w..(k + 1) * w].to_vec()).collect() }
        };
        let jw = if self.frames.is_empty() { 0 } else { self.positions.len() / self.frames.len().max(1) };
        let rw = if self.frames.is_empty() { 0 } else { self.rotations.len() / self.frames.len().max(1) };
        let kind = match &self.kind {
            Kind::Root2d { heading } => Kind::Root2d { heading: heading.as_ref().map(|h| keep.iter().map(|&k| h[k]).collect()) },
            k => k.clone(),
        };
        Self {
            kind,
            frames: keep.iter().map(|&k| self.frames[k] - start).collect(),
            smooth_root_2d: keep.iter().map(|&k| self.smooth_root_2d[k]).collect(),
            positions: per(&self.positions, jw),
            rotations: per(&self.rotations, rw),
        }
    }

    /// `update_constraints(data_dict, index_dict)`.
    pub fn update(&self, sk: &Skeleton, d: &mut ConditionData) -> Result<()> {
        let j = sk.nbjoints();
        for (k, &t) in self.frames.iter().enumerate() {
            d.smooth_root_2d.insert(t, self.smooth_root_2d[k]);
        }
        match &self.kind {
            Kind::Root2d { heading } => {
                if let Some(h) = heading {
                    for (k, &t) in self.frames.iter().enumerate() {
                        d.heading.insert(t, h[k]);
                    }
                }
            }
            Kind::FullBody => {
                for (k, &t) in self.frames.iter().enumerate() {
                    for jj in 0..j {
                        d.positions.insert((t, jj), geo::vec_at(&self.positions, k * j + jj));
                    }
                    d.root_y.insert(t, self.positions[k * j * 3 + 1]);
                    d.heading.insert(t, heading_cs(sk, &self.positions, k));
                }
            }
            Kind::EndEffector { joints, .. } => {
                let (rot_names, pos_names) = expand_joint_names(sk, joints)?;
                for (k, &t) in self.frames.iter().enumerate() {
                    for n in &pos_names {
                        let jj = sk.index_of(n)?;
                        d.positions.insert((t, jj), geo::vec_at(&self.positions, k * j + jj));
                    }
                    for n in &rot_names {
                        let jj = sk.index_of(n)?;
                        d.rotations.insert((t, jj), geo::mat_at(&self.rotations, k * j + jj));
                    }
                    d.root_y.insert(t, self.positions[k * j * 3 + 1]);
                    d.heading.insert(t, heading_cs(sk, &self.positions, k));
                }
            }
        }
        Ok(())
    }
}

/// `SkeletonBase.expand_joint_names`: base end-effector names → (rotation joints, position joints).
pub fn expand_joint_names(sk: &Skeleton, names: &[String]) -> Result<(Vec<String>, Vec<String>)> {
    let n = |ids: &[usize]| ids.iter().map(|&i| sk.joint_names[i].to_string()).collect::<Vec<_>>();
    let pelvis = sk.joint_names[0].to_string();
    let (mut rot, mut pos) = (Vec::new(), Vec::new());
    for name in names {
        let (r, p) = match name.as_str() {
            "LeftFoot" => (n(&sk.left_foot[..1]), n(&sk.left_foot)),
            "RightFoot" => (n(&sk.right_foot[..1]), n(&sk.right_foot)),
            "LeftHand" => (n(&sk.left_hand[..1]), n(&sk.left_hand)),
            "RightHand" => (n(&sk.right_hand[..1]), n(&sk.right_hand)),
            "Hips" => (vec![pelvis.clone()], vec![pelvis.clone()]),
            other => return Err(Error::Config(format!("unknown end effector `{other}`"))),
        };
        rot.extend(r);
        pos.extend(p);
    }
    Ok((rot, pos))
}

pub fn build_condition_data(sk: &Skeleton, cs: &[Constraint]) -> Result<ConditionData> {
    let mut d = ConditionData::default();
    for c in cs {
        c.update(sk, &mut d)?;
    }
    Ok(d)
}

fn floats(v: &Value) -> Vec<f32> {
    match v {
        Value::Array(a) => a.iter().flat_map(floats).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap_or(0.0) as f32],
        _ => vec![],
    }
}

fn pairs(v: &Value) -> Vec<[f32; 2]> {
    floats(v).chunks(2).map(|c| [c[0], c[1]]).collect()
}

/// `load_constraints_lst`: the demo's `constraints.json` (a list of
/// `{type, frame_indices, …}`; keyframes store axis-angle local rotations
/// (30 or 77 SOMA joints) and root positions, replayed through FK).
pub fn load_constraints_json(path: &Path, sk: &Skeleton) -> Result<Vec<Constraint>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    parse_constraints(&v, sk)
}

pub fn parse_constraints(v: &Value, sk: &Skeleton) -> Result<Vec<Constraint>> {
    let list = v.as_array().ok_or_else(|| Error::Config("constraints: expected a list".into()))?;
    let mut out = Vec::new();
    for el in list {
        let ty = el["type"].as_str().ok_or_else(|| Error::Config("constraint without a type".into()))?;
        let frames: Vec<usize> = floats(&el["frame_indices"]).iter().map(|&f| f as usize).collect();
        let smooth = el.get("smooth_root_2d").map(pairs);
        if ty == "root2d" {
            let heading = el.get("global_root_heading").map(pairs);
            out.push(Constraint::root2d(frames, smooth.unwrap_or_default(), heading));
            continue;
        }
        let aa = floats(&el["local_joints_rot"]);
        let k = frames.len();
        let jin = aa.len() / (3 * k.max(1));
        let mut local: Vec<f32> = Vec::with_capacity(k * jin * 9);
        for i in 0..k * jin {
            local.extend_from_slice(&geo::axis_angle_to_matrix(&[aa[i * 3], aa[i * 3 + 1], aa[i * 3 + 2]]));
        }
        if jin != sk.nbjoints() {
            if jin == 77 && sk.nbjoints() == 30 {
                local = skeleton::soma77_to_30(&local, sk, &Skeleton::soma77())?;
            } else {
                return Err(Error::Config(format!("constraint has {jin} joints, skeleton {}", sk.nbjoints())));
            }
        }
        let root = floats(&el["root_positions"]);
        let (g, p, _) = sk.fk(&local, &root);
        let c = match ty {
            "fullbody" => Constraint::full_body(sk, frames, p, g, smooth),
            "left-hand" | "right-hand" | "left-foot" | "right-foot" => {
                let base = match ty {
                    "left-hand" => "LeftHand",
                    "right-hand" => "RightHand",
                    "left-foot" => "LeftFoot",
                    _ => "RightFoot",
                };
                Constraint::end_effector(sk, ty, vec![base.to_string()], frames, p, g, smooth)
            }
            "end-effector" => {
                let names: Vec<String> = el["joint_names"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                Constraint::end_effector(sk, ty, names, frames, p, g, smooth)
            }
            other => return Err(Error::Config(format!("unknown constraint type `{other}`"))),
        };
        out.push(c);
    }
    Ok(out)
}
