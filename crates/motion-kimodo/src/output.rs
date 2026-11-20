//! Generated motion and its files: the Kimodo NPZ the CLI and demo write
//! (`kimodo/exports/motion_io.py`) and BVH (`kimodo/exports/bvh.py`).
//! SOMA-30 output is expanded to SOMA-77 (`output_to_SOMASkeleton77`: the
//! relaxed-hand rest pose for the fingers, FK on the 77-joint skeleton, six
//! foot-contact channels).

use crate::rep::Decoded;
use motion_core::geometry::{self as geo, Mat3};
use motion_core::skeleton::{self, Skeleton};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MotionOutput {
    pub skeleton: Skeleton,
    pub fps: f32,
    pub frames: usize,
    /// `[T, J, 9]` row-major 3×3.
    pub local_rot_mats: Vec<f32>,
    pub global_rot_mats: Vec<f32>,
    /// `[T, J, 3]`.
    pub posed_joints: Vec<f32>,
    pub root_positions: Vec<f32>,
    pub smooth_root_pos: Vec<f32>,
    /// `[T, 4]` (SOMA-77: `[T, 6]`), 0 / 1.
    pub foot_contacts: Vec<f32>,
    pub global_root_heading: Vec<f32>,
}

impl MotionOutput {
    pub fn from_decoded(d: Decoded, sk: &Skeleton, fps: f32) -> Self {
        let mut out = Self {
            skeleton: sk.clone(),
            fps,
            frames: d.frames,
            local_rot_mats: d.local_rot_mats,
            global_rot_mats: d.global_rot_mats,
            posed_joints: d.posed_joints,
            root_positions: d.root_positions,
            smooth_root_pos: d.smooth_root_pos,
            foot_contacts: d.foot_contacts,
            global_root_heading: d.global_root_heading,
        };
        if sk.name == "somaskel30" {
            let s77 = Skeleton::soma77();
            let l77 = skeleton::soma30_to_77(&out.local_rot_mats, sk, &s77).expect("soma30 ⊂ soma77");
            let (g, p, _) = s77.fk(&l77, &out.root_positions);
            let fc = &out.foot_contacts;
            out.foot_contacts = (0..out.frames)
                .flat_map(|t| {
                    let c = &fc[t * 4..t * 4 + 4];
                    [c[0], c[1], c[1], c[2], c[3], c[3]]
                })
                .collect();
            out.local_rot_mats = l77;
            out.global_rot_mats = g;
            out.posed_joints = p;
            out.skeleton = s77;
        }
        out
    }

    pub fn nbjoints(&self) -> usize {
        self.skeleton.nbjoints()
    }

    /// The Kimodo NPZ: float32 arrays, uncompressed.
    pub fn write_npz(&self, path: &Path) -> std::io::Result<()> {
        let (t, j) = (self.frames, self.nbjoints());
        let fc = self.foot_contacts.len() / t.max(1);
        let arrays: Vec<(&str, Vec<usize>, &[f32])> = vec![
            ("posed_joints", vec![t, j, 3], &self.posed_joints),
            ("global_rot_mats", vec![t, j, 3, 3], &self.global_rot_mats),
            ("local_rot_mats", vec![t, j, 3, 3], &self.local_rot_mats),
            ("foot_contacts", vec![t, fc], &self.foot_contacts),
            ("smooth_root_pos", vec![t, 3], &self.smooth_root_pos),
            ("root_positions", vec![t, 3], &self.root_positions),
            ("global_root_heading", vec![t, 2], &self.global_root_heading),
        ];
        let files: Vec<(String, Vec<u8>)> =
            arrays.into_iter().map(|(n, s, d)| (format!("{n}.npy"), npy_f32(&s, d))).collect();
        std::fs::write(path, zip_stored(&files))
    }

    /// BVH on SOMA-77 as `motion_to_bvh` writes it: a `Root` wrapper at the
    /// origin, `Hips` with absolute position channels, ZYX Euler channels, the
    /// rest pose of BONES-SEED (`bvh_joints.p`) unless `standard_tpose`.
    pub fn to_bvh(&self, standard_tpose: bool) -> String {
        let sk = &self.skeleton;
        let j = sk.nbjoints();
        let (local, neutral): (Vec<f32>, Vec<f32>) = if sk.name == "somaskel77" && !standard_tpose {
            (skeleton::soma77_from_standard_tpose(&self.local_rot_mats, sk).0, skeleton::soma77_bvh_neutral_joints().to_vec())
        } else {
            (self.local_rot_mats.clone(), sk.neutral_joints.clone())
        };
        let mut hips = [neutral[0] * 100.0, neutral[1] * 100.0, neutral[2] * 100.0];
        if hips == [0.0, 0.0, 0.0] {
            hips = [0.0, 100.0, 0.0];
        }
        let children: Vec<Vec<usize>> = (0..j).map(|p| (0..j).filter(|&c| sk.parents[c] == Some(p)).collect()).collect();
        let mut s = String::from("HIERARCHY\nROOT Root\n{\n  OFFSET 0.0 0.0 0.0\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n");
        let mut order = Vec::new();
        fn emit(s: &mut String, sk: &Skeleton, ch: &[Vec<usize>], neutral: &[f32], hips: [f32; 3], i: usize, depth: usize, order: &mut Vec<usize>) {
            order.push(i);
            let ind = "  ".repeat(depth);
            let off = match sk.parents[i] {
                None => hips,
                Some(p) => [
                    (neutral[i * 3] - neutral[p * 3]) * 100.0,
                    (neutral[i * 3 + 1] - neutral[p * 3 + 1]) * 100.0,
                    (neutral[i * 3 + 2] - neutral[p * 3 + 2]) * 100.0,
                ],
            };
            let _ = writeln!(s, "{ind}JOINT {}\n{ind}{{\n{ind}  OFFSET {:.6} {:.6} {:.6}", sk.joint_names[i], off[0], off[1], off[2]);
            if sk.parents[i].is_none() {
                let _ = writeln!(s, "{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation");
            } else {
                let _ = writeln!(s, "{ind}  CHANNELS 3 Zrotation Yrotation Xrotation");
            }
            for &c in &ch[i] {
                emit(s, sk, ch, neutral, hips, c, depth + 1, order);
            }
            let _ = writeln!(s, "{ind}}}");
        }
        emit(&mut s, sk, &children, &neutral, hips, 0, 1, &mut order);
        let _ = writeln!(s, "}}\nMOTION\nFrames: {}\nFrame Time: {}", self.frames, 1.0 / self.fps as f64);
        for t in 0..self.frames {
            let mut row: Vec<String> = vec!["0.0".into(); 6];
            for &i in &order {
                if sk.parents[i].is_none() {
                    for c in 0..3 {
                        row.push(format!("{:.6}", self.root_positions[t * 3 + c] * 100.0));
                    }
                }
                let e = euler_zyx(&geo::mat_at(&local, t * j + i));
                row.extend(e.iter().map(|v| format!("{v:.6}")));
            }
            let _ = writeln!(s, "{}", row.join(" "));
        }
        s
    }
}

/// `R = Rz(a)·Ry(b)·Rx(c)` → `(a, b, c)` in degrees.
pub fn euler_zyx(m: &Mat3) -> [f32; 3] {
    let b = (-m[6]).atan2((m[0] * m[0] + m[3] * m[3]).sqrt());
    let a = m[3].atan2(m[0]);
    let c = m[7].atan2(m[8]);
    [a.to_degrees(), b.to_degrees(), c.to_degrees()]
}

/// A `.npy` (format 1.0) of little-endian f32.
pub fn npy_f32(shape: &[usize], data: &[f32]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_s = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_s}, }}");
    let total = 10 + h.len() + 1;
    h.push_str(&" ".repeat((64 - total % 64) % 64));
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// A zip archive with stored (uncompressed) entries, as `np.savez` writes.
pub fn zip_stored(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let off = out.len() as u32;
        let crc = crc32(data);
        let n = name.as_bytes();
        let mut local = Vec::new();
        local.write_all(&0x0403_4b50u32.to_le_bytes()).unwrap();
        local.write_all(&[20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]).unwrap(); // version, flags, method 0, time, date
        local.write_all(&crc.to_le_bytes()).unwrap();
        local.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        local.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        local.write_all(&(n.len() as u16).to_le_bytes()).unwrap();
        local.write_all(&0u16.to_le_bytes()).unwrap();
        local.write_all(n).unwrap();
        out.extend_from_slice(&local);
        out.extend_from_slice(data);
        central.write_all(&0x0201_4b50u32.to_le_bytes()).unwrap();
        central.write_all(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]).unwrap();
        central.write_all(&crc.to_le_bytes()).unwrap();
        central.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        central.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        central.write_all(&(n.len() as u16).to_le_bytes()).unwrap();
        central.write_all(&[0; 8]).unwrap(); // extra, comment, disk, internal attrs
        central.write_all(&0u32.to_le_bytes()).unwrap(); // external attrs
        central.write_all(&off.to_le_bytes()).unwrap();
        central.write_all(n).unwrap();
    }
    let cd_off = out.len() as u32;
    out.extend_from_slice(&central);
    out.write_all(&0x0605_4b50u32.to_le_bytes()).unwrap();
    out.write_all(&[0, 0, 0, 0]).unwrap();
    out.write_all(&(files.len() as u16).to_le_bytes()).unwrap();
    out.write_all(&(files.len() as u16).to_le_bytes()).unwrap();
    out.write_all(&(central.len() as u32).to_le_bytes()).unwrap();
    out.write_all(&cd_off.to_le_bytes()).unwrap();
    out.write_all(&0u16.to_le_bytes()).unwrap();
    out
}
