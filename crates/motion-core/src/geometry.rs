//! Rotation geometry on flat `f32` slices: 3×3 matrices stored row-major
//! (`m[r*3 + c]`, the convention `x' = R x`), quaternions `(w, x, y, z)`,
//! the 6-D continuous representation (first two matrix *columns*), axis-angle,
//! and the Y-axis heading rotations the motion representations use.
//!
//! Every function mirrors the corresponding PyTorch code in
//! `kimodo/geometry.py` and `tools/onnx/models/soma/rotations.py` /
//! `g1/rig.py` of MotionBricks, including their normalisation guards.

pub type Mat3 = [f32; 9];
pub type Vec3 = [f32; 3];
pub type Quat = [f32; 4];

pub const IDENTITY: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    o
}

pub fn transpose(a: &Mat3) -> Mat3 {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

pub fn apply(a: &Mat3, v: &Vec3) -> Vec3 {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

pub fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: &Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn norm(a: &Vec3) -> f32 {
    dot(a, a).sqrt()
}

pub fn normalize(a: &Vec3) -> Vec3 {
    let n = norm(a);
    scale(a, 1.0 / n)
}

/// Rotation about +Y by `angle`: `[[cos, 0, sin], [0, 1, 0], [−sin, 0, cos]]`
/// (`angle_to_Y_rotation_matrix`).
pub fn rot_y(angle: f32) -> Mat3 {
    let (s, c) = angle.sin_cos();
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

/// 6-D → matrix by Gram–Schmidt on the two stored columns (`cont6d_to_matrix`):
/// `x = x̂_raw`, `z = x × y_raw` normalised, `y = z × x`; the matrix columns are `(x, y, z)`.
pub fn cont6d_to_matrix(c: &[f32]) -> Mat3 {
    let x_raw = [c[0], c[1], c[2]];
    let y_raw = [c[3], c[4], c[5]];
    let x = normalize(&x_raw);
    let z = normalize(&cross(&x, &y_raw));
    let y = cross(&z, &x);
    [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
}

/// Matrix → 6-D: the first two columns, `(m[:, 0], m[:, 1])`.
pub fn matrix_to_cont6d(m: &Mat3) -> [f32; 6] {
    [m[0], m[3], m[6], m[1], m[4], m[7]]
}

/// `(w, x, y, z)` → matrix (`quaternion_to_matrix`; normalises by `2/‖q‖²`).
pub fn quat_to_matrix(q: &Quat) -> Mat3 {
    let (r, i, j, k) = (q[0], q[1], q[2], q[3]);
    let two_s = 2.0 / (r * r + i * i + j * j + k * k);
    [
        1.0 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1.0 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1.0 - two_s * (i * i + j * j),
    ]
}

/// Matrix → `(w, x, y, z)` (PyTorch3D's `matrix_to_quaternion`, as in Kimodo
/// and the MotionBricks G1 rig: the candidate with the largest `q_abs`,
/// denominators floored at 0.1).
pub fn matrix_to_quat(m: &Mat3) -> Quat {
    let (m00, m01, m02, m10, m11, m12, m20, m21, m22) =
        (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    let sp = |x: f32| if x > 0.0 { x.sqrt() } else { 0.0 };
    let q_abs = [
        sp(1.0 + m00 + m11 + m22),
        sp(1.0 + m00 - m11 - m22),
        sp(1.0 - m00 + m11 - m22),
        sp(1.0 - m00 - m11 + m22),
    ];
    let rows = [
        [q_abs[0] * q_abs[0], m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, q_abs[1] * q_abs[1], m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, q_abs[2] * q_abs[2], m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, q_abs[3] * q_abs[3]],
    ];
    let mut best = 0;
    for i in 1..4 {
        if q_abs[i] > q_abs[best] {
            best = i;
        }
    }
    let d = 2.0 * q_abs[best].max(0.1);
    let r = rows[best];
    [r[0] / d, r[1] / d, r[2] / d, r[3] / d]
}

/// Axis-angle → matrix (Rodrigues; `axis_angle_to_matrix` with its `1e-6` guard).
pub fn axis_angle_to_matrix(aa: &Vec3) -> Mat3 {
    let eps = 1e-6;
    let angle = norm(aa);
    let axis = scale(aa, 1.0 / (angle + eps));
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let k: Mat3 = [0.0, -z, y, z, 0.0, -x, -y, x, 0.0];
    let kk = matmul(&k, &k);
    let (s, c) = angle.sin_cos();
    let mut r = IDENTITY;
    for i in 0..9 {
        r[i] += s * k[i] + (1.0 - c) * kk[i];
    }
    r
}

/// Matrix → axis-angle through the quaternion (`matrix_to_axis_angle`).
pub fn matrix_to_axis_angle(m: &Mat3) -> Vec3 {
    quat_to_axis_angle(&matrix_to_quat(m))
}

/// `(w, x, y, z)` → axis-angle (`quaternion_to_axis_angle`, canonical sign).
pub fn quat_to_axis_angle(q: &Quat) -> Vec3 {
    let eps = 1e-6f32;
    let mut q = *q;
    let flip = q[0] < -eps || (q[0].abs() <= eps && q[1] < 0.0);
    if flip {
        for v in q.iter_mut() {
            *v = -*v;
        }
    }
    let xyz = [q[1], q[2], q[3]];
    let sin_half = norm(&xyz);
    let angle = 2.0 * sin_half.atan2(q[0]);
    let s = if sin_half.abs() < eps { 2.0 } else { angle / sin_half.max(eps) };
    scale(&xyz, s)
}

/// Rotate a quaternion-free vector by a `(w, x, y, z)` quaternion (`quat_apply`).
pub fn quat_apply(q: &Quat, v: &Vec3) -> Vec3 {
    let xyz = [q[1], q[2], q[3]];
    let t = scale(&cross(&xyz, v), 2.0);
    add(&add(v, &scale(&t, q[0])), &cross(&xyz, &t))
}

pub fn quat_conjugate(q: &Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}

pub fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Views over flat storage of many 3×3 matrices.
pub fn mat_at(flat: &[f32], i: usize) -> Mat3 {
    let s = &flat[i * 9..i * 9 + 9];
    [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]]
}

pub fn set_mat(flat: &mut [f32], i: usize, m: &Mat3) {
    flat[i * 9..i * 9 + 9].copy_from_slice(m);
}

pub fn vec_at(flat: &[f32], i: usize) -> Vec3 {
    [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]
}

pub fn set_vec(flat: &mut [f32], i: usize, v: &Vec3) {
    flat[i * 3..i * 3 + 3].copy_from_slice(v);
}

/// Frame-to-frame angle difference by the shortest turn, scaled by `fps`
/// (`diff_angles`); output length `n − 1`.
pub fn diff_angles(angles: &[f32], fps: f32) -> Vec<f32> {
    angles
        .windows(2)
        .map(|w| {
            let (c0, s0) = (w[0].cos(), w[0].sin());
            let (c1, s1) = (w[1].cos(), w[1].sin());
            fps * (s1 * c0 - c1 * s0).atan2(c1 * c0 + s1 * s0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cont6d_round_trip() {
        let r = matmul(&rot_y(0.7), &axis_angle_to_matrix(&[0.2, -0.4, 0.9]));
        let c = matrix_to_cont6d(&r);
        let back = cont6d_to_matrix(&c);
        for i in 0..9 {
            assert!((r[i] - back[i]).abs() < 1e-5, "{i}: {} vs {}", r[i], back[i]);
        }
    }

    #[test]
    fn quat_round_trip_and_apply() {
        let r = axis_angle_to_matrix(&[1.0, 0.3, -0.2]);
        let q = matrix_to_quat(&r);
        let back = quat_to_matrix(&q);
        for i in 0..9 {
            assert!((r[i] - back[i]).abs() < 1e-5);
        }
        let v = [0.3, -1.2, 2.0];
        let a = apply(&r, &v);
        let b = quat_apply(&q, &v);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5);
        }
        let aa = matrix_to_axis_angle(&r);
        for i in 0..3 {
            assert!((aa[i] - [1.0, 0.3, -0.2][i]).abs() < 1e-5);
        }
    }

    #[test]
    fn y_rotation_convention() {
        // x' = R x: rotating +Z by +90° about Y gives +X.
        let r = rot_y(std::f32::consts::FRAC_PI_2);
        let v = apply(&r, &[0.0, 0.0, 1.0]);
        assert!((v[0] - 1.0).abs() < 1e-6 && v[2].abs() < 1e-6);
    }
}
