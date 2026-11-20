//! The CPU reference backend: plain f32, one obvious loop per op, rayon over
//! rows. It is the ground truth every GPU kernel is compared against
//! (`motion check`), so clarity beats speed; it still runs the real models at
//! useful speed because the motion transformers are small.

use motion_core::backend::{Act, Backend, BinOp, Conv1dShape};
use rayon::prelude::*;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Default)]
pub struct CpuBuf(Arc<RwLock<Vec<f32>>>);

impl CpuBuf {
    fn r(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.0.try_read().expect("buffer aliased with a written output")
    }
    fn w(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.0.try_write().expect("output buffer aliases an input")
    }
}

pub struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Default, Clone, Copy)]
pub struct Cpu;

pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// Exact GELU through an erf accurate to ~1e-12 (f64).
pub fn gelu_erf(x: f32) -> f32 {
    let z = x as f64 / std::f64::consts::SQRT_2;
    (0.5 * x as f64 * (1.0 + erf(z))) as f32
}

/// erf: Taylor series below 2, a continued fraction for erfc above.
pub fn erf(x: f64) -> f64 {
    let ax = x.abs();
    let v = if ax < 2.0 {
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 * sum.abs() || n > 200.0 {
                break;
            }
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // erfc(x) = exp(-x²)/(x√π) · 1/(1 + 1/(2x²)/(1 + 2/(2x²)/(1 + ...))) (Lentz)
        let x2 = ax * ax;
        let mut f = ax;
        for k in (1..=60).rev() {
            f = ax + (k as f64 * 0.5) / f;
        }
        1.0 - (-x2).exp() / (f * std::f64::consts::PI.sqrt())
    };
    if x < 0.0 { -v } else { v }
}

pub fn act(x: f32, a: Act) -> f32 {
    match a {
        Act::None => x,
        Act::Relu => x.max(0.0),
        Act::LeakyRelu => {
            if x >= 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        Act::GeluErf => gelu_erf(x),
        Act::GeluTanh => gelu_tanh(x),
        Act::Silu => x / (1.0 + (-x).exp()),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight accumulators: the compiler vectorises this on x86-64 / aarch64.
    let n = a.len();
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// Rows per register tile of [`gemm`]: each weight row is read once per tile.
const MR: usize = 4;
/// Output columns per parallel task.
const NC: usize = 64;

/// `y[i, c] = act(Σ_k x[i, k]·w[c, k] + b[c])`: tiles of [`MR`] rows ×
/// [`NC`] columns run in parallel; inside a tile every weight row is read once
/// for the [`MR`] rows, with eight-lane accumulators the compiler vectorises.
/// Summation order per output: lanes of eight, then the tail (fixed, so
/// results do not depend on the thread count).
#[allow(clippy::too_many_arguments)]
fn gemm(x: &[f32], w: &[f32], bias: Option<&[f32]>, rows: usize, n: usize, k: usize, a: Act, y: &mut [f32]) {
    let row_tiles = rows.div_ceil(MR);
    let col_tiles = n.div_ceil(NC);
    let out = SyncPtr(y.as_mut_ptr());
    (0..row_tiles * col_tiles).into_par_iter().for_each(|t| {
        let (rt, ct) = (t / col_tiles, t % col_tiles);
        let r0 = rt * MR;
        let mr = MR.min(rows - r0);
        let c0 = ct * NC;
        let nc = NC.min(n - c0);
        let k8 = k / 8 * 8;
        for c in c0..c0 + nc {
            let wr = &w[c * k..(c + 1) * k];
            let mut acc = [[0f32; 8]; MR];
            let mut kk = 0;
            while kk < k8 {
                let wv = &wr[kk..kk + 8];
                for r in 0..mr {
                    let xv = &x[(r0 + r) * k + kk..(r0 + r) * k + kk + 8];
                    for j in 0..8 {
                        acc[r][j] += xv[j] * wv[j];
                    }
                }
                kk += 8;
            }
            for r in 0..mr {
                let mut v = acc[r].iter().sum::<f32>();
                for q in k8..k {
                    v += x[(r0 + r) * k + q] * wr[q];
                }
                if let Some(b) = bias {
                    v += b[c];
                }
                // Each (row, column) is written by exactly one task.
                out.write((r0 + r) * n + c, act(v, a));
            }
        }
    });
}

/// A raw output pointer shared by the GEMM tasks, which write disjoint elements.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
#[allow(unsafe_code)]
unsafe impl Send for SyncPtr {}
#[allow(unsafe_code)]
unsafe impl Sync for SyncPtr {}
impl SyncPtr {
    #[allow(unsafe_code)]
    fn write(self, i: usize, v: f32) {
        // SAFETY: `gemm` hands each index to exactly one task, inside `y`'s bounds.
        unsafe { *self.0.add(i) = v }
    }
}

impl Backend for Cpu {
    type Buf = CpuBuf;
    type Weight = CpuWeight;

    fn name(&self) -> String {
        format!("cpu (rayon, {} threads)", rayon::current_num_threads())
    }

    fn alloc(&self, len: usize) -> CpuBuf {
        CpuBuf(Arc::new(RwLock::new(vec![0.0; len])))
    }
    fn upload(&self, data: &[f32]) -> CpuBuf {
        CpuBuf(Arc::new(RwLock::new(data.to_vec())))
    }
    fn write(&self, buf: &CpuBuf, off: usize, data: &[f32]) {
        buf.w()[off..off + data.len()].copy_from_slice(data);
    }
    fn download(&self, buf: &CpuBuf) -> Vec<f32> {
        buf.r().clone()
    }
    fn buf_len(&self, buf: &CpuBuf) -> usize {
        buf.r().len()
    }
    fn copy(&self, src: &CpuBuf, src_off: usize, dst: &CpuBuf, dst_off: usize, len: usize) {
        if Arc::ptr_eq(&src.0, &dst.0) {
            let mut d = dst.w();
            d.copy_within(src_off..src_off + len, dst_off);
        } else {
            let s = src.r();
            dst.w()[dst_off..dst_off + len].copy_from_slice(&s[src_off..src_off + len]);
        }
    }
    fn fill(&self, buf: &CpuBuf, off: usize, len: usize, value: f32) {
        buf.w()[off..off + len].fill(value);
    }
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> CpuWeight {
        assert_eq!(data.len(), rows * cols, "weight data does not match [{rows}, {cols}]");
        CpuWeight { data: data.to_vec(), rows, cols }
    }
    fn weight_bytes(&self, w: &CpuWeight) -> usize {
        w.data.len() * 4
    }
    fn sync(&self) {}

    fn linear(
        &self,
        x: &CpuBuf,
        x_off: usize,
        rows: usize,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        a: Act,
        y: &CpuBuf,
        y_off: usize,
    ) {
        let (n, k) = (w.rows, w.cols);
        let xr = x.r();
        let x = &xr[x_off..x_off + rows * k];
        let bias = bias.map(|b| b.r());
        let mut yw = y.w();
        let y = &mut yw[y_off..y_off + rows * n];
        let bias = bias.as_deref().map(|b| &b[..n]);
        gemm(x, &w.data, bias, rows, n, k, a, y);
    }

    fn layer_norm(
        &self,
        x: &CpuBuf,
        x_off: usize,
        rows: usize,
        dim: usize,
        w: &CpuBuf,
        b: &CpuBuf,
        eps: f32,
        out: &CpuBuf,
        out_off: usize,
    ) {
        let xr = x.r();
        let (w, b) = (w.r(), b.r());
        let mut o = out.w();
        o[out_off..out_off + rows * dim]
            .par_chunks_mut(dim)
            .zip(xr[x_off..x_off + rows * dim].par_chunks(dim))
            .for_each(|(o, x)| {
                let mean = x.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
                let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
                let inv = 1.0 / (var + eps as f64).sqrt();
                for i in 0..dim {
                    o[i] = ((x[i] as f64 - mean) * inv) as f32 * w[i] + b[i];
                }
            });
    }

    fn rms_norm(
        &self,
        x: &CpuBuf,
        x_off: usize,
        rows: usize,
        dim: usize,
        w: &CpuBuf,
        eps: f32,
        out: &CpuBuf,
        out_off: usize,
    ) {
        let xr = x.r();
        let w = w.r();
        let mut o = out.w();
        o[out_off..out_off + rows * dim]
            .par_chunks_mut(dim)
            .zip(xr[x_off..x_off + rows * dim].par_chunks(dim))
            .for_each(|(o, x)| {
                let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dim as f64;
                let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
                for i in 0..dim {
                    o[i] = x[i] * inv * w[i];
                }
            });
    }

    fn attention(
        &self,
        qkv: &CpuBuf,
        batch: usize,
        seq: usize,
        heads: usize,
        kv_heads: usize,
        d: usize,
        key_mask: Option<&CpuBuf>,
        out: &CpuBuf,
    ) {
        assert_eq!(heads % kv_heads, 0, "heads must be a multiple of kv_heads");
        let group = heads / kv_heads;
        let width = (heads + 2 * kv_heads) * d;
        let q = qkv.r();
        debug_assert!(q.len() >= batch * seq * width, "attention: qkv too short");
        let mask = key_mask.map(|m| m.r());
        let mut o = out.w();
        let scale = 1.0 / (d as f32).sqrt();
        o.par_chunks_mut(seq * heads * d).enumerate().for_each(|(b, ob)| {
            let rows = &q[b * seq * width..(b + 1) * seq * width];
            let mask = mask.as_ref().map(|m| &m[b * seq..(b + 1) * seq]);
            let mut scores = vec![0f32; seq];
            for h in 0..heads {
                let kh = h / group;
                let q_off = h * d;
                let k_off = heads * d + kh * d;
                let v_off = heads * d + kv_heads * d + kh * d;
                for i in 0..seq {
                    let qi = &rows[i * width + q_off..i * width + q_off + d];
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..seq {
                        let allowed = mask.is_none_or(|mk| mk[j] > 0.5);
                        if allowed {
                            let kj = &rows[j * width + k_off..j * width + k_off + d];
                            scores[j] = dot(qi, kj) * scale;
                            m = m.max(scores[j]);
                        } else {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    let acc = &mut ob[i * heads * d + h * d..i * heads * d + (h + 1) * d];
                    acc.fill(0.0);
                    if m == f32::NEG_INFINITY {
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for j in 0..seq {
                        if scores[j] == f32::NEG_INFINITY {
                            scores[j] = 0.0;
                            continue;
                        }
                        scores[j] = (scores[j] - m).exp();
                        sum += scores[j];
                    }
                    let inv = 1.0 / sum;
                    for j in 0..seq {
                        let p = scores[j] * inv;
                        if p == 0.0 {
                            continue;
                        }
                        let vj = &rows[j * width + v_off..j * width + v_off + d];
                        for c in 0..d {
                            acc[c] += p * vj[c];
                        }
                    }
                }
            }
        });
    }

    fn rope(
        &self,
        qkv: &CpuBuf,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        d: usize,
        cos: &CpuBuf,
        sin: &CpuBuf,
    ) {
        let width = (heads + 2 * kv_heads) * d;
        let half = d / 2;
        let (cos, sin) = (cos.r(), sin.r());
        let mut q = qkv.w();
        q[..rows * width].par_chunks_mut(width).enumerate().for_each(|(r, row)| {
            let c = &cos[r * half..(r + 1) * half];
            let s = &sin[r * half..(r + 1) * half];
            for h in 0..heads + kv_heads {
                let o = h * d;
                for i in 0..half {
                    let x1 = row[o + i];
                    let x2 = row[o + i + half];
                    row[o + i] = x1 * c[i] - x2 * s[i];
                    row[o + i + half] = x2 * c[i] + x1 * s[i];
                }
            }
        });
    }

    fn swiglu(&self, x: &CpuBuf, rows: usize, dim: usize, out: &CpuBuf) {
        let xr = x.r();
        let mut o = out.w();
        o[..rows * dim].par_chunks_mut(dim).zip(xr[..rows * 2 * dim].par_chunks(2 * dim)).for_each(
            |(o, x)| {
                for i in 0..dim {
                    o[i] = act(x[i], Act::Silu) * x[dim + i];
                }
            },
        );
    }

    fn conv1d(
        &self,
        x: &CpuBuf,
        s: Conv1dShape,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        a: Act,
        residual: Option<&CpuBuf>,
        out: &CpuBuf,
    ) {
        assert_eq!(w.rows, s.cout);
        assert_eq!(w.cols, s.k_len());
        let xr = x.r();
        let bias = bias.map(|b| b.r());
        let res = residual.map(|r| r.r());
        let (lin, lout) = (s.in_len(), s.out_len());
        let mut o = out.w();
        o[..s.batch * s.cout * lout]
            .par_chunks_mut(lout)
            .enumerate()
            .for_each(|(bc, orow)| {
                let (b, co) = (bc / s.cout, bc % s.cout);
                let xb = &xr[b * s.cin * s.len..(b + 1) * s.cin * s.len];
                let wrow = &w.data[co * s.k_len()..(co + 1) * s.k_len()];
                for (t, out) in orow.iter_mut().enumerate() {
                    let mut v = bias.as_ref().map_or(0.0, |bb| bb[co]);
                    for ci in 0..s.cin {
                        let xc = &xb[ci * s.len..(ci + 1) * s.len];
                        let wc = &wrow[ci * s.k..(ci + 1) * s.k];
                        for j in 0..s.k {
                            let pos = (t * s.stride + j * s.dilation) as isize - s.pad as isize;
                            if pos < 0 || pos as usize >= lin {
                                continue;
                            }
                            let p = pos as usize;
                            let xv = if s.upsample { xc[p / 2] } else { xc[p] };
                            v += xv * wc[j];
                        }
                    }
                    if let Some(r) = &res {
                        v += r[bc * lout + t];
                    }
                    *out = act(v, a);
                }
            });
    }

    fn unary(&self, x: &CpuBuf, x_off: usize, n: usize, a: Act, out: &CpuBuf, out_off: usize) {
        if Arc::ptr_eq(&x.0, &out.0) {
            let mut o = out.w();
            for i in 0..n {
                o[out_off + i] = act(o[x_off + i], a);
            }
            return;
        }
        let xr = x.r();
        let mut o = out.w();
        o[out_off..out_off + n]
            .par_iter_mut()
            .zip(xr[x_off..x_off + n].par_iter())
            .for_each(|(o, &x)| *o = act(x, a));
    }

    fn binary(
        &self,
        a: &CpuBuf,
        a_off: usize,
        b: &CpuBuf,
        b_off: usize,
        n: usize,
        op: BinOp,
        out: &CpuBuf,
        out_off: usize,
    ) {
        let f = |x: f32, y: f32| match op {
            BinOp::Add => x + y,
            BinOp::Mul => x * y,
        };
        let bv: Vec<f32> = b.r()[b_off..b_off + n].to_vec();
        if Arc::ptr_eq(&a.0, &out.0) {
            let mut o = out.w();
            for i in 0..n {
                o[out_off + i] = f(o[a_off + i], bv[i]);
            }
            return;
        }
        let ar = a.r();
        let mut o = out.w();
        for i in 0..n {
            o[out_off + i] = f(ar[a_off + i], bv[i]);
        }
    }

    fn axpby(&self, y: &CpuBuf, y_off: usize, a: f32, x: &CpuBuf, x_off: usize, b: f32, n: usize) {
        let xv: Vec<f32> = x.r()[x_off..x_off + n].to_vec();
        let mut yw = y.w();
        for i in 0..n {
            yw[y_off + i] = a * yw[y_off + i] + b * xv[i];
        }
    }

    fn mask_time(&self, x: &CpuBuf, batch: usize, channels: usize, len: usize, mask: &CpuBuf) {
        let m = mask.r();
        let mut xw = x.w();
        for b in 0..batch {
            for c in 0..channels {
                let row = &mut xw[(b * channels + c) * len..(b * channels + c + 1) * len];
                for t in 0..len {
                    row[t] *= m[b * len + t];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-12);
        assert!((erf(2.5) - 0.999_593_047_982_555_0).abs() < 1e-12);
        assert!((erf(-1.0) + 0.842_700_792_949_714_9).abs() < 1e-12);
    }

    #[test]
    fn linear_is_x_w_transpose_plus_bias() {
        let be = Cpu;
        let x = be.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 2 rows of 3
        let w = be.weight(&[1.0, 0.0, 0.0, 0.0, 1.0, 1.0], 2, 3); // [2, 3]
        let b = be.upload(&[0.5, -0.5]);
        let y = be.alloc(4);
        be.linear(&x, 0, 2, &w, Some(&b), Act::None, &y, 0);
        assert_eq!(be.download(&y), vec![1.5, 4.5, 4.5, 10.5]);
    }

    #[test]
    fn attention_uniform_keys_average_values() {
        let be = Cpu;
        // one batch, 2 tokens, 1 head, d=2; q=0 so softmax is uniform.
        let qkv = be.upload(&[0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0]);
        let out = be.alloc(4);
        be.attention(&qkv, 1, 2, 1, 1, 2, None, &out);
        assert_eq!(be.download(&out), vec![2.0, 3.0, 2.0, 3.0]);
        let mask = be.upload(&[1.0, 0.0]);
        be.attention(&qkv, 1, 2, 1, 1, 2, Some(&mask), &out);
        assert_eq!(be.download(&out), vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn conv1d_upsample_matches_explicit() {
        let be = Cpu;
        let s = Conv1dShape { batch: 1, cin: 1, cout: 1, len: 3, k: 3, stride: 1, pad: 1, dilation: 1, upsample: true };
        let x = be.upload(&[1.0, 2.0, 3.0]);
        let w = be.weight(&[1.0, 1.0, 1.0], 1, 3);
        let out = be.alloc(s.out_elems());
        be.conv1d(&x, s, &w, None, Act::None, None, &out);
        // upsampled: 1 1 2 2 3 3; box filter with zero pad
        assert_eq!(be.download(&out), vec![2.0, 4.0, 5.0, 7.0, 8.0, 6.0]);
    }

    #[test]
    fn conv1d_dilated_strided() {
        let be = Cpu;
        let s = Conv1dShape { batch: 1, cin: 1, cout: 1, len: 8, k: 3, stride: 2, pad: 1, dilation: 1, upsample: false };
        assert_eq!(s.out_len(), 4);
        let x = be.upload(&(0..8).map(|v| v as f32).collect::<Vec<_>>());
        let w = be.weight(&[0.0, 1.0, 0.0], 1, 3);
        let out = be.alloc(4);
        be.conv1d(&x, s, &w, None, Act::None, None, &out);
        assert_eq!(be.download(&out), vec![0.0, 2.0, 4.0, 6.0]);
        let s2 = Conv1dShape { dilation: 3, pad: 3, stride: 1, ..s };
        assert_eq!(s2.out_len(), 8);
    }
}
