//! Small host-side helpers for the glue between ops: row-major `Vec<f32>`
//! views with explicit shapes. Not a tensor library; just what the model
//! crates need to pack features and masks.

/// Concatenate rows: each input is `[rows, w_i]`, the output `[rows, Σ w_i]`.
pub fn cat_cols(parts: &[(&[f32], usize)], rows: usize) -> Vec<f32> {
    let total: usize = parts.iter().map(|p| p.1).sum();
    let mut out = Vec::with_capacity(rows * total);
    for r in 0..rows {
        for (data, w) in parts {
            debug_assert_eq!(data.len(), rows * w, "cat_cols: part length");
            out.extend_from_slice(&data[r * w..(r + 1) * w]);
        }
    }
    out
}

/// Columns `[c0, c1)` of a `[rows, width]` matrix.
pub fn slice_cols(x: &[f32], width: usize, c0: usize, c1: usize) -> Vec<f32> {
    let rows = x.len() / width;
    let mut out = Vec::with_capacity(rows * (c1 - c0));
    for r in 0..rows {
        out.extend_from_slice(&x[r * width + c0..r * width + c1]);
    }
    out
}

/// Gather columns `idx` of a `[rows, width]` matrix.
pub fn gather_cols(x: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    let rows = x.len() / width;
    let mut out = Vec::with_capacity(rows * idx.len());
    for r in 0..rows {
        for &i in idx {
            out.push(x[r * width + i]);
        }
    }
    out
}

/// `[rows, width]` → `[width, rows]`.
pub fn transpose(x: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..width {
            out[c * rows + r] = x[r * width + c];
        }
    }
    out
}

/// `[batch, len, c]` → `[batch, c, len]` (PyTorch `transpose(1, 2)` for Conv1d).
pub fn to_channels_first(x: &[f32], batch: usize, len: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for b in 0..batch {
        for t in 0..len {
            for ch in 0..c {
                out[(b * c + ch) * len + t] = x[(b * len + t) * c + ch];
            }
        }
    }
    out
}

/// `[batch, c, len]` → `[batch, len, c]`.
pub fn to_channels_last(x: &[f32], batch: usize, c: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for b in 0..batch {
        for ch in 0..c {
            for t in 0..len {
                out[(b * len + t) * c + ch] = x[(b * c + ch) * len + t];
            }
        }
    }
    out
}

pub fn softmax_inplace(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `max |a − b| / (max |b| + 1e-6)`.
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "rel_err: lengths differ");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| {
        let d = (x - y).abs();
        if d.is_nan() { f32::INFINITY } else { m.max(d) }
    });
    worst / scale
}
