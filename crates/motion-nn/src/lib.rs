//! Building blocks shared by the model crates, each the exact arithmetic of
//! its PyTorch namesake and loaded from the same state-dict keys:
//! [`Linear`] (`nn.Linear`), [`LayerNorm`], [`EncoderLayer`]
//! (`nn.TransformerEncoderLayer`, post-norm, `batch_first`, no dropout),
//! [`Encoder`] (`nn.TransformerEncoder`), [`Mlp`] (a chain of linears with an
//! activation between them) and the sinusoidal positional table.
//!
//! Activations live in flat `[rows, dim]` device buffers; a layer forward
//! allocates its scratch from the backend (the motion models are small
//! enough that this is not the cost that matters).

use motion_core::backend::{Act, Backend};
use motion_core::weights::WeightSource;
use motion_core::Result;

pub struct Linear<B: Backend> {
    pub w: B::Weight,
    pub b: Option<B::Buf>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl<B: Backend> Linear<B> {
    /// `<key>.weight [out, in]` and, when `bias`, `<key>.bias [out]`.
    pub fn load(be: &B, src: &dyn WeightSource, key: &str, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        let w = src.get(&format!("{key}.weight"), &[out_dim, in_dim])?;
        let b = if bias { Some(be.upload(&src.get(&format!("{key}.bias"), &[out_dim])?)) } else { None };
        Ok(Self { w: be.weight(&w, out_dim, in_dim), b, in_dim, out_dim })
    }

    pub fn from_parts(be: &B, w: &[f32], b: Option<&[f32]>, in_dim: usize, out_dim: usize) -> Self {
        Self { w: be.weight(w, out_dim, in_dim), b: b.map(|b| be.upload(b)), in_dim, out_dim }
    }

    pub fn forward(&self, be: &B, x: &B::Buf, x_off: usize, rows: usize, act: Act, y: &B::Buf, y_off: usize) {
        be.linear(x, x_off, rows, &self.w, self.b.as_ref(), act, y, y_off);
    }

    /// Allocating convenience: `act(x·Wᵀ + b)` for `rows` rows of `x`.
    pub fn apply(&self, be: &B, x: &B::Buf, rows: usize, act: Act) -> B::Buf {
        let y = be.alloc(rows * self.out_dim);
        self.forward(be, x, 0, rows, act, &y, 0);
        y
    }
}

pub struct LayerNorm<B: Backend> {
    pub w: B::Buf,
    pub b: B::Buf,
    pub dim: usize,
    pub eps: f32,
}

impl<B: Backend> LayerNorm<B> {
    pub fn load(be: &B, src: &dyn WeightSource, key: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            w: be.upload(&src.get(&format!("{key}.weight"), &[dim])?),
            b: be.upload(&src.get(&format!("{key}.bias"), &[dim])?),
            dim,
            eps,
        })
    }

    pub fn forward(&self, be: &B, x: &B::Buf, rows: usize, out: &B::Buf) {
        be.layer_norm(x, 0, rows, self.dim, &self.w, &self.b, self.eps, out, 0);
    }
}

/// `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation, batch_first=True, norm_first=False)`:
/// `x = norm1(x + out_proj(attn(in_proj(x))))`, `x = norm2(x + linear2(act(linear1(x))))`.
pub struct EncoderLayer<B: Backend> {
    pub in_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub d_model: usize,
    pub heads: usize,
    pub act: Act,
}

impl<B: Backend> EncoderLayer<B> {
    /// Keys under `prefix` (e.g. `seqTransEncoder.layers.3`): `self_attn.in_proj_weight`
    /// `[3d, d]`, `self_attn.in_proj_bias`, `self_attn.out_proj.{weight,bias}`,
    /// `linear1`, `linear2`, `norm1`, `norm2`.
    pub fn load(
        be: &B,
        src: &dyn WeightSource,
        prefix: &str,
        d_model: usize,
        heads: usize,
        ff: usize,
        act: Act,
        eps: f32,
    ) -> Result<Self> {
        let w = src.get(&format!("{prefix}.self_attn.in_proj_weight"), &[3 * d_model, d_model])?;
        let b = src.get(&format!("{prefix}.self_attn.in_proj_bias"), &[3 * d_model])?;
        Ok(Self {
            in_proj: Linear::from_parts(be, &w, Some(&b), d_model, 3 * d_model),
            out_proj: Linear::load(be, src, &format!("{prefix}.self_attn.out_proj"), d_model, d_model, true)?,
            linear1: Linear::load(be, src, &format!("{prefix}.linear1"), d_model, ff, true)?,
            linear2: Linear::load(be, src, &format!("{prefix}.linear2"), ff, d_model, true)?,
            norm1: LayerNorm::load(be, src, &format!("{prefix}.norm1"), d_model, eps)?,
            norm2: LayerNorm::load(be, src, &format!("{prefix}.norm2"), d_model, eps)?,
            d_model,
            heads,
            act,
        })
    }

    /// In place on `x` (`[batch·seq, d_model]`). `key_mask` is `[batch·seq]`
    /// with `1` for keys that may be attended (the negation of PyTorch's
    /// `src_key_padding_mask`).
    pub fn forward(&self, be: &B, x: &B::Buf, batch: usize, seq: usize, key_mask: Option<&B::Buf>) {
        let rows = batch * seq;
        let d = self.d_model;
        let qkv = be.alloc(rows * 3 * d);
        self.in_proj.forward(be, x, 0, rows, Act::None, &qkv, 0);
        let attn = be.alloc(rows * d);
        be.attention(&qkv, batch, seq, self.heads, self.heads, d / self.heads, key_mask, &attn);
        let proj = be.alloc(rows * d);
        self.out_proj.forward(be, &attn, 0, rows, Act::None, &proj, 0);
        be.axpby(&proj, 0, 1.0, x, 0, 1.0, rows * d);
        self.norm1.forward(be, &proj, rows, x);
        let h = be.alloc(rows * self.linear1.out_dim);
        self.linear1.forward(be, x, 0, rows, self.act, &h, 0);
        self.linear2.forward(be, &h, 0, rows, Act::None, &proj, 0);
        be.axpby(&proj, 0, 1.0, x, 0, 1.0, rows * d);
        self.norm2.forward(be, &proj, rows, x);
    }
}

/// `nn.TransformerEncoder(layer, num_layers)` (no final norm).
pub struct Encoder<B: Backend> {
    pub layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> Encoder<B> {
    /// Layers at `<prefix>.layers.{i}`.
    pub fn load(
        be: &B,
        src: &dyn WeightSource,
        prefix: &str,
        num_layers: usize,
        d_model: usize,
        heads: usize,
        ff: usize,
        act: Act,
        eps: f32,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|i| EncoderLayer::load(be, src, &format!("{prefix}.layers.{i}"), d_model, heads, ff, act, eps))
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, be: &B, x: &B::Buf, batch: usize, seq: usize, key_mask: Option<&B::Buf>) {
        for l in &self.layers {
            l.forward(be, x, batch, seq, key_mask);
        }
    }
}

/// A chain of linears with `act` after every layer but the last
/// (`nn.Sequential(Linear, SiLU, Linear)` and MotionBricks' `FCBlock`, whose
/// last `forward_projection` has no activation).
pub struct Mlp<B: Backend> {
    pub layers: Vec<Linear<B>>,
    pub act: Act,
}

impl<B: Backend> Mlp<B> {
    pub fn new(layers: Vec<Linear<B>>, act: Act) -> Self {
        Self { layers, act }
    }

    pub fn forward(&self, be: &B, x: &B::Buf, rows: usize) -> B::Buf {
        let mut cur = x.clone();
        let n = self.layers.len();
        for (i, l) in self.layers.iter().enumerate() {
            let a = if i + 1 < n { self.act } else { Act::None };
            cur = l.apply(be, &cur, rows, a);
        }
        cur
    }
}

/// The sinusoidal table `pe[pos, 2i] = sin(pos·10000^(−2i/d))`, `pe[pos, 2i+1] = cos(…)`,
/// `[max_len, dim]` (Kimodo's `PositionalEncoding` and MotionBricks' `PE1d_sincos`).
pub fn sincos_table(max_len: usize, dim: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; max_len * dim];
    for pos in 0..max_len {
        for i in (0..dim).step_by(2) {
            let div = 10000f64.powf(-(i as f64) / dim as f64);
            let a = pos as f64 * div;
            pe[pos * dim + i] = a.sin() as f32;
            if i + 1 < dim {
                pe[pos * dim + i + 1] = a.cos() as f32;
            }
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;
    use motion_core::weights::Synthetic;
    use motion_cpu::Cpu;

    #[test]
    fn encoder_layer_runs_and_masked_keys_do_not_leak() {
        let be = Cpu;
        let src = Synthetic::new(1);
        let layer = EncoderLayer::load(&be, &src, "layers.0", 16, 4, 32, Act::Relu, 1e-5).unwrap();
        let x0: Vec<f32> = (0..2 * 5 * 16).map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.5).collect();
        // Batch 0 and batch 1 identical except in the last (masked) token.
        let mut x1 = x0.clone();
        for v in &mut x1[4 * 16..5 * 16] {
            *v += 3.0;
        }
        for v in &mut x1[(5 + 4) * 16..(5 + 5) * 16] {
            *v -= 2.0;
        }
        let x = be.upload(&x1);
        let mask = be.upload(&[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        layer.forward(&be, &x, 2, 5, Some(&mask));
        let y = be.download(&x);
        for i in 0..4 * 16 {
            assert!((y[i] - y[5 * 16 + i]).abs() < 1e-5, "masked key leaked into token {}", i / 16);
        }
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sincos_matches_formula() {
        let pe = sincos_table(4, 6);
        assert_eq!(pe[0], 0.0);
        assert_eq!(pe[1], 1.0);
        let expect = (3.0f64 * 10000f64.powf(-2.0 / 6.0)).sin() as f32;
        assert!((pe[3 * 6 + 2] - expect).abs() < 1e-6);
    }
}
