//! The op vocabulary every backend implements.
//!
//! Small and fused, following fastwan-rs / h3-rs: each op is one kernel on a
//! GPU and the set is exactly what the Kimodo denoiser, the MotionBricks
//! modules (transformers plus 1-D convolution stacks) and the LLM2Vec text
//! encoder need. Everything else (feature packing, masks, gathers, sampling)
//! is host-side glue on small tensors in the model crates.
//!
//! Buffers are flat f32 (the backend may store narrower precision internally;
//! `upload` / `download` convert). Shapes are passed per call, never stored.
//! An output never aliases an input except where an op is documented as *in
//! place*.

/// Activation epilogues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    /// `nn.LeakyReLU` default slope 0.01 (MotionBricks `FCBlock`).
    LeakyRelu,
    /// `0.5·x·(1 + erf(x/√2))` (`nn.GELU()` / `activation="gelu"` in PyTorch).
    GeluErf,
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
    GeluTanh,
    /// `x·σ(x)`.
    Silu,
}

impl Act {
    pub fn code(self) -> u32 {
        match self {
            Act::None => 0,
            Act::Relu => 1,
            Act::LeakyRelu => 2,
            Act::GeluErf => 3,
            Act::GeluTanh => 4,
            Act::Silu => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Mul,
}

/// A 1-D convolution over `batch` signals of `cin` channels and `len` samples
/// (PyTorch `Conv1d` layout `[batch, cin, len]`, weight `[cout, cin, k]`
/// flattened to `[cout, cin·k]`). With `upsample` the input is read through a
/// nearest ×2 upsample first (`nn.Upsample(scale_factor=2)` + conv fused).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dShape {
    pub batch: usize,
    pub cin: usize,
    pub cout: usize,
    /// Input length before any upsample.
    pub len: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
    pub upsample: bool,
}

impl Conv1dShape {
    /// Length the convolution reads (after the optional upsample).
    pub fn in_len(&self) -> usize {
        if self.upsample { self.len * 2 } else { self.len }
    }
    /// `(L + 2p − d·(k − 1) − 1) / s + 1`.
    pub fn out_len(&self) -> usize {
        (self.in_len() + 2 * self.pad - self.dilation * (self.k - 1) - 1) / self.stride + 1
    }
    pub fn out_elems(&self) -> usize {
        self.batch * self.cout * self.out_len()
    }
    pub fn k_len(&self) -> usize {
        self.cin * self.k
    }
}

pub trait Backend {
    /// A flat device buffer of activations.
    type Buf: Clone;
    /// A `[rows, cols]` row-major matrix (`nn.Linear` layout `[out, in]`, or
    /// a convolution weight flattened to `[cout, cin·k]`).
    type Weight;

    fn name(&self) -> String;

    /// Zero-filled.
    fn alloc(&self, len: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// `buf[off..off + data.len()] = data`.
    fn write(&self, buf: &Self::Buf, off: usize, data: &[f32]);
    /// Blocking read of the whole buffer.
    fn download(&self, buf: &Self::Buf) -> Vec<f32>;
    fn buf_len(&self, buf: &Self::Buf) -> usize;
    /// `dst[dst_off..][..len] = src[src_off..][..len]`.
    fn copy(&self, src: &Self::Buf, src_off: usize, dst: &Self::Buf, dst_off: usize, len: usize);
    /// `buf[off..off + len] = value`.
    fn fill(&self, buf: &Self::Buf, off: usize, len: usize, value: f32);
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> Self::Weight;
    fn weight_bytes(&self, w: &Self::Weight) -> usize;
    /// Wait for queued work (timing only; downloads synchronise themselves).
    fn sync(&self);

    /// `y[y_off + i·n ..][..n] = act(x[x_off + i·k ..][..k] · Wᵀ + b)` for `i < rows`,
    /// `W` `[n, k]`.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: &Self::Buf,
        x_off: usize,
        rows: usize,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        act: Act,
        y: &Self::Buf,
        y_off: usize,
    );

    /// `out = LN(x)·w + b` over `rows` rows of width `dim` (biased variance, `eps` inside the sqrt).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &Self::Buf,
        x_off: usize,
        rows: usize,
        dim: usize,
        w: &Self::Buf,
        b: &Self::Buf,
        eps: f32,
        out: &Self::Buf,
        out_off: usize,
    );

    /// `out = x · rsqrt(mean(x²) + eps) · w`.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        x: &Self::Buf,
        x_off: usize,
        rows: usize,
        dim: usize,
        w: &Self::Buf,
        eps: f32,
        out: &Self::Buf,
        out_off: usize,
    );

    /// Softmax attention over packed projections. `qkv` holds `batch·seq` rows of
    /// `(heads + 2·kv_heads)·head_dim`: `[q | k | v]` with query head `h`
    /// reading KV head `h / (heads / kv_heads)`. Scale `1/√head_dim`.
    /// `key_mask` (optional) is `[batch·seq]`, `1` = may be attended, `0` =
    /// ignored (PyTorch `src_key_padding_mask` negated). `out` is
    /// `[batch·seq, heads·head_dim]`, ready for the output projection.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        qkv: &Self::Buf,
        batch: usize,
        seq: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        key_mask: Option<&Self::Buf>,
        out: &Self::Buf,
    );

    /// In place rotate-half RoPE on the q and k parts of `qkv` rows (same
    /// packing as [`Backend::attention`]): `cos` / `sin` are `[rows, head_dim/2]`,
    /// `(x₁, x₂) → (x₁·c − x₂·s, x₂·c + x₁·s)` with `x₂ = x₁ + head_dim/2`.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        qkv: &Self::Buf,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        cos: &Self::Buf,
        sin: &Self::Buf,
    );

    /// `out[i, :dim] = silu(x[i, :dim]) · x[i, dim:2·dim]` (Llama `[gate | up]` halves).
    fn swiglu(&self, x: &Self::Buf, rows: usize, dim: usize, out: &Self::Buf);

    /// `out[b, co, t] = act(bias[co] + Σ x[b, ci, t·s − p + d·j]·w[co, ci, j] (+ residual[b, co, t]))`.
    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        &self,
        x: &Self::Buf,
        shape: Conv1dShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        act: Act,
        residual: Option<&Self::Buf>,
        out: &Self::Buf,
    );

    /// `out[out_off + i] = act(x[x_off + i])` for `i < n`.
    fn unary(&self, x: &Self::Buf, x_off: usize, n: usize, act: Act, out: &Self::Buf, out_off: usize);

    /// `out[out_off + i] = a[a_off + i] (op) b[b_off + i]`. `out` may alias `a`.
    #[allow(clippy::too_many_arguments)]
    fn binary(
        &self,
        a: &Self::Buf,
        a_off: usize,
        b: &Self::Buf,
        b_off: usize,
        n: usize,
        op: BinOp,
        out: &Self::Buf,
        out_off: usize,
    );

    /// In place: `y[y_off + i] = a·y[y_off + i] + b·x[x_off + i]` for `i < n`.
    #[allow(clippy::too_many_arguments)]
    fn axpby(&self, y: &Self::Buf, y_off: usize, a: f32, x: &Self::Buf, x_off: usize, b: f32, n: usize);

    /// In place: `x[b, c, t] *= mask[b·len + t]` over `[batch, channels, len]`
    /// (the MotionBricks token mask between every decoder layer).
    fn mask_time(&self, x: &Self::Buf, batch: usize, channels: usize, len: usize, mask: &Self::Buf);
}
