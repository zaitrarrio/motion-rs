//! WebGPU backend (wgpu): one WGSL kernel per [`Backend`] op, f32 throughout.
//!
//! Runs on Vulkan / Metal / DX12 without CUDA. It is trusted on a card only
//! after `motion check --backend wgpu` passes there (the kernels are compared
//! op by op with the CPU reference by `motion-check`). So far that has
//! happened only on lavapipe, mesa's CPU Vulkan driver.
//!
//! Dispatches are recorded into one command encoder and submitted in batches;
//! every host write or read flushes first, so queue order is program order.

use motion_core::backend::{Act, Backend, BinOp, Conv1dShape};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const COMMON: &str = include_str!("shaders/common.wgsl");
const LINEAR: &str = include_str!("shaders/linear.wgsl");
const NORM: &str = include_str!("shaders/norm.wgsl");
const ATTN: &str = include_str!("shaders/attention.wgsl");
const ROPE: &str = include_str!("shaders/rope.wgsl");
const CONV: &str = include_str!("shaders/conv1d.wgsl");
const ELEM: &str = include_str!("shaders/elementwise.wgsl");

/// Submit after this many recorded dispatches.
const BATCH: usize = 64;
/// Max workgroups per dispatch dimension (WebGPU default limit).
const MAX_DIM: u32 = 65535;
/// The attention kernel keeps a query row and an accumulator of this many
/// dims in registers.
pub const MAX_HEAD_DIM: usize = 128;

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// A flat f32 device buffer.
#[derive(Clone)]
pub struct GBuf {
    buf: Arc<wgpu::Buffer>,
    len: usize,
}

impl GBuf {
    fn same(&self, o: &GBuf) -> bool {
        Arc::ptr_eq(&self.buf, &o.buf)
    }
}

pub struct GWeight {
    buf: GBuf,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WgpuOptions {
    /// Accept a CPU rasterizer (lavapipe, llvmpipe, SwiftShader, WARP). Off by
    /// default: wgpu falls back to one silently when the GPU's driver fails to
    /// load, and a run on it says nothing about the GPU.
    pub allow_software: bool,
}

impl WgpuOptions {
    /// `MOTION_ALLOW_SOFTWARE=1` sets `allow_software`.
    pub fn from_env() -> Self {
        Self { allow_software: std::env::var("MOTION_ALLOW_SOFTWARE").is_ok_and(|v| v == "1") }
    }
}

#[derive(Debug)]
pub struct InitError(pub String);

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InitError {}

struct Pipes {
    linear: wgpu::ComputePipeline,
    norm: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    conv: wgpu::ComputePipeline,
    elem: wgpu::ComputePipeline,
}

struct Pending {
    enc: Option<wgpu::CommandEncoder>,
    n: usize,
}

pub struct Wgpu {
    info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Pipes,
    pending: Mutex<Pending>,
    dummy: GBuf,
}

impl Wgpu {
    pub fn new() -> Result<Self, InitError> {
        Self::with_options(WgpuOptions::from_env())
    }

    pub fn with_options(opts: WgpuOptions) -> Result<Self, InitError> {
        pollster::block_on(Self::with_options_async(opts))
    }

    /// The first high-performance adapter with its own limits requested (the
    /// WebGPU defaults cap a binding at 128 MiB).
    pub async fn with_options_async(opts: WgpuOptions) -> Result<Self, InitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("no WebGPU adapter: {e}")))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && !opts.allow_software {
            return Err(InitError(format!(
                "the only WebGPU adapter is a CPU rasterizer ({} on {:?}); set \
                 MOTION_ALLOW_SOFTWARE=1 to use it for checks (timings on it mean nothing)",
                info.name, info.backend
            )));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("motion"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("request_device: {e}")))?;
        let make = |name: &str, body: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{body}").into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = Pipes {
            linear: make("linear", LINEAR),
            norm: make("norm", NORM),
            attn: make("attention", ATTN),
            rope: make("rope", ROPE),
            conv: make("conv1d", CONV),
            elem: make("elementwise", ELEM),
        };
        let dummy = GBuf {
            buf: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })),
            len: 4,
        };
        Ok(Self { info, device, queue, pipes, pending: Mutex::new(Pending { enc: None, n: 0 }), dummy })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// A CPU rasterizer: fine for conformance checks, meaningless for timing.
    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    fn create(&self, len: usize, contents: Option<&[f32]>) -> GBuf {
        let bytes = (len.max(1) * 4) as u64;
        let max = self.device.limits().max_storage_buffer_binding_size;
        assert!(
            bytes <= max,
            "a {len}-element buffer ({bytes} bytes) exceeds this adapter's storage binding limit ({max} bytes)"
        );
        let buf = match contents {
            Some(d) if !d.is_empty() => {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(d),
                    usage: STORAGE,
                })
            }
            _ => self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                usage: STORAGE,
                mapped_at_creation: false,
            }),
        };
        GBuf { buf: Arc::new(buf), len }
    }

    /// Record one dispatch. `params` is the uniform block as 32-bit words.
    fn dispatch(&self, pipe: &wgpu::ComputePipeline, params: &[u32], bufs: &[&GBuf], groups: (u32, u32, u32)) {
        let mut words = params.to_vec();
        while !words.len().is_multiple_of(4) {
            words.push(0);
        }
        let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.buf.as_entire_binding() });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pend = self.pending.lock().expect("pending");
        let enc = pend.enc.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        pend.n += 1;
        if pend.n >= BATCH {
            let enc = pend.enc.take().expect("encoder");
            pend.n = 0;
            drop(pend);
            self.queue.submit([enc.finish()]);
        }
    }

    fn flush(&self) {
        let mut pend = self.pending.lock().expect("pending");
        let enc = pend.enc.take();
        pend.n = 0;
        drop(pend);
        if let Some(enc) = enc {
            self.queue.submit([enc.finish()]);
        }
    }

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device lost");
    }

    /// `n` workgroups as an (x, y) grid within the per-dimension limit (the
    /// kernels fold it back with `wg_index`).
    fn grid(n: usize) -> (u32, u32, u32) {
        let n = n.max(1) as u32;
        if n <= MAX_DIM {
            (n, 1, 1)
        } else {
            let y = n.div_ceil(MAX_DIM);
            (n.div_ceil(y), y, 1)
        }
    }

    /// The elementwise kernel. `a` / `b` are the read bindings (the dummy when
    /// unused); an in-place op reads `out` itself through its `1x` twin.
    #[allow(clippy::too_many_arguments)]
    fn elem(&self, op: u32, n: usize, out: &GBuf, oo: usize, a: &GBuf, ao: usize, b: &GBuf, bo: usize, code: u32, dim: usize, ch: usize, len: usize, f0: f32, f1: f32) {
        if n == 0 {
            return;
        }
        let p = [
            n as u32, op, oo as u32, ao as u32, bo as u32, code, dim as u32, ch as u32, len as u32,
            f0.to_bits(), f1.to_bits(), 0,
        ];
        self.dispatch(&self.pipes.elem, &p, &[out, a, b], Self::grid(n.div_ceil(256)));
    }
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("size exceeds u32")
}

impl Backend for Wgpu {
    type Buf = GBuf;
    type Weight = GWeight;

    fn name(&self) -> String {
        let soft = if self.is_software() { ", software" } else { "" };
        format!("wgpu ({}, {:?}{soft})", self.info.name, self.info.backend)
    }

    fn alloc(&self, len: usize) -> GBuf {
        self.create(len, None)
    }

    fn upload(&self, data: &[f32]) -> GBuf {
        self.create(data.len(), Some(data))
    }

    fn write(&self, buf: &GBuf, off: usize, data: &[f32]) {
        assert!(off + data.len() <= buf.len, "write past the end of a buffer");
        if data.is_empty() {
            return;
        }
        // queue.write_buffer runs before the next submission's commands:
        // submit what is recorded first.
        self.flush();
        self.queue.write_buffer(&buf.buf, (off * 4) as u64, bytemuck::cast_slice(data));
    }

    fn download(&self, buf: &GBuf) -> Vec<f32> {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut pend = self.pending.lock().expect("pending");
            let enc = pend.enc.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait();
        rx.recv().expect("map callback").expect("map failed");
        let out = {
            let data = slice.get_mapped_range().expect("mapped range");
            bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec()
        };
        staging.unmap();
        out
    }

    fn buf_len(&self, buf: &GBuf) -> usize {
        buf.len
    }

    fn copy(&self, src: &GBuf, src_off: usize, dst: &GBuf, dst_off: usize, len: usize) {
        assert!(src_off + len <= src.len && dst_off + len <= dst.len, "copy out of bounds");
        if len == 0 {
            return;
        }
        if src.same(dst) {
            // A copy within one buffer must not overlap; stage it.
            let tmp = self.alloc(len);
            self.copy(src, src_off, &tmp, 0, len);
            self.copy(&tmp, 0, dst, dst_off, len);
            return;
        }
        let mut pend = self.pending.lock().expect("pending");
        let enc = pend.enc.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        enc.copy_buffer_to_buffer(&src.buf, (src_off * 4) as u64, &dst.buf, (dst_off * 4) as u64, (len * 4) as u64);
    }

    fn fill(&self, buf: &GBuf, off: usize, len: usize, value: f32) {
        assert!(off + len <= buf.len, "fill out of bounds");
        self.elem(4, len, buf, off, &self.dummy, 0, &self.dummy, 0, 0, 1, 1, 1, value, 0.0);
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> GWeight {
        assert_eq!(data.len(), rows * cols, "weight shape");
        GWeight { buf: self.upload(data), rows, cols }
    }

    fn weight_bytes(&self, w: &GWeight) -> usize {
        w.rows * w.cols * 4
    }

    fn sync(&self) {
        self.flush();
        self.wait();
    }

    fn linear(&self, x: &GBuf, x_off: usize, rows: usize, w: &GWeight, bias: Option<&GBuf>, act: Act, y: &GBuf, y_off: usize) {
        if rows == 0 {
            return;
        }
        let (n, k) = (w.rows, w.cols);
        assert!(x_off + rows * k <= x.len && y_off + rows * n <= y.len, "linear out of bounds");
        assert!(!x.same(y), "linear output aliases its input");
        let p = [u(rows), u(n), u(k), u(x_off), u(y_off), bias.is_some() as u32, act.code(), 0];
        let groups = (u(n.div_ceil(64)), u(rows.div_ceil(64)), 1);
        assert!(groups.1 <= MAX_DIM, "linear: too many rows for one dispatch");
        self.dispatch(&self.pipes.linear, &p, &[x, &w.buf, bias.unwrap_or(&self.dummy), y], groups);
    }

    fn layer_norm(&self, x: &GBuf, x_off: usize, rows: usize, dim: usize, w: &GBuf, b: &GBuf, eps: f32, out: &GBuf, out_off: usize) {
        if rows == 0 {
            return;
        }
        assert!(!x.same(out), "layer_norm output aliases its input");
        let p = [u(rows), u(dim), 0, u(x_off), u(out_off), 1, eps.to_bits(), 0];
        self.dispatch(&self.pipes.norm, &p, &[x, w, b, out], Self::grid(rows));
    }

    fn rms_norm(&self, x: &GBuf, x_off: usize, rows: usize, dim: usize, w: &GBuf, eps: f32, out: &GBuf, out_off: usize) {
        if rows == 0 {
            return;
        }
        assert!(!x.same(out), "rms_norm output aliases its input");
        let p = [u(rows), u(dim), 1, u(x_off), u(out_off), 0, eps.to_bits(), 0];
        self.dispatch(&self.pipes.norm, &p, &[x, w, &self.dummy, out], Self::grid(rows));
    }

    fn attention(&self, qkv: &GBuf, batch: usize, seq: usize, heads: usize, kv_heads: usize, head_dim: usize, key_mask: Option<&GBuf>, out: &GBuf) {
        assert!(head_dim <= MAX_HEAD_DIM, "attention: head_dim {head_dim} > {MAX_HEAD_DIM}");
        assert!(heads.is_multiple_of(kv_heads), "attention: heads must be a multiple of kv_heads");
        if batch * seq == 0 {
            return;
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let p = [u(batch), u(seq), u(heads), u(kv_heads), u(head_dim), key_mask.is_some() as u32, scale.to_bits(), 0];
        let groups = (u(seq.div_ceil(64)), u(heads), u(batch));
        self.dispatch(&self.pipes.attn, &p, &[qkv, key_mask.unwrap_or(&self.dummy), out], groups);
    }

    fn rope(&self, qkv: &GBuf, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, cos: &GBuf, sin: &GBuf) {
        let n = rows * (heads + kv_heads) * (head_dim / 2);
        if n == 0 {
            return;
        }
        let p = [u(rows), u(heads), u(kv_heads), u(head_dim), u(n), 0, 0, 0];
        self.dispatch(&self.pipes.rope, &p, &[qkv, cos, sin], Self::grid(n.div_ceil(256)));
    }

    fn swiglu(&self, x: &GBuf, rows: usize, dim: usize, out: &GBuf) {
        assert!(!x.same(out), "swiglu output aliases its input");
        self.elem(5, rows * dim, out, 0, x, 0, &self.dummy, 0, 0, dim, 1, 1, 0.0, 0.0);
    }

    fn conv1d(&self, x: &GBuf, s: Conv1dShape, w: &GWeight, bias: Option<&GBuf>, act: Act, residual: Option<&GBuf>, out: &GBuf) {
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()), "conv1d weight shape");
        assert!(!x.same(out), "conv1d output aliases its input");
        if let Some(r) = residual {
            assert!(!r.same(out), "conv1d residual aliases its output");
        }
        let n = s.out_elems();
        if n == 0 {
            return;
        }
        let p = [
            u(s.batch), u(s.cin), u(s.cout), u(s.len), u(s.k), u(s.stride), u(s.pad), u(s.dilation),
            s.upsample as u32, u(s.out_len()), bias.is_some() as u32, residual.is_some() as u32, act.code(), u(n), 0, 0,
        ];
        self.dispatch(
            &self.pipes.conv,
            &p,
            &[x, &w.buf, bias.unwrap_or(&self.dummy), residual.unwrap_or(&self.dummy), out],
            Self::grid(n.div_ceil(256)),
        );
    }

    fn unary(&self, x: &GBuf, x_off: usize, n: usize, act: Act, out: &GBuf, out_off: usize) {
        if x.same(out) {
            self.elem(10, n, out, out_off, &self.dummy, x_off, &self.dummy, 0, act.code(), 1, 1, 1, 0.0, 0.0);
        } else {
            self.elem(0, n, out, out_off, x, x_off, &self.dummy, 0, act.code(), 1, 1, 1, 0.0, 0.0);
        }
    }

    fn binary(&self, a: &GBuf, a_off: usize, b: &GBuf, b_off: usize, n: usize, op: BinOp, out: &GBuf, out_off: usize) {
        let code = match op {
            BinOp::Add => 1,
            BinOp::Mul => 2,
        };
        if b.same(out) {
            // `b` may only be read through `out` too: stage it.
            let tmp = self.alloc(n);
            self.copy(b, b_off, &tmp, 0, n);
            return self.binary(a, a_off, &tmp, 0, n, op, out, out_off);
        }
        if a.same(out) {
            self.elem(code + 10, n, out, out_off, &self.dummy, a_off, b, b_off, 0, 1, 1, 1, 0.0, 0.0);
        } else {
            self.elem(code, n, out, out_off, a, a_off, b, b_off, 0, 1, 1, 1, 0.0, 0.0);
        }
    }

    fn axpby(&self, y: &GBuf, y_off: usize, a: f32, x: &GBuf, x_off: usize, b: f32, n: usize) {
        if x.same(y) {
            self.elem(13, n, y, y_off, &self.dummy, x_off, &self.dummy, 0, 0, 1, 1, 1, a, b);
        } else {
            self.elem(3, n, y, y_off, x, x_off, &self.dummy, 0, 0, 1, 1, 1, a, b);
        }
    }

    fn mask_time(&self, x: &GBuf, batch: usize, channels: usize, len: usize, mask: &GBuf) {
        assert!(!x.same(mask), "mask_time mask aliases its buffer");
        self.elem(6, batch * channels * len, x, 0, mask, 0, &self.dummy, 0, 0, 1, channels, len, 0.0, 0.0);
    }
}
