//! Backend conformance: run every [`Backend`] op on random inputs on the
//! backend under test and on the CPU reference, and compare.
//!
//! A GPU backend is trusted on a card only after [`run_all`] passes there
//! (`motion check --backend <b>`). Each case states its tolerance as a bound
//! on `max |a − b| / (1 + max |ref|)`, so it reads as an absolute error near 0
//! and a relative one for large values. The bounds are f32 reduction-order
//! noise for the shape (sums of up to a few thousand terms), not model
//! tolerances.

use motion_core::backend::{Act, Backend, BinOp, Conv1dShape};
use motion_core::rng::Rng;
use motion_core::skeleton::Skeleton;
use motion_core::weights::Synthetic;
use motion_cpu::Cpu;
use motion_kimodo::{CfgType, KimodoConfig, KimodoModel, KimodoMotionRep, TextCond};
use motion_kimodo::stats::Stats;

#[derive(Debug, Clone)]
pub struct CaseResult {
    pub name: String,
    /// `max |a − b| / (1 + max |ref|)`.
    pub err: f32,
    pub tol: f32,
    /// A NaN or infinity appeared in the backend's output.
    pub non_finite: bool,
}

impl CaseResult {
    pub fn passed(&self) -> bool {
        !self.non_finite && self.err <= self.tol
    }
}

pub fn compare(name: &str, got: &[f32], want: &[f32], tol: f32) -> CaseResult {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = 1.0 + want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = got.iter().zip(want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    CaseResult {
        name: name.to_string(),
        err: diff / scale,
        tol,
        non_finite: got.iter().any(|v| !v.is_finite()),
    }
}

/// One pair of handles per tensor, uploaded to both backends.
struct Pair<'a, B: Backend> {
    be: &'a B,
    cpu: Cpu,
    rng: Rng,
}

impl<'a, B: Backend> Pair<'a, B> {
    fn data(&mut self, n: usize, scale: f32) -> Vec<f32> {
        self.rng.normals(n).into_iter().map(|v| v * scale).collect()
    }
    fn rand(&mut self, n: usize, scale: f32) -> (B::Buf, <Cpu as Backend>::Buf) {
        let d = self.data(n, scale);
        self.up(&d)
    }
    fn up(&self, d: &[f32]) -> (B::Buf, <Cpu as Backend>::Buf) {
        (self.be.upload(d), self.cpu.upload(d))
    }
    fn zeros(&self, n: usize) -> (B::Buf, <Cpu as Backend>::Buf) {
        (self.be.alloc(n), self.cpu.alloc(n))
    }
    fn weight(&self, d: &[f32], r: usize, c: usize) -> (B::Weight, <Cpu as Backend>::Weight) {
        (self.be.weight(d, r, c), self.cpu.weight(d, r, c))
    }
    fn check(&self, name: &str, t: &(B::Buf, <Cpu as Backend>::Buf), tol: f32) -> CaseResult {
        compare(name, &self.be.download(&t.0), &self.cpu.download(&t.1), tol)
    }
}

const ACTS: [Act; 6] = [Act::None, Act::Relu, Act::LeakyRelu, Act::GeluErf, Act::GeluTanh, Act::Silu];

/// Every op case, then the model case. Results in order; nothing panics on a
/// mismatch (a panic is a backend bug: shapes the backend refuses).
pub fn run_all<B: Backend>(be: &B) -> Vec<CaseResult> {
    let mut out = ops(be);
    out.push(kimodo_tiny(be));
    out
}

pub fn ops<B: Backend>(be: &B) -> Vec<CaseResult> {
    let mut p = Pair { be, cpu: Cpu, rng: Rng::new(0x5eed) };
    let mut r = Vec::new();

    // memory
    let d = p.data(1000, 1.0);
    let t = p.up(&d);
    r.push(p.check("upload/download", &t, 0.0));
    let w = p.data(37, 1.0);
    be.write(&t.0, 100, &w);
    p.cpu.write(&t.1, 100, &w);
    be.fill(&t.0, 500, 63, 2.5);
    p.cpu.fill(&t.1, 500, 63, 2.5);
    let dst = p.zeros(400);
    be.copy(&t.0, 7, &dst.0, 11, 300);
    p.cpu.copy(&t.1, 7, &dst.1, 11, 300);
    r.push(p.check("write+fill", &t, 0.0));
    r.push(p.check("copy", &dst, 0.0));

    // linear: tile edges, offsets, every activation, with and without bias
    for (i, &(m, k, n)) in [(1, 3, 5), (7, 64, 64), (65, 129, 70), (130, 1024, 96), (33, 4096, 17)].iter().enumerate() {
        let act = ACTS[i % ACTS.len()];
        let x = p.data(m * k + 5, 1.0);
        let wd = p.data(n * k, 1.0 / (k as f32).sqrt());
        let bd = p.data(n, 0.5);
        let xb = p.up(&x);
        let wb = p.weight(&wd, n, k);
        let bb = p.up(&bd);
        let y = p.zeros(m * n + 3);
        let bias = i % 2 == 0;
        be.linear(&xb.0, 5, m, &wb.0, bias.then_some(&bb.0), act, &y.0, 3);
        p.cpu.linear(&xb.1, 5, m, &wb.1, bias.then_some(&bb.1), act, &y.1, 3);
        r.push(p.check(&format!("linear {m}x{k}->{n} {act:?} bias={bias}"), &y, 2e-5));
    }

    // norms
    for &(rows, dim) in &[(3, 8), (50, 1024), (5, 4096), (2, 369)] {
        let x = p.data(rows * dim + 4, 3.0);
        let xb = p.up(&x);
        let wb = p.rand(dim, 1.0);
        let bb = p.rand(dim, 1.0);
        let o = p.zeros(rows * dim + 2);
        be.layer_norm(&xb.0, 4, rows, dim, &wb.0, &bb.0, 1e-5, &o.0, 2);
        p.cpu.layer_norm(&xb.1, 4, rows, dim, &wb.1, &bb.1, 1e-5, &o.1, 2);
        r.push(p.check(&format!("layer_norm {rows}x{dim}"), &o, 1e-5));
        let o = p.zeros(rows * dim);
        be.rms_norm(&xb.0, 0, rows, dim, &wb.0, 1e-6, &o.0, 0);
        p.cpu.rms_norm(&xb.1, 0, rows, dim, &wb.1, 1e-6, &o.1, 0);
        r.push(p.check(&format!("rms_norm {rows}x{dim}"), &o, 1e-5));
    }

    // attention: MHA (Kimodo / MotionBricks shapes), GQA (Llama), key masks
    for &(batch, seq, heads, kv, hd, masked) in &[
        (1, 5, 2, 2, 8, false),
        (3, 140, 8, 8, 128, true),
        (2, 70, 4, 4, 64, true),
        (1, 37, 8, 2, 128, false),
        (2, 9, 4, 1, 16, true),
    ] {
        let width = (heads + 2 * kv) * hd;
        let qkv = p.rand(batch * seq * width, 1.0);
        let mut m: Vec<f32> = (0..batch * seq).map(|i| if (i * 7) % 5 == 0 { 0.0 } else { 1.0 }).collect();
        if masked {
            // one sequence fully masked: the output must be zeros, not NaN
            for v in m.iter_mut().take(seq) {
                *v = 0.0;
            }
        }
        let mb = p.up(&m);
        let o = p.zeros(batch * seq * heads * hd);
        be.attention(&qkv.0, batch, seq, heads, kv, hd, masked.then_some(&mb.0), &o.0);
        p.cpu.attention(&qkv.1, batch, seq, heads, kv, hd, masked.then_some(&mb.1), &o.1);
        r.push(p.check(&format!("attention b{batch} s{seq} h{heads}/{kv} d{hd} mask={masked}"), &o, 2e-5));
    }

    // rope (in place)
    {
        let (rows, heads, kv, hd) = (11, 4, 2, 32);
        let qkv = p.rand(rows * (heads + 2 * kv) * hd, 1.0);
        let ang: Vec<f32> = (0..rows * hd / 2).map(|i| i as f32 * 0.37).collect();
        let c = p.up(&ang.iter().map(|a| a.cos()).collect::<Vec<_>>());
        let s = p.up(&ang.iter().map(|a| a.sin()).collect::<Vec<_>>());
        be.rope(&qkv.0, rows, heads, kv, hd, &c.0, &s.0);
        p.cpu.rope(&qkv.1, rows, heads, kv, hd, &c.1, &s.1);
        r.push(p.check("rope", &qkv, 1e-6));
    }

    // swiglu
    {
        let (rows, dim) = (9, 300);
        let x = p.rand(rows * 2 * dim, 2.0);
        let o = p.zeros(rows * dim);
        be.swiglu(&x.0, rows, dim, &o.0);
        p.cpu.swiglu(&x.1, rows, dim, &o.1);
        r.push(p.check("swiglu", &o, 1e-6));
    }

    // conv1d: the MotionBricks decoder shapes (k 3, pad 1, dilations, ×2 upsample, strided)
    for (i, &(batch, cin, cout, len, k, stride, pad, dil, up, res)) in [
        (1, 4, 6, 10, 3, 1, 1, 1, false, false),
        (2, 64, 48, 16, 3, 1, 3, 3, false, true),
        (2, 32, 32, 12, 3, 1, 1, 1, true, false),
        (1, 16, 24, 33, 4, 2, 1, 1, false, false),
        (3, 160, 128, 8, 3, 1, 9, 9, false, true),
        (1, 8, 8, 5, 1, 1, 0, 1, false, false),
    ]
    .iter()
    .enumerate()
    {
        let s = Conv1dShape { batch, cin, cout, len, k, stride, pad, dilation: dil, upsample: up };
        let act = ACTS[i % ACTS.len()];
        let x = p.rand(batch * cin * len, 1.0);
        let wd = p.data(cout * cin * k, 1.0 / ((cin * k) as f32).sqrt());
        let w = p.weight(&wd, cout, cin * k);
        let b = p.rand(cout, 0.3);
        let rs = p.rand(s.out_elems(), 1.0);
        let o = p.zeros(s.out_elems());
        be.conv1d(&x.0, s, &w.0, Some(&b.0), act, res.then_some(&rs.0), &o.0);
        p.cpu.conv1d(&x.1, s, &w.1, Some(&b.1), act, res.then_some(&rs.1), &o.1);
        r.push(p.check(&format!("conv1d {batch}x{cin}x{len} -> {cout} k{k} s{stride} p{pad} d{dil} up={up} res={res} {act:?}"), &o, 2e-5));
    }

    // elementwise
    let n = 70_000;
    let a = p.data(n + 10, 3.0);
    let bd = p.data(n + 10, 3.0);
    for act in ACTS {
        let x = p.up(&a);
        let o = p.zeros(n + 4);
        be.unary(&x.0, 10, n, act, &o.0, 4);
        p.cpu.unary(&x.1, 10, n, act, &o.1, 4);
        r.push(p.check(&format!("unary {act:?}"), &o, 1e-6));
        // in place
        be.unary(&x.0, 0, n, act, &x.0, 0);
        p.cpu.unary(&x.1, 0, n, act, &x.1, 0);
        r.push(p.check(&format!("unary {act:?} in place"), &x, 1e-6));
    }
    for op in [BinOp::Add, BinOp::Mul] {
        let x = p.up(&a);
        let y = p.up(&bd);
        let o = p.zeros(n);
        be.binary(&x.0, 3, &y.0, 7, n, op, &o.0, 0);
        p.cpu.binary(&x.1, 3, &y.1, 7, n, op, &o.1, 0);
        r.push(p.check(&format!("binary {op:?}"), &o, 1e-6));
        be.binary(&x.0, 0, &y.0, 0, n, op, &x.0, 0);
        p.cpu.binary(&x.1, 0, &y.1, 0, n, op, &x.1, 0);
        r.push(p.check(&format!("binary {op:?} in place"), &x, 1e-6));
    }
    {
        let y = p.up(&a);
        let x = p.up(&bd);
        be.axpby(&y.0, 2, 0.7, &x.0, 5, -1.3, n);
        p.cpu.axpby(&y.1, 2, 0.7, &x.1, 5, -1.3, n);
        r.push(p.check("axpby", &y, 1e-6));
        be.axpby(&y.0, 0, 0.5, &y.0, n / 2, 2.0, n / 4);
        p.cpu.axpby(&y.1, 0, 0.5, &y.1, n / 2, 2.0, n / 4);
        r.push(p.check("axpby same buffer", &y, 1e-6));
    }
    {
        let (batch, ch, len) = (3, 20, 17);
        let x = p.rand(batch * ch * len, 1.0);
        let m: Vec<f32> = (0..batch * len).map(|i| (i % 3 != 0) as u8 as f32).collect();
        let mb = p.up(&m);
        be.mask_time(&x.0, batch, ch, len, &mb.0);
        p.cpu.mask_time(&x.1, batch, ch, len, &mb.1);
        r.push(p.check("mask_time", &x, 0.0));
    }
    r
}

/// Three DDIM steps of a tiny two-stage Kimodo (random weights, separated
/// CFG, a root constraint at both ends): every op composed as the model
/// composes them.
pub fn kimodo_tiny<B: Backend>(be: &B) -> CaseResult {
    compare(
        "kimodo tiny: 3 DDIM steps, separated CFG, root constraint",
        &kimodo_sample(be),
        &kimodo_sample(&Cpu),
        1e-4,
    )
}

fn kimodo_sample<B: Backend>(be: &B) -> Vec<f32> {
    let frames = 24;
    let cfg = KimodoConfig::tiny(Skeleton::soma30());
    let st = |n: usize| Stats::new(vec![0.0; n], vec![1.0; n]);
    let rep = KimodoMotionRep::new(Skeleton::soma30(), 30.0, st(5), st(4), st(364));
    let d = rep.dim();
    let m = KimodoModel::from_parts(be, cfg, rep, &Synthetic::new(7)).expect("tiny kimodo");
    let noise = Rng::new(3).normals(frames * d);
    let mut obs = vec![0.0; frames * d];
    let mut mask = vec![0.0; frames * d];
    for f in [0, frames - 1] {
        for c in 0..3 {
            obs[f * d + c] = 0.25 * c as f32;
            mask[f * d + c] = 1.0;
        }
    }
    let cfg = CfgType::Separated(2.0, 2.0);
    m.sample(be, &TextCond::Empty, frames, 3, cfg, Some((&obs, &mask)), 0.3, noise, &mut |_, _| {})
}

/// A one-line-per-case report and whether everything passed.
pub fn report(results: &[CaseResult]) -> (String, bool) {
    let mut s = String::new();
    let mut ok = true;
    for c in results {
        let mark = if c.passed() { "ok  " } else { "FAIL" };
        ok &= c.passed();
        let nf = if c.non_finite { "  non-finite output" } else { "" };
        s += &format!("{mark} {:>9.2e} (tol {:.0e})  {}{nf}\n", c.err, c.tol, c.name);
    }
    let failed = results.iter().filter(|c| !c.passed()).count();
    s += &format!("{} cases, {failed} failed\n", results.len());
    (s, ok)
}
