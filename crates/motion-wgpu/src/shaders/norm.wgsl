// norm: one 256-thread workgroup per row.
//   mode 0 layer_norm: out = (x − mean)·rsqrt(var + eps)·w + b
//   mode 1 rms_norm:   out = x·rsqrt(mean(x²) + eps)·w
// Means are accumulated in f32 over a two-pass (mean, then centred variance)
// reduction, matching the f64 reference to ~1e-6 at the model widths.
struct P { rows: u32, dim: u32, mode: u32, x_off: u32, out_off: u32, has_b: u32, eps: f32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

fn block_sum(v: f32, t: u32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg_index(wid, nwg);
    if (row >= p.rows) { return; }
    let t = lid.x;
    let base = p.x_off + row * p.dim;
    let obase = p.out_off + row * p.dim;
    var s = 0.0;
    for (var j = t; j < p.dim; j = j + 256u) { s = s + x[base + j]; }
    let mean = block_sum(s, t) / f32(p.dim);
    var inv: f32;
    var sub = 0.0;
    if (p.mode == 0u) {
        var vv = 0.0;
        for (var j = t; j < p.dim; j = j + 256u) {
            let dv = x[base + j] - mean;
            vv = vv + dv * dv;
        }
        inv = inverseSqrt(block_sum(vv, t) / f32(p.dim) + p.eps);
        sub = mean;
    } else {
        var q = 0.0;
        for (var j = t; j < p.dim; j = j + 256u) { let v = x[base + j]; q = q + v * v; }
        inv = inverseSqrt(block_sum(q, t) / f32(p.dim) + p.eps);
    }
    for (var j = t; j < p.dim; j = j + 256u) {
        var o = (x[base + j] - sub) * inv * w[j];
        if (p.mode == 0u && p.has_b != 0u) { o = o + b[j]; }
        out[obase + j] = o;
    }
}
