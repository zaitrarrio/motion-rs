// linear: y[y_off + m·n + n'] = act(Σ_k x[x_off + m·k + k'] · w[n'·k + k'] + bias[n']).
// 64×64 output tile per workgroup, 16×16 threads, 4×4 register block per
// thread held as four vec4 (NVIDIA's Vulkan compiler spills array accumulators).
struct P { m: u32, n: u32, k: u32, x_off: u32, y_off: u32, has_bias: u32, act: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> xs: array<f32, 1024>;   // [16 k][64 rows]
var<workgroup> ws: array<f32, 1024>;   // [16 k][64 cols]

fn store(m: u32, n: u32, v: f32) {
    if (m >= p.m || n >= p.n) { return; }
    var o = v;
    if (p.has_bias != 0u) { o = o + bias[n]; }
    y[p.y_off + m * p.n + n] = act(o, p.act);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        for (var t = 0u; t < 4u; t = t + 1u) {
            let e = tid + t * 256u;
            let r = e / 16u;
            let kk = e % 16u;
            let gk = k0 + kk;
            var xv = 0.0;
            if (m0 + r < p.m && gk < p.k) { xv = x[p.x_off + (m0 + r) * p.k + gk]; }
            xs[kk * 64u + r] = xv;
            var wv = 0.0;
            if (n0 + r < p.n && gk < p.k) { wv = w[(n0 + r) * p.k + gk]; }
            ws[kk * 64u + r] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let b = vec4<f32>(ws[o + tx], ws[o + tx + 16u], ws[o + tx + 32u], ws[o + tx + 48u]);
            c0 = fma(vec4<f32>(xs[o + ty]), b, c0);
            c1 = fma(vec4<f32>(xs[o + ty + 16u]), b, c1);
            c2 = fma(vec4<f32>(xs[o + ty + 32u]), b, c2);
            c3 = fma(vec4<f32>(xs[o + ty + 48u]), b, c3);
        }
        workgroupBarrier();
    }
    let rows = array<vec4<f32>, 4>(c0, c1, c2, c3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + ty + 16u * i;
        let v = rows[i];
        store(m, n0 + tx, v.x);
        store(m, n0 + tx + 16u, v.y);
        store(m, n0 + tx + 32u, v.z);
        store(m, n0 + tx + 48u, v.w);
    }
}
