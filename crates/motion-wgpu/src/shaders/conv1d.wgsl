// conv1d: out[b, co, t] = act(bias[co] + Σ_{ci,j} x[b, ci, s(t·stride + j·dil − pad)]·w[co, ci·k + j] (+ res[b, co, t])),
// one thread per output element over [batch, cout, out_len]. With `up` the
// input is read through a nearest ×2 upsample (x[.., pos / 2]).
struct P { batch: u32, cin: u32, cout: u32, len: u32, k: u32, stride: u32, pad: u32, dil: u32,
           up: u32, ol: u32, has_bias: u32, has_res: u32, act: u32, n: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    let t = i % p.ol;
    let bc = i / p.ol;
    let co = bc % p.cout;
    let b = bc / p.cout;
    var lin = p.len;
    if (p.up != 0u) { lin = p.len * 2u; }
    var acc = 0.0;
    if (p.has_bias != 0u) { acc = bias[co]; }
    let xb = b * p.cin * p.len;
    let wb = co * p.cin * p.k;
    for (var j = 0u; j < p.k; j = j + 1u) {
        let pos = i32(t * p.stride + j * p.dil) - i32(p.pad);
        if (pos < 0 || pos >= i32(lin)) { continue; }
        var q = u32(pos);
        if (p.up != 0u) { q = q / 2u; }
        for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
            acc = fma(x[xb + ci * p.len + q], w[wb + ci * p.k + j], acc);
        }
    }
    if (p.has_res != 0u) { acc = acc + res[i]; }
    out[i] = act(acc, p.act);
}
