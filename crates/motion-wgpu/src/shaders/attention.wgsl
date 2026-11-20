// attention over packed rows [q | k | v] of width (heads + 2·kv)·d: one
// thread per query row, 64 queries per workgroup, grid (⌈seq/64⌉, heads,
// batch). Keys stream from global memory with an online softmax; a key with
// mask ≤ 0.5 is skipped; a row with no allowed key writes zeros. d ≤ 128.
struct P { batch: u32, seq: u32, heads: u32, kv: u32, d: u32, has_mask: u32, scale: f32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.y;
    let b = wid.z;
    let i = wid.x * 64u + lid.x;
    if (i >= p.seq) { return; }
    let d = p.d;
    let group = p.heads / p.kv;
    let kh = h / group;
    let width = (p.heads + 2u * p.kv) * d;
    let q_off = h * d;
    let k_off = p.heads * d + kh * d;
    let v_off = p.heads * d + p.kv * d + kh * d;
    var qr: array<f32, 128>;
    var acc: array<f32, 128>;
    for (var c = 0u; c < 128u; c = c + 1u) { qr[c] = 0.0; acc[c] = 0.0; }
    let qbase = (b * p.seq + i) * width + q_off;
    for (var c = 0u; c < d; c = c + 1u) { qr[c] = qkv[qbase + c] * p.scale; }
    var m = -3.0e38;
    var l = 0.0;
    var any = false;
    for (var j = 0u; j < p.seq; j = j + 1u) {
        if (p.has_mask != 0u && mask[b * p.seq + j] <= 0.5) { continue; }
        any = true;
        let kbase = (b * p.seq + j) * width + k_off;
        var s = 0.0;
        for (var c = 0u; c < d; c = c + 1u) { s = fma(qr[c], qkv[kbase + c], s); }
        let mt = max(m, s);
        let corr = exp(m - mt);
        let pj = exp(s - mt);
        l = l * corr + pj;
        let vbase = (b * p.seq + j) * width + v_off;
        for (var c = 0u; c < d; c = c + 1u) { acc[c] = fma(pj, qkv[vbase + c], acc[c] * corr); }
        m = mt;
    }
    let obase = (b * p.seq + i) * p.heads * d + h * d;
    if (!any) {
        for (var c = 0u; c < d; c = c + 1u) { out[obase + c] = 0.0; }
        return;
    }
    let inv = 1.0 / l;
    for (var c = 0u; c < d; c = c + 1u) { out[obase + c] = acc[c] * inv; }
}
