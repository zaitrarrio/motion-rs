// rope: in place rotate-half on the q and k heads of packed [q | k | v] rows.
// One thread per (row, head, i < d/2): (x1, x2) → (x1·c − x2·s, x2·c + x1·s),
// cos / sin [rows, d/2].
struct P { rows: u32, heads: u32, kv: u32, d: u32, n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(3) var<storage, read> sn: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg_index(wid, nwg) * 256u + lid.x;
    if (idx >= p.n) { return; }
    let half = p.d / 2u;
    let nh = p.heads + p.kv;
    let i = idx % half;
    let h = (idx / half) % nh;
    let r = idx / (half * nh);
    let width = (p.heads + 2u * p.kv) * p.d;
    let o = r * width + h * p.d;
    let c = cs[r * half + i];
    let s = sn[r * half + i];
    let x1 = qkv[o + i];
    let x2 = qkv[o + i + half];
    qkv[o + i] = x1 * c - x2 * s;
    qkv[o + i + half] = x2 * c + x1 * s;
}
