// elementwise: one thread per element.
//   op 0  unary:        out[oo+i] = act(a[ao+i], code)
//   op 1  add:          out[oo+i] = a[ao+i] + b[bo+i]
//   op 2  mul:          out[oo+i] = a[ao+i] · b[bo+i]
//   op 3  axpby:        out[oo+i] = f0·out[oo+i] + f1·a[ao+i]
//   op 4  fill:         out[oo+i] = f0
//   op 5  swiglu:       out[r·dim + j] = silu(a[r·2dim + j]) · a[r·2dim + dim + j]
//   op 6  mask_time:    out[i] *= a[(i / (ch·len))·len + i % len]
//   op 10/11/12/13: the in-place twins of 0/1/2/3 where `a` is `out` itself
//   (a buffer cannot be bound twice, once writable).
struct P { n: u32, op: u32, oo: u32, ao: u32, bo: u32, code: u32, dim: u32, ch: u32, len: u32, f0: f32, f1: f32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    switch p.op {
        case 0u: { out[p.oo + i] = act(a[p.ao + i], p.code); }
        case 1u: { out[p.oo + i] = a[p.ao + i] + b[p.bo + i]; }
        case 2u: { out[p.oo + i] = a[p.ao + i] * b[p.bo + i]; }
        case 3u: { out[p.oo + i] = p.f0 * out[p.oo + i] + p.f1 * a[p.ao + i]; }
        case 4u: { out[p.oo + i] = p.f0; }
        case 5u: {
            let r = i / p.dim;
            let j = i % p.dim;
            let g = a[r * 2u * p.dim + j];
            out[i] = silu(g) * a[r * 2u * p.dim + p.dim + j];
        }
        case 6u: {
            let bb = i / (p.ch * p.len);
            let t = i % p.len;
            out[i] = out[i] * a[bb * p.len + t];
        }
        case 10u: { out[p.oo + i] = act(out[p.ao + i], p.code); }
        case 11u: { out[p.oo + i] = out[p.ao + i] + b[p.bo + i]; }
        case 12u: { out[p.oo + i] = out[p.ao + i] * b[p.bo + i]; }
        case 13u: { out[p.oo + i] = p.f0 * out[p.oo + i] + p.f1 * out[p.ao + i]; }
        default: {}
    }
}
