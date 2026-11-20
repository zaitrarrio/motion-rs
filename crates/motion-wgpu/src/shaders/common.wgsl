// Shared by every kernel of motion-wgpu.
fn tanh_safe(u: f32) -> f32 {
    // tanh via exp, clamped: WGSL tanh overflows to NaN on some drivers.
    let e = exp(-2.0 * abs(u));
    return sign(u) * (1.0 - e) / (1.0 + e);
}
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608;
    return 0.5 * x * (1.0 + tanh_safe(c * (x + 0.044715 * x * x * x)));
}
// erf by Abramowitz & Stegun 7.1.26: |error| ≤ 1.5e-7 over the real line.
fn erf_f32(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
    return sign(x) * y;
}
fn gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_f32(x * 0.70710678118));
}
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}
// Activation codes (motion_core::backend::Act::code): 0 none, 1 relu,
// 2 leaky relu (0.01), 3 gelu (erf), 4 gelu (tanh), 5 silu.
fn act(v: f32, a: u32) -> f32 {
    if (a == 1u) { return max(v, 0.0); }
    if (a == 2u) { return select(0.01 * v, v, v >= 0.0); }
    if (a == 3u) { return gelu_erf(v); }
    if (a == 4u) { return gelu_tanh(v); }
    if (a == 5u) { return silu(v); }
    return v;
}
// Linear workgroup index of a (possibly 2-D folded) dispatch.
fn wg_index(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.x + wid.y * nwg.x;
}
