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
// erf by W. J. Cody's rational approximations (as in libm), |err| ~1e-7 in f32.
fn erf_f32(x: f32) -> f32 {
    let a = abs(x);
    var r: f32;
    if (a < 0.5) {
        let z = a * a;
        let p = ((((1.85777706184603153e-1 * z + 3.16112374387056560e0) * z + 1.13864154151050156e2) * z + 3.77485237685302021e2) * z + 3.20937758913846947e3);
        let q = ((((z + 2.36012909523441209e1) * z + 2.44024637934444173e2) * z + 1.28261652607737228e3) * z + 2.84423683343917062e3);
        r = a * p / q;
    } else if (a < 4.0) {
        let p = ((((((((5.64188496988670089e-1 * a + 8.88314979438837594e0) * a + 6.61191906371416295e1) * a + 2.98635138197400131e2) * a + 8.81952221241769090e2) * a + 1.71204761263407058e3) * a + 2.05107837782607147e3) * a + 1.23033935479799725e3) * a + 2.15311535474403846e-8);
        let q = (((((((a + 1.57449261107098347e1) * a + 1.17693950891312499e2) * a + 5.37181101862009858e2) * a + 1.62138957456669019e3) * a + 3.29079923573345963e3) * a + 4.36261909014324716e3) * a + 3.43936767414372164e3) * a + 1.23033935480374942e3;
        r = 1.0 - exp(-a * a) * p / q;
    } else {
        r = 1.0;
    }
    return sign(x) * r;
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
