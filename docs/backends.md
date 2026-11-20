# Backends

Every model is written against `motion_core::Backend` (thirteen fused ops,
listed in `design.md`). A backend is trusted on a device only after
`motion check --backend <b>` passes there.

| backend | crate | status |
|---|---|---|
| `cpu` | `motion-cpu` | the reference: plain f32, rayon over rows, a register-tiled GEMM. Defines every op. |
| `wgpu` | `motion-wgpu` | one WGSL kernel per op, f32, dispatches batched into one command encoder (a host read or write flushes first). Passed its checks on lavapipe only; **nothing has run on a real GPU yet**. |

## `motion check`

`motion-check` runs each op on random inputs on the backend under test and on
the CPU reference and compares `max |a − b| / (1 + max |ref|)` with a bound
per case (f32 reduction-order noise for the shape): memory ops, `linear` on
tile edges with every activation, both norms, attention (MHA, GQA, key masks,
a fully masked sequence), RoPE, SwiGLU, `conv1d` with the MotionBricks
decoder's strides, dilations, ×2 upsample and residuals, elementwise ops in
and out of place, `mask_time`; then three DDIM steps of a tiny two-stage
Kimodo with separated CFG and a root constraint.

```bash
motion check --backend wgpu
```

On lavapipe (mesa 25.2.8, LLVM 20.1.2) all 49 cases pass; the tiny Kimodo
run differs from the CPU by 2.9e-6. The suite found one kernel bug while it
was written: a wrong erf in the GELU epilogue (5e-3 error), now an
Abramowitz–Stegun 7.1.26 erf (≤ 1.5e-7).

End to end on lavapipe with the released weights:

- Kimodo, 2 s, 4 steps, separated CFG: posed joints within 6.0e-7 m of the
  CPU run, rotation matrices within 1.9e-6, identical foot contacts;
- MotionBricks SOMA, two targets: the same token counts (82 frames each),
  BVH channels within 4.4e-4.

## Software adapters

wgpu silently falls back to a CPU rasterizer (lavapipe, llvmpipe,
SwiftShader, WARP) when a GPU's driver fails to load. `--backend wgpu`
refuses one unless `MOTION_ALLOW_SOFTWARE=1`, and says so when it is used:
a check on it is meaningful, a timing is not.

## CI

`.github/workflows/ci.yml` runs clippy with `-D warnings` and
`cargo test --workspace` on lavapipe with `CI=1` and
`MOTION_ALLOW_SOFTWARE=1`; with `CI=1` the conformance test fails instead of
skipping when no adapter loads.

## Limits

- Attention keeps a query row in registers: head dim ≤ 128 (all three models
  use 64 or 128).
- Every buffer must fit the adapter's storage binding limit (the adapter's
  own limits are requested; the WebGPU default is 128 MiB).
