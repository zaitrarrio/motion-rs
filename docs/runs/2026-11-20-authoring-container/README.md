# 2026-11-20: the authoring container (CPU only)

| | |
|---|---|
| CPU | Intel Xeon @ 2.30 GHz, 4 vCPU (cloud container) |
| memory | 15 GB |
| GPU | none; WebGPU through lavapipe (Mesa 25.2.8, LLVM 20.1.2, Vulkan 1.4) |
| toolchain | rustc 1.97.0, `--release` (thin LTO) |
| build | commit after 53e0494 (WebGPU backend) |

These are CPU-reference timings on a small shared machine. They say nothing
about a GPU and are recorded only so no number elsewhere goes unsourced.

## CPU backend, released weights

| run | wall time |
|---|---|
| Kimodo-SOMA-RP-v1.2, 5 s clip (150 frames), 10 DDIM steps, separated CFG (3-way batch), empty prompt | 80.5 s (about 8 s per step) |
| Kimodo, 3 s clip, 10 steps, two waypoints, in `motion serve` | 58.4 s |
| MotionBricks SOMA, one target (1.4 s transition), in `motion serve` (models loaded) | 0.53 s |
| `motion inbetween`, two targets, including loading the three checkpoints | 1.65–1.79 s (3 runs) |
| model load for `motion serve` (Kimodo + MotionBricks) | 4.0 s |

## WebGPU on lavapipe

`motion check --backend wgpu`: 49 of 49 cases pass. Kimodo 2 s / 4 steps took
62.4 s on lavapipe against 17.0 s on the CPU backend; lavapipe is a CPU
rasterizer and this says nothing about WebGPU on a GPU.
