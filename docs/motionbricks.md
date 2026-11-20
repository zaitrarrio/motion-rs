# MotionBricks (`motion-bricks`)

A port of the [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks)
inference graph as released (the ONNX tensor interface), for the SOMA human
and Unitree G1 checkpoints of `nvidia/MotionBricks`.

## What is ported

| piece | file |
|---|---|
| configs (SOMA: 5 codebook heads × 16 codes, code dim 160; G1: 8 × 10, 256) and tiny test configs | `config.rs` |
| feature layouts (dual 370 / 418, global 366 / 414, local 365 / 413), statistics, global ↔ local root | `rep.rs` |
| networks: root module (12-class token-count head with out-of-reach and allowed masking, shared + root-token transformers, `DoubleCondDecoder` root path), masked pose transformer (one all-masked pass, argmax), VQ-VAE conv decoder | `nets.rs` |
| the inference graph: `predict_features`, `predict_soma`, `predict_g1` | `model.rs` |
| G1 rig: qpos forward kinematics | `rig_g1.rs` |
| SOMA T-pose change (`R · capture · stdᵀ`; the capture offsets are the identity) | `model.rs` |
| engines: heading alignment, encode, conditions, decode (drops 3 boundary frames), endpoint correction, chaining | `engine.rs` |
| BVH read / write, G1 MuJoCo CSV | `io.rs` |

## Use

```bash
motion download motionbricks --include soma/
```

```bash
motion inbetween --model-dir weights/MotionBricks/soma --clip clip.bvh --context-frame 10 --target-frames 120,200 --out inbetween.bvh
```

`--clip` takes a SOMA BVH (any Kimodo BVH works) or a G1 CSV. The context is
the four 30 Hz frames from `--context-frame`; each target is up to four
frames from its start frame. `--tokens 8-14` bounds the transition length
(one token is four frames).

## Parity

`scripts/reference/motionbricks_fixtures.py`; tests in
`crates/motion-bricks/tests/parity.rs`. Errors are relative, over every
intermediate the release graph exposes (masked logits excluded).

| check | tiny (CI) | released weights |
|---|---|---|
| root token-count logits | ≤ 1.0e-6 | ≤ 2.2e-6 |
| root path (global / local root values) | ≤ 2.7e-6 | ≤ 3.4e-5 |
| pose logits | ≤ 3.5e-7 | ≤ 4.2e-5 |
| decoded local poses (valid frames) | ≤ 5.2e-6 | ≤ 2.6e-4 |
| G1 qpos / SOMA global motion | ≤ 4.5e-6 | ≤ 4.0e-5 / 8.0e-4 |
| engine encode of a real clip vs the upstream kernel | — | ≤ 5.1e-6 |

Every argmax (token counts, codes) agrees with PyTorch on every fixture.

The released-weight fixtures built from NVIDIA's sample clips are
evaluation-only data and are not in the repository: point
`MOTION_BRICKS_CLIP_FIXTURES` at a directory holding them (the fixture script
writes them from a local upstream checkout).
