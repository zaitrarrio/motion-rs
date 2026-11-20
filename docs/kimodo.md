# Kimodo (`motion-kimodo`)

A port of [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) inference for
the SOMA checkpoints (Kimodo-SOMA-RP-v1.2 is the one tested).

## What is ported

| piece | file | upstream |
|---|---|---|
| `config.yaml` (Hydra tree) | `config.rs` | `kimodo/configs` |
| motion statistics (`global_root`, `local_root`, `body`; eps 1e-5) | `stats.rs` | `kimodo/model/stats` |
| `KimodoMotionRep`: smooth root 3, heading 2, local positions 90, 6-D rotations 180, velocities 90, foot contacts 4 = 369 | `rep.rs` | `kimodo/motion_rep` |
| smooth root (ADMM with a multigrid solve) | `smooth_root.rs` | `kimodo/motion_rep/smooth_root.py` |
| cosine schedule, `space_timesteps`, DDIM with η = 0 (f32 tables) | `diffusion.rs` | `kimodo/diffusion` |
| CFG: `nocfg`, `regular` (2 copies), `separated` (3 copies: text, constraint, uncond) | `cfg.rs` | `kimodo/model/cfg.py` |
| two-stage denoiser: a 50-token text prefix, a time token, a heading token, sinusoidal PE; root stage (5-d) → `global_root_to_local_root` → body stage (364-d); 16 post-norm layers each, d 1024, ff 2048, 8 heads, GELU (erf); masks concatenated | `denoiser.rs` | `kimodo/model/twostage_denoiser.py` |
| constraints: `root2d`, `fullbody`, end-effector (hands, feet, with joint-name groups); the demo's `constraints.json` | `constraints.rs` | `kimodo/constraints` |
| multi-prompt sequences with transition blending | `model.rs` | `kimodo/model/kimodo.py` |
| SOMA-30 → SOMA-77 (relaxed-hand rest pose), NPZ and BVH export (`bvh_joints`, `from_standard_tpose`) | `output.rs` | `kimodo/skeleton`, `kimodo/exports` |

Text arrives as one pooled 4096-d LLM2Vec embedding (`TextCond::Embedding`)
or as the empty prompt (`TextCond::Empty`: zero features, as upstream encodes
`""`). See `text-encoder.md` for the encoder.

## Use

```bash
motion download kimodo-soma-rp
```

```bash
motion generate --model-dir weights/Kimodo-SOMA-RP-v1.2 --duration 5 --steps 50 --bvh --out walk
```

Without embeddings a prompt must be empty: the motion then follows only the
constraints (`--constraints constraints.json`, as saved by the Kimodo demo).
With `motion encode` output, pass `--embeddings prompts.safetensors` and any
prompt it holds; several prompts separated by periods make a multi-prompt
sequence, with one `--duration` per prompt.

## Parity

`scripts/reference/kimodo_fixtures.py` runs the upstream code; the tests are
in `crates/motion-kimodo/tests/parity.rs`. Errors are
`max |a − b| / max |ref|`.

| check | tiny (random weights, CI) | Kimodo-SOMA-RP-v1.2 |
|---|---|---|
| diffusion tables (1, 4, 50, 100 steps) | ≤ 6.0e-7 | same code |
| positional table (1024) | 3.0e-5 (torch.pow is off by an ulp for some divisors; the port uses a correctly rounded power) | same |
| motion rep `encode` / `inverse` / smooth root | ≤ 5.4e-7 | — |
| `create_conditions` (observed, mask) | 1.0e-6, exact | 1.7e-6, exact |
| denoiser, 3-way CFG batch | 1.6e-6 | 9.8e-5 |
| 4 DDIM steps (features) | 7.4e-6 | 1.1e-4 |
| posed joints after `inverse` | 5.6e-6 | 1.1e-5 |
| local rotation matrices | 1.1e-4 (Gram–Schmidt amplifies the feature error) | 3.8e-5 |

The BVH export matches upstream's to 0.002° per channel on a real sample.

## Notes

- Only the SOMA checkpoints are tested. Upstream's G1 forward kinematics reads
  `rest_local_rots`, which nothing in the release defines, so Kimodo-G1 is
  not supported yet.
- Kimodo's BVH rest pose (`bvh_joints.p`) is exactly the rest pose of the
  MotionBricks sample clips, so a Kimodo BVH is a valid MotionBricks clip;
  the studio relies on it.
