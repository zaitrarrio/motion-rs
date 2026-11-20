# motion-rs

NVIDIA's [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) and
[MotionBricks](https://nvlabs.github.io/motionbricks/) motion models in Rust,
following the approach of [fastwan-rs](https://github.com/zaitrarrio/fastwan-rs)
and [h3-rs](https://github.com/zaitrarrio/h3-rs): one small fused op
vocabulary (`motion_core::Backend`), a CPU reference that defines every op,
a WebGPU backend checked against it, and parity fixtures generated from the
released PyTorch code and weights.

| model | what it does | checkpoint | status |
|---|---|---|---|
| **Kimodo** (`motion-kimodo`) | text + kinematic constraints (keyframes, end-effectors, 2-D paths / waypoints) → 3-D human motion on the SOMA skeleton, 30 fps, up to 10 s; two-stage transformer denoiser, DDIM, separated classifier-free guidance | [nvidia/Kimodo-SOMA-RP-v1.2](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2) (1.1 GB, NVIDIA Open Model Licence) | see `docs/kimodo.md` |
| **MotionBricks** (`motion-bricks`) | four context poses + 1–4 target poses → a 0.6–1.9 s transition, chained in real time; root module + masked pose transformer + VQ-VAE decoder; SOMA human and Unitree G1 robot | [nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks) (`soma/`, `g1/`, NVIDIA Open Model Agreement) | see `docs/motionbricks.md` |
| LLM2Vec text encoder (`motion-text`) | Kimodo's prompt conditioner: Meta-Llama-3-8B-Instruct, bidirectional, mean-pooled, two LoRA adapters | gated base weights + [McGill-NLP adapters](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp) | see `docs/text-encoder.md` |

| backend | runs on | precision | status |
|---|---|---|---|
| `cpu` | anywhere | f32 | the reference every backend is checked against |
| `wgpu` | NVIDIA, AMD, Intel, Apple (Vulkan / DX12 / Metal) | f32 | conformance suite on lavapipe only (no GPU in the authoring container); see `docs/backends.md` |

No speed numbers: nothing has been measured on named hardware yet
(`docs/runs/`).

## Use

```bash
cargo build --release
```

See `docs/` for each piece until the CLI section below is filled in.

## Layout

| crate | contents |
|---|---|
| `motion-core` | `Backend` op vocabulary, weight sources (safetensors, `.npy`, synthetic), rotation geometry, SOMA / G1 / SMPL-X skeletons with forward kinematics, embedded skeleton assets |
| `motion-cpu` | the f32 reference backend |
| `motion-nn` | PyTorch-shaped blocks: `Linear`, `LayerNorm`, post-norm `TransformerEncoderLayer`, MLP, sinusoidal table |
| `motion-kimodo` | Kimodo: denoiser, diffusion, CFG, motion representation, constraints, smooth root, NPZ / BVH export |
| `motion-bricks` | MotionBricks: networks, inference graph, SOMA and G1 rigs, runtime engines, BVH / CSV |
| `motion-text` | LLM2Vec Llama encoder, LoRA merge, tokenizer, precomputed embeddings |
| `motion-wgpu` | WGSL kernels for every op |
| `motion-check` | the conformance suite (`motion check`) |
| `motion-cli` | the `motion` binary |

## Licences

Code: Apache-2.0. The embedded skeleton tables come from the upstream
repositories (Apache-2.0). Model weights are downloaded from Hugging Face
under NVIDIA's model licences; nothing of them is redistributed here.
