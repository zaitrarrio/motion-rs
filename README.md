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
| `wgpu` | NVIDIA, AMD, Intel, Apple (Vulkan / DX12 / Metal) | f32 | all 49 conformance cases pass on lavapipe, and both real models match the CPU there; **not yet run on a real GPU**. See `docs/backends.md` |

Parity with the released PyTorch code and weights (details in each doc):

| | tiny random weights (CI) | released weights |
|---|---|---|
| Kimodo, 4 DDIM steps, posed joints | 5.6e-6 | 1.1e-5 |
| MotionBricks, decoded poses (SOMA / G1) | ≤ 5.2e-6 | ≤ 2.6e-4, same argmax everywhere |
| LLM2Vec Llama, pooled embedding | 5.6e-7 | not run: the base weights are gated |

No GPU speed numbers: nothing has run on a GPU yet. The only timings are CPU
reference runs in the authoring container
(`docs/runs/2026-11-20-authoring-container/`).

## Use

```bash
cargo build --release
```

```bash
./target/release/motion download kimodo-soma-rp
```

```bash
./target/release/motion download motionbricks --include soma/
```

Generate with Kimodo (an empty prompt follows only the constraints; prompts
need embeddings from `motion encode`, see `docs/text-encoder.md`):

```bash
./target/release/motion generate --model-dir weights/Kimodo-SOMA-RP-v1.2 --duration 5 --steps 50 --bvh --out walk
```

Continue it with MotionBricks toward two poses of the same clip:

```bash
./target/release/motion inbetween --model-dir weights/MotionBricks/soma --clip walk.bvh --context-frame 100 --target-frames 20,60 --out more.bvh
```

The Motion Studio, a local web page for both models (`docs/studio.md`):

```bash
./target/release/motion serve --kimodo-dir weights/Kimodo-SOMA-RP-v1.2 --bricks-dir weights/MotionBricks/soma
```

Check the WebGPU backend on your GPU before trusting it there:

```bash
./target/release/motion check --backend wgpu
```

Every command takes `--backend cpu|wgpu`.

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
| `motion-cli` | the `motion` binary: `download`, `generate`, `inbetween`, `encode`, `serve`, `check` |
| `web/studio.html` | the Motion Studio page, compiled into the binary |

## Licences

Code: Apache-2.0. The embedded skeleton tables come from the upstream
repositories (Apache-2.0). Model weights are downloaded from Hugging Face
under NVIDIA's model licences; nothing of them is redistributed here.
