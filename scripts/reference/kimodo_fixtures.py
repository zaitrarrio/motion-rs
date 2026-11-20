"""PyTorch parity fixtures for motion-kimodo.

Runs the upstream Kimodo code (nv-tlabs/kimodo) on fixed inputs and writes safetensors under
crates/motion-kimodo/tests/fixtures:

  python3 -I scripts/reference/kimodo_fixtures.py --repo /home/user/nv-tlabs/kimodo \
      --weights <Kimodo-SOMA-RP-v1.2 dir> --out crates/motion-kimodo/tests/fixtures

  * tiny.safetensors: a two-stage denoiser with random weights and shrunk widths (latent 32, ff 64,
    2 layers, 2 heads, 16-d text) and random statistics, so `cargo test` runs without the download.
  * real.safetensors: the released weights (inputs, intermediates and outputs only).
  * rep.safetensors: diffusion tables, the positional table, the representation functions and the
    smooth-root solver (no weights).

The text encoder is never run (Llama-3 is gated): prompts are random 4096-d vectors or the empty
prompt's zero features. No real motion data is stored; keyframes come from random rotations.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)


def to_np(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            v = v.to(torch.int32) if v.dtype in (torch.bool, torch.int64, torch.int32) else v.to(torch.float32)
            out[k] = np.ascontiguousarray(v.numpy())
        else:
            a = np.asarray(v)
            out[k] = np.ascontiguousarray(a.astype(np.int32 if (a.dtype == np.bool_ or np.issubdtype(a.dtype, np.integer)) else np.float32))
    return out


def write_stats(root, dims, rng):
    for name, n in dims.items():
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "mean.npy", (rng.randn(n) * 0.3).astype(np.float64))
        np.save(d / "std.npy", (rng.rand(n) * 0.8 + 0.4).astype(np.float64))


def random_keyframe(sk, frames, rng):
    """FK of random small local rotations: global positions and rotations of `frames` poses."""
    from kimodo.geometry import axis_angle_to_matrix
    J = sk.nbjoints
    aa = torch.tensor(rng.randn(len(frames), J, 3) * 0.25, dtype=torch.float32)
    local = axis_angle_to_matrix(aa)
    root = torch.tensor([[0.2 * i, 0.92, 0.1 * i] for i in range(len(frames))], dtype=torch.float32)
    g, p, _ = sk.fk(local, root)
    return local, root, g, p


def build_conditions(rep, sk, T, rng):
    from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet, LeftHandConstraintSet
    frames = torch.tensor([0, T // 2])
    local, root, g, p = random_keyframe(sk, frames, rng)
    fb = FullBodyConstraintSet(sk, frames, p, g)
    path_frames = torch.arange(T // 2, T, 3)
    path = torch.tensor(np.stack([np.linspace(0.0, 1.0, len(path_frames)), np.linspace(0.0, 0.5, len(path_frames))], -1), dtype=torch.float32)
    r2 = Root2DConstraintSet(sk, path_frames, path)
    hf = torch.tensor([T - 2])
    _, _, g2, p2 = random_keyframe(sk, hf, rng)
    lh = LeftHandConstraintSet(sk, hf, p2, g2, None)
    cons = [fb, r2, lh]
    obs, mask = rep.create_conditions_from_constraints(cons, T, to_normalize=False, device="cpu")
    obs_n, _ = rep.create_conditions_from_constraints(cons, T, to_normalize=True, device="cpu")
    return cons, obs, obs_n, mask


def save_constraints_json(cons, path):
    from kimodo.constraints import save_constraints_lst
    save_constraints_lst(str(path), cons)


def denoiser_case(denoiser, rep, T, text_dim, rng, store, prefix, cons_obs_n, cons_mask):
    """One forward of the CFG-batched (3 copies, separated) two-stage denoiser."""
    B = 3
    x = torch.randn(B, T, rep.motion_rep_dim)
    pad = torch.ones(B, T, dtype=torch.bool)
    pad[2, T - 3:] = False
    text = torch.randn(B, 1, text_dim)
    text[1:] = 0.0
    tmask = torch.tensor([[True], [False], [False]])
    t = torch.tensor([999, 500, 3])
    heading = torch.tensor([0.0, 0.7, -2.0])
    mask = torch.stack([cons_mask.float() * 0, cons_mask.float(), cons_mask.float() * 0])
    obs = cons_obs_n[None].repeat(B, 1, 1)
    with torch.inference_mode():
        out = denoiser(x, pad, text, tmask, t, heading, mask, obs)
    for k, v in dict(x=x, pad=pad, text=text, tmask=tmask, t=t, heading=heading, mask=mask, obs=obs, out=out).items():
        store[f"{prefix}den.{k}"] = v


def sample_case(model, rep, T, text_dim, steps, store, prefix, obs_n, mask, text):
    from kimodo.motion_rep.feature_utils import length_to_mask
    shape = (1, T, rep.motion_rep_dim)
    torch.manual_seed(123)
    noise = torch.randn(shape)
    torch.manual_seed(123)
    text_feat = text[None, None] if text is not None else torch.zeros(1, 1, text_dim)
    text_pad = torch.tensor([[text is not None]])
    with torch.inference_mode():
        motion = model._generate(
            [""], T, steps, pad_mask=length_to_mask(torch.tensor([T])), first_heading_angle=torch.tensor([0.3]),
            motion_mask=mask[None].int(), observed_motion=obs_n[None], cfg_weight=[2.0, 2.0],
            text_feat=text_feat, text_pad_mask=text_pad, cfg_type="separated", progress_bar=lambda x: x)
    out = rep.inverse(motion, is_normalized=True)
    store[f"{prefix}sample.noise"] = noise[0]
    store[f"{prefix}sample.text"] = text_feat[0, 0]
    store[f"{prefix}sample.has_text"] = torch.tensor([int(text is not None)])
    store[f"{prefix}sample.motion"] = motion[0]
    for k in ("posed_joints", "global_rot_mats", "local_rot_mats", "root_positions", "smooth_root_pos", "foot_contacts", "global_root_heading"):
        store[f"{prefix}sample.out.{k}"] = out[k][0]


def tiny(args, out):
    from kimodo.model.twostage_denoiser import TwostageDenoiser
    from kimodo.model.kimodo_model import Kimodo
    from kimodo.motion_rep import KimodoMotionRep
    from kimodo.skeleton import SOMASkeleton30
    rng = np.random.RandomState(7)
    sk = SOMASkeleton30()
    tmp = tempfile.mkdtemp()
    write_stats(tmp, {"global_root": 5, "local_root": 4, "body": 364}, rng)
    rep = KimodoMotionRep(skeleton=sk, fps=30, stats_path=tmp)
    den = TwostageDenoiser(rep, "concat", llm_shape=[1, 16], use_text_mask=False, latent_dim=32, ff_size=64,
                           num_layers=2, num_heads=2, activation="gelu", dropout=0.0, pe_dropout=0.0, norm_first=False,
                           input_first_heading_angle=True, num_text_tokens_override=50).eval()
    with torch.no_grad():
        for n, p in den.named_parameters():
            p.normal_(0.0, 0.3 if p.dim() == 1 else 1.0 / np.sqrt(p.shape[-1]))
            if "norm" in n and n.endswith("weight"):
                p.add_(1.0)
    store = {f"weights.{k}": v for k, v in den.state_dict().items()}
    for name in ("global_root", "local_root", "body"):
        store[f"stats.{name}.mean"] = np.load(Path(tmp) / name / "mean.npy")
        store[f"stats.{name}.std"] = np.load(Path(tmp) / name / "std.npy")
    T = 12
    cons, obs, obs_n, mask = build_conditions(rep, sk, T, rng)
    store["cond.obs"] = obs
    store["cond.mask"] = mask
    save_constraints_json(cons, out / "tiny_constraints.json")
    denoiser_case(den, rep, T, 16, rng, store, "", obs_n, mask)
    model = Kimodo(den, None, num_base_steps=1000, device="cpu", cfg_type="separated")
    sample_case(model, rep, T, 16, 4, store, "", obs_n, mask, torch.randn(16))
    save_file(to_np(store), str(out / "tiny.safetensors"))
    print("tiny written", len(store), "tensors")


def real(args, out):
    from kimodo.model.twostage_denoiser import TwostageDenoiser
    from kimodo.model.kimodo_model import Kimodo
    from kimodo.motion_rep import KimodoMotionRep
    from kimodo.skeleton import SOMASkeleton30
    rng = np.random.RandomState(11)
    sk = SOMASkeleton30()
    w = Path(args.weights)
    rep = KimodoMotionRep(skeleton=sk, fps=30, stats_path=str(w / "stats" / "motion"))
    den = TwostageDenoiser(rep, "concat", llm_shape=[1, 4096], use_text_mask=False, latent_dim=1024, ff_size=2048,
                           num_layers=16, num_heads=8, activation="gelu", dropout=0.0, pe_dropout=0.0,
                           norm_first=False, input_first_heading_angle=True, num_text_tokens_override=50,
                           ckpt_path=str(w / "model.safetensors")).eval()
    store = {}
    T = 24
    cons, obs, obs_n, mask = build_conditions(rep, sk, T, rng)
    store["cond.obs"] = obs
    store["cond.mask"] = mask
    save_constraints_json(cons, out / "real_constraints.json")
    denoiser_case(den, rep, T, 4096, rng, store, "", obs_n, mask)
    model = Kimodo(den, None, num_base_steps=1000, device="cpu", cfg_type="separated")
    sample_case(model, rep, T, 4096, 4, store, "", obs_n, mask, None)
    save_file(to_np(store), str(out / "real.safetensors"))
    print("real written", len(store), "tensors")


def rep_fixture(args, out):
    from kimodo.model.diffusion import Diffusion
    from kimodo.model.backbone import PositionalEncoding
    from kimodo.motion_rep import KimodoMotionRep
    from kimodo.motion_rep.smooth_root import get_smooth_root_pos
    from kimodo.skeleton import SOMASkeleton30
    from kimodo.geometry import axis_angle_to_matrix
    rng = np.random.RandomState(3)
    store = {}
    diff = Diffusion(1000)
    for n in (1, 4, 50, 100):
        use, _ = diff.space_timesteps(n)
        diff.calc_diffusion_vars(use)
        store[f"diff.{n}.timesteps"] = use
        for k in ("alphas_cumprod", "alphas_cumprod_prev", "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod"):
            store[f"diff.{n}.{k}"] = getattr(diff, k)
    for d in (32, 1024):
        pe = PositionalEncoding(d, 0.0).pe[0]
        rows = torch.tensor([0, 1, 2, 51, 75, 333, 500, 998, 999])
        store[f"pe.{d}.rows"] = rows
        store[f"pe.{d}.values"] = pe[rows]
    sk = SOMASkeleton30()
    tmp = tempfile.mkdtemp()
    write_stats(tmp, {"global_root": 5, "local_root": 4, "body": 364}, rng)
    for name in ("global_root", "local_root", "body"):
        store[f"stats.{name}.mean"] = np.load(Path(tmp) / name / "mean.npy")
        store[f"stats.{name}.std"] = np.load(Path(tmp) / name / "std.npy")
    rep = KimodoMotionRep(skeleton=sk, fps=30, stats_path=tmp)
    root = torch.randn(1, 40, 5)
    store["g2l.in"] = root[0]
    store["g2l.out"] = rep.global_root_to_local_root(root, normalized=True, lengths=torch.tensor([33]))[0]
    feats = torch.randn(1, 20, rep.motion_rep_dim)
    store["inv.in"] = feats[0]
    inv = rep.inverse(feats, is_normalized=True)
    for k in ("posed_joints", "global_rot_mats", "local_rot_mats", "root_positions", "smooth_root_pos", "foot_contacts"):
        store[f"inv.{k}"] = inv[k][0]
    walk = torch.cumsum(torch.randn(1, 61, 3) * 0.03, dim=1)
    walk[..., 1] = 0.9
    store["smooth.in"] = walk[0]
    store["smooth.out"] = get_smooth_root_pos(walk)[0]
    aa = torch.tensor(rng.randn(16, 30, 3) * 0.3, dtype=torch.float32)
    local = axis_angle_to_matrix(aa)
    rp = torch.cumsum(torch.randn(16, 3) * 0.02, 0) + torch.tensor([0.0, 0.93, 0.0])
    store["enc.local"] = local
    store["enc.root"] = rp
    store["enc.out"] = rep(local[None], rp[None], to_normalize=False)[0]
    save_file(to_np(store), str(out / "rep.safetensors"))
    print("rep written", len(store), "tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep_fixture(args, out)
    tiny(args, out)
    if args.weights:
        real(args, out)


if __name__ == "__main__":
    main()
