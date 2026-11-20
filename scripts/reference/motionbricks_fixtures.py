"""PyTorch parity fixtures for motion-bricks.

Runs the MotionBricks ONNX exporter's PyTorch inference graphs (tools/onnx/models) on fixed inputs
and saves inputs, outputs and intermediates as safetensors under crates/motion-bricks/tests/fixtures.

  python3 -I scripts/reference/motionbricks_fixtures.py --repo /home/user/nvlabs/motionbricks \
      --weights <dir with soma/ and g1/ HF trees> --clips <dir with a SOMA .bvh and a G1 .csv> \
      --out crates/motion-bricks/tests/fixtures

Two kinds of fixtures:
  * *_tiny.safetensors: random-weight networks with shrunk widths (their weights are inside the
    file under the checkpoint key names), so `cargo test` runs without the 1.5 GB download.
  * *_real_*.safetensors: the released weights on the exporter's example inputs and on poses from
    a sample clip. Only inputs / outputs / intermediates are stored (the weights are not).
    The *_real_clip fixtures derive from NVIDIA's evaluation-only sample clips: keep them out of
    the repository and point MOTION_BRICKS_CLIP_FIXTURES at them.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)


# ----------------------------------------------------------------------------------------------
# helpers

def capture(model_inner, store):
    """Wrap the inference class's stages and the networks' forwards to record intermediates."""
    inner = model_inner  # mmm_inference_class / motion_inference
    root_net = getattr(inner, "_mmm_root_model", None) or inner._root_model
    pose_net = getattr(inner, "_mmm_pose_model", None) or inner._pose_model
    root_backbone = getattr(root_net, "mmm_net", None) or root_net.backbone_net
    pose_backbone = getattr(pose_net, "mmm_net", None) or pose_net.backbone_net

    orig_root = root_backbone.forward

    def root_fwd(*a, **k):
        out = orig_root(*a, **k)
        store["root.num_token_logits"] = out["num_token_logits"]
        store["root.pred_num_tokens"] = out["pred_num_tokens"]
        store["root.pred_global_root_values"] = out["pred_global_root_values"]
        return out

    root_backbone.forward = root_fwd
    orig_pose = pose_backbone.forward

    def pose_fwd(*a, **k):
        out = orig_pose(*a, **k)
        store["pose.logits"] = out["pose_logits"]
        return out

    pose_backbone.forward = pose_fwd

    orig_rt = inner._predict_root_trajectories

    def rt(batch, config={}):
        r = orig_rt(batch, config)
        store["in.global_root_values_norm"] = batch["global_root_values"]
        store["in.local_root_values_norm"] = batch["local_root_values"]
        store["in.local_poses_norm"] = batch["local_poses"]
        store["pred_num_tokens"] = r[0]
        store["pred_global_root_values"] = r[1]
        store["pred_local_root_values"] = r[2]
        return r

    inner._predict_root_trajectories = rt
    orig_pt = inner._predict_pose_tokens

    def pt(batch, config={}, info={}):
        r = orig_pt(batch, config, info)
        store["pred_pose_tokens"] = r[0]
        store["pred_pose_cond"] = r[1]
        store["pred_has_pose_cond"] = r[2]
        return r

    inner._predict_pose_tokens = pt
    orig_dec = inner._decode_motions_from_predicted_root_and_pose_tokens

    def dec(batch, config={}, info={}):
        r = orig_dec(batch, config, info)
        store["pred_local_poses"] = r[0]
        store["pred_global_poses_pre_reapply"] = r[1].clone()
        return r

    inner._decode_motions_from_predicted_root_and_pose_tokens = dec


def to_np(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if v.dtype == torch.bool:
                v = v.to(torch.int32)
            elif v.dtype in (torch.int64, torch.int32):
                v = v.to(torch.int32)
            else:
                v = v.to(torch.float32)
            out[k] = np.ascontiguousarray(v.numpy())
        else:
            a = np.asarray(v)
            if a.dtype == np.bool_ or np.issubdtype(a.dtype, np.integer):
                a = a.astype(np.int32)
            else:
                a = a.astype(np.float32)
            out[k] = np.ascontiguousarray(a)
    return out


def run_batched(model, kind, inputs, prefix, store):
    names = SOMA_INPUTS if kind == "soma" else G1_INPUTS
    with torch.inference_mode():
        outs = model(*[torch.as_tensor(inputs[n]) for n in names])
    for n in names:
        store[f"{prefix}input.{n}"] = inputs[n]
    if kind == "soma":
        store[f"{prefix}output.pred_global_motions"] = outs[0]
        store[f"{prefix}output.pred_num_tokens"] = outs[1]
    else:
        store[f"{prefix}output.pred_qpos"] = outs[0]
        store[f"{prefix}output.pred_num_tokens"] = outs[1]
        store[f"{prefix}output.pred_foot_contacts"] = outs[2]


SOMA_INPUTS = ("global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
               "local_poses", "has_local_poses", "allowed_pred_num_tokens")
G1_INPUTS = ("context_qpos", "target_qpos", "has_global_root_values", "has_local_root_values", "has_local_poses",
             "allowed_pred_num_tokens")


def example_inputs(kind):
    """The exporter's trace inputs (tools/onnx/export.py `example_inputs`), copied so onnx is not needed."""
    allowed = torch.eye(11, dtype=torch.int32)[[0, 10]]
    masks = [torch.ones(2, 8, dtype=torch.int32) for _ in range(3)]
    for mask in masks:
        mask[:, 4:6] = 0
    if kind == "g1":
        context = torch.zeros(2, 4, 36)
        context[..., 2] = 0.75
        context[..., 6] = 1.0
        target = context.clone()
        target[..., 0] = 1.0
        vals = (context, target, *masks, allowed)
    else:
        global_root = torch.zeros(2, 8, 5)
        global_root[..., 1] = 1.0
        global_root[..., 3] = 1.0
        local_root = torch.zeros(2, 8, 4)
        local_root[..., 3] = 1.0
        local_pose = torch.zeros(2, 8, 267)
        rotations = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(30)
        local_pose[..., 87:] = rotations
        vals = (global_root, masks[0], local_root, masks[1], local_pose, masks[2], allowed)
    names = G1_INPUTS if kind == "g1" else SOMA_INPUTS
    return {n: v.numpy().copy() for n, v in zip(names, vals)}


def conditions(num_target_poses, allowed=range(8, 15)):
    """write_motionbricks_conditions_kernel for one request."""
    has_g = np.zeros((1, 8), np.int32)
    has_l = np.zeros((1, 8), np.int32)
    has_p = np.zeros((1, 8), np.int32)
    for i in range(8):
        v = 1 if i < 4 + num_target_poses else 0
        has_g[0, i] = v
        has_p[0, i] = v
        has_l[0, i] = 0 if (i == 3 or i == 4 + num_target_poses - 1) else v
    allowed_arr = np.zeros((1, 11), np.int32)
    for a in allowed:
        allowed_arr[0, a - 6] = 1
    return has_g, has_l, has_p, allowed_arr


# ----------------------------------------------------------------------------------------------
# BVH → SOMA engine inputs (numpy port of motionbricks/model/soma/kernels.py)

def rot_axis(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parse_bvh(path):
    names, parents, offsets, channels = [], [], [], []
    stack = []
    in_end = False
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        t = lines[i].split()
        i += 1
        if not t:
            continue
        if t[0] == "MOTION":
            break
        if t[0] in ("ROOT", "JOINT"):
            names.append(t[1])
            parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
        elif t[:2] == ["End", "Site"]:
            in_end = True
        elif t[0] == "OFFSET" and not in_end:
            offsets[stack[-1]] = [float(v) for v in t[1:4]]
        elif t[0] == "CHANNELS":
            channels[stack[-1]] = t[2:]
        elif t[0] == "}":
            if in_end:
                in_end = False
            else:
                stack.pop()
    frames = int(lines[i].split()[1])
    frame_time = float(lines[i + 1].split()[2])
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 2:i + 2 + frames]])
    return names, parents, np.array(offsets) * 0.01, channels, data, frame_time


def bvh_world_poses(bvh, frame):
    names, parents, offsets, channels, data, _ = bvh
    row = data[frame]
    col = 0
    world_p, world_r = [], []
    for j, ch in enumerate(channels):
        p = offsets[j].copy()
        r = np.eye(3)
        for c in ch:
            v = row[col]
            col += 1
            cl = c.lower()
            if cl.endswith("position"):
                p["xyz".index(cl[0])] = v * 0.01
            else:
                r = r @ rot_axis("xyz".index(cl[0]), v)
        if parents[j] < 0:
            wp_, wr = p, r
        else:
            wp_ = world_p[parents[j]] + world_r[parents[j]] @ p
            wr = world_r[parents[j]] @ r
        world_p.append(wp_)
        world_r.append(wr)
    return np.array(world_p), np.array(world_r)


SOMA30 = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
          "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
          "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]


def heading_from_hips(right, left):
    across = np.array([right[0] - left[0], 0.0, right[2] - left[2]])
    n = np.linalg.norm(across)
    if n <= 1e-6:
        return 0.0
    across /= n
    forward = np.cross([0.0, 1.0, 0.0], across)
    return math.atan2(forward[0], forward[2])


def encode_soma(world_p, world_r, right_leg, left_leg, desired=(1.0, 0.0)):
    """world_p [8, 30, 3], world_r [8, 30, 3, 3] in SOMA model space (model joint order)."""
    hips = world_p[3, 0]
    cur = heading_from_hips(world_p[3, right_leg], world_p[3, left_leg])
    des = math.atan2(desired[0], desired[1])
    ang = cur - des
    a_p = np.array([hips[0], 0.0, hips[2]])
    a_r = rot_axis(1, math.degrees(ang))
    inv = a_r.T

    def apos(p):
        return inv @ (p - a_p)

    g = np.zeros((8, 5), np.float32)
    l = np.zeros((8, 4), np.float32)
    lp = np.zeros((8, 267), np.float32)
    for f in range(8):
        nxt = f if f == 3 else min(f + 1, 7)
        ch = apos(world_p[f, 0])
        nh = apos(world_p[nxt, 0])
        c_head = heading_from_hips(apos(world_p[f, right_leg]), apos(world_p[f, left_leg]))
        n_head = heading_from_hips(apos(world_p[nxt, right_leg]), apos(world_p[nxt, left_leg]))
        d = math.atan2(math.sin(n_head - c_head), math.cos(n_head - c_head))
        g[f] = [ch[0], ch[1], ch[2], math.cos(c_head), math.sin(c_head)]
        l[f] = [d * 30.0, (nh[0] - ch[0]) * 30.0, (nh[2] - ch[2]) * 30.0, ch[1]]
        root = np.array([ch[0], 0.0, ch[2]])
        for j in range(30):
            jp = apos(world_p[f, j])
            jr = inv @ world_r[f, j]
            if j > 0:
                lp[f, (j - 1) * 3:(j - 1) * 3 + 3] = jp - root
            o = 29 * 3 + j * 6
            lp[f, o:o + 3] = jr[:, 0]
            lp[f, o + 3:o + 6] = jr[:, 1]
    return g, l, lp, np.array([a_p[0], a_p[1], a_p[2], ang], np.float32)


def soma_clip_inputs(bvh_path, context_frame, target_frame, num_target=4):
    bvh = parse_bvh(bvh_path)
    names = bvh[0]
    idx = [names.index(n) for n in SOMA30]
    step = int(round(1.0 / bvh[5] / 30.0))
    frames = [context_frame + step * i for i in range(4)] + [target_frame + step * i for i in range(4)]
    wp_, wr = [], []
    for f in frames:
        p, r = bvh_world_poses(bvh, f)
        wp_.append(p[idx])
        wr.append(r[idx])
    wp_, wr = np.array(wp_), np.array(wr)
    g, l, lp, align = encode_soma(wp_, wr, SOMA30.index("RightLeg"), SOMA30.index("LeftLeg"))
    has_g, has_l, has_p, allowed = conditions(num_target)
    inputs = dict(global_root_values=g[None], has_global_root_values=has_g, local_root_values=l[None],
                  has_local_root_values=has_l, local_poses=lp[None], has_local_poses=has_p,
                  allowed_pred_num_tokens=allowed)
    extra = {"clip.world_positions": wp_.astype(np.float32), "clip.world_rotations": wr.astype(np.float32),
             "clip.alignment": align}
    return inputs, extra


def g1_clip_inputs(csv_path, context_frame, target_frame, num_target=4):
    from scipy.spatial.transform import Rotation as R
    rows = [l.strip().split(",") for l in Path(csv_path).read_text().splitlines() if l.strip()]
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    q = np.empty((len(data), 36), np.float32)
    q[:, :3] = data[:, 1:4] * 0.01
    q[:, 3:7] = R.from_euler("xyz", data[:, 4:7], degrees=True).as_quat()
    q[:, 7:] = np.deg2rad(data[:, 7:])
    step = 4  # 120 Hz clips
    ctx = q[[context_frame + step * i for i in range(4)]]
    tgt = q[[target_frame + step * i for i in range(4)]]
    # G1 alignment (unitree_g1/kernels.py): last context root to face +X, position on the ground.
    root_q = ctx[3, 3:7] / np.linalg.norm(ctx[3, 3:7])
    fwd = R.from_quat(root_q).apply([1.0, 0.0, 0.0])
    cur = math.atan2(fwd[1], fwd[0])
    a_p = np.array([ctx[3, 0], ctx[3, 1], 0.0])
    a_r = R.from_rotvec([0, 0, cur])

    def enc(frames):
        out = frames.copy()
        for f in range(4):
            pos = a_r.inv().apply(frames[f, :3] - a_p)
            rot = (a_r.inv() * R.from_quat(frames[f, 3:7] / np.linalg.norm(frames[f, 3:7]))).as_quat()
            out[f, :3] = [pos[0], pos[1], frames[f, 2]]
            out[f, 3:7] = rot
        return out

    has_g, has_l, has_p, allowed = conditions(num_target)
    inputs = dict(context_qpos=enc(ctx)[None].astype(np.float32), target_qpos=enc(tgt)[None].astype(np.float32),
                  has_global_root_values=has_g, has_local_root_values=has_l, has_local_poses=has_p,
                  allowed_pred_num_tokens=allowed)
    extra = {"clip.context_qpos_world": ctx, "clip.target_qpos_world": tgt,
             "clip.alignment": np.array([a_p[0], a_p[1], a_p[2], cur], np.float32)}
    return inputs, extra


# ----------------------------------------------------------------------------------------------
# real models

def load_sources_dir(root):
    sources = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix == ".safetensors":
            sources[str(p.relative_to(root))] = load_file(str(p), device="cpu")
        elif p.suffix == ".npy":
            sources[str(p.relative_to(root))] = np.load(p, allow_pickle=False)
    return sources


def real_fixtures(args, out):
    from models.soma import build_soma
    from models.g1 import build_g1
    soma_root = Path(args.weights) / "soma" / "soma-v6-nova-4n8g-ea43dd8-fp32-v1"
    g1_root = Path(args.weights) / "g1" / "out"
    soma = build_soma(load_sources_dir(soma_root))
    g1 = build_g1(load_sources_dir(g1_root))
    for kind, model, inner in (("soma", soma, soma.model), ("g1", g1, g1.core.inferencer)):
        store = {}
        capture(inner, store)
        inputs = example_inputs(kind)
        run_batched(model, kind, inputs, "", store)
        save_file(to_np(store), str(out / f"{kind}_real_example.safetensors"))
        print(kind, "example: tokens", store["output.pred_num_tokens"])
        store = {}
        capture(inner, store)
        if kind == "soma":
            inputs, extra = soma_clip_inputs(args.soma_clip, 200, 320)
        else:
            inputs, extra = g1_clip_inputs(args.g1_clip, 200, 320)
        run_batched(model, kind, inputs, "", store)
        store.update(extra)
        save_file(to_np(store), str(out / f"{kind}_real_clip.safetensors"))
        print(kind, "clip: tokens", store["output.pred_num_tokens"], "pose tokens", store["pred_pose_tokens"][0, :3])


# ----------------------------------------------------------------------------------------------
# tiny models

def tiny_fixture(kind, out, rig_npz_dir):
    from models.representation import GlobalRootGlobalJoints, Stats
    from models.vqvae import VQVAE
    if kind == "soma":
        from models.soma.networks import pose_mmm_network as PoseNet, root_mmm_network as RootNet
        from models.soma.layers import DoubleCondDecoder
        from models.soma.inference import mmm_inference_class as Inference
        from models.soma.model import BatchedSOMA as Batched, Predictor
        nbj, codes, nb_code, name = 30, 4, 16, "novaskel30_dual_root_global_joints"
    else:
        from models.g1.networks import pose_backbone_network as PoseNet, root_backbone_network as RootNet
        from models.layers import DoubleCondDecoder
        from models.g1.inference import motion_inference as Inference
        from models.g1.model import BatchedG1 as Batched, Predictor
        nbj, codes, nb_code, name = 34, 10, 100, "g1skel34_dual_root_global_joints"
    dim = 9 + (nbj - 1) * 3 + nbj * 6 + nbj * 3 + 4
    mean = torch.randn(dim) * 0.3
    std = torch.rand(dim) * 1.0 + 0.5
    stats = Stats()
    stats.register_from_tensors(mean.float(), std.float())
    motion_rep = GlobalRootGlobalJoints(fps=30, skeleton=SimpleNamespace(nbjoints=nbj), name=name, stats=stats)
    common = dict(min_tokens=6, max_tokens=16, down_t=2, local_pose_feature="joint_positions_and_rotations_and_hip_height")
    pose_args = dict(common, pose_root_mode="pose", pose_vqvae_motion_rep="local",
                     cond_root_feature="root_without_hip_height", cond_root_feature_is_from_motion_rep="global")
    root_args = dict(common, pose_root_mode="root", local_root_feature="root", global_root_feature="root",
                     use_hard_num_token_emb_for_root_prediction=True)
    heads = 2
    code_dim = 8
    pose_network_args = dict(pose_args, pose_token_mlp_num_layers=2, root_feat_width=16, pose_feat_width=64,
                             token_length_feat_width=16, text_emb_dim=4096, text_embeddings=None, n_embd=32,
                             n_head=2, n_layers=2,
                             pose_vqvae=dict(nb_code=nb_code, code_dim=code_dim, num_heads=heads, has_codebook=True),
                             root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1))
    root_network_args = dict(root_args, pose_feat_dim=16, local_root_feat_dim=8, global_root_feat_dim=8,
                             text_emb_dim=4096, text_embeddings=None, input_feat_mlp_num_layers=2, n_embd=32,
                             n_head=2, n_layers_shared=2, n_layers_root_token=1, width=32, depth=2,
                             dilation_growth_rate=3, activation="relu", norm="None",
                             pose_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
                             root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1))
    pose_backbone = PoseNet(motion_rep, pose_network_args)
    root_backbone = RootNet(root_network_args, motion_rep)
    vq_kwargs = dict(pose_root_mode="pose", motion_rep=motion_rep.dual_rep.local_motion_rep,
                     feature_mode=["joint_positions_and_rotations_and_hip_height", "pose",
                                   "joint_positions_and_rotations_and_hip_height",
                                   "root_without_hip_height_without_heading"],
                     nb_code=nb_code, code_dim=code_dim, output_emb_width=code_dim, down_t=2, stride_t=2, width=32,
                     depth=2, dilation_growth_rate=3, activation="relu", num_heads=heads, norm="None",
                     cond_fusion_last_layer=False)
    if kind == "soma":
        vqvae = VQVAE(decoder_type=DoubleCondDecoder, **vq_kwargs)
    else:
        vqvae = VQVAE(**vq_kwargs)
    # Random codebook (the checkpoint buffer is zero-initialised) and random parameters.
    with torch.no_grad():
        for net in (pose_backbone, root_backbone, vqvae):
            for p in net.parameters():
                if p.dim() >= 2:
                    torch.nn.init.normal_(p, std=1.0 / math.sqrt(p.shape[-1]) if p.shape[-1] > 0 else 0.02)
                else:
                    torch.nn.init.normal_(p, std=0.1)
            for n, b in net.named_buffers():
                if n.endswith("embed"):
                    b.normal_()
        pose_backbone.initted.fill_(True)
    weights = {}
    for net, prefix in ((pose_backbone, "mmm_net." if kind == "soma" else "backbone_net."),
                        (root_backbone, "mmm_net." if kind == "soma" else "backbone_net."),
                        (vqvae, "pose_net.")):
        tag = {"pose_net.": "vqvae", }.get(prefix, None)
        role = "vqvae" if net is vqvae else ("pose" if net is pose_backbone else "root")
        for k, v in net.state_dict().items():
            weights[f"weights.{role}.{prefix}{k}"] = v
    weights["stats.mean"] = mean
    weights["stats.std"] = std
    weights["skeleton.joints"] = torch.zeros(nbj, 3)
    models = dict(pose=Predictor(pose_backbone, motion_rep, pose_args, vqvae), root=Predictor(root_backbone, motion_rep, root_args))
    inner = Inference(models, pose_args, device="cpu")
    batched = Batched(inner).eval()
    store = dict(weights)
    capture(inner, store)
    inputs = example_inputs(kind)
    # Make the example a bit less degenerate: small random perturbations of the pose values.
    rng = np.random.RandomState(1)
    if kind == "soma":
        inputs["local_poses"] = inputs["local_poses"] + rng.randn(*inputs["local_poses"].shape).astype(np.float32) * 0.05
        inputs["global_root_values"][..., :3] += rng.randn(2, 8, 3).astype(np.float32) * 0.1
        inputs["local_root_values"] += rng.randn(2, 8, 4).astype(np.float32) * 0.1
    else:
        inputs["context_qpos"][..., 7:] += rng.randn(2, 4, 29).astype(np.float32) * 0.2
        inputs["target_qpos"][..., 7:] += rng.randn(2, 4, 29).astype(np.float32) * 0.2
    run_batched(batched, kind, inputs, "", store)
    save_file(to_np(store), str(out / f"{kind}_tiny.safetensors"))
    print(kind, "tiny: tokens", store["output.pred_num_tokens"], "pose tokens", store["pred_pose_tokens"][0, :2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--soma-clip")
    ap.add_argument("--g1-clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny-only", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.repo) / "tools" / "onnx"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tiny_fixture("soma", out, None)
    tiny_fixture("g1", out, None)
    if not args.tiny_only:
        real_fixtures(args, out)


if __name__ == "__main__":
    main()
