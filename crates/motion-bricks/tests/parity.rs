//! PyTorch parity of the release graph (`scripts/reference/motionbricks_fixtures.py`).
//!
//! `*_tiny` fixtures carry random weights (shrunk widths) and run in CI;
//! `*_real_*` fixtures hold only inputs / intermediates / outputs and need the
//! `nvidia/MotionBricks` download (`MOTION_BRICKS_DIR`, with `soma/` and `g1/`).

use motion_bricks::config::BricksConfig;
use motion_bricks::model::{G1Inputs, MotionBricks, SomaInputs, Trace, COUNT_CLASSES, SLOTS};
use motion_core::tensor::rel_err;
use motion_core::weights::{Prefixed, SafetensorsDir, WeightSource};
use motion_cpu::Cpu;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> SafetensorsDir {
    SafetensorsDir::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
}

/// Fixtures built from NVIDIA's evaluation-only sample clips stay out of the
/// repository: `MOTION_BRICKS_CLIP_FIXTURES` points at a directory holding them.
fn clip_fixture(name: &str) -> Option<SafetensorsDir> {
    let d = std::env::var("MOTION_BRICKS_CLIP_FIXTURES").ok()?;
    SafetensorsDir::open(&Path::new(&d).join(name)).ok()
}

fn get(f: &SafetensorsDir, k: &str) -> Vec<f32> {
    let s = f.shape(k).unwrap();
    f.get(k, &s).unwrap()
}

fn row(v: &[f32], b: usize, n: usize) -> Vec<f32> {
    v[b * n..(b + 1) * n].to_vec()
}

fn mask8(v: &[f32], b: usize) -> [bool; SLOTS] {
    std::array::from_fn(|i| v[b * SLOTS + i] > 0.5)
}

fn allowed(v: &[f32], b: usize) -> [bool; COUNT_CLASSES] {
    std::array::from_fn(|i| v[b * COUNT_CLASSES + i] > 0.5)
}

/// Relative error ignoring the positions where the reference is ±inf (masked logits).
fn rel_finite(a: &[f32], b: &[f32]) -> f32 {
    let (x, y): (Vec<f32>, Vec<f32>) =
        a.iter().zip(b).filter(|(_, y)| y.is_finite()).map(|(x, y)| (*x, *y)).unzip();
    assert!(a.iter().zip(b).all(|(x, y)| y.is_finite() || x == y), "masked logits differ");
    rel_err(&x, &y)
}

struct Report {
    worst: f32,
    lines: Vec<String>,
}

impl Report {
    fn check(&mut self, name: &str, got: &[f32], want: &[f32]) {
        let e = rel_finite(got, want);
        self.lines.push(format!("{name:32} rel {e:.2e}"));
        self.worst = self.worst.max(e);
    }
}

fn compare_trace(rep: &mut Report, f: &SafetensorsDir, b: usize, tr: &Trace, model: &MotionBricks<Cpu>) {
    let lay = model.layout();
    let lpw = lay.local_pose_feature();
    let heads = model.cfg.pose.heads;
    let codes = model.cfg.pose.codes;
    rep.check("in.global_root_values_norm", &tr.global_root_norm, &row(&get(f, "in.global_root_values_norm"), b, 40));
    rep.check("in.local_root_values_norm", &tr.local_root_norm, &row(&get(f, "in.local_root_values_norm"), b, 32));
    rep.check("in.local_poses_norm", &tr.local_poses_norm, &row(&get(f, "in.local_poses_norm"), b, 8 * lpw));
    rep.check("root.num_token_logits", &tr.count_logits, &row(&get(f, "root.num_token_logits"), b, 12));
    let n_ref = get(f, "pred_num_tokens")[b] as usize;
    assert_eq!(tr.pred_num_tokens, n_ref, "token count differs from PyTorch");
    rep.check("pred_global_root_values", &tr.pred_global_root, &row(&get(f, "pred_global_root_values"), b, 64 * 5));
    rep.check("pred_local_root_values", &tr.pred_local_root, &row(&get(f, "pred_local_root_values"), b, 64 * 4));
    rep.check("pred_pose_cond", &tr.pose_cond, &row(&get(f, "pred_pose_cond"), b, 64 * lpw));
    rep.check("pose.logits", &tr.pose_logits, &row(&get(f, "pose.logits"), b, 16 * heads * codes));
    let tok_ref = row(&get(f, "pred_pose_tokens"), b, 16 * heads);
    let valid = n_ref * heads;
    let flips = (0..valid).filter(|&i| tr.pose_tokens[i] != tok_ref[i] as usize).count();
    rep.lines.push(format!("{:32} {flips} of {valid} valid tokens differ", "pred_pose_tokens"));
    assert_eq!(flips, 0, "pose tokens differ from PyTorch");
    let w = lay.local();
    let fr = n_ref * 4;
    rep.check(
        "pred_local_poses (valid frames)",
        &tr.pred_local_poses[..fr * w],
        &row(&get(f, "pred_local_poses"), b, 64 * w)[..fr * w],
    );
    let gw = lay.global();
    rep.check(
        "pred_global_pre_reapply (valid)",
        &tr.pred_global_pre_reapply[..fr * gw],
        &row(&get(f, "pred_global_poses_pre_reapply"), b, 64 * gw)[..fr * gw],
    );
}

fn tiny_model(f: &SafetensorsDir, cfg: BricksConfig, net_prefix: &str) -> MotionBricks<Cpu> {
    let mean = get(f, "stats.mean");
    let std = get(f, "stats.std");
    MotionBricks::new(
        &Cpu,
        cfg,
        &Prefixed { inner: f, prefix: format!("weights.pose.{net_prefix}") },
        &Prefixed { inner: f, prefix: format!("weights.root.{net_prefix}") },
        &Prefixed { inner: f, prefix: "weights.vqvae.pose_net.".into() },
        mean,
        std,
    )
    .unwrap()
}

fn run_soma(f: &SafetensorsDir, model: &MotionBricks<Cpu>, tol: f32) -> Report {
    let mut rep = Report { worst: 0.0, lines: Vec::new() };
    let gw = model.layout().global();
    let batch = get(f, "input.global_root_values").len() / 40;
    for b in 0..batch {
        let x = SomaInputs {
            global_root_values: row(&get(f, "input.global_root_values"), b, 40),
            has_global_root_values: mask8(&get(f, "input.has_global_root_values"), b),
            local_root_values: row(&get(f, "input.local_root_values"), b, 32),
            has_local_root_values: mask8(&get(f, "input.has_local_root_values"), b),
            local_poses: row(&get(f, "input.local_poses"), b, 8 * 267),
            has_local_poses: mask8(&get(f, "input.has_local_poses"), b),
            allowed: allowed(&get(f, "input.allowed_pred_num_tokens"), b),
        };
        let mut tr = Trace::default();
        let out = model.predict_soma(&Cpu, &x, Some(&mut tr));
        compare_trace(&mut rep, f, b, &tr, model);
        let n = out.pred_num_tokens;
        assert_eq!(n, get(f, "output.pred_num_tokens")[b] as usize);
        rep.check(
            "output.pred_global_motions (valid)",
            &out.pred_global_motions[..n * 4 * gw],
            &row(&get(f, "output.pred_global_motions"), b, 64 * gw)[..n * 4 * gw],
        );
    }
    for l in &rep.lines {
        println!("{l}");
    }
    assert!(rep.worst <= tol, "worst relative error {:.2e} above {tol:.0e}", rep.worst);
    rep
}

fn run_g1(f: &SafetensorsDir, model: &MotionBricks<Cpu>, tol: f32) -> Report {
    let mut rep = Report { worst: 0.0, lines: Vec::new() };
    let batch = get(f, "input.context_qpos").len() / (4 * 36);
    for b in 0..batch {
        let x = G1Inputs {
            context_qpos: row(&get(f, "input.context_qpos"), b, 4 * 36),
            target_qpos: row(&get(f, "input.target_qpos"), b, 4 * 36),
            has_global_root_values: mask8(&get(f, "input.has_global_root_values"), b),
            has_local_root_values: mask8(&get(f, "input.has_local_root_values"), b),
            has_local_poses: mask8(&get(f, "input.has_local_poses"), b),
            allowed: allowed(&get(f, "input.allowed_pred_num_tokens"), b),
        };
        let mut tr = Trace::default();
        let out = model.predict_g1(&Cpu, &x, Some(&mut tr));
        compare_trace(&mut rep, f, b, &tr, model);
        let n = out.pred_num_tokens;
        rep.check(
            "output.pred_qpos (valid)",
            &out.pred_qpos[..n * 4 * 36],
            &row(&get(f, "output.pred_qpos"), b, 64 * 36)[..n * 4 * 36],
        );
        rep.check(
            "output.pred_foot_contacts (valid)",
            &out.pred_foot_contacts[..n * 4 * 4],
            &row(&get(f, "output.pred_foot_contacts"), b, 64 * 4)[..n * 4 * 4],
        );
    }
    for l in &rep.lines {
        println!("{l}");
    }
    assert!(rep.worst <= tol, "worst relative error {:.2e} above {tol:.0e}", rep.worst);
    rep
}

#[test]
fn soma_tiny_matches_pytorch() {
    let f = fixture("soma_tiny.safetensors");
    let m = tiny_model(&f, BricksConfig::tiny_soma(), "mmm_net.");
    run_soma(&f, &m, 1e-4);
}

#[test]
fn g1_tiny_matches_pytorch() {
    let f = fixture("g1_tiny.safetensors");
    let m = tiny_model(&f, BricksConfig::tiny_g1(), "backbone_net.");
    run_g1(&f, &m, 1e-4);
}

fn real_dir() -> Option<PathBuf> {
    let d = std::env::var("MOTION_BRICKS_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/tmp/claude-0/-home-user-motion-rs/e7ce916f-ca91-5408-824c-c590d325e681/scratchpad/weights/motionbricks")
    });
    d.join("soma").is_dir().then_some(d)
}

#[test]
#[ignore = "needs the nvidia/MotionBricks download (MOTION_BRICKS_DIR)"]
fn soma_real_matches_pytorch() {
    let Some(d) = real_dir() else { return };
    let m = MotionBricks::load(&Cpu, &d.join("soma")).unwrap();
    println!("== soma_real_example");
    run_soma(&fixture("soma_real_example.safetensors"), &m, 1e-3);
    if let Some(f) = clip_fixture("soma_real_clip.safetensors") {
        println!("== soma_real_clip");
        run_soma(&f, &m, 1e-3);
    }
}

#[test]
#[ignore = "needs the nvidia/MotionBricks download (MOTION_BRICKS_DIR)"]
fn g1_real_matches_pytorch() {
    let Some(d) = real_dir() else { return };
    let m = MotionBricks::load(&Cpu, &d.join("g1")).unwrap();
    println!("== g1_real_example");
    run_g1(&fixture("g1_real_example.safetensors"), &m, 1e-3);
    if let Some(f) = clip_fixture("g1_real_clip.safetensors") {
        println!("== g1_real_clip");
        run_g1(&f, &m, 1e-3);
    }
}

/// The engine's encode against the numpy port of `encode_motionbricks_inputs_kernel`
/// in the fixture script, and the full generation against the recorded output.
#[test]
#[ignore = "needs the sample BVH (MOTION_BRICKS_SOMA_BVH), its clip fixture and the weights"]
fn soma_engine_encode_matches_reference() {
    use motion_bricks::engine::{GenerateConfig, SomaEngine};
    use motion_bricks::io::read_bvh;
    let (Some(d), Some(f), Ok(bvh)) =
        (real_dir(), clip_fixture("soma_real_clip.safetensors"), std::env::var("MOTION_BRICKS_SOMA_BVH"))
    else {
        return;
    };
    let clip = read_bvh(Path::new(&bvh)).unwrap().rebase("Hips").unwrap();
    let m = MotionBricks::load(&Cpu, &d.join("soma")).unwrap();
    let eng = SomaEngine::new(&m, clip.skeleton.clone()).unwrap();
    let ctx: Vec<_> = (0..4).map(|i| clip.poses[200 + 4 * i].clone()).collect();
    let tgt: Vec<_> = (0..4).map(|i| clip.poses[320 + 4 * i].clone()).collect();
    let (x, _) = eng.encode(&ctx, &tgt, &GenerateConfig::default()).unwrap();
    let eg = rel_err(&x.global_root_values, &get(&f, "input.global_root_values"));
    let el = rel_err(&x.local_root_values, &get(&f, "input.local_root_values"));
    let ep = rel_err(&x.local_poses, &get(&f, "input.local_poses"));
    println!("encode: global {eg:.2e} local {el:.2e} poses {ep:.2e}");
    assert!(eg < 1e-4 && el < 1e-3 && ep < 1e-4);
    let out = m.predict_soma(&Cpu, &x, None);
    assert_eq!(out.pred_num_tokens, get(&f, "output.pred_num_tokens")[0] as usize);
    let motion = eng.generate(&Cpu, &ctx, &tgt, &GenerateConfig::default()).unwrap();
    println!("engine: {} tokens, {} frames", motion.tokens, motion.poses.len());
}
