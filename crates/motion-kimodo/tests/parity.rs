//! PyTorch parity (`scripts/reference/kimodo_fixtures.py`): diffusion tables,
//! the positional table, the motion representation, constraints, the
//! two-stage denoiser and full DDIM sampling. `tiny` / `rep` fixtures run in
//! CI; `real` needs the Kimodo-SOMA-RP-v1.2 download (`MOTION_KIMODO_DIR`).

use motion_core::skeleton::Skeleton;
use motion_core::tensor::rel_err;
use motion_core::weights::{Prefixed, SafetensorsDir, WeightSource};
use motion_cpu::Cpu;
use motion_kimodo::cfg::CfgType;
use motion_kimodo::constraints::{build_condition_data, load_constraints_json};
use motion_kimodo::denoiser::{kimodo_pe, Cond};
use motion_kimodo::diffusion::Schedule;
use motion_kimodo::rep::KimodoMotionRep;
use motion_kimodo::smooth_root::get_smooth_root_pos;
use motion_kimodo::stats::Stats;
use motion_kimodo::{KimodoConfig, KimodoModel, TextCond};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn open(name: &str) -> SafetensorsDir {
    SafetensorsDir::open(&fixtures().join(name)).unwrap()
}

fn get(f: &SafetensorsDir, k: &str) -> Vec<f32> {
    let s = f.shape(k).unwrap();
    f.get(k, &s).unwrap()
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let e = rel_err(got, want);
    println!("{name:36} rel {e:.2e}");
    assert!(e <= tol, "{name}: {e:.2e} > {tol:.0e}");
}

fn rep_from(f: &SafetensorsDir) -> KimodoMotionRep {
    let st = |n: &str| Stats::new(get(f, &format!("stats.{n}.mean")), get(f, &format!("stats.{n}.std")));
    KimodoMotionRep::new(Skeleton::soma30(), 30.0, st("global_root"), st("local_root"), st("body"))
}

#[test]
fn diffusion_tables_match() {
    let f = open("rep.safetensors");
    for n in [1usize, 4, 50, 100] {
        let s = Schedule::new(1000, n);
        let ts: Vec<f32> = s.timesteps.iter().map(|&t| t as f32).collect();
        assert_eq!(ts, get(&f, &format!("diff.{n}.timesteps")), "timesteps for {n} steps");
        check(&format!("diff.{n}.alphas_cumprod"), &s.alphas_cumprod, &get(&f, &format!("diff.{n}.alphas_cumprod")), 1e-6);
        check(&format!("diff.{n}.alphas_cumprod_prev"), &s.alphas_cumprod_prev, &get(&f, &format!("diff.{n}.alphas_cumprod_prev")), 1e-6);
        check(&format!("diff.{n}.sqrt_recip"), &s.sqrt_recip_alphas_cumprod, &get(&f, &format!("diff.{n}.sqrt_recip_alphas_cumprod")), 1e-5);
        check(&format!("diff.{n}.sqrt_recipm1"), &s.sqrt_recipm1_alphas_cumprod, &get(&f, &format!("diff.{n}.sqrt_recipm1_alphas_cumprod")), 1e-5);
    }
}

#[test]
fn positional_table_matches() {
    let f = open("rep.safetensors");
    for d in [32usize, 1024] {
        let pe = kimodo_pe(1000, d);
        let rows = get(&f, &format!("pe.{d}.rows"));
        let got: Vec<f32> = rows.iter().flat_map(|&r| pe[r as usize * d..(r as usize + 1) * d].to_vec()).collect();
        // torch.pow's last-ulp error on a few divisors, times positions up to 999 (see `kimodo_pe`).
        check(&format!("pe.{d}"), &got, &get(&f, &format!("pe.{d}.values")), 5e-5);
    }
}

#[test]
fn representation_matches() {
    let f = open("rep.safetensors");
    let rep = rep_from(&f);
    check("global_root_to_local_root", &rep.global_root_to_local_root(&get(&f, "g2l.in"), true, 33), &get(&f, "g2l.out"), 1e-5);
    let inv = rep.inverse(&get(&f, "inv.in"), true);
    check("inverse.global_rot_mats", &inv.global_rot_mats, &get(&f, "inv.global_rot_mats"), 1e-5);
    check("inverse.local_rot_mats", &inv.local_rot_mats, &get(&f, "inv.local_rot_mats"), 1e-5);
    check("inverse.posed_joints", &inv.posed_joints, &get(&f, "inv.posed_joints"), 1e-5);
    check("inverse.root_positions", &inv.root_positions, &get(&f, "inv.root_positions"), 1e-5);
    check("inverse.foot_contacts", &inv.foot_contacts, &get(&f, "inv.foot_contacts"), 0.0);
    check("smooth_root", &get_smooth_root_pos(&get(&f, "smooth.in")), &get(&f, "smooth.out"), 1e-4);
    check("encode (__call__)", &rep.encode(&get(&f, "enc.local"), &get(&f, "enc.root")), &get(&f, "enc.out"), 1e-4);
}

fn tiny_model(f: &SafetensorsDir) -> KimodoModel<Cpu> {
    let rep = rep_from(f);
    let cfg = KimodoConfig::tiny(Skeleton::soma30());
    KimodoModel::from_parts(&Cpu, cfg, rep, &Prefixed { inner: f, prefix: "weights.".into() }).unwrap()
}

fn check_conditions(model: &KimodoModel<Cpu>, f: &SafetensorsDir, json: &str, frames: usize) {
    let cs = load_constraints_json(&fixtures().join(json), &model.rep.skeleton).unwrap();
    let data = build_condition_data(&model.rep.skeleton, &cs).unwrap();
    let (obs, mask) = model.rep.create_conditions(&data, frames).unwrap();
    check("create_conditions.mask", &mask, &get(f, "cond.mask"), 0.0);
    check("create_conditions.observed", &obs, &get(f, "cond.obs"), 1e-4);
}

fn check_denoiser(model: &KimodoModel<Cpu>, f: &SafetensorsDir, tol: f32) {
    let x = get(f, "den.x");
    let pad = get(f, "den.pad");
    let b = 3;
    let frames = pad.len() / b;
    let ts: Vec<usize> = get(f, "den.t").iter().map(|&t| t as usize).collect();
    let c = Cond {
        text: &get(f, "den.text"),
        text_tokens_in: 1,
        text_mask: &get(f, "den.tmask"),
        timesteps: &ts,
        first_heading: &get(f, "den.heading"),
        pad_mask: &pad,
    };
    let out = model.denoiser.forward(&Cpu, &model.rep, &x, b, frames, &get(f, "den.mask"), &get(f, "den.obs"), &c);
    // Padded frames of sample 2 are garbage in both implementations; compare valid frames.
    let d = model.rep.dim();
    let want = get(f, "den.out");
    let keep = |v: &[f32]| -> Vec<f32> {
        (0..b * frames).filter(|&r| pad[r] > 0.5).flat_map(|r| v[r * d..(r + 1) * d].to_vec()).collect()
    };
    check("denoiser (3-way CFG batch)", &keep(&out), &keep(&want), tol);
}

fn check_sample(model: &KimodoModel<Cpu>, f: &SafetensorsDir, tol: f32) {
    let noise = get(f, "sample.noise");
    let d = model.rep.dim();
    let frames = noise.len() / d;
    let text = if get(f, "sample.has_text")[0] > 0.5 { TextCond::Embedding(get(f, "sample.text")) } else { TextCond::Empty };
    let mut obs = get(f, "cond.obs");
    model.rep.normalize(&mut obs);
    let mask = get(f, "cond.mask");
    let x = model.sample(&Cpu, &text, frames, 4, CfgType::Separated(2.0, 2.0), Some((&obs, &mask)), 0.3, noise, &mut |_, _| {});
    check("sample (4 DDIM steps)", &x, &get(f, "sample.motion"), tol);
    let out = model.rep.inverse(&x, true);
    check("sample.posed_joints", &out.posed_joints, &get(f, "sample.out.posed_joints"), tol);
    // Rotations pass through Gram–Schmidt of the 6-D features, which amplifies
    // the feature error where a column is short (random tiny weights): 10× slack.
    check("sample.local_rot_mats", &out.local_rot_mats, &get(f, "sample.out.local_rot_mats"), 10.0 * tol);
}

#[test]
fn tiny_model_matches_pytorch() {
    let f = open("tiny.safetensors");
    let m = tiny_model(&f);
    check_conditions(&m, &f, "tiny_constraints.json", 12);
    check_denoiser(&m, &f, 1e-4);
    check_sample(&m, &f, 1e-4);
}

#[test]
#[ignore = "needs the Kimodo-SOMA-RP-v1.2 download (MOTION_KIMODO_DIR)"]
fn real_model_matches_pytorch() {
    let dir = std::env::var("MOTION_KIMODO_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/tmp/claude-0/-home-user-motion-rs/e7ce916f-ca91-5408-824c-c590d325e681/scratchpad/weights/kimodo-soma-rp-v1.2")
    });
    if !dir.join("model.safetensors").exists() {
        return;
    }
    let f = open("real.safetensors");
    let m = KimodoModel::load(&Cpu, &dir).unwrap();
    check_conditions(&m, &f, "real_constraints.json", 24);
    check_denoiser(&m, &f, 1e-3);
    check_sample(&m, &f, 1e-3);
}
