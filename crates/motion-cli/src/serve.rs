//! `motion serve`: the Motion Studio. A local page (`web/studio.html`) and a
//! small job API in front of the models, in the shape of fastwan-rs's Prompt
//! Studio: the page posts a request, polls the job, and plays the result.
//!
//! * Kimodo jobs: a timeline of prompts (precomputed embeddings, the live text
//!   encoder, or the empty prompt), a root path drawn on the ground, and
//!   keyframes pinned from earlier results.
//! * MotionBricks jobs: continue any earlier motion toward target poses taken
//!   from earlier motions and placed on the ground.
//!
//! Every motion is kept on the SOMA-77 capture skeleton (the sample clips' and
//! Kimodo's BVH rest pose), so the two models feed each other.

use crate::convert::{kimodo_to_capture, soma77_capture, world_positions};
use anyhow::{anyhow, Context, Result};
use clap::Args;
use motion_bricks::config::BricksConfig;
use motion_bricks::engine::{Chain, GenerateConfig, RigSkeleton, SomaEngine, Transform};
use motion_bricks::MotionBricks;
use motion_core::backend::Backend;
use motion_core::geometry as geo;
use motion_core::skeleton::{self, Skeleton};
use motion_core::weights::Synthetic;
use motion_kimodo::constraints::Constraint;
use motion_kimodo::stats::Stats;
use motion_kimodo::{CfgType, GenerateRequest, KimodoConfig, KimodoModel, MotionOutput, TextCond};
use motion_text::{Embeddings, TextEncoder};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PAGE: &str = include_str!("../../../web/studio.html");
const KEEP_JOBS: usize = 24;

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// A Kimodo checkpoint directory (Kimodo-SOMA-RP-v1.2).
    #[arg(long)]
    pub kimodo_dir: Option<PathBuf>,
    /// The MotionBricks SOMA checkpoint directory (`nvidia/MotionBricks` `soma/`).
    #[arg(long)]
    pub bricks_dir: Option<PathBuf>,
    /// Precomputed prompt embeddings (`motion encode`): the prompt gallery.
    #[arg(long)]
    pub embeddings: Option<PathBuf>,
    /// Meta-Llama-3-8B-Instruct, to encode any prompt (with --adapters).
    #[arg(long)]
    pub llama_dir: Option<PathBuf>,
    #[arg(long, num_args = 1..)]
    pub adapters: Vec<PathBuf>,
    /// Tiny random models instead of the checkpoints: try the page without a download.
    #[arg(long)]
    pub synthetic: bool,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 7860)]
    pub port: u16,
    /// Answer CORS for one other origin that hosts the page.
    #[arg(long)]
    pub allow_origin: Option<String>,
}

/// A finished motion on the SOMA-77 capture skeleton.
struct Motion {
    label: String,
    kind: &'static str,
    fps: f32,
    poses: Vec<Vec<Transform>>,
    contacts: Vec<[f32; 4]>,
    kimodo: Option<MotionOutput>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Queued,
    Running,
    Done,
    Failed,
}

struct Job {
    id: u64,
    kind: &'static str,
    label: String,
    state: State,
    step: usize,
    steps: usize,
    segment: usize,
    segments: usize,
    message: String,
    seconds: f32,
    motion: Option<Arc<Motion>>,
}

#[derive(Default)]
struct Jobs {
    next: u64,
    map: HashMap<u64, Job>,
    order: VecDeque<u64>,
}

impl Jobs {
    fn add(&mut self, kind: &'static str, label: String, segments: usize) -> u64 {
        self.next += 1;
        let id = self.next;
        self.map.insert(
            id,
            Job { id, kind, label, state: State::Queued, step: 0, steps: 0, segment: 0, segments, message: String::new(), seconds: 0.0, motion: None },
        );
        self.order.push_back(id);
        while self.order.len() > KEEP_JOBS {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        id
    }

    fn motion(&self, id: u64) -> Result<Arc<Motion>> {
        self.map
            .get(&id)
            .and_then(|j| j.motion.clone())
            .ok_or_else(|| anyhow!("motion {id} is not available (unfinished, failed, or expired)"))
    }
}

#[derive(Deserialize, Clone)]
struct Segment {
    prompt: String,
    seconds: f32,
}

#[derive(Deserialize, Clone)]
struct Waypoint {
    t: f32,
    x: f32,
    z: f32,
}

#[derive(Deserialize, Clone)]
struct Keyframe {
    job: u64,
    frame: usize,
    at: f32,
}

#[derive(Deserialize, Clone)]
struct KimodoReq {
    segments: Vec<Segment>,
    #[serde(default = "d_steps")]
    steps: usize,
    #[serde(default = "d_cfg")]
    cfg: [f32; 2],
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    heading: f32,
    #[serde(default)]
    waypoints: Vec<Waypoint>,
    #[serde(default)]
    keyframes: Vec<Keyframe>,
    #[serde(default = "d_trans")]
    transition: usize,
}

fn d_steps() -> usize {
    50
}
fn d_cfg() -> [f32; 2] {
    [2.0, 2.0]
}
fn d_trans() -> usize {
    5
}

#[derive(Deserialize, Clone)]
struct Target {
    job: u64,
    frame: usize,
    #[serde(default = "d_poses")]
    poses: usize,
    x: f32,
    z: f32,
    /// Degrees, 0 faces +Z.
    heading: f32,
}

fn d_poses() -> usize {
    4
}

#[derive(Deserialize, Clone)]
struct BricksReq {
    /// The motion to continue; its last four frames are the context.
    context: u64,
    targets: Vec<Target>,
    #[serde(default = "d_tokens")]
    tokens: [usize; 2],
    #[serde(default)]
    transition_early: usize,
}

fn d_tokens() -> [usize; 2] {
    [8, 14]
}

enum Work {
    Kimodo(KimodoReq),
    Bricks(BricksReq),
}

struct Engine<B: Backend> {
    be: B,
    kimodo: Option<KimodoModel<B>>,
    bricks: Option<MotionBricks<B>>,
    emb: Option<Embeddings>,
    encoder: Option<TextEncoder>,
    synthetic: bool,
    rig: RigSkeleton,
}

fn heading_of(rig: &RigSkeleton, pose: &[Transform]) -> f32 {
    let w = rig.world(pose);
    let (r, l) = (rig.index("RightLeg").expect("SOMA"), rig.index("LeftLeg").expect("SOMA"));
    let d = geo::sub(&w[r].p, &w[l].p);
    // Kimodo's convention: atan2(dz, −dx) of right − left hip, 0 faces +Z.
    d[2].atan2(-d[0])
}

fn quat_y(a: f32) -> [f32; 4] {
    let (s, c) = (a / 2.0).sin_cos();
    [c, 0.0, s, 0.0]
}

impl<B: Backend> Engine<B> {
    fn text(&self, prompt: &str) -> Result<TextCond> {
        if prompt.trim().is_empty() || self.synthetic {
            return Ok(TextCond::Empty);
        }
        let k = self.kimodo.as_ref().expect("checked");
        if let Some(v) = self.emb.as_ref().and_then(|e| e.get(prompt)) {
            return Ok(TextCond::Embedding(v.to_vec()));
        }
        if let Some(enc) = &self.encoder {
            let v = enc.encode(&self.be, &[motion_text::sanitize_text(prompt)])?.remove(0);
            if v.len() != k.cfg.llm_dim() {
                return Err(anyhow!("text encoder gives {} values, Kimodo expects {}", v.len(), k.cfg.llm_dim()));
            }
            return Ok(TextCond::Embedding(v));
        }
        Err(anyhow!(
            "“{}” has no embedding and no text encoder is loaded: pick a prompt from the gallery, leave it empty, or start the engine with --llama-dir",
            motion_text::sanitize_text(prompt)
        ))
    }

    /// Global positions / rotations of one frame on SOMA-30 in Kimodo's standard T-pose.
    fn keyframe_pose(&self, m: &Motion, frame: usize) -> (Vec<f32>, Vec<f32>) {
        let s30 = Skeleton::soma30();
        let s77 = Skeleton::soma77();
        let f = frame.min(m.poses.len() - 1);
        let w = self.rig.world(&m.poses[f]);
        let map = s30.slice_into(&s77).expect("soma30 ⊂ soma77");
        let mut pos = Vec::with_capacity(90);
        let mut rot = Vec::with_capacity(270);
        for &j in &map {
            pos.extend_from_slice(&w[j].p);
            // Capture → standard: G_std = G_cap · Oᵀ (`to_standard_tpose`).
            let g = geo::quat_to_matrix(&w[j].q);
            let o = geo::mat_at(&motion_core::assets::SOMASKEL77_STANDARD_TPOSE_GLOBAL_OFFSETS, j);
            rot.extend_from_slice(&geo::matmul(&g, &geo::transpose(&o)));
        }
        (pos, rot)
    }

    fn run_kimodo(&self, r: &KimodoReq, jobs: &Mutex<Jobs>, id: u64) -> Result<Motion> {
        let k = self.kimodo.as_ref().ok_or_else(|| anyhow!("this engine was started without --kimodo-dir"))?;
        if r.segments.is_empty() {
            return Err(anyhow!("no prompts"));
        }
        let fps = k.fps();
        let frames: Vec<usize> = r.segments.iter().map(|s| ((s.seconds.clamp(0.5, 10.0)) * fps) as usize).collect();
        let total: usize = frames.iter().sum();
        let prompts = r.segments.iter().map(|s| self.text(&s.prompt)).collect::<Result<Vec<_>>>()?;
        let mut constraints = Vec::new();
        if !r.waypoints.is_empty() {
            let mut wp = r.waypoints.clone();
            wp.sort_by(|a, b| a.t.total_cmp(&b.t));
            let fr: Vec<usize> = wp.iter().map(|w| ((w.t * fps).round() as usize).min(total - 1)).collect();
            constraints.push(Constraint::root2d(fr, wp.iter().map(|w| [w.x, w.z]).collect(), None));
        }
        for kf in &r.keyframes {
            let m = jobs.lock().expect("jobs").motion(kf.job)?;
            let (pos, rot) = self.keyframe_pose(&m, kf.frame);
            let at = ((kf.at * fps).round() as usize).min(total - 1);
            constraints.push(Constraint::full_body(&k.rep.skeleton, vec![at], pos, rot, None));
        }
        let req = GenerateRequest {
            multi_prompt: prompts.len() > 1,
            prompts,
            num_frames: frames,
            steps: r.steps.clamp(1, 1000),
            cfg: CfgType::Separated(r.cfg[0], r.cfg[1]),
            seed: r.seed,
            noise: None,
            constraints,
            num_transition_frames: r.transition.max(1),
            first_heading_angle: r.heading.to_radians(),
        };
        let out = k.generate(&self.be, &req, &mut |seg, s, n| {
            if let Some(j) = jobs.lock().expect("jobs").map.get_mut(&id) {
                j.segment = seg;
                j.step = s;
                j.steps = n;
            }
        })?;
        let poses = kimodo_to_capture(&out);
        let contacts = (0..out.frames)
            .map(|t| {
                let c = &out.foot_contacts[t * 6..t * 6 + 6];
                [c[0], c[1], c[3], c[4]]
            })
            .collect();
        let label = r.segments.iter().map(|s| if s.prompt.trim().is_empty() { "(no prompt)".to_string() } else { motion_text::sanitize_text(&s.prompt) }).collect::<Vec<_>>().join(" → ");
        Ok(Motion { label, kind: "kimodo", fps, poses, contacts, kimodo: Some(out) })
    }

    fn run_bricks(&self, r: &BricksReq, jobs: &Mutex<Jobs>, id: u64) -> Result<Motion> {
        let model = self.bricks.as_ref().ok_or_else(|| anyhow!("this engine was started without --bricks-dir"))?;
        let eng = SomaEngine::new(model, self.rig.clone())?;
        let ctx = jobs.lock().expect("jobs").motion(r.context)?;
        if ctx.poses.len() < 4 {
            return Err(anyhow!("the context motion is shorter than four frames"));
        }
        let cfg = GenerateConfig { allowed_token_counts: (r.tokens[0].max(6)..=r.tokens[1].min(16)).collect(), ..Default::default() };
        let mut chain = Chain { played: ctx.poses[ctx.poses.len() - 4..].to_vec(), contacts: vec![[1.0; 4]; 4] };
        for (i, t) in r.targets.iter().enumerate() {
            let src = jobs.lock().expect("jobs").motion(t.job)?;
            let n = t.poses.clamp(1, 4);
            let f0 = t.frame.min(src.poses.len().saturating_sub(n));
            let poses = &src.poses[f0..f0 + n];
            // Place: move the first target pose's root onto (x, z) facing `heading`.
            let p0 = poses[0][0].p;
            let h0 = heading_of(&self.rig, &poses[0]);
            let rot = quat_y(t.heading.to_radians() - h0);
            let place = Transform::new([t.x, 0.0, t.z], rot).mul(&Transform::new([-p0[0], 0.0, -p0[2]], [1.0, 0.0, 0.0, 0.0]));
            let placed: Vec<Vec<Transform>> = poses
                .iter()
                .map(|p| {
                    let mut q = p.clone();
                    q[0] = place.mul(&p[0]);
                    q
                })
                .collect();
            chain.step(&self.be, &eng, &placed, r.transition_early, &cfg)?;
            if let Some(j) = jobs.lock().expect("jobs").map.get_mut(&id) {
                j.step = i + 1;
                j.steps = r.targets.len();
            }
        }
        let label = format!("MotionBricks: {} target{} after #{}", r.targets.len(), if r.targets.len() == 1 { "" } else { "s" }, r.context);
        Ok(Motion { label, kind: "bricks", fps: 30.0, poses: chain.played, contacts: chain.contacts, kimodo: None })
    }
}

fn synthetic_kimodo<B: Backend>(be: &B) -> Result<KimodoModel<B>> {
    let cfg = KimodoConfig::tiny(Skeleton::soma30());
    let st = |n: usize| Stats::new(vec![0.0; n], vec![1.0; n]);
    let rep = motion_kimodo::KimodoMotionRep::new(Skeleton::soma30(), 30.0, st(5), st(4), st(364));
    Ok(KimodoModel::from_parts(be, cfg, rep, &Synthetic::new(1))?)
}

fn synthetic_bricks<B: Backend>(be: &B) -> Result<MotionBricks<B>> {
    let cfg = BricksConfig::tiny_soma();
    let n = cfg.layout.dual();
    let src = Synthetic::new(2);
    Ok(MotionBricks::new(be, cfg, &src, &src, &src, vec![0.0; n], vec![1.0; n])?)
}

fn motion_json(id: u64, rig: &RigSkeleton, m: &Motion) -> Value {
    let pos = world_positions(rig, &m.poses);
    let flat: Vec<f32> = pos.iter().flatten().map(|v| (v * 1e4).round() / 1e4).collect();
    let contacts: Vec<f32> = m.contacts.iter().flatten().map(|v| (v * 100.0).round() / 100.0).collect();
    json!({ "id": id, "label": m.label, "kind": m.kind, "fps": m.fps, "frames": m.poses.len(), "positions": flat, "contacts": contacts })
}

fn job_json(j: &Job) -> Value {
    json!({
        "id": j.id, "kind": j.kind, "label": j.label,
        "state": match j.state { State::Queued => "queued", State::Running => "running", State::Done => "done", State::Failed => "failed" },
        "step": j.step, "steps": j.steps, "segment": j.segment, "segments": j.segments,
        "message": j.message, "seconds": j.seconds,
        "frames": j.motion.as_ref().map(|m| m.poses.len()),
    })
}

fn header(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).expect("ascii header")
}

fn reply(req: tiny_http::Request, code: u16, body: Vec<u8>, ctype: &str, extra: &[tiny_http::Header]) {
    let mut r = tiny_http::Response::from_data(body).with_status_code(code).with_header(header("Content-Type", ctype));
    for h in extra {
        r = r.with_header(h.clone());
    }
    let _ = req.respond(r);
}

fn reply_json(req: tiny_http::Request, code: u16, v: &Value, cors: &[tiny_http::Header]) {
    reply(req, code, v.to_string().into_bytes(), "application/json", cors);
}

pub fn run<B>(be: B, a: ServeArgs) -> Result<()>
where
    B: Backend + Send + 'static,
    B::Buf: Send,
    B::Weight: Send,
{
    let t0 = Instant::now();
    let (kimodo, bricks) = if a.synthetic {
        (Some(synthetic_kimodo(&be)?), Some(synthetic_bricks(&be)?))
    } else {
        let k = a.kimodo_dir.as_ref().map(|d| KimodoModel::load(&be, d).with_context(|| format!("loading {}", d.display()))).transpose()?;
        let b = a.bricks_dir.as_ref().map(|d| MotionBricks::load(&be, d).with_context(|| format!("loading {}", d.display()))).transpose()?;
        (k, b)
    };
    if kimodo.is_none() && bricks.is_none() {
        return Err(anyhow!("nothing to serve: pass --kimodo-dir and/or --bricks-dir, or --synthetic"));
    }
    if let Some(b) = &bricks {
        if b.cfg.rig != motion_bricks::Rig::Soma {
            return Err(anyhow!("the studio drives the SOMA MotionBricks model (soma/), not G1"));
        }
    }
    let emb = a.embeddings.as_ref().map(|p| Embeddings::load(p)).transpose()?;
    let encoder = match (&a.llama_dir, a.synthetic) {
        (Some(d), false) => Some(TextEncoder::load(d, &a.adapters).context("loading the text encoder")?),
        _ => None,
    };
    eprintln!("models ready in {:.1} s", t0.elapsed().as_secs_f32());
    let rig = soma77_capture();
    let s77 = Skeleton::soma77();
    let feet: Vec<usize> = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
        .iter()
        .map(|n| s77.index(n).expect("SOMA-77 foot joint"))
        .collect();
    let parents: Vec<i64> = s77.parents.iter().map(|p| p.map_or(-1, |x| x as i64)).collect();
    let info = json!({
        "engine": "motion-rs",
        "version": env!("CARGO_PKG_VERSION"),
        "backend": be.name(),
        "synthetic": a.synthetic,
        "kimodo": kimodo.is_some(),
        "bricks": bricks.is_some(),
        "text_encoder": encoder.is_some(),
        "gallery": emb.as_ref().map(|e| e.prompts.clone()).unwrap_or_default(),
        "fps": kimodo.as_ref().map_or(30.0, |k| k.fps()),
        "joints": s77.joint_names,
        "parents": parents,
        "feet": feet,
        "max_seconds": 10.0,
    });
    let jobs = Arc::new(Mutex::new(Jobs::default()));
    let (tx, rx) = mpsc::channel::<(u64, Work)>();
    let engine = Engine { be, kimodo, bricks, emb, encoder, synthetic: a.synthetic, rig: rig.clone() };
    {
        let jobs = jobs.clone();
        std::thread::spawn(move || {
            for (id, work) in rx {
                if let Some(j) = jobs.lock().expect("jobs").map.get_mut(&id) {
                    j.state = State::Running;
                }
                let t = Instant::now();
                let res = match &work {
                    Work::Kimodo(r) => engine.run_kimodo(r, &jobs, id),
                    Work::Bricks(r) => engine.run_bricks(r, &jobs, id),
                };
                let mut g = jobs.lock().expect("jobs");
                if let Some(j) = g.map.get_mut(&id) {
                    j.seconds = t.elapsed().as_secs_f32();
                    match res {
                        Ok(m) => {
                            j.label = m.label.clone();
                            j.motion = Some(Arc::new(m));
                            j.state = State::Done;
                        }
                        Err(e) => {
                            j.message = format!("{e:#}");
                            j.state = State::Failed;
                        }
                    }
                }
            }
        });
    }
    let addr = format!("{}:{}", a.host, a.port);
    let server = tiny_http::Server::http(&addr).map_err(|e| anyhow!("binding {addr}: {e}"))?;
    eprintln!("Motion Studio on http://{addr}/");
    for req in server.incoming_requests() {
        let origin = req.headers().iter().find(|h| h.field.equiv("Origin")).map(|h| h.value.as_str().to_string());
        let mut cors = Vec::new();
        if let (Some(allow), Some(o)) = (&a.allow_origin, &origin) {
            if o == allow {
                cors.push(header("Access-Control-Allow-Origin", o));
                cors.push(header("Access-Control-Allow-Headers", "Content-Type"));
                cors.push(header("Access-Control-Allow-Private-Network", "true"));
                cors.push(header("Vary", "Origin"));
            }
        }
        let url = req.url().to_string();
        let path = url.split('?').next().unwrap_or("").to_string();
        let method = req.method().clone();
        match (method, path.as_str()) {
            (tiny_http::Method::Options, _) => reply(req, 204, Vec::new(), "text/plain", &cors),
            (tiny_http::Method::Get, "/") | (tiny_http::Method::Get, "/studio") => {
                reply(req, 200, PAGE.as_bytes().to_vec(), "text/html; charset=utf-8", &cors)
            }
            (tiny_http::Method::Get, "/api/info") => reply_json(req, 200, &info, &cors),
            (tiny_http::Method::Get, "/api/jobs") => {
                let g = jobs.lock().expect("jobs");
                let list: Vec<Value> = g.order.iter().rev().filter_map(|id| g.map.get(id)).map(job_json).collect();
                drop(g);
                reply_json(req, 200, &json!({ "jobs": list }), &cors)
            }
            (tiny_http::Method::Post, p @ ("/api/kimodo" | "/api/bricks")) => {
                let is_json = req.headers().iter().any(|h| h.field.equiv("Content-Type") && h.value.as_str().starts_with("application/json"));
                let cross = origin.as_ref().is_some_and(|o| !o.ends_with(&format!("//{addr}")) && Some(o) != a.allow_origin.as_ref());
                if !is_json || cross {
                    reply_json(req, 403, &json!({"error": "requests must be JSON from this page or --allow-origin"}), &cors);
                    continue;
                }
                let p = p.to_string();
                let mut req = req;
                let mut body = String::new();
                if req.as_reader().read_to_string(&mut body).is_err() {
                    reply_json(req, 400, &json!({"error": "unreadable body"}), &cors);
                    continue;
                }
                let parsed: std::result::Result<(Work, &'static str, String, usize), String> = if p == "/api/kimodo" {
                    serde_json::from_str::<KimodoReq>(&body).map_err(|e| e.to_string()).map(|r| {
                        let label = r.segments.iter().map(|s| s.prompt.clone()).collect::<Vec<_>>().join(" → ");
                        let n = r.segments.len();
                        (Work::Kimodo(r), "kimodo", label, n)
                    })
                } else {
                    serde_json::from_str::<BricksReq>(&body).map_err(|e| e.to_string()).map(|r| (Work::Bricks(r), "bricks", "MotionBricks".to_string(), 1))
                };
                match parsed {
                    Ok((work, kind, label, segs)) => {
                        let id = jobs.lock().expect("jobs").add(kind, label, segs);
                        let _ = tx.send((id, work));
                        reply_json(req, 200, &json!({ "id": id }), &cors)
                    }
                    Err(e) => reply_json(req, 400, &json!({ "error": e }), &cors),
                }
            }
            (tiny_http::Method::Get, p) if p.starts_with("/api/jobs/") => {
                let rest = &p["/api/jobs/".len()..];
                let (id_s, what) = rest.split_once('/').unwrap_or((rest, ""));
                let Ok(id) = id_s.parse::<u64>() else {
                    reply_json(req, 404, &json!({"error": "no such job"}), &cors);
                    continue;
                };
                let g = jobs.lock().expect("jobs");
                let Some(job) = g.map.get(&id) else {
                    drop(g);
                    reply_json(req, 404, &json!({"error": "no such job"}), &cors);
                    continue;
                };
                let status = job_json(job);
                let motion = job.motion.clone();
                drop(g);
                match (what, motion) {
                    ("", _) => reply_json(req, 200, &status, &cors),
                    ("motion", Some(m)) => reply_json(req, 200, &motion_json(id, &rig, &m), &cors),
                    ("motion.bvh", Some(m)) => {
                        let bvh = match &m.kimodo {
                            Some(k) => k.to_bvh(false),
                            None => motion_bricks::io::write_bvh(&rig, &m.poses, m.fps),
                        };
                        let cd = header("Content-Disposition", &format!("attachment; filename=\"motion-{id}.bvh\""));
                        reply(req, 200, bvh.into_bytes(), "text/plain", &[cors.clone(), vec![cd]].concat())
                    }
                    ("motion.npz", Some(m)) if m.kimodo.is_some() => {
                        let tmp = std::env::temp_dir().join(format!("motion-studio-{id}-{}.npz", std::process::id()));
                        let res = m.kimodo.as_ref().expect("checked").write_npz(&tmp).and_then(|_| std::fs::read(&tmp));
                        let _ = std::fs::remove_file(&tmp);
                        match res {
                            Ok(bytes) => {
                                let cd = header("Content-Disposition", &format!("attachment; filename=\"motion-{id}.npz\""));
                                reply(req, 200, bytes, "application/octet-stream", &[cors.clone(), vec![cd]].concat())
                            }
                            Err(e) => reply_json(req, 500, &json!({"error": e.to_string()}), &cors),
                        }
                    }
                    _ => reply_json(req, 404, &json!({"error": "not available for this job"}), &cors),
                }
            }
            _ => reply_json(req, 404, &json!({"error": "not found"}), &cors),
        }
    }
    let _ = skeleton::soma77_bvh_neutral_joints;
    Ok(())
}
