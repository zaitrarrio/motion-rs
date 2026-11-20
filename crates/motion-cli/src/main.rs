//! `motion`: the command-line front end of motion-rs.

mod backend;
mod bricks_cmd;
mod convert;
mod download;
mod encode_cmd;
mod kimodo_cmd;
mod serve;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "motion", version, about = "Kimodo and MotionBricks motion models in Rust")]
struct Cli {
    #[arg(long, value_enum, default_value_t = backend::BackendKind::Cpu, global = true)]
    backend: backend::BackendKind,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download a model tree from Hugging Face (or use `hf download`).
    Download {
        /// A known name (`motion download --list`) or any `owner/repo`.
        name: Option<String>,
        /// Destination directory (default `weights/<repo tail>`).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only files whose path starts with one of these prefixes (e.g. `soma/`).
        #[arg(long)]
        include: Vec<String>,
        /// List the known model names.
        #[arg(long)]
        list: bool,
    },
    /// Kimodo: text prompts and kinematic constraints to motion (NPZ, BVH).
    Generate(kimodo_cmd::GenerateArgs),
    /// MotionBricks: transitions between context and target poses from clips.
    Inbetween(bricks_cmd::InbetweenArgs),
    /// Precompute LLM2Vec text embeddings for Kimodo prompts.
    Encode(encode_cmd::EncodeArgs),
    /// The Motion Studio: a local web page for both models.
    Serve(serve::ServeArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let be = motion_cpu::Cpu;
    match cli.cmd {
        Cmd::Download { name, out, include, list } => {
            if list || name.is_none() {
                for (k, repo, what) in download::KNOWN {
                    println!("{k:22} {repo:58} {what}");
                }
                return Ok(());
            }
            let name = name.expect("checked");
            let repo = download::resolve_repo(&name);
            let out = out.unwrap_or_else(|| PathBuf::from("weights").join(repo.rsplit('/').next().unwrap_or(&repo)));
            eprintln!("{repo} → {}", out.display());
            let files = download::snapshot(&repo, &out, &include)?;
            eprintln!("{} files", files.len());
            Ok(())
        }
        Cmd::Generate(a) => kimodo_cmd::run(&be, &a),
        Cmd::Inbetween(a) => bricks_cmd::run(&be, &a),
        Cmd::Encode(a) => encode_cmd::run(&be, &a),
        Cmd::Serve(a) => serve::run(be, a),
    }
}
