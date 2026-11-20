//! `motion`: the command-line front end of motion-rs.

mod download;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "motion", version, about = "Kimodo and MotionBricks motion models in Rust")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Wgpu,
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
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
            let out = out.unwrap_or_else(|| {
                PathBuf::from("weights").join(repo.rsplit('/').next().unwrap_or(&repo))
            });
            eprintln!("{repo} → {}", out.display());
            let files = download::snapshot(&repo, &out, &include)?;
            eprintln!("{} files", files.len());
            Ok(())
        }
    }
}
