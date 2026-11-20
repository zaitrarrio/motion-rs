//! Every op and the tiny Kimodo run on the WebGPU backend against the CPU
//! reference. Skips without an adapter, except under `CI=1`, where a missing
//! adapter fails (CI installs lavapipe and sets `MOTION_ALLOW_SOFTWARE=1`).

use motion_wgpu::{Wgpu, WgpuOptions};

#[test]
fn wgpu_matches_cpu() {
    let be = match Wgpu::with_options(WgpuOptions::from_env()) {
        Ok(be) => be,
        Err(e) => {
            if std::env::var("CI").is_ok_and(|v| v == "1" || v == "true") {
                panic!("no usable WebGPU adapter in CI: {e}");
            }
            eprintln!("skipped: {e}");
            return;
        }
    };
    eprintln!("adapter: {}", motion_core::backend::Backend::name(&be));
    let results = motion_check::run_all(&be);
    let (text, ok) = motion_check::report(&results);
    eprintln!("{text}");
    assert!(ok, "WebGPU backend differs from the CPU reference:\n{text}");
}
