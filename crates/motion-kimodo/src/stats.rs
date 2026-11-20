//! Feature normalisation statistics (`kimodo/motion_rep/stats.py`):
//! `normalize(x) = (x − mean) / sqrt(std² + eps)`, `eps = 1e-5`.

use motion_core::npy::Npy;
use motion_core::{Error, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Stats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// `sqrt(std² + eps)`.
    pub std_eps: Vec<f32>,
}

impl Stats {
    pub const EPS: f32 = 1e-5;

    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(mean.len(), std.len());
        let std_eps = std.iter().map(|s| (s * s + Self::EPS).sqrt()).collect();
        Self { mean, std, std_eps }
    }

    /// `<dir>/mean.npy` and `<dir>/std.npy`.
    pub fn load(dir: &Path) -> Result<Self> {
        let mean = Npy::read(&dir.join("mean.npy"))?;
        let std = Npy::read(&dir.join("std.npy"))?;
        if mean.data.len() != std.data.len() {
            return Err(Error::Weights(format!("{}: mean and std differ in length", dir.display())));
        }
        Ok(Self::new(mean.to_f32(), std.to_f32()))
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    pub fn concat(parts: &[&Stats]) -> Self {
        let mean = parts.iter().flat_map(|p| p.mean.iter().copied()).collect();
        let std = parts.iter().flat_map(|p| p.std.iter().copied()).collect();
        Self::new(mean, std)
    }

    /// Rows of width `dim()`, in place.
    pub fn normalize(&self, x: &mut [f32]) {
        let d = self.dim();
        for row in x.chunks_exact_mut(d) {
            for i in 0..d {
                row[i] = (row[i] - self.mean[i]) / self.std_eps[i];
            }
        }
    }

    pub fn unnormalize(&self, x: &mut [f32]) {
        let d = self.dim();
        for row in x.chunks_exact_mut(d) {
            for i in 0..d {
                row[i] = row[i] * self.std_eps[i] + self.mean[i];
            }
        }
    }

    pub fn normalized(&self, x: &[f32]) -> Vec<f32> {
        let mut v = x.to_vec();
        self.normalize(&mut v);
        v
    }

    pub fn unnormalized(&self, x: &[f32]) -> Vec<f32> {
        let mut v = x.to_vec();
        self.unnormalize(&mut v);
        v
    }
}
