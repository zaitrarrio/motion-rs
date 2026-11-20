//! `kimodo/model/diffusion.py`: the cosine schedule over `num_base_steps`
//! (1000), the respacing to `n` denoising steps, and the deterministic DDIM
//! update (η = 0). Arithmetic is f32 as in PyTorch (the betas come from
//! Python floats, then every table is a float32 tensor).

/// `get_beta_schedule(num_diffusion_timesteps, max_beta=0.999)`.
pub fn cosine_betas(n: usize) -> Vec<f32> {
    let ab = |t: f64| ((t + 0.008) / 1.008 * std::f64::consts::PI / 2.0).cos().powi(2);
    (0..n)
        .map(|i| {
            let (t1, t2) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
            (1.0 - ab(t2) / ab(t1)).min(0.999) as f32
        })
        .collect()
}

fn cumprod(v: &[f32]) -> Vec<f32> {
    let mut acc = 1f32;
    v.iter()
        .map(|&x| {
            acc *= x;
            acc
        })
        .collect()
}

/// The respaced tables for `n` steps (`space_timesteps` + `calc_diffusion_vars`).
#[derive(Debug, Clone)]
pub struct Schedule {
    /// `use_timesteps` (length `num_base_steps`); the sampler reads the first `n`.
    pub timesteps: Vec<usize>,
    pub alphas_cumprod: Vec<f32>,
    pub alphas_cumprod_prev: Vec<f32>,
    pub sqrt_recip_alphas_cumprod: Vec<f32>,
    pub sqrt_recipm1_alphas_cumprod: Vec<f32>,
    pub steps: usize,
}

impl Schedule {
    pub fn new(num_base_steps: usize, steps: usize) -> Self {
        let base = cosine_betas(num_base_steps);
        let acp_base = cumprod(&base.iter().map(|b| 1.0 - b).collect::<Vec<_>>());
        // space_timesteps: round(arange(N)·(N−1)/max(1, n−1)), clamped to N−1.
        let frac = (num_base_steps as f32 - 1.0) / (steps.max(2) - 1) as f32;
        let frac = if steps <= 1 { num_base_steps as f32 - 1.0 } else { frac };
        let timesteps: Vec<usize> = (0..num_base_steps)
            .map(|i| ((i as f32 * frac).round_ties_even() as usize).min(num_base_steps - 1))
            .collect();
        let acp: Vec<f32> = timesteps.iter().map(|&t| acp_base[t]).collect();
        let mut last = vec![1f32];
        last.extend_from_slice(&acp[..acp.len() - 1]);
        let alphas: Vec<f32> = acp.iter().zip(&last).map(|(a, l)| 1.0 - (1.0 - a / l)).collect();
        let alphas_cumprod: Vec<f32> = cumprod(&alphas).into_iter().map(|v| v.max(1e-9)).collect();
        let mut prev = vec![1f32];
        prev.extend_from_slice(&alphas_cumprod[..alphas_cumprod.len() - 1]);
        let sqrt_recip = alphas_cumprod.iter().map(|a| 1.0 / a.sqrt()).collect();
        let sqrt_recipm1 = alphas_cumprod.iter().map(|a| 1.0 / (a / (1.0 - a)).sqrt()).collect();
        Self {
            timesteps,
            alphas_cumprod,
            alphas_cumprod_prev: prev,
            sqrt_recip_alphas_cumprod: sqrt_recip,
            sqrt_recipm1_alphas_cumprod: sqrt_recipm1,
            steps,
        }
    }

    /// `DDIMSampler.__call__` at step `t` (η = 0): `x_{t−1}` from `x_t` and the clean prediction.
    pub fn ddim(&self, t: usize, x_t: &[f32], x0: &[f32]) -> Vec<f32> {
        let (sr, srm1) = (self.sqrt_recip_alphas_cumprod[t], self.sqrt_recipm1_alphas_cumprod[t]);
        let prev = self.alphas_cumprod_prev[t];
        let (sp, sq) = (prev.sqrt(), (1.0 - prev).sqrt());
        x_t.iter()
            .zip(x0)
            .map(|(&x, &p)| {
                let eps = (sr * x - p) / srm1;
                p * sp + sq * eps
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respacing_matches_upstream_formula() {
        let s = Schedule::new(1000, 100);
        assert_eq!(s.timesteps[0], 0);
        assert_eq!(s.timesteps[99], 999);
        assert_eq!(s.timesteps[1], 10);
        assert_eq!(*s.timesteps.last().unwrap(), 999);
        assert!(s.alphas_cumprod[0] > 0.99);
        assert!(s.alphas_cumprod[99] < 1e-3);
    }
}
