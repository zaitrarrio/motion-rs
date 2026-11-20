//! The smooth root trajectory (`kimodo/motion_rep/smooth_root.py`): a
//! multigrid ADMM smoother that keeps the ground-plane root path within a
//! 6 cm margin of the pelvis while minimising its acceleration. Ported
//! step for step, including the final-level neighbour averaging the NumPy
//! slice arithmetic produces.

/// `TrajectorySmoother(margins, pos_weight, loop=False, admm_iters, alpha_overrelax)`.
struct Smoother {
    n: usize,
    dim: usize,
    margins: Vec<f64>,
    pos_weight: f64,
    iters: usize,
    alpha: f64,
    step: f64,
    /// Cholesky factor (lower, dense) of `pos_weight·I + AᵀA + step·I`.
    chol: Vec<f64>,
}

impl Smoother {
    fn new(margins: &[f64], dim: usize, pos_weight: f64, iters: usize, alpha: f64) -> Self {
        let n = margins.len();
        // A: second differences on rows 1..n-1 (no loop).
        let mut m = vec![0.0f64; n * n];
        for i in 1..n.saturating_sub(1) {
            let row = [(i - 1, -1.0), (i, 2.0), (i + 1, -1.0)];
            for (a, va) in row {
                for (b, vb) in row {
                    m[a * n + b] += va * vb;
                }
            }
        }
        for i in 0..n {
            m[i * n + i] += pos_weight;
        }
        let diag_max = (0..n).map(|i| m[i * n + i].abs()).fold(0.0, f64::max);
        let step = 0.25 * diag_max.sqrt();
        for i in 0..n {
            m[i * n + i] += step;
        }
        let chol = cholesky(&m, n);
        Self { n, dim, margins: margins.to_vec(), pos_weight, iters, alpha, step, chol }
    }

    fn solve(&self, r: &[f64]) -> Vec<f64> {
        chol_solve(&self.chol, self.n, r)
    }

    /// `smooth(targets, x0)`: `x` is `[n, dim]`.
    fn smooth(&self, targets: &[f64], x0: &[f64]) -> Vec<f64> {
        let (n, d) = (self.n, self.dim);
        let mut x = x0.to_vec();
        let mut z = vec![0.0; n * d];
        let mut u = vec![0.0; n * d];
        for _ in 0..self.iters {
            // z update
            for i in 0..n {
                let mut nrm = 0.0;
                for k in 0..d {
                    z[i * d + k] = x[i * d + k] + u[i * d + k] - targets[i * d + k];
                    nrm += z[i * d + k] * z[i * d + k];
                }
                let nrm = nrm.sqrt();
                if nrm > self.margins[i] {
                    let s = self.margins[i] / nrm;
                    for k in 0..d {
                        z[i * d + k] *= s;
                    }
                }
                for k in 0..d {
                    z[i * d + k] += targets[i * d + k];
                }
            }
            // u update
            for i in 0..n * d {
                u[i] += self.alpha * (x[i] - z[i]);
            }
            // x update, one solve per column
            let mut col = vec![0.0; n];
            for k in 0..d {
                for i in 0..n {
                    col[i] = self.pos_weight * targets[i * d + k] + self.step * (z[i * d + k] - u[i * d + k]);
                }
                let sol = self.solve(&col);
                for i in 0..n {
                    x[i * d + k] = sol[i];
                }
            }
        }
        x
    }
}

fn cholesky(m: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = s.max(1e-300).sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    l
}

fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// `smooth_signal(x, margins, pos_weight=0, alpha_overrelax=1.8, admm_iters=500)`,
/// `x` `[T, dim]` row-major.
pub fn smooth_signal(x: &[f64], dim: usize, margins: &[f64]) -> Vec<f64> {
    let n = x.len() / dim;
    assert_eq!(margins.len(), n);
    let (pos_weight, alpha, iters) = (0.0, 1.8, 500);
    // x_smoothed[:] = mean over time
    let mut xs = vec![0.0; n * dim];
    for k in 0..dim {
        let mean = (0..n).map(|i| x[i * dim + k]).sum::<f64>() / n as f64;
        for i in 0..n {
            xs[i * dim + k] = mean;
        }
    }
    let levels = ((n as f64).log2().floor() as i64 - 4).max(1) as u32;
    let mut stepsize = 1usize << levels;
    loop {
        let idx: Vec<usize> = (0..n).step_by(stepsize).collect();
        let num_steps = idx.len();
        let sm = Smoother::new(
            &idx.iter().map(|&i| margins[i]).collect::<Vec<_>>(),
            dim,
            pos_weight,
            iters,
            alpha,
        );
        let targets: Vec<f64> = idx.iter().flat_map(|&i| x[i * dim..(i + 1) * dim].iter().copied()).collect();
        let x0: Vec<f64> = idx.iter().flat_map(|&i| xs[i * dim..(i + 1) * dim].iter().copied()).collect();
        let res = sm.smooth(&targets, &x0);
        for (c, &i) in idx.iter().enumerate() {
            xs[i * dim..(i + 1) * dim].copy_from_slice(&res[c * dim..(c + 1) * dim]);
        }
        // interpolate to the next level: x_smoothed[next::step]
        let next = stepsize / 2;
        let inter: Vec<usize> = if stepsize == 1 {
            // Python: x_smoothed[0::1] (step 0 is the whole array)
            (0..n).collect()
        } else {
            (next..n).step_by(stepsize).collect()
        };
        let mut num_inter = inter.len();
        let coarse: Vec<usize> = idx.clone();
        if num_inter == num_steps {
            // linearly extrapolate the last value
            let a = coarse[num_steps - 1];
            let b = coarse[num_steps - 2];
            let last = inter[num_inter - 1];
            for k in 0..dim {
                let va = xs[a * dim + k];
                let vb = xs[b * dim + k];
                xs[last * dim + k] = va + (va - vb) / 2.0;
            }
            num_inter -= 1;
        }
        // RHS is evaluated before assignment (NumPy): snapshot the coarse values.
        let snap: Vec<f64> = coarse.iter().flat_map(|&i| xs[i * dim..(i + 1) * dim].iter().copied()).collect();
        for c in 0..num_inter {
            let t = inter[c];
            for k in 0..dim {
                xs[t * dim + k] = (snap[c * dim + k] + snap[(c + 1) * dim + k]) / 2.0;
            }
        }
        if stepsize == 1 {
            break;
        }
        stepsize /= 2;
    }
    xs
}

/// `get_smooth_root_pos(hip_translations)`: `[T, 3]` → `[T, 3]` with the
/// ground-plane path smoothed (margin 0.06 m) and the height kept.
pub fn get_smooth_root_pos(root: &[f32]) -> Vec<f32> {
    let n = root.len() / 3;
    if n < 3 {
        return root.to_vec();
    }
    let xz: Vec<f64> = (0..n).flat_map(|t| [root[t * 3] as f64, root[t * 3 + 2] as f64]).collect();
    let margins = vec![0.06; n];
    let s = smooth_signal(&xz, 2, &margins);
    let mut out = root.to_vec();
    for t in 0..n {
        out[t * 3] = s[t * 2] as f32;
        out[t * 3 + 2] = s[t * 2 + 1] as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_stays_within_margin() {
        let root: Vec<f32> = (0..40).flat_map(|t| [t as f32 * 0.02, 0.9, 0.0]).collect();
        let s = get_smooth_root_pos(&root);
        for t in 0..40 {
            assert!((s[t * 3] - root[t * 3]).abs() < 0.07, "frame {t}");
            assert_eq!(s[t * 3 + 1], 0.9);
        }
    }
}
