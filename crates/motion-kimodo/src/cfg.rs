//! `kimodo/model/cfg.py`: classifier-free guidance by batching the
//! conditional and unconditional passes.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CfgType {
    /// One pass.
    NoCfg,
    /// `uncond + w·(cond − uncond)` with text and constraints together.
    Regular(f32),
    /// `uncond + w_text·(text − uncond) + w_constraint·(constraint − uncond)`.
    Separated(f32, f32),
}

impl CfgType {
    /// Passes per sample.
    pub fn copies(&self) -> usize {
        match self {
            CfgType::NoCfg => 1,
            CfgType::Regular(_) => 2,
            CfgType::Separated(..) => 3,
        }
    }

    /// `(text on, constraints on)` of each copy, in the order cfg.py concatenates them.
    pub fn conditions(&self) -> &'static [(bool, bool)] {
        match self {
            CfgType::NoCfg => &[(true, true)],
            CfgType::Regular(_) => &[(true, true), (false, false)],
            CfgType::Separated(..) => &[(true, false), (false, true), (false, false)],
        }
    }

    /// Combine the per-copy outputs (each `n` values).
    pub fn combine(&self, outs: &[Vec<f32>]) -> Vec<f32> {
        match *self {
            CfgType::NoCfg => outs[0].clone(),
            CfgType::Regular(w) => outs[1].iter().zip(&outs[0]).map(|(u, c)| u + w * (c - u)).collect(),
            CfgType::Separated(wt, wc) => outs[2]
                .iter()
                .zip(outs[0].iter().zip(&outs[1]))
                .map(|(u, (t, c))| u + wt * (t - u) + wc * (c - u))
                .collect(),
        }
    }

    pub fn parse(kind: &str, weights: &[f32]) -> Result<Self, String> {
        match kind {
            "nocfg" => Ok(CfgType::NoCfg),
            "regular" => match weights {
                [w] => Ok(CfgType::Regular(*w)),
                [] => Ok(CfgType::Regular(2.0)),
                _ => Err("regular CFG takes one weight".into()),
            },
            "separated" => match weights {
                [t, c] => Ok(CfgType::Separated(*t, *c)),
                [] => Ok(CfgType::Separated(2.0, 2.0)),
                _ => Err("separated CFG takes two weights (text, constraint)".into()),
            },
            other => Err(format!("unknown CFG type `{other}` (nocfg, regular, separated)")),
        }
    }
}
