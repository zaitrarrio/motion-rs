//! Checkpoint access: safetensors (mmap'd; F32 / F16 / BF16 / F64; any number
//! of files, later files shadow earlier ones) or deterministic synthetic
//! weights of the same names and shapes (tests and benchmarks without a
//! download). Every model asks for every tensor by its exact PyTorch
//! state-dict key and shape; a missing key or a shape mismatch is an error,
//! never a default.

use crate::rng::Rng;
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub trait WeightSource {
    /// The tensor as f32, checked against `shape` (size-1 dims are ignored on
    /// both sides, so a `[C, 1]` may be asked for as `[C]`).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>>;
    fn contains(&self, name: &str) -> bool;
    /// The stored shape of a tensor.
    fn shape(&self, name: &str) -> Result<Vec<usize>>;
    /// Every key, for manifests and diagnostics.
    fn keys(&self) -> Vec<String>;
}

fn squeeze(s: &[usize]) -> Vec<usize> {
    s.iter().copied().filter(|&d| d != 1).collect()
}

fn check_shape(name: &str, got: &[usize], want: &[usize]) -> Result<()> {
    if squeeze(got) != squeeze(want) {
        return Err(Error::Weights(format!("`{name}`: stored shape {got:?}, model expects {want:?}")));
    }
    Ok(())
}

/// One or more `.safetensors` files (or directories of them).
pub struct SafetensorsDir {
    maps: Vec<memmap2::Mmap>,
    files: Vec<PathBuf>,
    index: HashMap<String, usize>,
}

impl SafetensorsDir {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_many(&[path.to_path_buf()])
    }

    pub fn open_many(paths: &[PathBuf]) -> Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        for path in paths {
            if path.is_dir() {
                let mut v: Vec<PathBuf> = std::fs::read_dir(path)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                    .collect();
                v.sort();
                files.extend(v);
            } else {
                files.push(path.clone());
            }
        }
        if files.is_empty() {
            return Err(Error::Weights(format!("no .safetensors under {paths:?}")));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)
                .map_err(|e| Error::Weights(format!("{}: {e}", f.display())))?;
            // Read-only mapping of a file we do not modify.
            #[allow(unsafe_code)]
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| Error::Weights(format!("{}: {e}", f.display())))?;
            let st = safetensors::SafeTensors::deserialize(&map)
                .map_err(|e| Error::Weights(format!("{}: {e}", f.display())))?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
            maps.push(map);
        }
        Ok(Self { maps, files, index })
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// Rows `rows` of a 2-D tensor as f32, without converting the rest (a
    /// 128k-row embedding table read for a dozen tokens).
    pub fn get_rows(&self, name: &str, rows: &[usize]) -> Result<Vec<f32>> {
        let (dtype, shape, data) = self.view(name)?;
        if shape.len() != 2 {
            return Err(Error::Weights(format!("`{name}`: get_rows needs a 2-D tensor, got {shape:?}")));
        }
        let width = shape[1];
        let esize = data.len() / (shape[0] * width);
        let mut out = Vec::with_capacity(rows.len() * width);
        for &r in rows {
            if r >= shape[0] {
                return Err(Error::Weights(format!("`{name}`: row {r} out of {}", shape[0])));
            }
            out.extend(bytes_to_f32(dtype, &data[r * width * esize..(r + 1) * width * esize])?);
        }
        Ok(out)
    }

    fn view(&self, name: &str) -> Result<(safetensors::Dtype, Vec<usize>, &[u8])> {
        let &i = self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let st = safetensors::SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor(name).map_err(|e| Error::Weights(e.to_string()))?;
        // The view borrows `st`; copy the shape and keep the byte range by offset into the map.
        let shape = t.shape().to_vec();
        let dtype = t.dtype();
        let data = t.data();
        let base = self.maps[i].as_ptr() as usize;
        let start = data.as_ptr() as usize - base;
        Ok((dtype, shape, &self.maps[i][start..start + data.len()]))
    }
}

pub fn bytes_to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    Ok(match dtype {
        D::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        D::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        D::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        D::I32 => data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
        D::BOOL | D::U8 => data.iter().map(|&b| b as f32).collect(),
        other => return Err(Error::Weights(format!("unsupported dtype {other:?}"))),
    })
}

impl WeightSource for SafetensorsDir {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (dtype, got, data) = self.view(name)?;
        check_shape(name, &got, shape)?;
        bytes_to_f32(dtype, data)
    }
    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.view(name)?.1)
    }
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.index.keys().cloned().collect();
        k.sort();
        k
    }
}

/// Deterministic random weights for any key: `N(0, scale)` where `scale` is
/// `1/√fan_in` for matrices (`cols` = last dim product) and 0.02 otherwise;
/// norm weights (`*norm*.weight`, `*.ln*.weight`) are `1 + N(0, 0.02)`.
/// The seed is derived from the key, so a tensor is the same whatever the
/// order of requests.
pub struct Synthetic {
    pub seed: u64,
}

impl Synthetic {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
    fn rng_for(&self, name: &str) -> Rng {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ self.seed;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        Rng::new(h)
    }
}

impl WeightSource for Synthetic {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let n: usize = shape.iter().product();
        let mut rng = self.rng_for(name);
        let is_norm = (name.contains("norm") || name.contains(".ln")) && name.ends_with("weight");
        let fan_in: usize = if shape.len() >= 2 { shape[1..].iter().product() } else { 0 };
        let scale = if is_norm {
            0.02
        } else if fan_in > 0 {
            1.0 / (fan_in as f32).sqrt()
        } else {
            0.02
        };
        let base = if is_norm { 1.0 } else { 0.0 };
        Ok((0..n).map(|_| base + scale * rng.normal()).collect())
    }
    fn contains(&self, _name: &str) -> bool {
        true
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Err(Error::Weights(format!("`{name}`: synthetic weights have no stored shape")))
    }
    fn keys(&self) -> Vec<String> {
        Vec::new()
    }
}

/// A source that strips a key prefix before asking the inner one
/// (`denoiser.backbone.` on Kimodo checkpoints, `mmm_net.` / `backbone_net.` /
/// `pose_net.` on MotionBricks ones).
pub struct Prefixed<'a> {
    pub inner: &'a dyn WeightSource,
    pub prefix: String,
}

impl WeightSource for Prefixed<'_> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.inner.get(&format!("{}{name}", self.prefix), shape)
    }
    fn contains(&self, name: &str) -> bool {
        self.inner.contains(&format!("{}{name}", self.prefix))
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.inner.shape(&format!("{}{name}", self.prefix))
    }
    fn keys(&self) -> Vec<String> {
        self.inner
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_prefix(&self.prefix).map(str::to_string))
            .collect()
    }
}

/// An in-memory source (tests, fixtures).
#[derive(Default)]
pub struct MemorySource {
    pub tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl MemorySource {
    pub fn insert(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}: shape/data mismatch");
        self.tensors.insert(name.to_string(), (shape.to_vec(), data));
    }
}

impl WeightSource for MemorySource {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (got, data) =
            self.tensors.get(name).ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        check_shape(name, got, shape)?;
        Ok(data.clone())
    }
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.tensors
            .get(name)
            .map(|(s, _)| s.clone())
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))
    }
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.tensors.keys().cloned().collect();
        k.sort();
        k
    }
}

/// Write f32 tensors as one safetensors file (fixtures, caches).
pub fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<()> {
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(n, s, d)| {
            let bytes: &[u8] = bytemuck_f32(d);
            (
                n.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), bytes)
                    .expect("valid tensor view"),
            )
        })
        .collect();
    let bytes = safetensors::serialize(views, None).map_err(|e| Error::Weights(e.to_string()))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn bytemuck_f32(d: &[f32]) -> &[u8] {
    // f32 has no padding and any bit pattern is valid u8; the slice stays aligned.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4)
    }
}
