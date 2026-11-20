//! NumPy `.npy` (v1 / v2 / v3) reader for little-endian f32 / f64 / int /
//! bool arrays; the checkpoints ship their normalisation statistics and
//! skeleton tables this way.

use crate::{Error, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Npy {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Weights(format!("{}: {e}", path.display())))?;
        Self::parse(&bytes).map_err(|e| Error::Weights(format!("{}: {e}", path.display())))
    }

    pub fn parse(bytes: &[u8]) -> std::result::Result<Self, String> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err("not an .npy file".into());
        }
        let major = bytes[6];
        let (hlen, off) = if major == 1 {
            (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        };
        let header = std::str::from_utf8(&bytes[off..off + hlen]).map_err(|e| e.to_string())?;
        let descr = field(header, "descr").ok_or("no descr")?;
        let fortran = field(header, "fortran_order").ok_or("no fortran_order")?;
        if fortran.trim() != "False" {
            return Err("fortran_order arrays are not supported".into());
        }
        let shape_s = field(header, "shape").ok_or("no shape")?;
        let shape: Vec<usize> = shape_s
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
            .collect::<std::result::Result<_, _>>()?;
        let n: usize = shape.iter().product();
        let body = &bytes[off + hlen..];
        let descr = descr.trim().trim_matches('\'').trim_matches('"');
        let data = match descr {
            "<f4" | "=f4" | "f4" => read::<4>(body, n, |b| f32::from_le_bytes(b) as f64)?,
            "<f8" | "=f8" | "f8" => read::<8>(body, n, f64::from_le_bytes)?,
            "<i4" | "=i4" | "i4" => read::<4>(body, n, |b| i32::from_le_bytes(b) as f64)?,
            "<i8" | "=i8" | "i8" => read::<8>(body, n, |b| i64::from_le_bytes(b) as f64)?,
            "<i2" | "=i2" | "i2" => read::<2>(body, n, |b| i16::from_le_bytes(b) as f64)?,
            "|u1" | "u1" => read::<1>(body, n, |b| b[0] as f64)?,
            "|i1" | "i1" => read::<1>(body, n, |b| b[0] as i8 as f64)?,
            "|b1" | "b1" => read::<1>(body, n, |b| (b[0] != 0) as u8 as f64)?,
            "<f2" | "=f2" | "f2" => {
                read::<2>(body, n, |b| half::f16::from_le_bytes(b).to_f64())?
            }
            other => return Err(format!("unsupported dtype {other}")),
        };
        Ok(Self { shape, data })
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32).collect()
    }
}

fn read<const W: usize>(
    body: &[u8],
    n: usize,
    f: impl Fn([u8; W]) -> f64,
) -> std::result::Result<Vec<f64>, String> {
    if body.len() < n * W {
        return Err(format!("truncated: {} bytes for {n} elements of {W}", body.len()));
    }
    Ok(body[..n * W]
        .chunks_exact(W)
        .map(|c| {
            let mut a = [0u8; W];
            a.copy_from_slice(c);
            f(a)
        })
        .collect())
}

/// The value of `'key': value` in the header dict (up to the next top-level `,`).
fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("'{key}':");
    let start = header.find(&k)? + k.len();
    let rest = &header[start..];
    let mut depth = 0i32;
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' | '}' if depth <= 0 => return Some(&rest[..i]),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v1_header() {
        let mut v = b"\x93NUMPY\x01\x00".to_vec();
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }";
        let pad = 64 - (10 + header.len() + 1) % 64;
        let mut h = header.to_string();
        h.push_str(&" ".repeat(pad));
        h.push('\n');
        v.extend_from_slice(&(h.len() as u16).to_le_bytes());
        v.extend_from_slice(h.as_bytes());
        for i in 0..6 {
            v.extend_from_slice(&(i as f64).to_le_bytes());
        }
        let a = Npy::parse(&v).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data[5], 5.0);
    }
}
