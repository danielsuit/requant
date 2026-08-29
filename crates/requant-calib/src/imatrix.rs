//! llama.cpp importance-matrix (imatrix) file loader.
//!
//! The imatrix is the per-input-channel mean of squared activations — equivalently the
//! diagonal of the GPTQ Hessian `H = 2XXᵀ` up to a global scale (§2.4 of the design). The
//! k-quant scale search minimizes `Σ gᵢ·(xᵢ − d·qᵢ)²`; because a global positive scale on every
//! `gᵢ` leaves the argmin `d` unchanged, we can feed the *raw* stored values directly without
//! normalizing by `ncall` and the scale search picks the identical grid.
//!
//! Modern llama.cpp (`common/imatrix.cpp` `save_imatrix`) writes the imatrix as a **GGUF
//! container**: `general.type = "imatrix"`, with one f32 tensor per weight named
//! `<weight>.in_sum2` (the accumulated Σ x², length = #input-channels) and a scalar f32
//! `<weight>.counts` (the accumulation call count). We parse that with the requant-io GGUF
//! reader and expose the `.in_sum2` values keyed by the bare weight name.
//!
//! For older builds that wrote a flat record stream, we keep a legacy fallback:
//!   ```text
//!   int32  n_tensors
//!   repeat n_tensors:
//!     int32  name_len
//!     char   name[name_len]        // utf-8, no terminator
//!     int32  ncall                  // number of accumulation calls
//!     int32  nval                   // == number of input channels
//!     f32    values[nval]           // accumulated Σ x² over the calibration set
//!   ```

use anyhow::{bail, Context, Result};
use requant_io::GgufReader;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// One tensor's importance vector plus its accumulation metadata.
#[derive(Debug, Clone)]
pub struct ImatrixEntry {
    /// Per-input-channel importance (raw stored values; see module docs on scale invariance).
    pub values: Vec<f32>,
    /// Number of accumulation calls recorded for this tensor.
    pub ncall: u32,
}

/// A parsed imatrix: tensor name -> entry.
#[derive(Debug, Clone, Default)]
pub struct Imatrix {
    pub entries: HashMap<String, ImatrixEntry>,
}

impl Imatrix {
    /// Per-input-channel importance for a tensor, if present.
    pub fn get(&self, tensor_name: &str) -> Option<&[f32]> {
        self.entries.get(tensor_name).map(|e| e.values.as_slice())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Load an imatrix from a llama.cpp imatrix file (GGUF container, or the legacy flat stream).
pub fn load_imatrix<P: AsRef<Path>>(path: P) -> Result<Imatrix> {
    let p = path.as_ref();
    let mut f = File::open(p).with_context(|| format!("opening imatrix {}", p.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .with_context(|| format!("reading imatrix {}", p.display()))?;

    if buf.starts_with(b"GGUF") {
        return load_imatrix_gguf(p);
    }
    load_imatrix_flat(&buf)
}

/// Modern format: the imatrix is a GGUF with `<weight>.in_sum2` (f32, Σ x² per input channel)
/// and `<weight>.counts` (f32 scalar) tensors.
fn load_imatrix_gguf(p: &Path) -> Result<Imatrix> {
    let r = GgufReader::open(p)
        .with_context(|| format!("opening imatrix GGUF {}", p.display()))?;
    let mut entries = HashMap::new();
    for (idx, t) in r.tensors.iter().enumerate() {
        let Some(weight) = t.name.strip_suffix(".in_sum2") else {
            continue;
        };
        let values = r
            .tensor_to_f32(idx)
            .with_context(|| format!("imatrix: decoding `{}`", t.name))?;
        // ncall lives in the matching `<weight>.counts` scalar (stored as f32).
        let ncall = r
            .find_tensor(&format!("{weight}.counts"))
            .and_then(|c| r.tensor_to_f32(c).ok())
            .and_then(|v| v.first().copied())
            .map(|v| v as u32)
            .unwrap_or(0);
        entries.insert(weight.to_string(), ImatrixEntry { values, ncall });
    }
    if entries.is_empty() {
        bail!("imatrix GGUF has no `.in_sum2` tensors (not an imatrix file?)");
    }
    Ok(Imatrix { entries })
}

/// Legacy flat format: `int32 n_tensors` then a record stream.
fn load_imatrix_flat(buf: &[u8]) -> Result<Imatrix> {
    let mut cur = 0usize;
    let n_tensors = read_i32(buf, &mut cur)? as usize;
    if n_tensors > 1_000_000 {
        bail!("imatrix n_tensors={n_tensors} implausibly large (not an imatrix file?)");
    }

    let mut entries = HashMap::with_capacity(n_tensors);
    for i in 0..n_tensors {
        let name_len = read_i32(buf, &mut cur)? as usize;
        if name_len > 4096 {
            bail!("imatrix tensor {i}: name_len={name_len} implausible");
        }
        let name = std::str::from_utf8(read_bytes(buf, &mut cur, name_len)?)
            .with_context(|| format!("imatrix tensor {i}: non-utf8 name"))?
            .to_string();
        let ncall = read_i32(buf, &mut cur)? as u32;
        let nval = read_i32(buf, &mut cur)? as usize;
        // Sanity bound against a corrupt/truncated file before allocating.
        if nval > 1 << 28 {
            bail!("imatrix tensor `{name}`: nval={nval} implausible");
        }
        let nbytes = nval * 4;
        if cur + nbytes > buf.len() {
            bail!("imatrix tensor `{name}`: truncated (need {nbytes} bytes, have {})", buf.len() - cur);
        }
        let values = (0..nval)
            .map(|_| {
                let v = f32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap());
                cur += 4;
                v
            })
            .collect();
        entries.insert(name, ImatrixEntry { values, ncall });
    }

    Ok(Imatrix { entries })
}

fn read_bytes<'a>(buf: &'a [u8], cur: &mut usize, n: usize) -> Result<&'a [u8]> {
    if *cur + n > buf.len() {
        bail!("imatrix read past EOF at {cur}+{n}");
    }
    let s = &buf[*cur..*cur + n];
    *cur += n;
    Ok(s)
}
fn read_i32(buf: &[u8], cur: &mut usize) -> Result<i32> {
    Ok(i32::from_le_bytes(read_bytes(buf, cur, 4)?.try_into().unwrap()))
}

/// Summarize non-finite entries (NaN/Inf) — a common artifact of a bad calibration run. Returns
/// the count of bad values across all tensors. Useful for a CLI sanity check.
pub fn count_nonfinite(im: &Imatrix) -> usize {
    im.entries
        .values()
        .flat_map(|e| e.values.iter())
        .filter(|v| !v.is_finite() || **v <= 0.0)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_imatrix(entries: &[(&str, u32, &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(entries.len() as i32).to_le_bytes());
        for (name, ncall, vals) in entries {
            buf.extend_from_slice(&(name.len() as i32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&ncall.to_le_bytes());
            buf.extend_from_slice(&(vals.len() as i32).to_le_bytes());
            for v in *vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn round_trips_a_small_imatrix() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5];
        let bytes = write_imatrix(&[("blk.0.ffn_up.weight", 128, &a), ("blk.1.attn_q.weight", 64, &b)]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.imatrix");
        std::fs::write(&path, &bytes).unwrap();

        let im = load_imatrix(&path).unwrap();
        assert_eq!(im.len(), 2);
        assert_eq!(im.get("blk.0.ffn_up.weight").unwrap(), &a[..]);
        assert_eq!(im.get("blk.1.attn_q.weight").unwrap(), &b[..]);
        assert_eq!(im.entries.get("blk.0.ffn_up.weight").unwrap().ncall, 128);
        assert!(im.get("nope").is_none());
    }

    #[test]
    fn rejects_truncated_file() {
        let a = vec![1.0f32, 2.0, 3.0];
        let mut bytes = write_imatrix(&[("t", 1, &a)]);
        // chop off the last 4 bytes (one f32) -> truncated values
        bytes.truncate(bytes.len() - 4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc.imatrix");
        std::fs::write(&path, &bytes).unwrap();
        let res = load_imatrix(&path);
        assert!(res.is_err(), "truncated imatrix must error");
    }

    /// Build a minimal GGUF imatrix (the format modern llama.cpp `llama-imatrix` writes):
    /// `general.type = "imatrix"` plus `<weight>.in_sum2` (f32, Σ x² per input channel) and
    /// `<weight>.counts` (f32 scalar) tensors.
    fn write_imatrix_gguf(entries: &[(&str, f32, &[f32])]) -> Vec<u8> {
        use requant_io::{GgufValue, GgufWriter, TensorSpec};
        let mut w = GgufWriter::new();
        w.add_kv("general.type", GgufValue::String("imatrix".to_string()));
        for (name, ncall, vals) in entries {
            let mut data = Vec::with_capacity(vals.len() * 4);
            for v in *vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            w.add_tensor(TensorSpec {
                name: format!("{name}.in_sum2"),
                dims: vec![vals.len() as u64],
                ggml_type: 0, // F32
                data,
            });
            w.add_tensor(TensorSpec {
                name: format!("{name}.counts"),
                dims: vec![1],
                ggml_type: 0, // F32
                data: ncall.to_le_bytes().to_vec(),
            });
        }
        w.to_bytes()
    }

    #[test]
    fn loads_modern_gguf_imatrix() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5];
        let bytes = write_imatrix_gguf(&[
            ("blk.0.ffn_up.weight", 128.0, &a),
            ("blk.1.attn_q.weight", 64.0, &b),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.imatrix");
        std::fs::write(&path, &bytes).unwrap();

        let im = load_imatrix(&path).unwrap();
        // `.counts` and `.in_sum2` tensors must not appear as separate entries — only the bare
        // weight name is exposed.
        assert_eq!(im.len(), 2);
        assert_eq!(im.get("blk.0.ffn_up.weight").unwrap(), &a[..]);
        assert_eq!(im.get("blk.1.attn_q.weight").unwrap(), &b[..]);
        assert_eq!(im.entries.get("blk.0.ffn_up.weight").unwrap().ncall, 128);
        assert_eq!(im.entries.get("blk.1.attn_q.weight").unwrap().ncall, 64);
        assert!(im.get("blk.0.ffn_up.weight.in_sum2").is_none());
        assert!(im.get("blk.0.ffn_up.weight.counts").is_none());
    }

    #[test]
    fn gguf_imatrix_without_in_sum2_tensors_errors() {
        use requant_io::{GgufValue, GgufWriter, TensorSpec};
        let mut w = GgufWriter::new();
        w.add_kv("general.type", GgufValue::String("imatrix".to_string()));
        w.add_tensor(TensorSpec {
            name: "unrelated.weight".to_string(),
            dims: vec![2],
            ggml_type: 0,
            data: [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.imatrix");
        std::fs::write(&path, w.to_bytes()).unwrap();
        let res = load_imatrix(&path);
        assert!(res.is_err(), "GGUF with no .in_sum2 tensors must error");
    }
}
