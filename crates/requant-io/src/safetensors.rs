//! Minimal safetensors reader, including the sharded (`*.index.json`) case.
//!
//! GGUF is the container for the llama.cpp world; safetensors is the container for everything
//! that serves through vLLM/SGLang — which is where the FP4/FP8 checkpoints in
//! [`crate::blockfloat`] actually live. The reader is deliberately small: mmap the file, parse the
//! JSON header, hand out zero-copy byte slices. The interesting work is in the *layout* helpers
//! at the bottom, which turn a `(weight, weight_scale, weight_scale_2)` triple into the plain
//! `f32` that `requant_quant::quantize_tensor` consumes.
//!
//! Format: `u64 header_len` (LE) | `header_len` bytes of JSON | tensor data blob. Every entry is
//! `{"dtype": "...", "shape": [...], "data_offsets": [start, end]}` with offsets relative to the
//! start of the data blob. The reserved key `__metadata__` maps to a string→string table.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;

use crate::blockfloat::{
    dequant_fp8, dequant_nvfp4, e4m3_to_f32, e5m2_to_f32, Fp8Scale, RQ_TYPE_FP8_E4M3,
    RQ_TYPE_FP8_E5M2,
};

/// safetensors element types. The FP8 pair is why this reader exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl StDtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E4M3" => Self::F8E4M3,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "I64" => Self::I64,
            "U64" => Self::U64,
            other => bail!("unknown safetensors dtype `{other}`"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E5M2 | Self::F8E4M3 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// Canonical safetensors header spelling for this dtype.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E4M3 => "F8_E4M3",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::I64 => "I64",
            Self::U64 => "U64",
        }
    }

    /// True for dtypes [`SafeTensors::to_f32`] can widen directly.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::BF16 | Self::F32 | Self::F64 | Self::F8E4M3 | Self::F8E5M2
        )
    }
}

/// One tensor's header entry.
#[derive(Debug, Clone)]
pub struct StEntry {
    pub name: String,
    pub dtype: StDtype,
    /// Logical shape, outermost dim first (torch order: a Linear weight is `[out, in]`).
    pub shape: Vec<u64>,
    /// Byte range within the data blob.
    pub start: usize,
    pub end: usize,
}

impl StEntry {
    pub fn n_elems(&self) -> u64 {
        if self.shape.is_empty() {
            1 // 0-D scalar
        } else {
            self.shape.iter().product()
        }
    }

    /// `(rows, cols)` for a 2-D tensor; `(1, n)` for 1-D; `(1, 1)` for a scalar.
    pub fn rows_cols(&self) -> (usize, usize) {
        match self.shape.len() {
            0 => (1, 1),
            1 => (1, self.shape[0] as usize),
            _ => (
                self.shape[0] as usize,
                self.shape[1..].iter().product::<u64>() as usize,
            ),
        }
    }
}

/// A single memory-mapped safetensors file.
pub struct SafeTensors {
    #[allow(dead_code)]
    file: std::fs::File,
    mmap: Mmap,
    data_start: usize,
    pub path: PathBuf,
    pub metadata: BTreeMap<String, String>,
    entries: BTreeMap<String, StEntry>,
}

impl SafeTensors {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)
            .with_context(|| format!("opening safetensors {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            bail!("{}: too short to be a safetensors file", path.display());
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_len;
        if data_start > mmap.len() {
            bail!(
                "{}: header length {header_len} runs past EOF",
                path.display()
            );
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..data_start])
            .with_context(|| format!("parsing safetensors header of {}", path.display()))?;
        let obj = header.as_object().ok_or_else(|| {
            anyhow!(
                "{}: safetensors header is not a JSON object",
                path.display()
            )
        })?;

        let mut metadata = BTreeMap::new();
        let mut entries = BTreeMap::new();
        for (key, val) in obj {
            if key == "__metadata__" {
                if let Some(m) = val.as_object() {
                    for (k, v) in m {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = StDtype::parse(
                val.get("dtype")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| anyhow!("tensor `{key}`: missing dtype"))?,
            )?;
            let shape: Vec<u64> = val
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| anyhow!("tensor `{key}`: missing shape"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| anyhow!("tensor `{key}`: non-integer shape entry"))
                })
                .collect::<Result<_>>()?;
            let offs = val
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .ok_or_else(|| anyhow!("tensor `{key}`: missing data_offsets"))?;
            if offs.len() != 2 {
                bail!("tensor `{key}`: data_offsets must have 2 entries");
            }
            let start = offs[0].as_u64().unwrap_or(0) as usize;
            let end = offs[1].as_u64().unwrap_or(0) as usize;
            if end < start || data_start + end > mmap.len() {
                bail!("tensor `{key}`: data_offsets [{start}, {end}] run past EOF");
            }
            entries.insert(
                key.clone(),
                StEntry {
                    name: key.clone(),
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            file,
            mmap,
            data_start,
            path,
            metadata,
            entries,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    pub fn entry(&self, name: &str) -> Option<&StEntry> {
        self.entries.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Zero-copy byte view of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` not in {}", self.path.display()))?;
        Ok(&self.mmap[self.data_start + e.start..self.data_start + e.end])
    }

    /// Widen a float tensor to `f32`. FP8 dtypes decode with a unit scale — apply the
    /// checkpoint's `weight_scale` yourself, or use [`load_fp8_linear`] which does it for you.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` not in {}", self.path.display()))?;
        let b = self.bytes(name)?;
        Ok(match e.dtype {
            StDtype::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            StDtype::F64 => b
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            StDtype::F16 => b
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDtype::BF16 => b
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDtype::F8E4M3 => b.iter().map(|&x| e4m3_to_f32(x)).collect(),
            StDtype::F8E5M2 => b.iter().map(|&x| e5m2_to_f32(x)).collect(),
            other => bail!("tensor `{name}`: dtype {other:?} is not a float type"),
        })
    }

    /// Read a 0-D or 1-element tensor as a scalar.
    pub fn scalar_f32(&self, name: &str) -> Result<f32> {
        let v = self.to_f32(name)?;
        v.first()
            .copied()
            .ok_or_else(|| anyhow!("tensor `{name}` is empty; expected a scalar"))
    }
}

/// A model split across several safetensors shards, resolved through `*.index.json`
/// (`{"weight_map": {"tensor.name": "model-00001-of-000NN.safetensors", ...}}`).
pub struct ShardedSafeTensors {
    shards: Vec<SafeTensors>,
    /// tensor name -> index into `shards`
    map: BTreeMap<String, usize>,
}

impl ShardedSafeTensors {
    /// Open a checkpoint directory. Uses `model.safetensors.index.json` when present, otherwise
    /// opens every `*.safetensors` in the directory and merges their headers.
    pub fn open_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let index = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = Vec::new();
        if index.exists() {
            let txt = std::fs::read_to_string(&index)
                .with_context(|| format!("reading {}", index.display()))?;
            let json: serde_json::Value = serde_json::from_str(&txt)
                .with_context(|| format!("parsing {}", index.display()))?;
            let map = json
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow!("{}: missing weight_map", index.display()))?;
            let mut seen: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            seen.sort();
            seen.dedup();
            files = seen.into_iter().map(|f| dir.join(f)).collect();
        } else {
            let rd =
                std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))?;
            for ent in rd {
                let p = ent?.path();
                if p.extension().map_or(false, |e| e == "safetensors") {
                    files.push(p);
                }
            }
            files.sort();
        }
        if files.is_empty() {
            bail!("no .safetensors shards found in {}", dir.display());
        }
        Self::open_files(&files)
    }

    pub fn open_files(paths: &[PathBuf]) -> Result<Self> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut map = BTreeMap::new();
        for (i, p) in paths.iter().enumerate() {
            let st = SafeTensors::open(p)?;
            for n in st.names() {
                map.insert(n.to_string(), i);
            }
            shards.push(st);
        }
        Ok(Self { shards, map })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|s| s.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn shard_of(&self, name: &str) -> Option<&SafeTensors> {
        self.map.get(name).map(|&i| &self.shards[i])
    }

    pub fn entry(&self, name: &str) -> Option<&StEntry> {
        self.shard_of(name).and_then(|s| s.entry(name))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.shard_of(name)
            .ok_or_else(|| anyhow!("tensor `{name}` not found in any shard"))?
            .bytes(name)
    }

    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.shard_of(name)
            .ok_or_else(|| anyhow!("tensor `{name}` not found in any shard"))?
            .to_f32(name)
    }

    pub fn scalar_f32(&self, name: &str) -> Result<f32> {
        self.shard_of(name)
            .ok_or_else(|| anyhow!("tensor `{name}` not found in any shard"))?
            .scalar_f32(name)
    }
}

// ---------------------------------------------------------------------------
// Layout helpers: checkpoint triples -> f32
// ---------------------------------------------------------------------------

/// A tensor source that may be a single file or a sharded checkpoint.
pub trait TensorSource {
    fn st_entry(&self, name: &str) -> Option<&StEntry>;
    fn st_bytes(&self, name: &str) -> Result<&[u8]>;
    fn st_f32(&self, name: &str) -> Result<Vec<f32>>;
    fn st_has(&self, name: &str) -> bool {
        self.st_entry(name).is_some()
    }
}

impl TensorSource for SafeTensors {
    fn st_entry(&self, name: &str) -> Option<&StEntry> {
        self.entry(name)
    }
    fn st_bytes(&self, name: &str) -> Result<&[u8]> {
        self.bytes(name)
    }
    fn st_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.to_f32(name)
    }
}

impl TensorSource for ShardedSafeTensors {
    fn st_entry(&self, name: &str) -> Option<&StEntry> {
        self.entry(name)
    }
    fn st_bytes(&self, name: &str) -> Result<&[u8]> {
        self.bytes(name)
    }
    fn st_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.to_f32(name)
    }
}

/// A linear weight reconstructed from a quantized checkpoint, in logical `[out, in]` order.
pub struct DequantizedLinear {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    /// The on-disk format we read it from, for the quant→quant warning path.
    pub source_type: u32,
}

/// Load an NVFP4 linear from a checkpoint: `{prefix}.weight` (uint8, `[out, in/2]`),
/// `{prefix}.weight_scale` (float8_e4m3fn, `[out, in/16]`), `{prefix}.weight_scale_2` (fp32 scalar).
///
/// Reconstruction is `w = e2m1(q) · e4m3(block_scale) · weight_scale_2`, inverting NVIDIA
/// ModelOpt's exporter (`weight_scale_2 = amax / (6 · 448)`).
///
/// The block scales are read **linearly**, which is how they are stored in a compressed-tensors /
/// ModelOpt checkpoint. vLLM applies its 128×4 swizzle at load time, not at export time — do not
/// un-swizzle here.
pub fn load_nvfp4_linear<S: TensorSource + ?Sized>(
    src: &S,
    prefix: &str,
) -> Result<DequantizedLinear> {
    let w_name = format!("{prefix}.weight");
    let s_name = format!("{prefix}.weight_scale");
    let g_name = format!("{prefix}.weight_scale_2");

    let we = src
        .st_entry(&w_name)
        .ok_or_else(|| anyhow!("NVFP4 load: `{w_name}` not found"))?;
    if we.shape.len() != 2 {
        bail!(
            "NVFP4 load: `{w_name}` has shape {:?}, expected 2-D [out, in/2]",
            we.shape
        );
    }
    let rows = we.shape[0] as usize;
    let cols = we.shape[1] as usize * 2; // two E2M1 codes per byte
    let data = src.st_bytes(&w_name)?;
    let scales = src
        .st_bytes(&s_name)
        .with_context(|| format!("NVFP4 load: `{s_name}` (per-block E4M3 scales) is required"))?;
    // `weight_scale_2` is a scalar in every exporter we've seen; tolerate a 1-element tensor.
    let global = if src.st_has(&g_name) {
        let v = src.st_f32(&g_name)?;
        v.first().copied().unwrap_or(1.0)
    } else {
        // Absent global scale means the block scales are already absolute.
        1.0
    };

    let n = rows * cols;
    let mut out = vec![0f32; n];
    dequant_nvfp4(data, scales, global, n, &mut out)
        .with_context(|| format!("NVFP4 load: dequantizing `{w_name}` ({rows}×{cols})"))?;
    Ok(DequantizedLinear {
        rows,
        cols,
        data: out,
        source_type: crate::blockfloat::RQ_TYPE_NVFP4,
    })
}

/// Load a dense FP8 linear: `{prefix}.weight` (float8_*) plus whichever scale tensor the producer
/// wrote — `weight_scale` (multiply) or `weight_scale_inv` (DeepSeek's 2-D block form, also a
/// multiply despite the name: it holds `amax/448` per 128×128 tile).
///
/// `block` is the 2-D scale tile size to assume when the scale tensor is 2-D; pass `(128, 128)`
/// for DeepSeek-style checkpoints.
pub fn load_fp8_linear<S: TensorSource + ?Sized>(
    src: &S,
    prefix: &str,
    block: (usize, usize),
) -> Result<DequantizedLinear> {
    let w_name = format!("{prefix}.weight");
    let we = src
        .st_entry(&w_name)
        .ok_or_else(|| anyhow!("FP8 load: `{w_name}` not found"))?;
    let ty = match we.dtype {
        StDtype::F8E4M3 => RQ_TYPE_FP8_E4M3,
        StDtype::F8E5M2 => RQ_TYPE_FP8_E5M2,
        other => bail!("FP8 load: `{w_name}` has dtype {other:?}, expected F8_E4M3 or F8_E5M2"),
    };
    let (rows, cols) = we.rows_cols();
    let data = src.st_bytes(&w_name)?;

    let scale_name = [
        format!("{prefix}.weight_scale_inv"),
        format!("{prefix}.weight_scale"),
    ]
    .into_iter()
    .find(|n| src.st_has(n));
    let (scale_vals, scale_shape) = match &scale_name {
        Some(n) => (
            src.st_f32(n)?,
            src.st_entry(n).map(|e| e.shape.clone()).unwrap_or_default(),
        ),
        None => (Vec::new(), Vec::new()),
    };

    let scale = if scale_vals.is_empty() {
        Fp8Scale::Unit
    } else if scale_vals.len() == 1 {
        Fp8Scale::PerTensor(scale_vals[0])
    } else if scale_shape.len() >= 2 && scale_shape[0] as usize != rows {
        Fp8Scale::Block2d {
            scales: &scale_vals,
            bh: block.0,
            bw: block.1,
        }
    } else if scale_vals.len() == rows {
        Fp8Scale::PerChannel(&scale_vals)
    } else {
        Fp8Scale::Block2d {
            scales: &scale_vals,
            bh: block.0,
            bw: block.1,
        }
    };

    let mut out = vec![0f32; rows * cols];
    dequant_fp8(ty, data, rows, cols, scale, &mut out)
        .with_context(|| format!("FP8 load: dequantizing `{w_name}` ({rows}×{cols})"))?;
    Ok(DequantizedLinear {
        rows,
        cols,
        data: out,
        source_type: ty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockfloat::{f32_to_e2m1, f32_to_e4m3};

    /// Build an in-memory safetensors file from `(name, dtype, shape, bytes)` tuples.
    fn build(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut blob: Vec<u8> = Vec::new();
        for (i, (name, dtype, shape, data)) in entries.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let start = blob.len();
            blob.extend_from_slice(data);
            let shape_s = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_s}],\"data_offsets\":[{start},{}]}}",
                blob.len()
            ));
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&blob);
        out
    }

    #[test]
    fn reads_header_and_widens_floats() {
        let f32s: Vec<u8> = [1.0f32, -2.0, 3.5, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let buf = build(&[("a", "F32", vec![2, 2], f32s)]);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.safetensors");
        std::fs::write(&p, &buf).unwrap();

        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(st.entry("a").unwrap().rows_cols(), (2, 2));
        assert_eq!(st.to_f32("a").unwrap(), vec![1.0, -2.0, 3.5, 0.0]);
    }

    #[test]
    fn nvfp4_linear_round_trips_a_known_block() {
        // 1 row × 16 cols: one NVFP4 block. Values chosen to sit exactly on the E2M1 grid so the
        // reconstruction is exact.
        let vals: Vec<f32> = vec![
            6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
        ];
        let global = 1.0f32 / 448.0; // weight_scale_2
        let block_scale = 1.0f32 / global; // so block_scale * global == 1.0
        let sb = f32_to_e4m3(block_scale);
        let mut packed = vec![0u8; 8];
        for j in 0..8 {
            let lo = f32_to_e2m1(vals[2 * j]);
            let hi = f32_to_e2m1(vals[2 * j + 1]);
            packed[j] = lo | (hi << 4);
        }
        let buf = build(&[
            ("l.weight", "U8", vec![1, 8], packed),
            ("l.weight_scale", "F8_E4M3", vec![1, 1], vec![sb]),
            (
                "l.weight_scale_2",
                "F32",
                vec![],
                global.to_le_bytes().to_vec(),
            ),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.safetensors");
        std::fs::write(&p, &buf).unwrap();

        let st = SafeTensors::open(&p).unwrap();
        let lin = load_nvfp4_linear(&st, "l").unwrap();
        assert_eq!((lin.rows, lin.cols), (1, 16));
        assert_eq!(lin.data, vals);
    }

    #[test]
    fn fp8_linear_applies_a_per_tensor_scale() {
        let data: Vec<u8> = (0..4).map(|_| f32_to_e4m3(2.0)).collect();
        let buf = build(&[
            ("l.weight", "F8_E4M3", vec![2, 2], data),
            (
                "l.weight_scale",
                "F32",
                vec![],
                3.0f32.to_le_bytes().to_vec(),
            ),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.safetensors");
        std::fs::write(&p, &buf).unwrap();

        let st = SafeTensors::open(&p).unwrap();
        let lin = load_fp8_linear(&st, "l", (128, 128)).unwrap();
        assert_eq!(lin.data, vec![6.0, 6.0, 6.0, 6.0]);
    }
}
