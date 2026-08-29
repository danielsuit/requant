//! requant-quant: the Quantizer trait + format-family implementations.
//!
//! Families shipped:
//!   - `legacy`: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 (bit-exact ports of ggml `_ref` kernels)
//!   - `kquant`: Q2_K..Q6_K (super-block k-quants with imatrix-weighted scale search)
//!   - `mxfp`:   MXFP4 / NVFP4 / MXFP8 / dense FP8 (block-scaled floats; the Blackwell fast path)
//! The trait + `StatKind` are structured so GPTQ/AWQ/i-quants can slot in later.
//!
//! Two entry points, because the two format worlds have different shapes:
//!
//! - [`quantize_tensor`] returns one flat `Vec<u8>` and is the GGUF path. Every ggml block type
//!   carries its scales inside the block, so a tensor really is just bytes.
//! - [`quantize_tensor_ex`] returns a [`QuantOutput`], which for NVFP4/FP8 is an element buffer
//!   *plus named sidecar scale tensors* — because that is how a safetensors checkpoint stores
//!   them, and flattening them would produce something no serving stack can read.
//!
//! [`roundtrip_tensor`] papers over the difference for the search/eval paths, which only ever
//! want the reconstructed f32 back.

pub mod legacy;
pub mod kquant;
pub mod mxfp;
pub mod iquant;
pub mod recipe;

use anyhow::{bail, Result};
use rayon::prelude::*;

use requant_io::blockfloat::{
    GGML_TYPE_MXFP4, RQ_TYPE_FP8_E4M3, RQ_TYPE_FP8_E5M2, RQ_TYPE_MXFP4_OCP, RQ_TYPE_MXFP8_E4M3,
    RQ_TYPE_NVFP4,
};

pub use mxfp::{
    Fp8Checkpoint, Fp8Granularity, MxFp4Checkpoint, MxFp8Checkpoint, MxScaleRule, MxfpPolicy,
    NvFp4Checkpoint,
};
pub use recipe::{
    Bits, Defaults, Floors, Policy, QuantMethod, Recipe, Rule, StatKind, DEFAULT_RECIPE_TOML,
};

/// Quantize a 2-D weight (row-major `x`, `rows`×`cols`, `cols` contiguous) into packed bytes
/// for `ggml_type`. `imatrix` is an optional per-input-channel importance vector (length `cols`).
pub fn quantize_tensor(
    ggml_type: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    imatrix: Option<&[f32]>,
) -> Result<Vec<u8>> {
    if x.len() != rows * cols {
        bail!("quantize_tensor: data len {} != {rows}×{cols}", x.len());
    }
    let (block, bpb) = requant_io::block_layout(ggml_type)
        .ok_or_else(|| anyhow::anyhow!("unsupported target type {ggml_type}"))?;
    if cols % block != 0 {
        bail!("quantize_tensor: cols {cols} not divisible by block {block} for type {ggml_type}");
    }
    // Block-float types that are not ggml types cannot be expressed as one flat buffer that any
    // loader understands — their scales are sibling tensors. Send the caller to the API that
    // preserves that structure rather than emitting bytes nothing can read.
    if !requant_io::is_gguf_type(ggml_type) {
        bail!(
            "quantize_tensor: {} has no ggml type — its block scales are separate tensors. \
             Use quantize_tensor_ex() and write the resulting sidecar scale tensors, or \
             roundtrip_tensor() if you only need the reconstruction.",
            requant_io::ggml_type_name(ggml_type)
        );
    }
    if ggml_type == GGML_TYPE_MXFP4 {
        return mxfp::quantize_mxfp4_ggml(x, rows, cols, imatrix, &MxfpPolicy::default());
    }

    // i-quants (codebook family). These are ggml types but have their own row driver.
    if iquant::handles(ggml_type) {
        let bs = iquant::block_size_of(ggml_type);
        let row_bytes = (cols / bs) * iquant::bytes_per_block(ggml_type);
        let mut out = vec![0u8; rows * row_bytes];
        iquant::quantize_rows(ggml_type, x, &mut out, rows, cols, imatrix)?;
        return Ok(out);
    }

    let blocks_per_row = cols / block;
    let row_bytes = blocks_per_row * bpb;
    let mut out = vec![0u8; rows * row_bytes];

    // Legacy + k-quants are row-independent; parallelize across rows.
    let is_legacy = matches!(ggml_type, 2 | 3 | 6 | 7 | 8 | 9);
    if is_legacy {
        out.par_chunks_mut(row_bytes)
            .zip(x.par_chunks(cols))
            .for_each(|(out_row, in_row)| {
                legacy::quantize_row(ggml_type, in_row, out_row);
            });
    } else {
        kquant::quantize_rows(ggml_type, x, &mut out, rows, cols, imatrix)?;
    }
    Ok(out)
}

/// Dequantize packed `bytes` (type `ggml_type`, `rows`×`cols`) back to f32 (logical order).
pub fn dequantize_tensor(
    ggml_type: u32,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let (block, bpb) = requant_io::block_layout(ggml_type)
        .ok_or_else(|| anyhow::anyhow!("unsupported source type {ggml_type}"))?;
    if cols % block != 0 {
        bail!("dequantize_tensor: cols {cols} not divisible by block {block} for type {ggml_type}");
    }
    if ggml_type == GGML_TYPE_MXFP4 {
        return mxfp::dequantize_mxfp4_ggml(bytes, rows, cols);
    }
    if iquant::handles(ggml_type) {
        let mut out = vec![0f32; rows * cols];
        iquant::dequantize_rows(ggml_type, bytes, &mut out, rows, cols)?;
        return Ok(out);
    }
    match ggml_type {
        // Self-contained requant-internal containers: element bytes then scale bytes.
        RQ_TYPE_MXFP4_OCP => {
            let n = rows * cols;
            let mut out = vec![0f32; n];
            requant_io::dequant_mxfp4_ocp(&bytes[..n / 2], &bytes[n / 2..], n, &mut out)?;
            return Ok(out);
        }
        RQ_TYPE_MXFP8_E4M3 => {
            let n = rows * cols;
            let mut out = vec![0f32; n];
            requant_io::dequant_mxfp8(&bytes[..n], &bytes[n..], n, &mut out)?;
            return Ok(out);
        }
        // NVFP4 needs the per-tensor `weight_scale_2` and dense FP8 needs its scale tensor;
        // neither is in the byte buffer, so there is no honest answer here.
        RQ_TYPE_NVFP4 => bail!(
            "dequantize_tensor: NVFP4 also needs the per-tensor weight_scale_2 — use \
             NvFp4Checkpoint::from_packed(bytes, rows, cols, weight_scale_2).dequantize()"
        ),
        RQ_TYPE_FP8_E4M3 | RQ_TYPE_FP8_E5M2 => bail!(
            "dequantize_tensor: dense FP8 also needs its weight_scale tensor — use \
             requant_io::blockfloat::dequant_fp8() or requant_io::load_fp8_linear()"
        ),
        _ => {}
    }
    let blocks_per_row = cols / block;
    let row_bytes = blocks_per_row * bpb;
    if bytes.len() < rows * row_bytes {
        bail!("dequantize_tensor: {} bytes < needed {}", bytes.len(), rows * row_bytes);
    }
    let mut out = vec![0f32; rows * cols];
    let is_legacy = matches!(ggml_type, 2 | 3 | 6 | 7 | 8 | 9);
    if is_legacy {
        out.par_chunks_mut(cols)
            .zip(bytes.par_chunks(row_bytes))
            .for_each(|(out_row, in_row)| {
                legacy::dequantize_row(ggml_type, in_row, out_row);
            });
    } else {
        kquant::dequantize_rows(ggml_type, bytes, &mut out, rows, cols)?;
    }
    Ok(out)
}

/// Bits-per-weight for a type (informational; from IO geometry).
pub fn bpw(ggml_type: u32) -> Option<f64> {
    requant_io::bpw(ggml_type)
}

/// Resolve the actual on-disk ggml type for a tensor given the recipe's requested `target_type`
/// and the tensor's contiguous dimension `cols` (the in-features, i.e. `ne[0]`).
///
/// Block-quant types require `cols` to be divisible by their block size. When it is not, ggml
/// falls back to a compatible type rather than padding — matching this exactly is what keeps the
/// output loadable by llama.cpp and size-equivalent to `llama-quantize`. The fallback table is
/// ggml's `tensor_type_fallback` (llama-quant.cpp):
///
///   Q2_K/Q3_K -> Q4_0,  Q4_K -> Q5_0,  Q5_K -> Q5_1,  Q6_K -> Q8_0
///
/// with a secondary fallback to F16 when the first fallback's block (32) *also* fails to divide
/// `cols`. Returns `(actual_type, fell_back)`; `fell_back` is true iff a fallback was applied.
pub fn fallback_type(target_type: u32, cols: usize) -> (u32, bool) {
    use requant_io::block_layout;
    // F16
    const F16: u32 = 1;
    let fits = |t: u32| block_layout(t).map_or(false, |(b, _)| cols % b == 0);
    if fits(target_type) {
        return (target_type, false);
    }
    // Primary fallback: k-quants drop to a legacy quant; legacy quants drop straight to F16.
    let primary = match target_type {
        10 | 11 => 2,  // Q2_K, Q3_K -> Q4_0
        12 => 6,       // Q4_K -> Q5_0
        13 => 7,       // Q5_K -> Q5_1
        14 => 8,       // Q6_K -> Q8_0
        // Super-block i-quants (block 256) -> IQ4_NL (block 32), matching llama-quantize's
        // incompatible-shape fallback. IQ4_NL itself is block 32; if 32 still doesn't divide
        // cols the secondary fallback below drops to F16.
        16 | 17 | 18 | 19 | 21 | 22 | 23 | 29 => 20, // IQ2/IQ3/IQ1/IQ4_XS -> IQ4_NL
        // Block-float families fall back to dense FP8, whose "block" is a single element and so
        // always divides. Falling back from FP4 to F16 would quadruple the tensor and blow the
        // budget that motivated FP4 in the first place; FP8 keeps the dense-path floor intact.
        GGML_TYPE_MXFP4 | RQ_TYPE_MXFP4_OCP | RQ_TYPE_NVFP4 | RQ_TYPE_MXFP8_E4M3 => RQ_TYPE_FP8_E4M3,
        _ => F16,      // legacy / other block quant that doesn't fit -> F16
    };
    // Secondary fallback: if the legacy quant's block (32) also doesn't divide cols, use F16.
    let actual = if fits(primary) { primary } else { F16 };
    (actual, true)
}

/// Pack f32 values into a float ggml type (F32/F16/BF16) as little-endian bytes. Used for the
/// "target is a float type" path (e.g. keep the router at F16, convert an F32 norm to F16).
pub fn pack_float(target_type: u32, x: &[f32]) -> Result<Vec<u8>> {
    use half::{bf16, f16};
    match target_type {
        0 => {
            // F32
            let mut out = Vec::with_capacity(x.len() * 4);
            for v in x {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        1 => {
            // F16
            let mut out = Vec::with_capacity(x.len() * 2);
            for v in x {
                out.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
            }
            Ok(out)
        }
        30 => {
            // BF16
            let mut out = Vec::with_capacity(x.len() * 2);
            for v in x {
                out.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
            }
            Ok(out)
        }
        other => bail!("pack_float: target type {other} is not a float type (0=F32,1=F16,30=BF16)"),
    }
}

/// A sidecar scale tensor emitted alongside a block-float weight.
#[derive(Debug, Clone)]
pub enum ScaleTensor {
    /// fp32 values (dense FP8 `weight_scale` / NVFP4 `weight_scale_2`).
    F32(Vec<f32>),
    /// float8_e4m3fn bytes (NVFP4 `weight_scale`).
    E4M3(Vec<u8>),
    /// E8M0 shared-exponent bytes (MX formats).
    E8M0(Vec<u8>),
}

impl ScaleTensor {
    pub fn nbytes(&self) -> usize {
        match self {
            ScaleTensor::F32(v) => v.len() * 4,
            ScaleTensor::E4M3(v) | ScaleTensor::E8M0(v) => v.len(),
        }
    }

    /// safetensors dtype string for this scale tensor.
    pub fn dtype(&self) -> &'static str {
        match self {
            ScaleTensor::F32(_) => "F32",
            ScaleTensor::E4M3(_) => "F8_E4M3",
            ScaleTensor::E8M0(_) => "U8",
        }
    }
}

/// The result of quantizing one tensor.
#[derive(Debug, Clone)]
pub enum QuantOutput {
    /// A real ggml type: one flat buffer, writable straight into a GGUF.
    Ggml { ggml_type: u32, data: Vec<u8> },
    /// A safetensors-style group: element bytes plus named sidecar scale tensors. The names are
    /// suffixes to append to the tensor's own name (`weight_scale`, `weight_scale_2`).
    Grouped {
        ggml_type: u32,
        weight: Vec<u8>,
        scales: Vec<(&'static str, ScaleTensor)>,
    },
}

impl QuantOutput {
    /// Total bytes on disk, including sidecar scales.
    pub fn nbytes(&self) -> usize {
        match self {
            QuantOutput::Ggml { data, .. } => data.len(),
            QuantOutput::Grouped { weight, scales, .. } => {
                weight.len() + scales.iter().map(|(_, s)| s.nbytes()).sum::<usize>()
            }
        }
    }

    pub fn ggml_type(&self) -> u32 {
        match self {
            QuantOutput::Ggml { ggml_type, .. } | QuantOutput::Grouped { ggml_type, .. } => {
                *ggml_type
            }
        }
    }
}

/// Quantize a tensor into whichever shape its format actually has.
///
/// Use this instead of [`quantize_tensor`] when the target may be NVFP4 or dense FP8, whose scales
/// are separate tensors. The `Grouped` variant's scale names are exactly what
/// [`requant_io::load_nvfp4_linear`] / [`requant_io::load_fp8_linear`] read back.
pub fn quantize_tensor_ex(
    ggml_type: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    imatrix: Option<&[f32]>,
    policy: &MxfpPolicy,
) -> Result<QuantOutput> {
    match ggml_type {
        RQ_TYPE_NVFP4 => {
            let ck = NvFp4Checkpoint::quantize(x, rows, cols, imatrix, policy)?;
            Ok(QuantOutput::Grouped {
                ggml_type,
                weight: ck.weight,
                scales: vec![
                    ("weight_scale", ScaleTensor::E4M3(ck.weight_scale)),
                    ("weight_scale_2", ScaleTensor::F32(vec![ck.weight_scale_2])),
                ],
            })
        }
        RQ_TYPE_MXFP4_OCP => {
            let ck = MxFp4Checkpoint::quantize(x, rows, cols, imatrix, policy)?;
            Ok(QuantOutput::Grouped {
                ggml_type,
                weight: ck.weight,
                scales: vec![("weight_scale", ScaleTensor::E8M0(ck.weight_scale))],
            })
        }
        RQ_TYPE_MXFP8_E4M3 => {
            let ck = MxFp8Checkpoint::quantize(x, rows, cols)?;
            Ok(QuantOutput::Grouped {
                ggml_type,
                weight: ck.weight,
                scales: vec![("weight_scale", ScaleTensor::E8M0(ck.weight_scale))],
            })
        }
        RQ_TYPE_FP8_E4M3 | RQ_TYPE_FP8_E5M2 => {
            let ck = Fp8Checkpoint::quantize(ggml_type, x, rows, cols, Fp8Granularity::PerChannel)?;
            Ok(QuantOutput::Grouped {
                ggml_type,
                weight: ck.weight,
                scales: vec![("weight_scale", ScaleTensor::F32(ck.weight_scale))],
            })
        }
        _ => Ok(QuantOutput::Ggml {
            ggml_type,
            data: quantize_tensor(ggml_type, x, rows, cols, imatrix)?,
        }),
    }
}

/// Quantize then dequantize, returning the reconstruction.
///
/// This is what the sensitivity harness and the search proxy consume: they never need the packed
/// bytes, only "what does the model see after this format eats the tensor". Handling float types
/// here too means a ladder can contain F16 without the caller special-casing it.
pub fn roundtrip_tensor(
    ggml_type: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    imatrix: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if requant_io::is_float_type(ggml_type) {
        // F32 is exact; F16/BF16 lose mantissa bits, which a sensitivity curve should see.
        let packed = pack_float(ggml_type, x)?;
        return unpack_float(ggml_type, &packed, x.len());
    }
    if mxfp::handles(ggml_type) {
        return mxfp::roundtrip(ggml_type, x, rows, cols, imatrix, &MxfpPolicy::default());
    }
    if iquant::handles(ggml_type) {
        let bs = iquant::block_size_of(ggml_type);
        let row_bytes = (cols / bs) * iquant::bytes_per_block(ggml_type);
        let mut packed = vec![0u8; rows * row_bytes];
        iquant::quantize_rows(ggml_type, x, &mut packed, rows, cols, imatrix)?;
        let mut out = vec![0f32; rows * cols];
        iquant::dequantize_rows(ggml_type, &packed, &mut out, rows, cols)?;
        return Ok(out);
    }
    let packed = quantize_tensor(ggml_type, x, rows, cols, imatrix)?;
    dequantize_tensor(ggml_type, &packed, rows, cols)
}

/// Inverse of [`pack_float`].
pub fn unpack_float(ggml_type: u32, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    use half::{bf16, f16};
    let mut out = Vec::with_capacity(n);
    match ggml_type {
        0 => {
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        1 => {
            for c in bytes.chunks_exact(2) {
                out.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
            }
        }
        30 => {
            for c in bytes.chunks_exact(2) {
                out.push(bf16::from_le_bytes([c[0], c[1]]).to_f32());
            }
        }
        other => bail!("unpack_float: target type {other} is not a float type"),
    }
    out.truncate(n);
    Ok(out)
}
