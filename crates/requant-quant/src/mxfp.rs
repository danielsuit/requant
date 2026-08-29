//! The MX / NVFP4 quantizer family — the **emit** path for block-scaled float formats.
//!
//! Reading these formats (`requant_io::blockfloat`) is the easy direction: you decode bytes you
//! were handed. Emitting them is the hard direction, and for a different reason than the k-quants.
//! With k-quants the risk is numerical (does our scale search pick the same scale ggml's does).
//! Here the arithmetic is nearly trivial — amax, divide, round — and the risk is entirely
//! **layout**: the serving stack reconstructs `W` from three tensors whose packing, nibble order,
//! scale dtype and scale *convention* all have to match what its kernels assume. "Internally
//! consistent" is not a defence; a checkpoint that round-trips perfectly through our own
//! dequantizer and disagrees with vLLM by one nibble order produces fluent garbage.
//!
//! So every layout decision below cites the thing it is matching:
//!
//! - **NVFP4 element packing** — `byte[j] = q[2j] | (q[2j+1] << 4)`, adjacent pairs, even element
//!   in the low nibble. This is NVIDIA ModelOpt's `_cast_fp4` + pack, and what vLLM's
//!   `cutlass_scaled_fp4_mm` unpacks.
//! - **NVFP4 scale convention** — `weight_scale_2 = amax / (6 · 448)` (fp32, per tensor);
//!   `weight_scale[b] = e4m3(amax_b / 6 / weight_scale_2)` (per 16-element block); reconstruction
//!   is `e2m1(q) · weight_scale[b] · weight_scale_2`. In vLLM the second-level scale arrives
//!   folded into `alpha = input_scale · weight_scale_2`, which is why it must be the *small*
//!   number `amax/(6·448)` and not its reciprocal.
//! - **NVFP4 scale ordering** — linear, row-major `[out, in/16]`. vLLM applies its 128×4 swizzle
//!   in `process_weights_after_loading`, i.e. at load time. Pre-swizzling here double-applies it.
//! - **MXFP4 (ggml)** — `block_mxfp4 { e: u8, qs: [u8; 16] }` with ggml's split-half nibble
//!   interleave (`qs[j] = q[j] | (q[j+16] << 4)`), the same one `Q4_0` uses. Different from NVFP4.
//! - **MXFP4 shared scale** — OCP MX §6.3: `X = 2^(floor(log2(amax)) − emax_elem)`, `emax_elem = 2`
//!   for E2M1. This intentionally allows up to 8/6 clipping; [`MxScaleRule::NoClip`] trades one
//!   exponent step of resolution for guaranteed no saturation.
//!
//! Until the `ggml-oracle-mxfp4` test feature has been run against a ggml build that actually has
//! `GGML_TYPE_MXFP4`, treat the GGUF MXFP4 emit as unverified — the same discipline the k-quants
//! got, just not yet discharged.

use anyhow::{bail, Result};

use requant_io::blockfloat::{
    e2m1_to_f32, e4m3_to_f32, e8m0_to_f32, f32_to_e2m1_with, f32_to_e4m3, f32_to_e8m0,
    E2m1Rounding, E2M1_EMAX, E2M1_MAX, E4M3_MAX, GGML_TYPE_MXFP4, RQ_TYPE_FP8_E4M3,
    RQ_TYPE_FP8_E5M2, RQ_TYPE_MXFP4_OCP, RQ_TYPE_MXFP8_E4M3, RQ_TYPE_NVFP4,
};

pub const MXFP4_BLOCK: usize = 32;
pub const MXFP4_BYTES: usize = 17;
pub const NVFP4_BLOCK: usize = 16;

/// How to pick the power-of-two shared scale for an MX block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MxScaleRule {
    /// OCP MX spec: `2^(floor(log2(amax)) − emax_elem)`. Elements up to `8/6 ≈ 1.33×` the grid max
    /// clip to ±6. Spec-conformant, and what a hardware-oriented reference implementation does.
    #[default]
    OcpFloor,
    /// Smallest power of two `≥ amax / 6`, so nothing ever saturates. Costs up to one exponent
    /// step of resolution on blocks whose amax sits just above a power of two.
    NoClip,
}

/// Knobs for the MX/NVFP4 quantizers.
#[derive(Debug, Clone, Copy)]
pub struct MxfpPolicy {
    /// FP4 tie-breaking. Default [`E2m1Rounding::HalfDown`] matches both ggml's
    /// `best_index_mxfp4` and ModelOpt's `_cast_fp4`.
    pub rounding: E2m1Rounding,
    pub scale_rule: MxScaleRule,
    /// When an imatrix is available, search neighbouring in-format scales for the one minimising
    /// the importance-weighted block error (§2.3, same objective as the k-quant scale search).
    /// This changes the *scale value*, never the layout, so the result stays a conformant
    /// checkpoint. Off by default so the output matches a reference exporter byte-for-byte.
    pub refine_with_imatrix: bool,
}

impl Default for MxfpPolicy {
    fn default() -> Self {
        Self {
            rounding: E2m1Rounding::HalfDown,
            scale_rule: MxScaleRule::OcpFloor,
            refine_with_imatrix: false,
        }
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

#[inline]
fn amax(x: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// Importance-weighted squared error of encoding `x` with scale `s`.
#[inline]
fn block_err(x: &[f32], g: Option<&[f32]>, s: f32, rounding: E2m1Rounding) -> f64 {
    if !(s > 0.0) || !s.is_finite() {
        // Degenerate scale: everything encodes to zero.
        return match g {
            Some(w) => x.iter().zip(w).map(|(v, gi)| (*v as f64) * (*v as f64) * (*gi as f64)).sum(),
            None => x.iter().map(|v| (*v as f64) * (*v as f64)).sum(),
        };
    }
    let inv = 1.0 / s;
    let mut acc = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        let q = e2m1_to_f32(f32_to_e2m1_with(v * inv, rounding));
        let e = (v as f64) - (q as f64) * (s as f64);
        let w = g.map_or(1.0, |w| (w[i] as f64).max(1e-12));
        acc += e * e * w;
    }
    acc
}

/// Shared power-of-two exponent for one MX block, as an E8M0 code.
fn mx_shared_exponent(block: &[f32], g: Option<&[f32]>, policy: &MxfpPolicy) -> u8 {
    let m = amax(block);
    if !(m > 0.0) || !m.is_finite() {
        return 0;
    }
    let base = match policy.scale_rule {
        MxScaleRule::OcpFloor => f32_to_e8m0(m) as i32 - E2M1_EMAX,
        MxScaleRule::NoClip => {
            // Smallest power of two >= m/6.
            let target = m / E2M1_MAX;
            let e = f32_to_e8m0(target) as i32;
            if e8m0_to_f32(e.clamp(0, 254) as u8) < target {
                e + 1
            } else {
                e
            }
        }
    };
    let e0 = base.clamp(0, 254) as u8;
    if !policy.refine_with_imatrix {
        return e0;
    }
    // In-format refinement: the scale must stay a power of two, so the search space is the
    // neighbouring exponents. One step either way is enough — the objective is unimodal in the
    // exponent for a fixed rounding rule.
    let mut best = e0;
    let mut best_err = block_err(block, g, e8m0_to_f32(e0), policy.rounding);
    for delta in [-1i32, 1] {
        let e = (e0 as i32 + delta).clamp(0, 254) as u8;
        if e == e0 {
            continue;
        }
        let err = block_err(block, g, e8m0_to_f32(e), policy.rounding);
        if err < best_err {
            best_err = err;
            best = e;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// MXFP4 — ggml layout (GGUF-writable)
// ---------------------------------------------------------------------------

/// Quantize one row (`cols` elements, a multiple of 32) into ggml `block_mxfp4` bytes.
pub fn quant_mxfp4_ggml_row(x: &[f32], out: &mut [u8], im: Option<&[f32]>, policy: &MxfpPolicy) {
    let nb = x.len() / MXFP4_BLOCK;
    for b in 0..nb {
        let blk = &x[b * MXFP4_BLOCK..(b + 1) * MXFP4_BLOCK];
        let g = im.map(|w| &w[b * MXFP4_BLOCK..(b + 1) * MXFP4_BLOCK]);
        let e = mx_shared_exponent(blk, g, policy);
        let s = e8m0_to_f32(e);
        let inv = if s > 0.0 && s.is_finite() { 1.0 / s } else { 0.0 };
        let dst = &mut out[b * MXFP4_BYTES..(b + 1) * MXFP4_BYTES];
        dst[0] = e;
        // ggml packs element j in the low nibble and element j+16 in the high nibble.
        for j in 0..MXFP4_BLOCK / 2 {
            let q0 = f32_to_e2m1_with(blk[j] * inv, policy.rounding);
            let q1 = f32_to_e2m1_with(blk[j + MXFP4_BLOCK / 2] * inv, policy.rounding);
            dst[1 + j] = q0 | (q1 << 4);
        }
    }
}

/// Quantize a whole `rows × cols` weight into ggml MXFP4 bytes.
pub fn quantize_mxfp4_ggml(
    x: &[f32],
    rows: usize,
    cols: usize,
    im: Option<&[f32]>,
    policy: &MxfpPolicy,
) -> Result<Vec<u8>> {
    if cols % MXFP4_BLOCK != 0 {
        bail!("MXFP4: cols {cols} not divisible by block {MXFP4_BLOCK}");
    }
    if let Some(w) = im {
        if w.len() < cols {
            bail!("MXFP4: imatrix length {} < cols {cols}", w.len());
        }
    }
    let row_bytes = cols / MXFP4_BLOCK * MXFP4_BYTES;
    let mut out = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        quant_mxfp4_ggml_row(
            &x[r * cols..(r + 1) * cols],
            &mut out[r * row_bytes..(r + 1) * row_bytes],
            im,
            policy,
        );
    }
    Ok(out)
}

/// Dequantize ggml MXFP4 bytes back to f32.
pub fn dequantize_mxfp4_ggml(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if cols % MXFP4_BLOCK != 0 {
        bail!("MXFP4: cols {cols} not divisible by block {MXFP4_BLOCK}");
    }
    let row_bytes = cols / MXFP4_BLOCK * MXFP4_BYTES;
    if bytes.len() < rows * row_bytes {
        bail!("MXFP4: {} bytes < needed {}", bytes.len(), rows * row_bytes);
    }
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        requant_io::blockfloat::dequant_mxfp4_ggml_row(
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &mut out[r * cols..(r + 1) * cols],
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// MXFP4 — OCP layout (safetensors)
// ---------------------------------------------------------------------------

/// MXFP4 in OCP layout: adjacent nibbles, E8M0 scales in a sidecar tensor.
pub struct MxFp4Checkpoint {
    pub rows: usize,
    pub cols: usize,
    /// `[rows, cols/2]` uint8, `byte[j] = q[2j] | (q[2j+1] << 4)`.
    pub weight: Vec<u8>,
    /// `[rows, cols/32]` uint8 E8M0 shared exponents.
    pub weight_scale: Vec<u8>,
}

impl MxFp4Checkpoint {
    pub fn quantize(
        x: &[f32],
        rows: usize,
        cols: usize,
        im: Option<&[f32]>,
        policy: &MxfpPolicy,
    ) -> Result<Self> {
        if cols % MXFP4_BLOCK != 0 {
            bail!("MXFP4(OCP): cols {cols} not divisible by block {MXFP4_BLOCK}");
        }
        let nblk_row = cols / MXFP4_BLOCK;
        let mut weight = vec![0u8; rows * cols / 2];
        let mut weight_scale = vec![0u8; rows * nblk_row];
        for r in 0..rows {
            for b in 0..nblk_row {
                let off = r * cols + b * MXFP4_BLOCK;
                let blk = &x[off..off + MXFP4_BLOCK];
                let g = im.map(|w| &w[b * MXFP4_BLOCK..(b + 1) * MXFP4_BLOCK]);
                let e = mx_shared_exponent(blk, g, policy);
                weight_scale[r * nblk_row + b] = e;
                let s = e8m0_to_f32(e);
                let inv = if s > 0.0 && s.is_finite() { 1.0 / s } else { 0.0 };
                for j in 0..MXFP4_BLOCK / 2 {
                    let lo = f32_to_e2m1_with(blk[2 * j] * inv, policy.rounding);
                    let hi = f32_to_e2m1_with(blk[2 * j + 1] * inv, policy.rounding);
                    weight[(off + 2 * j) / 2] = lo | (hi << 4);
                }
            }
        }
        Ok(Self { rows, cols, weight, weight_scale })
    }

    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let n = self.rows * self.cols;
        let mut out = vec![0f32; n];
        requant_io::blockfloat::dequant_mxfp4_ocp(&self.weight, &self.weight_scale, n, &mut out)?;
        Ok(out)
    }

    /// The requant-internal contiguous container for `RQ_TYPE_MXFP4_OCP`: element bytes followed
    /// by scale bytes, total `n/32 · 17` — matching `block_layout(RQ_TYPE_MXFP4_OCP)`.
    pub fn to_packed(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.weight.len() + self.weight_scale.len());
        v.extend_from_slice(&self.weight);
        v.extend_from_slice(&self.weight_scale);
        v
    }
}

// ---------------------------------------------------------------------------
// NVFP4 — the Blackwell fast path
// ---------------------------------------------------------------------------

/// The three tensors a vLLM/compressed-tensors NVFP4 checkpoint stores per linear.
///
/// Emit these under the names `{prefix}.weight`, `{prefix}.weight_scale`,
/// `{prefix}.weight_scale_2`; [`requant_io::load_nvfp4_linear`] reads exactly that triple back.
pub struct NvFp4Checkpoint {
    pub rows: usize,
    pub cols: usize,
    /// `[rows, cols/2]` uint8. Even element in the low nibble.
    pub weight: Vec<u8>,
    /// `[rows, cols/16]` float8_e4m3fn, linear order (**not** swizzled).
    pub weight_scale: Vec<u8>,
    /// fp32 scalar, `amax / (6 · 448)`.
    pub weight_scale_2: f32,
}

impl NvFp4Checkpoint {
    pub fn quantize(
        x: &[f32],
        rows: usize,
        cols: usize,
        im: Option<&[f32]>,
        policy: &MxfpPolicy,
    ) -> Result<Self> {
        if cols % NVFP4_BLOCK != 0 {
            bail!("NVFP4: cols {cols} not divisible by block {NVFP4_BLOCK}");
        }
        if x.len() != rows * cols {
            bail!("NVFP4: data len {} != {rows}×{cols}", x.len());
        }
        if let Some(w) = im {
            if w.len() < cols {
                bail!("NVFP4: imatrix length {} < cols {cols}", w.len());
            }
        }
        let tensor_amax = amax(x);
        // Guard the degenerate all-zero tensor: a zero global scale would make every block scale
        // NaN (0/0) and poison the whole weight.
        let weight_scale_2 = if tensor_amax > 0.0 && tensor_amax.is_finite() {
            tensor_amax / (E2M1_MAX * E4M3_MAX)
        } else {
            1.0
        };

        let nblk_row = cols / NVFP4_BLOCK;
        let mut weight = vec![0u8; rows * cols / 2];
        let mut weight_scale = vec![0u8; rows * nblk_row];

        for r in 0..rows {
            for b in 0..nblk_row {
                let off = r * cols + b * NVFP4_BLOCK;
                let blk = &x[off..off + NVFP4_BLOCK];
                let g = im.map(|w| &w[b * NVFP4_BLOCK..(b + 1) * NVFP4_BLOCK]);

                let code = Self::block_scale_code(blk, g, weight_scale_2, policy);
                weight_scale[r * nblk_row + b] = code;

                let s = e4m3_to_f32(code) * weight_scale_2;
                let inv = if s > 0.0 && s.is_finite() { 1.0 / s } else { 0.0 };
                for j in 0..NVFP4_BLOCK / 2 {
                    let lo = f32_to_e2m1_with(blk[2 * j] * inv, policy.rounding);
                    let hi = f32_to_e2m1_with(blk[2 * j + 1] * inv, policy.rounding);
                    weight[(off + 2 * j) / 2] = lo | (hi << 4);
                }
            }
        }
        Ok(Self { rows, cols, weight, weight_scale, weight_scale_2 })
    }

    /// E4M3 code for one block's scale. RTN is ModelOpt's rule; refinement searches the
    /// neighbouring E4M3 codes, which keeps the result a legal NVFP4 scale.
    fn block_scale_code(
        blk: &[f32],
        g: Option<&[f32]>,
        global: f32,
        policy: &MxfpPolicy,
    ) -> u8 {
        let m = amax(blk);
        if !(m > 0.0) || !m.is_finite() {
            return 0;
        }
        // `amax_b / 6 / weight_scale_2` is ≤ 448 by construction, so this never saturates.
        let c0 = f32_to_e4m3(m / E2M1_MAX / global);
        if !policy.refine_with_imatrix {
            return c0;
        }
        let mut best = c0;
        let mut best_err = block_err(blk, g, e4m3_to_f32(c0) * global, policy.rounding);
        // Walk a few codes in each direction. E4M3 codes are monotone in magnitude within a sign,
        // so ±3 covers roughly ±40% of scale — far more than the RTN choice can be off by.
        for delta in [-3i32, -2, -1, 1, 2, 3] {
            let c = c0 as i32 + delta;
            if !(0..=0x7e).contains(&c) {
                continue;
            }
            let s = e4m3_to_f32(c as u8) * global;
            if !(s > 0.0) || !s.is_finite() {
                continue;
            }
            let err = block_err(blk, g, s, policy.rounding);
            if err < best_err {
                best_err = err;
                best = c as u8;
            }
        }
        best
    }

    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let n = self.rows * self.cols;
        let mut out = vec![0f32; n];
        requant_io::blockfloat::dequant_nvfp4(
            &self.weight,
            &self.weight_scale,
            self.weight_scale_2,
            n,
            &mut out,
        )?;
        Ok(out)
    }

    /// The requant-internal contiguous container for `RQ_TYPE_NVFP4`: element bytes followed by
    /// block-scale bytes, total `n/16 · 9` — matching `block_layout(RQ_TYPE_NVFP4)`.
    ///
    /// `weight_scale_2` is **not** in here; it is a per-tensor fp32 that the caller carries
    /// alongside. This container exists for byte accounting and internal round-trips, not as a
    /// serving format — the serving format is the three-tensor safetensors group.
    pub fn to_packed(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.weight.len() + self.weight_scale.len());
        v.extend_from_slice(&self.weight);
        v.extend_from_slice(&self.weight_scale);
        v
    }

    pub fn from_packed(bytes: &[u8], rows: usize, cols: usize, weight_scale_2: f32) -> Result<Self> {
        let n = rows * cols;
        if cols % NVFP4_BLOCK != 0 {
            bail!("NVFP4: cols {cols} not divisible by block {NVFP4_BLOCK}");
        }
        let split = n / 2;
        if bytes.len() < split + n / NVFP4_BLOCK {
            bail!("NVFP4: packed buffer {} too small for {rows}×{cols}", bytes.len());
        }
        Ok(Self {
            rows,
            cols,
            weight: bytes[..split].to_vec(),
            weight_scale: bytes[split..split + n / NVFP4_BLOCK].to_vec(),
            weight_scale_2,
        })
    }
}

// ---------------------------------------------------------------------------
// FP8 (dense + MX)
// ---------------------------------------------------------------------------

/// Scale granularity for a dense FP8 emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Granularity {
    PerTensor,
    /// One scale per output row — the safest default for weights, and what most exporters use.
    PerChannel,
    /// DeepSeek-style 2-D tiles, typically 128×128.
    Block2d(usize, usize),
}

/// An FP8 linear plus its scale tensor.
pub struct Fp8Checkpoint {
    pub rows: usize,
    pub cols: usize,
    /// `RQ_TYPE_FP8_E4M3` or `RQ_TYPE_FP8_E5M2`.
    pub ggml_type: u32,
    pub weight: Vec<u8>,
    pub weight_scale: Vec<f32>,
    pub granularity: Fp8Granularity,
}

impl Fp8Checkpoint {
    pub fn quantize(
        ggml_type: u32,
        x: &[f32],
        rows: usize,
        cols: usize,
        granularity: Fp8Granularity,
    ) -> Result<Self> {
        let (fmax, encode): (f32, fn(f32) -> u8) = match ggml_type {
            RQ_TYPE_FP8_E4M3 => (requant_io::blockfloat::E4M3_MAX, f32_to_e4m3),
            RQ_TYPE_FP8_E5M2 => (
                requant_io::blockfloat::E5M2_MAX,
                requant_io::blockfloat::f32_to_e5m2,
            ),
            other => bail!("FP8 emit: type {other} is not a dense FP8 type"),
        };
        if x.len() != rows * cols {
            bail!("FP8: data len {} != {rows}×{cols}", x.len());
        }
        let mut weight = vec![0u8; rows * cols];
        let scales: Vec<f32>;
        match granularity {
            Fp8Granularity::PerTensor => {
                let s = (amax(x) / fmax).max(f32::MIN_POSITIVE);
                let inv = 1.0 / s;
                for (i, &v) in x.iter().enumerate() {
                    weight[i] = encode(v * inv);
                }
                scales = vec![s];
            }
            Fp8Granularity::PerChannel => {
                let mut sc = Vec::with_capacity(rows);
                for r in 0..rows {
                    let row = &x[r * cols..(r + 1) * cols];
                    let s = (amax(row) / fmax).max(f32::MIN_POSITIVE);
                    let inv = 1.0 / s;
                    for (j, &v) in row.iter().enumerate() {
                        weight[r * cols + j] = encode(v * inv);
                    }
                    sc.push(s);
                }
                scales = sc;
            }
            Fp8Granularity::Block2d(bh, bw) => {
                if bh == 0 || bw == 0 {
                    bail!("FP8: block dims must be non-zero");
                }
                let sh = rows.div_ceil(bh);
                let sw = cols.div_ceil(bw);
                let mut sc = vec![f32::MIN_POSITIVE; sh * sw];
                for br in 0..sh {
                    for bc in 0..sw {
                        let mut m = 0.0f32;
                        for r in br * bh..((br + 1) * bh).min(rows) {
                            for c in bc * bw..((bc + 1) * bw).min(cols) {
                                let a = x[r * cols + c].abs();
                                if a > m {
                                    m = a;
                                }
                            }
                        }
                        sc[br * sw + bc] = (m / fmax).max(f32::MIN_POSITIVE);
                    }
                }
                for r in 0..rows {
                    for c in 0..cols {
                        let s = sc[(r / bh) * sw + (c / bw)];
                        weight[r * cols + c] = encode(x[r * cols + c] / s);
                    }
                }
                scales = sc;
            }
        }
        Ok(Self { rows, cols, ggml_type, weight, weight_scale: scales, granularity })
    }

    pub fn dequantize(&self) -> Result<Vec<f32>> {
        use requant_io::blockfloat::Fp8Scale;
        let scale = match self.granularity {
            Fp8Granularity::PerTensor => Fp8Scale::PerTensor(self.weight_scale[0]),
            Fp8Granularity::PerChannel => Fp8Scale::PerChannel(&self.weight_scale),
            Fp8Granularity::Block2d(bh, bw) => {
                Fp8Scale::Block2d { scales: &self.weight_scale, bh, bw }
            }
        };
        let mut out = vec![0f32; self.rows * self.cols];
        requant_io::blockfloat::dequant_fp8(
            self.ggml_type,
            &self.weight,
            self.rows,
            self.cols,
            scale,
            &mut out,
        )?;
        Ok(out)
    }
}

/// MXFP8: E4M3 elements with an E8M0 shared scale per 32 elements.
pub struct MxFp8Checkpoint {
    pub rows: usize,
    pub cols: usize,
    pub weight: Vec<u8>,
    /// `[rows, cols/32]` uint8 E8M0.
    pub weight_scale: Vec<u8>,
}

impl MxFp8Checkpoint {
    pub fn quantize(x: &[f32], rows: usize, cols: usize) -> Result<Self> {
        const BLOCK: usize = 32;
        if cols % BLOCK != 0 {
            bail!("MXFP8: cols {cols} not divisible by block {BLOCK}");
        }
        let nblk_row = cols / BLOCK;
        let mut weight = vec![0u8; rows * cols];
        let mut weight_scale = vec![0u8; rows * nblk_row];
        for r in 0..rows {
            for b in 0..nblk_row {
                let off = r * cols + b * BLOCK;
                let blk = &x[off..off + BLOCK];
                let m = amax(blk);
                let e = if m > 0.0 && m.is_finite() {
                    // OCP: emax_elem for E4M3 is 8.
                    (f32_to_e8m0(m) as i32 - requant_io::blockfloat::E4M3_EMAX).clamp(0, 254) as u8
                } else {
                    0
                };
                weight_scale[r * nblk_row + b] = e;
                let s = e8m0_to_f32(e);
                let inv = if s > 0.0 && s.is_finite() { 1.0 / s } else { 0.0 };
                for j in 0..BLOCK {
                    weight[off + j] = f32_to_e4m3(blk[j] * inv);
                }
            }
        }
        Ok(Self { rows, cols, weight, weight_scale })
    }

    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let n = self.rows * self.cols;
        let mut out = vec![0f32; n];
        requant_io::blockfloat::dequant_mxfp8(&self.weight, &self.weight_scale, n, &mut out)?;
        Ok(out)
    }

    pub fn to_packed(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.weight.len() + self.weight_scale.len());
        v.extend_from_slice(&self.weight);
        v.extend_from_slice(&self.weight_scale);
        v
    }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

/// True for the block-float types this module can emit.
pub fn handles(ggml_type: u32) -> bool {
    matches!(
        ggml_type,
        GGML_TYPE_MXFP4
            | RQ_TYPE_MXFP4_OCP
            | RQ_TYPE_NVFP4
            | RQ_TYPE_MXFP8_E4M3
            | RQ_TYPE_FP8_E4M3
            | RQ_TYPE_FP8_E5M2
    )
}

/// Block size this family requires `cols` to be divisible by.
pub fn required_block(ggml_type: u32) -> Option<usize> {
    Some(match ggml_type {
        GGML_TYPE_MXFP4 | RQ_TYPE_MXFP4_OCP | RQ_TYPE_MXFP8_E4M3 => 32,
        RQ_TYPE_NVFP4 => 16,
        RQ_TYPE_FP8_E4M3 | RQ_TYPE_FP8_E5M2 => 1,
        _ => return None,
    })
}

/// Quantize then immediately dequantize — the round-trip the sensitivity/proxy search needs,
/// without forcing every format into a single flat byte buffer.
pub fn roundtrip(
    ggml_type: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    im: Option<&[f32]>,
    policy: &MxfpPolicy,
) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_MXFP4 => {
            let b = quantize_mxfp4_ggml(x, rows, cols, im, policy)?;
            dequantize_mxfp4_ggml(&b, rows, cols)
        }
        RQ_TYPE_MXFP4_OCP => MxFp4Checkpoint::quantize(x, rows, cols, im, policy)?.dequantize(),
        RQ_TYPE_NVFP4 => NvFp4Checkpoint::quantize(x, rows, cols, im, policy)?.dequantize(),
        RQ_TYPE_MXFP8_E4M3 => MxFp8Checkpoint::quantize(x, rows, cols)?.dequantize(),
        RQ_TYPE_FP8_E4M3 | RQ_TYPE_FP8_E5M2 => {
            Fp8Checkpoint::quantize(ggml_type, x, rows, cols, Fp8Granularity::PerChannel)?
                .dequantize()
        }
        other => bail!("mxfp::roundtrip: type {other} is not a block-float type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect()
    }

    #[test]
    fn mxfp4_ggml_round_trip_is_close() {
        let cols = 256;
        let x = ramp(cols * 4);
        let p = MxfpPolicy::default();
        let b = quantize_mxfp4_ggml(&x, 4, cols, None, &p).unwrap();
        assert_eq!(b.len(), 4 * (cols / 32) * 17);
        let y = dequantize_mxfp4_ggml(&b, 4, cols).unwrap();
        // MXFP4 is 4.25 bpw with a power-of-two scale; ~10% RMS is the expected regime.
        let num: f64 = x.iter().zip(&y).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = x.iter().map(|a| (*a as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        assert!(rel < 0.25, "MXFP4 relative RMS error {rel} too high");
    }

    #[test]
    fn mxfp4_exact_grid_values_survive() {
        // Values already on the grid at a power-of-two scale must reconstruct exactly.
        let mut x = vec![0f32; 32];
        for (i, v) in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].iter().enumerate() {
            x[i] = *v;
            x[i + 8] = -*v;
        }
        let p = MxfpPolicy { scale_rule: MxScaleRule::NoClip, ..Default::default() };
        let b = quantize_mxfp4_ggml(&x, 1, 32, None, &p).unwrap();
        let y = dequantize_mxfp4_ggml(&b, 1, 32).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn nvfp4_layout_matches_the_documented_convention() {
        let cols = 64;
        let x = ramp(cols * 2);
        let ck = NvFp4Checkpoint::quantize(&x, 2, cols, None, &MxfpPolicy::default()).unwrap();
        assert_eq!(ck.weight.len(), 2 * cols / 2, "one byte per two elements");
        assert_eq!(ck.weight_scale.len(), 2 * cols / 16, "one E4M3 scale per 16-element block");
        // weight_scale_2 = amax / (6 * 448)
        let m = x.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!((ck.weight_scale_2 - m / (6.0 * 448.0)).abs() < 1e-12);
        // Every block scale must fit E4M3's finite range (that's what the global scale buys).
        for &s in &ck.weight_scale {
            assert_ne!(s & 0x7f, 0x7f, "block scale encoded as NaN");
        }
    }

    #[test]
    fn nvfp4_packed_container_matches_block_layout() {
        let cols = 64;
        let rows = 3;
        let x = ramp(cols * rows);
        let ck = NvFp4Checkpoint::quantize(&x, rows, cols, None, &MxfpPolicy::default()).unwrap();
        let packed = ck.to_packed();
        let (block, bytes) = requant_io::block_layout(RQ_TYPE_NVFP4).unwrap();
        assert_eq!(packed.len(), rows * cols / block * bytes);
        let back = NvFp4Checkpoint::from_packed(&packed, rows, cols, ck.weight_scale_2).unwrap();
        assert_eq!(back.dequantize().unwrap(), ck.dequantize().unwrap());
    }

    #[test]
    fn nvfp4_beats_mxfp4_on_the_same_data() {
        // NVFP4's E4M3 block scale + smaller block is strictly more expressive than MXFP4's
        // power-of-two scale over 32. If this ever inverts, a scale convention is wrong.
        let cols = 512;
        let x = ramp(cols);
        let p = MxfpPolicy::default();
        let err = |y: &[f32]| -> f64 {
            x.iter().zip(y).map(|(a, b)| ((a - b) as f64).powi(2)).sum()
        };
        let e_nv = err(&roundtrip(RQ_TYPE_NVFP4, &x, 1, cols, None, &p).unwrap());
        let e_mx = err(&roundtrip(GGML_TYPE_MXFP4, &x, 1, cols, None, &p).unwrap());
        assert!(e_nv < e_mx, "NVFP4 err {e_nv} should beat MXFP4 err {e_mx}");
    }

    #[test]
    fn imatrix_refinement_never_increases_weighted_error() {
        let cols = 256;
        let x = ramp(cols);
        // Importance concentrated on a few channels — the case refinement should exploit.
        let im: Vec<f32> = (0..cols).map(|i| if i % 17 == 0 { 100.0 } else { 0.01 }).collect();
        let plain = MxfpPolicy::default();
        let refined = MxfpPolicy { refine_with_imatrix: true, ..Default::default() };
        let werr = |y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .zip(&im)
                .map(|((a, b), g)| ((a - b) as f64).powi(2) * (*g as f64))
                .sum()
        };
        for ty in [RQ_TYPE_NVFP4, GGML_TYPE_MXFP4] {
            let a = werr(&roundtrip(ty, &x, 1, cols, Some(&im), &plain).unwrap());
            let b = werr(&roundtrip(ty, &x, 1, cols, Some(&im), &refined).unwrap());
            assert!(b <= a * 1.0000001, "type {ty}: refinement made it worse ({a} -> {b})");
        }
    }

    #[test]
    fn fp8_per_channel_round_trips_within_e4m3_resolution() {
        let (rows, cols) = (4, 64);
        let x = ramp(rows * cols);
        let ck =
            Fp8Checkpoint::quantize(RQ_TYPE_FP8_E4M3, &x, rows, cols, Fp8Granularity::PerChannel)
                .unwrap();
        assert_eq!(ck.weight_scale.len(), rows);
        let y = ck.dequantize().unwrap();
        for (a, b) in x.iter().zip(&y) {
            // E4M3 has 3 mantissa bits: ~6% worst-case relative step, plus the row-scale headroom.
            assert!((a - b).abs() <= a.abs() * 0.07 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn fp8_block2d_handles_a_ragged_tail() {
        // rows/cols not multiples of the tile: the last tiles are partial.
        let (rows, cols) = (5, 6);
        let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32) - 15.0).collect();
        let ck =
            Fp8Checkpoint::quantize(RQ_TYPE_FP8_E4M3, &x, rows, cols, Fp8Granularity::Block2d(2, 4))
                .unwrap();
        assert_eq!(ck.weight_scale.len(), 3 * 2);
        let y = ck.dequantize().unwrap();
        assert_eq!(y.len(), rows * cols);
    }

    #[test]
    fn all_zero_tensor_does_not_produce_nan() {
        let x = vec![0f32; 64];
        for ty in [RQ_TYPE_NVFP4, GGML_TYPE_MXFP4, RQ_TYPE_FP8_E4M3, RQ_TYPE_MXFP8_E4M3] {
            let y = roundtrip(ty, &x, 1, 64, None, &MxfpPolicy::default()).unwrap();
            assert!(y.iter().all(|v| *v == 0.0), "type {ty} produced {:?}", &y[..4]);
        }
    }
}
