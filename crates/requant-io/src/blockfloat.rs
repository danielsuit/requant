//! Block-scaled float formats: FP4 (E2M1), FP8 (E4M3 / E5M2), and the MX / NVFP4 block layouts.
//!
//! This is the **read path** (DESIGN §4, "Block-scaled float formats"): element codecs plus
//! dequant kernels that produce the plain `f32` that `requant_quant::quantize_tensor` already
//! consumes. Emitting these formats lives in `requant-quant::mxfp`.
//!
//! # The three layouts that matter
//!
//! | layout | element | block | shared scale | second-level scale |
//! |---|---|---|---|---|
//! | **MXFP4** (OCP MX spec) | E2M1 | 32 | E8M0 (power-of-two), 1 byte | — |
//! | **NVFP4** (NVIDIA) | E2M1 | 16 | E4M3 (FP8), 1 byte | per-tensor FP32 |
//! | **FP8 dense** | E4M3 / E5M2 | 1 | per-tensor / per-channel / 2-D block | — |
//!
//! GGML now has self-contained MXFP4 and NVFP4 tensor types. The ModelOpt/vLLM NVFP4 checkpoint
//! layout is still distinct: it and dense FP8 live in safetensors, where block/channel scales are
//! **separate tensors** — hence the `ScaleSource` indirection below.
//!
//! # Nibble order is a real compatibility trap
//!
//! Two FP4 codes share a byte, and the two ecosystems disagree about which:
//!
//! - **ggml** (`quantize_row_mxfp4_ref`) packs `qs[j] = q[j] | (q[j + 16] << 4)` — the same
//!   split-half interleave it uses for `Q4_0`. See [`NibbleOrder::SplitHalf`].
//! - **NVFP4 / OCP reference / vLLM** pack adjacent pairs: `byte[j] = q[2j] | (q[2j+1] << 4)`.
//!   See [`NibbleOrder::Adjacent`].
//!
//! Getting this wrong produces a checkpoint that loads, runs, and emits garbage, so the order is
//! an explicit parameter everywhere rather than a per-format assumption.

use anyhow::{bail, Result};

// ---------------------------------------------------------------------------
// Type ids
// ---------------------------------------------------------------------------

/// `GGML_TYPE_MXFP4` in llama.cpp/ggml (added for gpt-oss). Block 32, 17 bytes:
/// `{ uint8_t e; uint8_t qs[16]; }`.
///
/// NOTE: this id is stable in current ggml but was assigned well after the 0.11.0 we pin the
/// k-quant oracle against. Verify it against the ggml build you intend to serve with before
/// emitting a GGUF that uses it (`ggml_type_name()` on the C side, or the `ggml-oracle-mxfp4`
/// test feature).
pub const GGML_TYPE_MXFP4: u32 = 39;
/// Current ggml's self-contained NVFP4 block (four 16-value sub-blocks per 64-value block).
/// This is distinct from ModelOpt/vLLM NVFP4 checkpoints, which also carry a tensor-level
/// `weight_scale_2` sidecar and use [`RQ_TYPE_NVFP4`].
pub const GGML_TYPE_NVFP4: u32 = 40;

/// Base of the requant-internal type-id namespace. These ids describe formats that have **no
/// ggml type** — they exist so the IR, the recipe language, and the byte-budget search can name
/// and cost NVFP4 / FP8 tensors uniformly. They must never be written into a GGUF tensor header;
/// see [`is_gguf_type`].
pub const RQ_TYPE_BASE: u32 = 0x8000;

/// NVFP4: E2M1 elements, block 16, per-block E4M3 scale, per-tensor FP32 second-level scale.
pub const RQ_TYPE_NVFP4: u32 = RQ_TYPE_BASE;
/// Dense FP8 E4M3 (`float8_e4m3fn`), scale supplied out-of-band.
pub const RQ_TYPE_FP8_E4M3: u32 = RQ_TYPE_BASE + 1;
/// Dense FP8 E5M2, scale supplied out-of-band.
pub const RQ_TYPE_FP8_E5M2: u32 = RQ_TYPE_BASE + 2;
/// MXFP8: E4M3 elements, block 32, E8M0 shared scale (OCP MX).
pub const RQ_TYPE_MXFP8_E4M3: u32 = RQ_TYPE_BASE + 3;
/// MXFP4 in OCP layout (adjacent nibbles, sidecar E8M0 scales) as opposed to ggml's packing.
pub const RQ_TYPE_MXFP4_OCP: u32 = RQ_TYPE_BASE + 4;

/// True for ids that are real ggml types and may appear in a GGUF tensor header.
pub fn is_gguf_type(ty: u32) -> bool {
    ty < RQ_TYPE_BASE
}

/// True for the FP4-family formats — the ones where a quant→quant requant has no higher-precision
/// master underneath it (DESIGN §4). Callers use this to decide how loudly to warn.
pub fn is_fp4_family(ty: u32) -> bool {
    matches!(ty, GGML_TYPE_MXFP4 | GGML_TYPE_NVFP4 | RQ_TYPE_NVFP4 | RQ_TYPE_MXFP4_OCP)
}

/// True for the FP8-family formats.
pub fn is_fp8_family(ty: u32) -> bool {
    matches!(ty, RQ_TYPE_FP8_E4M3 | RQ_TYPE_FP8_E5M2 | RQ_TYPE_MXFP8_E4M3)
}

/// Which half of a byte holds the even-indexed FP4 code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleOrder {
    /// `byte[j] = q[2j] | (q[2j+1] << 4)` — NVFP4, OCP reference, vLLM/compressed-tensors.
    Adjacent,
    /// `byte[j] = q[j] | (q[j + block/2] << 4)` — ggml's `Q4_0`/`MXFP4` interleave.
    SplitHalf,
}

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// Largest magnitude representable in E2M1.
pub const E2M1_MAX: f32 = 6.0;
/// Largest finite magnitude in E4M3 (`float8_e4m3fn`: no infinities, 0x7F/0xFF are NaN).
pub const E4M3_MAX: f32 = 448.0;
/// Largest finite magnitude in E5M2.
pub const E5M2_MAX: f32 = 57344.0;
/// Max *exponent* of a normal E2M1 value (6.0 = 1.5 · 2^2). The OCP shared-scale rule is
/// `scale = 2^(floor(log2(amax)) - E2M1_EMAX)`.
pub const E2M1_EMAX: i32 = 2;
/// Max exponent of a normal E4M3 value (448 = 1.75 · 2^8).
pub const E4M3_EMAX: i32 = 8;

/// The 16 E2M1 values, indexed by the 4-bit code (bit 3 = sign).
pub const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Magnitudes of the 8 non-negative E2M1 codes.
const E2M1_ABS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Midpoints between consecutive `E2M1_ABS` entries — the decision boundaries.
const E2M1_CUT: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
/// For a tie at `E2M1_CUT[k]` (between codes `k` and `k+1`), the code with the even mantissa.
const E2M1_TIE_EVEN: [u8; 7] = [0, 2, 2, 4, 4, 6, 6];

/// Tie-breaking rule when a value lands exactly on an FP4 decision boundary.
///
/// The two ecosystems differ, and the difference is observable in the packed bytes:
///
/// - [`HalfDown`](E2m1Rounding::HalfDown) — ties toward the smaller magnitude. This is what
///   ggml's `best_index_mxfp4` does (linear scan, `err < best_err` keeps the first/lower index)
///   **and** what NVIDIA ModelOpt's `_cast_fp4` does (strict `>` against the cutoff table). It is
///   therefore the right default for matching a *checkpoint* produced by either.
/// - [`NearestEven`](E2m1Rounding::NearestEven) — IEEE round-to-nearest-even, which is what the
///   Blackwell `cvt.rn.satfinite.e2m1x2.f32` PTX instruction does. That instruction quantizes
///   *activations* at runtime; use this mode only when you are deliberately matching the hardware
///   path rather than the reference exporter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum E2m1Rounding {
    #[default]
    HalfDown,
    NearestEven,
}

// ---------------------------------------------------------------------------
// Element codecs
// ---------------------------------------------------------------------------

/// `2^n` as an `f32`, exact across the subnormal range. Cheaper and more precise than `powi`.
#[inline]
pub fn exp2i(n: i32) -> f32 {
    if n > 127 {
        return f32::INFINITY;
    }
    if n >= -126 {
        return f32::from_bits(((n + 127) as u32) << 23);
    }
    if n < -149 {
        return 0.0;
    }
    // Subnormal: the mantissa bit at position (n + 149) is 2^n.
    f32::from_bits(1u32 << (n + 149))
}

/// Decode a 4-bit E2M1 code (sign in bit 3).
#[inline]
pub fn e2m1_to_f32(code: u8) -> f32 {
    E2M1_TABLE[(code & 0x0f) as usize]
}

/// Encode to a 4-bit E2M1 code with the default ([`E2m1Rounding::HalfDown`]) tie rule.
#[inline]
pub fn f32_to_e2m1(v: f32) -> u8 {
    f32_to_e2m1_with(v, E2m1Rounding::HalfDown)
}

/// Encode to a 4-bit E2M1 code, saturating at ±6.
#[inline]
pub fn f32_to_e2m1_with(v: f32, rounding: E2m1Rounding) -> u8 {
    let sign: u8 = if v.is_sign_negative() { 0x8 } else { 0x0 };
    let a = v.abs();
    if a.is_nan() {
        // E2M1 has no NaN; saturate rather than silently producing zero.
        return sign | 0x7;
    }
    for (k, &cut) in E2M1_CUT.iter().enumerate() {
        if a < cut {
            return sign | k as u8;
        }
        if a == cut {
            return sign
                | match rounding {
                    E2m1Rounding::HalfDown => k as u8,
                    E2m1Rounding::NearestEven => E2M1_TIE_EVEN[k],
                };
        }
    }
    sign | 0x7
}

/// Decode E4M3 (`float8_e4m3fn`: bias 7, no infinities, 0x7F/0xFF = NaN).
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 0x07) as f32;
    if e == 0x0f && (b & 0x07) == 0x07 {
        return f32::NAN;
    }
    if e == 0 {
        // Subnormal: m · 2^-3 · 2^(1-7) = m · 2^-9.
        sign * m * exp2i(-9)
    } else {
        sign * (1.0 + m / 8.0) * exp2i(e - 7)
    }
}

/// Encode to E4M3, round-to-nearest-even, saturating at ±448 (`satfinite` semantics).
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0x00 };
    let a = v.abs();
    if a >= E4M3_MAX {
        return sign | 0x7e; // largest finite: e=15, m=6
    }
    // Smallest subnormal is 2^-9; anything under half of that rounds to zero.
    if a < exp2i(-10) {
        return sign;
    }
    let bits = a.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127;
    if exp >= -6 {
        // Normal in E4M3: keep 3 mantissa bits out of 23, round-to-nearest-even.
        let mant = bits & 0x007f_ffff;
        let mut m3 = mant >> 20;
        let rem = mant & 0x000f_ffff;
        const HALF: u32 = 0x0008_0000;
        if rem > HALF || (rem == HALF && (m3 & 1) == 1) {
            m3 += 1;
        }
        if m3 == 8 {
            m3 = 0;
            exp += 1;
        }
        if exp > E4M3_EMAX || (exp == E4M3_EMAX && m3 > 6) {
            return sign | 0x7e;
        }
        sign | (((exp + 7) as u8) << 3) | m3 as u8
    } else {
        // Subnormal in E4M3: value = m · 2^-9, m ∈ 0..=7.
        let scaled = a * exp2i(9);
        let m = round_half_even(scaled) as u32;
        if m >= 8 {
            // Rounded up into the smallest normal (e=1, m=0) = 2^-6.
            return sign | (1 << 3);
        }
        sign | m as u8
    }
}

/// Decode ggml's unsigned E4M3 scale. ggml's E2M1 table is doubled, so the decoded scale is
/// halved to keep the product equal to ordinary E2M1 × E4M3.
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0x0f) as i32;
    let man = (x & 7) as f32;
    let raw = if exp == 0 {
        man * exp2i(-9)
    } else {
        (1.0 + man / 8.0) * exp2i(exp - 7)
    };
    raw * 0.5
}

/// Encode a positive ggml UE4M3 scale, matching `ggml_fp32_to_ue4m3`.
pub fn f32_to_ue4m3(mut x: f32) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    x = x.min(448.0);
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let fp32_man = ((bits >> 20) & 7) as i32;
    let mut exp = fp32_exp + 7;
    if exp <= 0 {
        return (x * 512.0 + 0.5).floor().clamp(0.0, 7.0) as u8;
    }
    if exp >= 15 {
        return 0x7e;
    }
    let mut man = fp32_man + ((bits >> 19) & 1) as i32;
    if man > 7 {
        man = 0;
        exp += 1;
    }
    if exp >= 15 { 0x7e } else { ((exp << 3) | man) as u8 }
}

/// Decode E5M2 (bias 15, IEEE-like: has infinities and NaN).
#[inline]
pub fn e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = ((b >> 2) & 0x1f) as i32;
    let m = (b & 0x03) as f32;
    if e == 0x1f {
        return if m == 0.0 { sign * f32::INFINITY } else { f32::NAN };
    }
    if e == 0 {
        // Subnormal: m · 2^-2 · 2^(1-15) = m · 2^-16.
        sign * m * exp2i(-16)
    } else {
        sign * (1.0 + m / 4.0) * exp2i(e - 15)
    }
}

/// Encode to E5M2, round-to-nearest-even, saturating at ±57344.
pub fn f32_to_e5m2(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0x00 };
    let a = v.abs();
    if a >= E5M2_MAX {
        return sign | 0x7b; // largest finite: e=30, m=3
    }
    // Smallest subnormal is 2^-16.
    if a < exp2i(-17) {
        return sign;
    }
    let bits = a.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127;
    if exp >= -14 {
        let mant = bits & 0x007f_ffff;
        let mut m2 = mant >> 21;
        let rem = mant & 0x001f_ffff;
        const HALF: u32 = 0x0010_0000;
        if rem > HALF || (rem == HALF && (m2 & 1) == 1) {
            m2 += 1;
        }
        if m2 == 4 {
            m2 = 0;
            exp += 1;
        }
        if exp > 15 || (exp == 15 && m2 > 3) {
            return sign | 0x7b;
        }
        sign | (((exp + 15) as u8) << 2) | m2 as u8
    } else {
        let scaled = a * exp2i(16);
        let m = round_half_even(scaled) as u32;
        if m >= 4 {
            return sign | (1 << 2);
        }
        sign | m as u8
    }
}

/// Decode an E8M0 shared scale to `2^(e - 127)`, OCP semantics (0xFF = NaN).
#[inline]
pub fn e8m0_to_f32(e: u8) -> f32 {
    if e == 0xff {
        return f32::NAN;
    }
    exp2i(e as i32 - 127)
}

/// Decode an E8M0 shared scale to `2^(e - 128)` — ggml's `GGML_E8M0_TO_FP32_HALF`.
///
/// ggml stores the MXFP4 element table as *doubled* integers (`{0,1,2,3,4,6,8,12}` rather than
/// `{0,.5,1,1.5,2,3,4,6}`), so it halves the shared scale to compensate. The product is identical
/// to `e2m1_to_f32(q) * e8m0_to_f32(e)`; this exists so the ggml kernel can be transcribed
/// literally. ggml has no NaN encoding here, so 0xFF decodes to `2^127`.
#[inline]
pub fn e8m0_to_f32_half(e: u8) -> f32 {
    exp2i(e as i32 - 128)
}

/// Encode a positive scale as E8M0 by *truncating* to a power of two: `e = floor(log2(v)) + 127`.
///
/// Non-positive / non-finite inputs encode as 0 (the smallest representable scale) rather than
/// NaN — a NaN scale poisons an entire block, and a degenerate block is better than a lost one.
#[inline]
pub fn f32_to_e8m0(v: f32) -> u8 {
    if !(v > 0.0) || !v.is_finite() {
        return 0;
    }
    let bits = v.to_bits();
    let e = ((bits >> 23) & 0xff) as i32;
    if e == 0 {
        // f32 subnormal: far below anything a real scale reaches; clamp to the floor.
        return 0;
    }
    e.clamp(0, 254) as u8
}

/// Round half to even. `f32::round` is half-away-from-zero, which biases block scales upward.
#[inline]
fn round_half_even(x: f32) -> f32 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

// ---------------------------------------------------------------------------
// Scale plumbing
// ---------------------------------------------------------------------------

/// How the per-element scale for a dense FP8 tensor is stored.
///
/// GGUF block quants carry their scale inside the block; safetensors FP8 checkpoints do not —
/// the scale is a sibling tensor whose granularity varies by producer. All four shapes below
/// appear in the wild, so the reader takes them explicitly instead of guessing.
#[derive(Debug, Clone, Copy)]
pub enum Fp8Scale<'a> {
    /// Values are already in their final units.
    Unit,
    /// One scale for the whole tensor (`weight_scale` scalar).
    PerTensor(f32),
    /// One scale per output row (`weight_scale` of shape `[rows]`).
    PerChannel(&'a [f32]),
    /// DeepSeek-style 2-D blocking: `weight_scale_inv` of shape
    /// `[ceil(rows/bh), ceil(cols/bw)]`, typically 128×128.
    Block2d { scales: &'a [f32], bh: usize, bw: usize },
}

impl Fp8Scale<'_> {
    /// Scale applying to element `(r, c)` of a `rows × cols` tensor.
    #[inline]
    fn at(&self, r: usize, c: usize, cols: usize) -> f32 {
        match self {
            Fp8Scale::Unit => 1.0,
            Fp8Scale::PerTensor(s) => *s,
            Fp8Scale::PerChannel(s) => s.get(r).copied().unwrap_or(1.0),
            Fp8Scale::Block2d { scales, bh, bw } => {
                let sw = (cols + bw - 1) / bw;
                let idx = (r / bh) * sw + (c / bw);
                scales.get(idx).copied().unwrap_or(1.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dequant kernels
// ---------------------------------------------------------------------------

/// Unpack `n` FP4 codes from `bytes` in the given nibble order.
///
/// `block` is only consulted for [`NibbleOrder::SplitHalf`], where the interleave pairs element
/// `j` with element `j + block/2` *within a block*.
#[inline]
fn unpack_fp4(bytes: &[u8], n: usize, block: usize, order: NibbleOrder, out: &mut [u8]) {
    match order {
        NibbleOrder::Adjacent => {
            for i in 0..n {
                let b = bytes[i / 2];
                out[i] = if i % 2 == 0 { b & 0x0f } else { b >> 4 };
            }
        }
        NibbleOrder::SplitHalf => {
            let half = block / 2;
            let nb = n / block;
            for blk in 0..nb {
                let src = &bytes[blk * half..blk * half + half];
                let dst = &mut out[blk * block..blk * block + block];
                for j in 0..half {
                    dst[j] = src[j] & 0x0f;
                    dst[j + half] = src[j] >> 4;
                }
            }
        }
    }
}

/// Dequantize one ggml `block_mxfp4` row: blocks of 32, each `{ e: u8, qs: [u8; 16] }`,
/// split-half nibbles, halved E8M0 scale.
pub fn dequant_mxfp4_ggml_row(bytes: &[u8], out: &mut [f32]) {
    const BLOCK: usize = 32;
    const BYTES: usize = 17;
    let nb = out.len() / BLOCK;
    for b in 0..nb {
        let blk = &bytes[b * BYTES..b * BYTES + BYTES];
        // ggml's table is 2× the E2M1 values and its scale is halved; the product matches OCP.
        let d = e8m0_to_f32_half(blk[0]);
        let dst = &mut out[b * BLOCK..b * BLOCK + BLOCK];
        for j in 0..BLOCK / 2 {
            let byte = blk[1 + j];
            dst[j] = 2.0 * e2m1_to_f32(byte & 0x0f) * d;
            dst[j + BLOCK / 2] = 2.0 * e2m1_to_f32(byte >> 4) * d;
        }
    }
}

/// Dequantize current ggml `block_nvfp4` bytes (64 values, four UE4M3 scales, 32 E2M1 bytes).
pub fn dequant_nvfp4_ggml_row(bytes: &[u8], out: &mut [f32]) {
    const BLOCK: usize = 64;
    const SUB: usize = 16;
    const BYTES: usize = 36;
    for (b, dst) in out.chunks_exact_mut(BLOCK).enumerate() {
        let src = &bytes[b * BYTES..(b + 1) * BYTES];
        for s in 0..BLOCK / SUB {
            let d = ue4m3_to_f32(src[s]);
            for j in 0..SUB / 2 {
                let q = src[4 + s * (SUB / 2) + j];
                dst[s * SUB + j] = 2.0 * e2m1_to_f32(q & 15) * d;
                dst[s * SUB + j + SUB / 2] = 2.0 * e2m1_to_f32(q >> 4) * d;
            }
        }
    }
}

/// Dequantize MXFP4 in OCP layout: element bytes and E8M0 scale bytes in separate buffers,
/// adjacent nibble packing, block 32.
///
/// `data.len()` must be `n/2`, `scales.len()` must be `n/32`.
pub fn dequant_mxfp4_ocp(data: &[u8], scales: &[u8], n: usize, out: &mut [f32]) -> Result<()> {
    const BLOCK: usize = 32;
    if n % BLOCK != 0 {
        bail!("mxfp4: element count {n} not divisible by block {BLOCK}");
    }
    if data.len() < n / 2 {
        bail!("mxfp4: {} data bytes < needed {}", data.len(), n / 2);
    }
    if scales.len() < n / BLOCK {
        bail!("mxfp4: {} scale bytes < needed {}", scales.len(), n / BLOCK);
    }
    if out.len() < n {
        bail!("mxfp4: output buffer {} < {n}", out.len());
    }
    let mut codes = vec![0u8; n];
    unpack_fp4(data, n, BLOCK, NibbleOrder::Adjacent, &mut codes);
    for b in 0..n / BLOCK {
        let d = e8m0_to_f32(scales[b]);
        // An all-NaN block would silently propagate; treat a NaN shared scale as zero, which is
        // what the OCP spec's "block is NaN" case degenerates to for weights.
        let d = if d.is_finite() { d } else { 0.0 };
        for j in 0..BLOCK {
            out[b * BLOCK + j] = e2m1_to_f32(codes[b * BLOCK + j]) * d;
        }
    }
    Ok(())
}

/// Dequantize NVFP4: E2M1 elements in blocks of 16, one E4M3 scale byte per block, and a single
/// per-tensor FP32 `weight_scale_2`.
///
/// Reconstruction is `w = e2m1(q) · e4m3(block_scale) · global_scale`, which is exactly what
/// NVIDIA ModelOpt's exporter inverts and what vLLM's `cutlass_scaled_fp4_mm` reconstructs
/// (the global scale arrives there folded into `alpha`).
///
/// `data.len()` must be `n/2`, `block_scales.len()` must be `n/16`.
pub fn dequant_nvfp4(
    data: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    const BLOCK: usize = 16;
    if n % BLOCK != 0 {
        bail!("nvfp4: element count {n} not divisible by block {BLOCK}");
    }
    if data.len() < n / 2 {
        bail!("nvfp4: {} data bytes < needed {}", data.len(), n / 2);
    }
    if block_scales.len() < n / BLOCK {
        bail!("nvfp4: {} scale bytes < needed {}", block_scales.len(), n / BLOCK);
    }
    if out.len() < n {
        bail!("nvfp4: output buffer {} < {n}", out.len());
    }
    let mut codes = vec![0u8; n];
    unpack_fp4(data, n, BLOCK, NibbleOrder::Adjacent, &mut codes);
    for b in 0..n / BLOCK {
        let s = e4m3_to_f32(block_scales[b]);
        let s = if s.is_finite() { s * global_scale } else { 0.0 };
        for j in 0..BLOCK {
            out[b * BLOCK + j] = e2m1_to_f32(codes[b * BLOCK + j]) * s;
        }
    }
    Ok(())
}

/// Dequantize MXFP8 (E4M3 elements, block 32, E8M0 shared scale in a sidecar buffer).
pub fn dequant_mxfp8(data: &[u8], scales: &[u8], n: usize, out: &mut [f32]) -> Result<()> {
    const BLOCK: usize = 32;
    if n % BLOCK != 0 {
        bail!("mxfp8: element count {n} not divisible by block {BLOCK}");
    }
    if data.len() < n || scales.len() < n / BLOCK || out.len() < n {
        bail!("mxfp8: buffer too small (data {}, scales {}, out {})", data.len(), scales.len(), out.len());
    }
    for b in 0..n / BLOCK {
        let d = e8m0_to_f32(scales[b]);
        let d = if d.is_finite() { d } else { 0.0 };
        for j in 0..BLOCK {
            out[b * BLOCK + j] = e4m3_to_f32(data[b * BLOCK + j]) * d;
        }
    }
    Ok(())
}

/// Dequantize a dense FP8 tensor (`RQ_TYPE_FP8_E4M3` / `RQ_TYPE_FP8_E5M2`) with an out-of-band
/// scale of any of the four supported granularities.
pub fn dequant_fp8(
    ggml_type: u32,
    data: &[u8],
    rows: usize,
    cols: usize,
    scale: Fp8Scale<'_>,
    out: &mut [f32],
) -> Result<()> {
    let n = rows * cols;
    if data.len() < n {
        bail!("fp8: {} bytes < needed {n}", data.len());
    }
    if out.len() < n {
        bail!("fp8: output buffer {} < {n}", out.len());
    }
    let decode: fn(u8) -> f32 = match ggml_type {
        RQ_TYPE_FP8_E4M3 => e4m3_to_f32,
        RQ_TYPE_FP8_E5M2 => e5m2_to_f32,
        other => bail!("dequant_fp8: type {other} is not a dense FP8 type"),
    };
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            out[i] = decode(data[i]) * scale.at(r, c, cols);
        }
    }
    Ok(())
}

/// Where a block-float tensor's scales come from, for the one-shot [`dequantize_blockfloat`] entry
/// point. GGUF-resident formats interleave them; safetensors-resident formats do not.
#[derive(Debug, Clone, Copy)]
pub enum ScaleSource<'a> {
    /// Scales live inside the element blocks (ggml MXFP4).
    Interleaved,
    /// Scales are a sibling byte tensor: E8M0 for MX, E4M3 for NVFP4. `global` is NVFP4's
    /// per-tensor `weight_scale_2` (ignored by MX formats; pass 1.0).
    Sidecar { scales: &'a [u8], global: f32 },
    /// Dense FP8 with a float scale tensor.
    Float(Fp8Scale<'a>),
}

/// Dequantize any block-float tensor to `f32` in logical (row-major, `cols` contiguous) order.
///
/// This is the single entry point the quantize/search paths call so that ingesting an FP4 or FP8
/// checkpoint looks exactly like ingesting an fp16 one.
pub fn dequantize_blockfloat(
    ggml_type: u32,
    data: &[u8],
    rows: usize,
    cols: usize,
    scales: ScaleSource<'_>,
) -> Result<Vec<f32>> {
    let n = rows * cols;
    let mut out = vec![0f32; n];
    match (ggml_type, scales) {
        (GGML_TYPE_MXFP4, ScaleSource::Interleaved) => {
            if cols % 32 != 0 {
                bail!("mxfp4: cols {cols} not divisible by 32");
            }
            let row_bytes = cols / 32 * 17;
            for r in 0..rows {
                dequant_mxfp4_ggml_row(
                    &data[r * row_bytes..r * row_bytes + row_bytes],
                    &mut out[r * cols..r * cols + cols],
                );
            }
        }
        (GGML_TYPE_NVFP4, ScaleSource::Interleaved) => {
            if cols % 64 != 0 {
                bail!("nvfp4 gguf: cols {cols} not divisible by 64");
            }
            let row_bytes = cols / 64 * 36;
            for r in 0..rows {
                dequant_nvfp4_ggml_row(
                    &data[r * row_bytes..r * row_bytes + row_bytes],
                    &mut out[r * cols..r * cols + cols],
                );
            }
        }
        (RQ_TYPE_MXFP4_OCP, ScaleSource::Sidecar { scales, .. }) => {
            dequant_mxfp4_ocp(data, scales, n, &mut out)?;
        }
        (RQ_TYPE_NVFP4, ScaleSource::Sidecar { scales, global }) => {
            dequant_nvfp4(data, scales, global, n, &mut out)?;
        }
        (RQ_TYPE_MXFP8_E4M3, ScaleSource::Sidecar { scales, .. }) => {
            dequant_mxfp8(data, scales, n, &mut out)?;
        }
        (RQ_TYPE_FP8_E4M3 | RQ_TYPE_FP8_E5M2, ScaleSource::Float(s)) => {
            dequant_fp8(ggml_type, data, rows, cols, s, &mut out)?;
        }
        (ty, _) => bail!(
            "dequantize_blockfloat: type {ty} with this scale source is not a block-float layout \
             (MXFP4 needs Interleaved or Sidecar, NVFP4/MXFP8 need Sidecar, dense FP8 needs Float)"
        ),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_round_trips_every_code() {
        for code in 0u8..16 {
            let v = e2m1_to_f32(code);
            let back = f32_to_e2m1(v);
            // -0.0 (code 8) re-encodes to code 8 because is_sign_negative() is true for -0.0.
            assert_eq!(e2m1_to_f32(back), v, "code {code} -> {v} -> {back}");
        }
    }

    #[test]
    fn e2m1_saturates_and_rounds() {
        assert_eq!(e2m1_to_f32(f32_to_e2m1(100.0)), 6.0);
        assert_eq!(e2m1_to_f32(f32_to_e2m1(-100.0)), -6.0);
        assert_eq!(e2m1_to_f32(f32_to_e2m1(0.6)), 0.5);
        assert_eq!(e2m1_to_f32(f32_to_e2m1(0.8)), 1.0);
        // Tie at 0.75: HalfDown keeps 0.5, NearestEven promotes to 1.0.
        assert_eq!(e2m1_to_f32(f32_to_e2m1_with(0.75, E2m1Rounding::HalfDown)), 0.5);
        assert_eq!(e2m1_to_f32(f32_to_e2m1_with(0.75, E2m1Rounding::NearestEven)), 1.0);
        // Tie at 5.0: HalfDown keeps 4.0; NearestEven also keeps 4.0 (code 6 is even).
        assert_eq!(e2m1_to_f32(f32_to_e2m1_with(5.0, E2m1Rounding::HalfDown)), 4.0);
        assert_eq!(e2m1_to_f32(f32_to_e2m1_with(5.0, E2m1Rounding::NearestEven)), 4.0);
    }

    #[test]
    fn e4m3_hits_the_documented_extremes() {
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0xfe), -448.0);
        assert!(e4m3_to_f32(0x7f).is_nan());
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        // Smallest subnormal = 2^-9.
        assert_eq!(e4m3_to_f32(0x01), exp2i(-9));
        // Smallest normal = 2^-6.
        assert_eq!(e4m3_to_f32(0x08), exp2i(-6));
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1.0)), 1.0);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1000.0)), 448.0);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(-1000.0)), -448.0);
    }

    #[test]
    fn ue4m3_matches_ggml_scale_convention() {
        for v in [2.0f32.powi(-8), 0.25, 0.5, 1.0, 3.0, 12.0, 448.0] {
            let code = f32_to_ue4m3(v);
            let decoded_twice = 2.0 * ue4m3_to_f32(code);
            assert!(decoded_twice.is_finite() && decoded_twice > 0.0, "{v} -> {code:#x}");
        }
        assert_eq!(f32_to_ue4m3(0.0), 0);
        assert_eq!(ue4m3_to_f32(0), 0.0);
    }

    #[test]
    fn e4m3_round_trips_every_finite_code() {
        for b in 0u8..=255 {
            if (b & 0x7f) == 0x7f {
                continue; // NaN
            }
            let v = e4m3_to_f32(b);
            assert_eq!(f32_to_e4m3(v), b, "byte {b:#04x} -> {v}");
        }
    }

    #[test]
    fn e5m2_round_trips_every_finite_code() {
        for b in 0u8..=255 {
            if (b >> 2) & 0x1f == 0x1f {
                continue; // inf / NaN
            }
            let v = e5m2_to_f32(b);
            assert_eq!(f32_to_e5m2(v), b, "byte {b:#04x} -> {v}");
        }
    }

    #[test]
    fn e8m0_half_is_exactly_half() {
        for e in 1u8..=254 {
            assert_eq!(e8m0_to_f32_half(e) * 2.0, e8m0_to_f32(e), "e={e}");
        }
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(f32_to_e8m0(1.0), 127);
        assert_eq!(f32_to_e8m0(0.5), 126);
        // Truncating: 1.9 has floor(log2) = 0.
        assert_eq!(f32_to_e8m0(1.9), 127);
        assert_eq!(f32_to_e8m0(0.0), 0);
        assert_eq!(f32_to_e8m0(-3.0), 0);
    }

    #[test]
    fn nvfp4_reconstructs_a_hand_built_block() {
        // One block of 16: codes 0..15 with scale 1.0 and global 1.0 reproduce E2M1_TABLE.
        let mut data = vec![0u8; 8];
        for j in 0..8 {
            data[j] = (2 * j) as u8 | (((2 * j + 1) as u8) << 4);
        }
        let scales = [f32_to_e4m3(1.0)];
        let mut out = vec![0f32; 16];
        dequant_nvfp4(&data, &scales, 1.0, 16, &mut out).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], E2M1_TABLE[j], "element {j}");
        }
    }

    #[test]
    fn ggml_mxfp4_uses_split_half_nibbles() {
        // e = 127 -> half-scale 0.5, doubled table -> element scale 1.0.
        let mut blk = vec![0u8; 17];
        blk[0] = 127;
        blk[1] = 0x07; // low nibble: element 0 = code 7 (6.0); high nibble: element 16 = code 0
        let mut out = vec![0f32; 32];
        dequant_mxfp4_ggml_row(&blk, &mut out);
        assert_eq!(out[0], 6.0);
        assert_eq!(out[16], 0.0);
    }

    #[test]
    fn fp8_block2d_indexes_the_right_scale() {
        // 4x4 tensor, 2x2 scale blocks -> 4 scales laid out row-major.
        let scales = [1.0f32, 2.0, 4.0, 8.0];
        let data: Vec<u8> = (0..16).map(|_| f32_to_e4m3(1.0)).collect();
        let mut out = vec![0f32; 16];
        dequant_fp8(
            RQ_TYPE_FP8_E4M3,
            &data,
            4,
            4,
            Fp8Scale::Block2d { scales: &scales, bh: 2, bw: 2 },
            &mut out,
        )
        .unwrap();
        assert_eq!(out[0], 1.0); // (0,0) -> block (0,0)
        assert_eq!(out[2], 2.0); // (0,2) -> block (0,1)
        assert_eq!(out[8], 4.0); // (2,0) -> block (1,0)
        assert_eq!(out[10], 8.0); // (2,2) -> block (1,1)
    }

    #[test]
    fn type_id_namespaces_do_not_overlap() {
        assert!(is_gguf_type(GGML_TYPE_MXFP4));
        assert!(!is_gguf_type(RQ_TYPE_NVFP4));
        assert!(!is_gguf_type(RQ_TYPE_FP8_E4M3));
        assert!(is_fp4_family(RQ_TYPE_NVFP4));
        assert!(!is_fp4_family(RQ_TYPE_FP8_E4M3));
        assert!(is_fp8_family(RQ_TYPE_MXFP8_E4M3));
    }
}
