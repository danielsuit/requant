//! GGML ternary quants for BitNet/TriLM-style tensors.
//!
//! `TQ1_0` packs five balanced ternary values into a byte (plus four tail groups) for
//! 1.6875 bpw. `TQ2_0` uses two bits per ternary value for 2.0625 bpw. The field order and
//! rounding below are direct Rust ports of ggml's reference kernels.

use anyhow::{bail, Result};
use half::f16;

const QK: usize = 256;
const TQ1_QS: usize = 48;
const TQ1_QH: usize = 4;
const TQ1_BYTES: usize = 54;
const TQ2_QS: usize = 64;
const TQ2_BYTES: usize = 66;

#[inline]
fn round_away(v: f32) -> i32 {
    if v >= 0.0 {
        (v + 0.5).floor() as i32
    } else {
        (v - 0.5).ceil() as i32
    }
}

pub fn handles(t: u32) -> bool {
    matches!(t, 34 | 35)
}

pub fn quantize_rows(
    ggml_type: u32,
    x: &[f32],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if cols % QK != 0 {
        bail!("ternary type {ggml_type}: cols {cols} not divisible by {QK}");
    }
    let bpb = if ggml_type == 34 {
        TQ1_BYTES
    } else if ggml_type == 35 {
        TQ2_BYTES
    } else {
        bail!("ternary type {ggml_type}: kernel not implemented");
    };
    let row_bytes = cols / QK * bpb;
    if x.len() < rows * cols || out.len() < rows * row_bytes {
        bail!("ternary type {ggml_type}: input/output buffer too small");
    }
    for r in 0..rows {
        for b in 0..cols / QK {
            let xb = &x[r * cols + b * QK..r * cols + (b + 1) * QK];
            let ob = &mut out[r * row_bytes + b * bpb..r * row_bytes + (b + 1) * bpb];
            if ggml_type == 34 {
                quant_tq1(xb, ob);
            } else {
                quant_tq2(xb, ob);
            }
        }
    }
    Ok(())
}

pub fn dequantize_rows(
    ggml_type: u32,
    bytes: &[u8],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if cols % QK != 0 {
        bail!("ternary type {ggml_type}: cols {cols} not divisible by {QK}");
    }
    let bpb = if ggml_type == 34 {
        TQ1_BYTES
    } else if ggml_type == 35 {
        TQ2_BYTES
    } else {
        bail!("ternary type {ggml_type}: dequant kernel not implemented");
    };
    let row_bytes = cols / QK * bpb;
    if bytes.len() < rows * row_bytes || out.len() < rows * cols {
        bail!("ternary type {ggml_type}: input/output buffer too small");
    }
    for r in 0..rows {
        for b in 0..cols / QK {
            let ib = &bytes[r * row_bytes + b * bpb..r * row_bytes + (b + 1) * bpb];
            let ob = &mut out[r * cols + b * QK..r * cols + (b + 1) * QK];
            if ggml_type == 34 {
                dequant_tq1(ib, ob);
            } else {
                dequant_tq2(ib, ob);
            }
        }
    }
    Ok(())
}

fn quant_tq1(x: &[f32], out: &mut [u8]) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let id = if amax != 0.0 { 1.0 / amax } else { 0.0 };
    let mut pos = 0;
    // 48 bytes = one 32-byte lane followed by one 16-byte lane, five trits per byte.
    for (base, lane) in [(0usize, 32usize), (32, 16)] {
        for m in 0..lane {
            let mut q = 0u8;
            for n in 0..5 {
                let xi = (round_away(x[pos + m + n * lane] * id) + 1).clamp(0, 2) as u8;
                q = q * 3 + xi;
            }
            out[base + m] = (((q as u16 * 256) + 242) / 243) as u8;
        }
        pos += 5 * lane;
    }
    for j in 0..TQ1_QH {
        let mut q = 0u8;
        for m in 0..4 {
            let xi = (round_away(x[pos + j + m * TQ1_QH] * id) + 1).clamp(0, 2) as u8;
            q = q * 3 + xi;
        }
        q *= 3;
        out[TQ1_QS + j] = (((q as u16 * 256) + 242) / 243) as u8;
    }
    out[TQ1_QS + TQ1_QH..TQ1_BYTES].copy_from_slice(&f16::from_f32(amax).to_le_bytes());
}

fn dequant_tq1(bytes: &[u8], out: &mut [f32]) {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let d = f16::from_le_bytes([bytes[TQ1_BYTES - 2], bytes[TQ1_BYTES - 1]]).to_f32();
    let mut pos = 0;
    for (base, lane) in [(0usize, 32usize), (32, 16)] {
        for &p in &POW3 {
            for m in 0..lane {
                let q = bytes[base + m].wrapping_mul(p);
                let xi = ((q as u16 * 3) >> 8) as i16;
                out[pos] = (xi - 1) as f32 * d;
                pos += 1;
            }
        }
    }
    for &p in POW3.iter().take(4) {
        for j in 0..TQ1_QH {
            let q = bytes[TQ1_QS + j].wrapping_mul(p);
            let xi = ((q as u16 * 3) >> 8) as i16;
            out[pos] = (xi - 1) as f32 * d;
            pos += 1;
        }
    }
}

fn quant_tq2(x: &[f32], out: &mut [u8]) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let id = if amax != 0.0 { 1.0 / amax } else { 0.0 };
    for j in (0..TQ2_QS).step_by(32) {
        for m in 0..32 {
            let mut q = 0u8;
            for n in 0..4 {
                let xi = (round_away(x[j * 4 + m + n * 32] * id) + 1).clamp(0, 2) as u8;
                q |= xi << (2 * n);
            }
            out[j + m] = q;
        }
    }
    out[TQ2_QS..TQ2_BYTES].copy_from_slice(&f16::from_f32(amax).to_le_bytes());
}

fn dequant_tq2(bytes: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([bytes[TQ2_QS], bytes[TQ2_QS + 1]]).to_f32();
    let mut pos = 0;
    for j in (0..TQ2_QS).step_by(32) {
        for shift in 0..4 {
            for m in 0..32 {
                let q = (bytes[j + m] >> (2 * shift)) & 3;
                out[pos] = (q as i8 - 1) as f32 * d;
                pos += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_reference_layouts_round_trip_synthetic_rows() {
        let x: Vec<f32> = (0..QK)
            .map(|i| match i % 7 {
                0 | 1 => -1.0,
                2 | 3 => 0.0,
                _ => 1.0,
            })
            .collect();
        for (ty, nbytes) in [(34, TQ1_BYTES), (35, TQ2_BYTES)] {
            let mut packed = vec![0u8; nbytes];
            quantize_rows(ty, &x, &mut packed, 1, QK).unwrap();
            let mut got = vec![0.0; QK];
            dequantize_rows(ty, &packed, &mut got, 1, QK).unwrap();
            assert_eq!(got, x, "type {ty}");
        }
    }
}
