//! Standard block quants: Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1.
//!
//! Bit-exact port of ggml's `quantize_row_*_ref` / `dequantize_row_*` (ggml-quants.c).
//! Each operates on a row split into blocks of 32. Within a block, Q4/Q5 pack the first 16
//! elements into the low nibbles and the second 16 into the high nibbles of `qs`; Q8 stores
//! sequentially. f16 conversion is round-to-nearest-even (`half::f16::from_f32`), matching
//! `GGML_FP32_TO_FP16`.

use half::f16;

pub const QK: usize = 32;

#[inline(always)]
fn f32_to_f16(v: f32) -> f16 {
    f16::from_f32(v)
}
#[inline(always)]
fn f16_to_f32(v: f16) -> f32 {
    v.to_f32()
}

/// Quantize one row (`x` length multiple of 32) into `out` for the given ggml type.
/// Returns the number of bytes written (== out.len() on success).
pub fn quantize_row(ggml_type: u32, x: &[f32], out: &mut [u8]) -> usize {
    match ggml_type {
        41 => quant_q1_0(x, out),
        42 => quant_q2_0(x, out),
        2 => quant_q4_0(x, out),
        3 => quant_q4_1(x, out),
        6 => quant_q5_0(x, out),
        7 => quant_q5_1(x, out),
        8 => quant_q8_0(x, out),
        9 => quant_q8_1(x, out),
        _ => panic!("legacy::quantize_row: unsupported type {ggml_type}"),
    }
}

/// Dequantize one row (`bytes` for the given type) into `out` (length = n elems).
pub fn dequantize_row(ggml_type: u32, bytes: &[u8], out: &mut [f32]) {
    match ggml_type {
        41 => dequant_q1_0(bytes, out),
        42 => dequant_q2_0(bytes, out),
        2 => dequant_q4_0(bytes, out),
        3 => dequant_q4_1(bytes, out),
        6 => dequant_q5_0(bytes, out),
        7 => dequant_q5_1(bytes, out),
        8 => dequant_q8_0(bytes, out),
        9 => dequant_q8_1(bytes, out),
        _ => panic!("legacy::dequantize_row: unsupported type {ggml_type}"),
    }
}

// ============================== Q1_0 / Q2_0 ==============================

// These are the current ggml reference layouts (introduced after the original legacy family).
// Q1_0 stores one sign bit per value and the mean absolute value as its block scale.
fn quant_q1_0(x: &[f32], out: &mut [u8]) -> usize {
    const QK1: usize = 128;
    const BYTES: usize = 18;
    let nb = x.len() / QK1;
    assert!(out.len() >= nb * BYTES);
    for i in 0..nb {
        let blk = &x[i * QK1..(i + 1) * QK1];
        let d = blk.iter().map(|v| v.abs()).sum::<f32>() / QK1 as f32;
        let off = i * BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[off + 2..off + BYTES].fill(0);
        for (j, &v) in blk.iter().enumerate() {
            if v >= 0.0 {
                out[off + 2 + j / 8] |= 1 << (j % 8);
            }
        }
    }
    nb * BYTES
}

fn dequant_q1_0(bytes: &[u8], out: &mut [f32]) {
    const QK1: usize = 128;
    const BYTES: usize = 18;
    for i in 0..out.len() / QK1 {
        let off = i * BYTES;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        for j in 0..QK1 {
            let positive = (bytes[off + 2 + j / 8] >> (j % 8)) & 1 != 0;
            out[i * QK1 + j] = if positive { d } else { -d };
        }
    }
}

// Q2_0 stores {-d, 0, d, 2d} in two bits, with a 64-value block.
fn quant_q2_0(x: &[f32], out: &mut [u8]) -> usize {
    const QK2: usize = 64;
    const BYTES: usize = 18;
    let nb = x.len() / QK2;
    assert!(out.len() >= nb * BYTES);
    for i in 0..nb {
        let blk = &x[i * QK2..(i + 1) * QK2];
        let d = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let off = i * BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[off + 2..off + BYTES].fill(0);
        for (j, &v) in blk.iter().enumerate() {
            // C roundf/lroundf: halfway cases go away from zero.
            let scaled = v * id;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5).floor()
            } else {
                (scaled - 0.5).ceil()
            } as i32;
            let q = (rounded + 1).clamp(0, 3) as u8;
            out[off + 2 + j / 4] |= q << (2 * (j % 4));
        }
    }
    nb * BYTES
}

fn dequant_q2_0(bytes: &[u8], out: &mut [f32]) {
    const QK2: usize = 64;
    const BYTES: usize = 18;
    for i in 0..out.len() / QK2 {
        let off = i * BYTES;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        for j in 0..QK2 {
            let q = (bytes[off + 2 + j / 4] >> (2 * (j % 4))) & 3;
            out[i * QK2 + j] = (q as i32 - 1) as f32 * d;
        }
    }
}

// ============================== Q4_0 ==============================
// d = signed_max / -8 ; q = clamp(round(x/d + 8), 0, 15) packed low/high halves.
fn quant_q4_0(x: &[f32], out: &mut [u8]) -> usize {
    let nb = x.len() / QK;
    assert!(out.len() >= nb * 18);
    for i in 0..nb {
        let blk = &x[i * QK..i * QK + QK];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in blk {
            let a = v.abs();
            if amax < a {
                amax = a;
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * 18;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let qs = &mut out[off + 2..off + 18];
        for j in 0..16 {
            let x0 = blk[j] * id;
            let x1 = blk[j + 16] * id;
            let xi0 = ((x0 + 8.5) as i8).min(15) as u8;
            let xi1 = ((x1 + 8.5) as i8).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
    }
    nb * 18
}

fn dequant_q4_0(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QK;
    for i in 0..nb {
        let off = i * 18;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let qs = &bytes[off + 2..off + 18];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            out[i * QK + j] = x0 as f32 * d;
            out[i * QK + j + 16] = x1 as f32 * d;
        }
    }
}

// ============================== Q4_1 ==============================
// d = (max-min)/15, m = min ; q = clamp(round((x-m)/d), 0, 15).
fn quant_q4_1(x: &[f32], out: &mut [u8]) -> usize {
    let nb = x.len() / QK;
    assert!(out.len() >= nb * 20);
    for i in 0..nb {
        let blk = &x[i * QK..i * QK + QK];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in blk {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * 20;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[off + 2..off + 4].copy_from_slice(&f32_to_f16(min).to_le_bytes());
        let qs = &mut out[off + 4..off + 20];
        for j in 0..16 {
            let x0 = (blk[j] - min) * id;
            let x1 = (blk[j + 16] - min) * id;
            let xi0 = ((x0 + 0.5) as i8).min(15) as u8;
            let xi1 = ((x1 + 0.5) as i8).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
    }
    nb * 20
}

fn dequant_q4_1(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QK;
    for i in 0..nb {
        let off = i * 20;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let m = f16_to_f32(f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]));
        let qs = &bytes[off + 4..off + 20];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32;
            let x1 = (qs[j] >> 4) as i32;
            out[i * QK + j] = x0 as f32 * d + m;
            out[i * QK + j + 16] = x1 as f32 * d + m;
        }
    }
}

// ============================== Q5_0 ==============================
// d = signed_max / -16 ; 5-bit q in [0,31], low nibble in qs, high bit in qh (u32 LE).
fn quant_q5_0(x: &[f32], out: &mut [u8]) -> usize {
    let nb = x.len() / QK;
    assert!(out.len() >= nb * 22);
    for i in 0..nb {
        let blk = &x[i * QK..i * QK + QK];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in blk {
            let a = v.abs();
            if amax < a { amax = a; max = v; }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * 22;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let mut qh: u32 = 0;
        let qs = &mut out[off + 6..off + 22];
        for j in 0..16 {
            let x0 = blk[j] * id;
            let x1 = blk[j + 16] * id;
            let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
            let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
            qs[j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            qh |= (((xi0 & 0x10) as u32) >> 4) << j;
            qh |= (((xi1 & 0x10) as u32) >> 4) << (j + 16);
        }
        out[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
    }
    nb * 22
}

fn dequant_q5_0(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QK;
    for i in 0..nb {
        let off = i * 22;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let qh = u32::from_le_bytes([bytes[off + 2], bytes[off + 3], bytes[off + 4], bytes[off + 5]]);
        let qs = &bytes[off + 6..off + 22];
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = (((qh >> (j + 12))) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0f) | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
            out[i * QK + j] = x0 as f32 * d;
            out[i * QK + j + 16] = x1 as f32 * d;
        }
    }
}

// ============================== Q5_1 ==============================
// d = (max-min)/31, m = min ; 5-bit q, low nibble in qs, high bit in qh.
fn quant_q5_1(x: &[f32], out: &mut [u8]) -> usize {
    let nb = x.len() / QK;
    assert!(out.len() >= nb * 24);
    for i in 0..nb {
        let blk = &x[i * QK..i * QK + QK];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in blk {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * 24;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[off + 2..off + 4].copy_from_slice(&f32_to_f16(min).to_le_bytes());
        let mut qh: u32 = 0;
        let qs = &mut out[off + 8..off + 24];
        for j in 0..16 {
            let x0 = (blk[j] - min) * id;
            let x1 = (blk[j + 16] - min) * id;
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;
            qs[j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            qh |= (((xi0 & 0x10) as u32) >> 4) << j;
            qh |= (((xi1 & 0x10) as u32) >> 4) << (j + 16);
        }
        out[off + 4..off + 8].copy_from_slice(&qh.to_le_bytes());
    }
    nb * 24
}

fn dequant_q5_1(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QK;
    for i in 0..nb {
        let off = i * 24;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let m = f16_to_f32(f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]));
        let qh = u32::from_le_bytes([bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]]);
        let qs = &bytes[off + 8..off + 24];
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = (((qh >> (j + 12))) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0f) | xh_0) as i32;
            let x1 = ((qs[j] >> 4) | xh_1) as i32;
            out[i * QK + j] = x0 as f32 * d + m;
            out[i * QK + j + 16] = x1 as f32 * d + m;
        }
    }
}

// ============================== Q8_0 ==============================
// d = amax/127 ; q = round(x/d) as i8.
fn quant_q8_0(x: &[f32], out: &mut [u8]) -> usize {
    let nb = x.len() / QK;
    assert!(out.len() >= nb * 34);
    for i in 0..nb {
        let blk = &x[i * QK..i * QK + QK];
        let mut amax = 0.0f32;
        for &v in blk { amax = amax.max(v.abs()); }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * 34;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let qs = &mut out[off + 2..off + 34];
        for j in 0..QK {
            qs[j] = (blk[j] * id).round() as i8 as u8;
        }
    }
    nb * 34
}

fn dequant_q8_0(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QK;
    for i in 0..nb {
        let off = i * 34;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let qs = &bytes[off + 2..off + 34];
        for j in 0..QK {
            out[i * QK + j] = (qs[j] as i8) as f32 * d;
        }
    }
}

// ============================== Q8_1 ==============================
// d = amax/127, s = d * sum(round(x/d)) ; q = round(x/d) as i8.
fn quant_q8_1(x: &[f32], out: &mut [u8]) -> usize {
    let nb = x.len() / QK;
    assert!(out.len() >= nb * 36);
    for i in 0..nb {
        let blk = &x[i * QK..i * QK + QK];
        let mut amax = 0.0f32;
        for &v in blk { amax = amax.max(v.abs()); }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * 36;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let qs = &mut out[off + 4..off + 36];
        let mut sum: i32 = 0;
        for j in 0..16 {
            let v0 = blk[j] * id;
            let v1 = blk[j + 16] * id;
            let q0 = v0.round() as i8;
            let q1 = v1.round() as i8;
            qs[j] = q0 as u8;
            qs[j + 16] = q1 as u8;
            sum += q0 as i32;
            sum += q1 as i32;
        }
        let s = d * sum as f32;
        out[off + 2..off + 4].copy_from_slice(&f32_to_f16(s).to_le_bytes());
    }
    nb * 36
}

fn dequant_q8_1(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QK;
    for i in 0..nb {
        let off = i * 36;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let qs = &bytes[off + 4..off + 36];
        for j in 0..QK {
            out[i * QK + j] = (qs[j] as i8) as f32 * d;
        }
    }
}

/// Bytes per block for a legacy type (block = 32 elements).
pub fn bytes_per_block(ggml_type: u32) -> usize {
    match ggml_type {
        2 => 18, 3 => 20, 6 => 22, 7 => 24, 8 => 34, 9 => 36, _ => 0,
    }
}
