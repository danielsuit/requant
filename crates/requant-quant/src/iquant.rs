//! i-quant codebook family (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
//! IQ4_XS) — bit-exact port of ggml's `quantize_iqX` / `quantize_row_iqX_ref` /
//! `dequantize_row_iqX` kernels (ggml-quants.c).
//!
//! i-quants are imatrix-driven codebook formats. IQ2_XXS, IQ2_XS, IQ1_S require an imatrix
//! (ggml aborts without one); the rest accept NULL (uniform/x^2 weights). The codebook grids
//! and the per-grid neighbour tables are precomputed lazily on first use, mirroring ggml's
//! `ggml_quantize_init()`. Index loops are kept verbatim for bit-exact auditability.

use anyhow::{bail, Result};
use half::f16;
use std::sync::OnceLock;

use crate::kquant::{make_qp_quants, nearest_int};

#[path = "iquant_tables.rs"]
mod iquant_tables;
use iquant_tables::*;

pub const QKK: usize = 256;
const QK4_NL: usize = 32;
const NGRID_IQ1S: usize = 2048;
const IQ3S_N_SCALE: usize = QKK / 64;
const IQ3S_BLOCK_SIZE: usize = 32;
const IQ1S_BLOCK_SIZE: usize = 32;
const IQ1M_BLOCK_SIZE: usize = 16;

const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;
const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;
const GROUP_MAX_EPS_IQ1_M: f32 = 1e-7;
const GROUP_MAX_EPS_IQ1_S: f32 = 1e-12;
const IQ1S_DELTA: f32 = 0.125;
const IQ1M_DELTA: f32 = 0.125;

#[inline(always)]
fn f16_from(v: f32) -> f16 {
    f16::from_f32(v)
}
#[inline(always)]
fn f16_to(v: f16) -> f32 {
    v.to_f32()
}

#[inline(always)]
fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

/// ggml's `best_index_int8`: binary search for the nearest of `values[0..n]` to `x`.
#[inline(always)]
fn best_index_int8(n: usize, values: &[i8], x: f32) -> usize {
    if x <= values[0] as f32 {
        return 0;
    }
    if x >= values[n - 1] as f32 {
        return n - 1;
    }
    let mut ml = 0usize;
    let mut mu = n - 1;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < values[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if (x - values[mu - 1] as f32) < (values[mu] as f32 - x) {
        mu - 1
    } else {
        mu
    }
}

#[inline(always)]
fn gbyte64(g: u64, k: usize) -> i8 {
    // The grid stores small positive ints (1,3,5,...) in each byte; little-endian byte order.
    g.to_le_bytes()[k] as i8
}
#[inline(always)]
fn gbyte32(g: u32, k: usize) -> i8 {
    g.to_le_bytes()[k] as i8
}

// ============================== dequantize ==============================

fn dequant_iq2_xxs(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 66;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 66..i * 66 + 66];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let qs = &blk[2..66];
            let mut aux32 = [0u32; 2];
            let mut oi = 0;
            for ib32 in 0..QKK / 32 {
                aux32[0] = u32::from_le_bytes([qs[8 * ib32], qs[8 * ib32 + 1], qs[8 * ib32 + 2], qs[8 * ib32 + 3]]);
                aux32[1] = u32::from_le_bytes([qs[8 * ib32 + 4], qs[8 * ib32 + 5], qs[8 * ib32 + 6], qs[8 * ib32 + 7]]);
                // `aux8` is the 8-byte little-endian view of the two u32s; ggml casts aux32 to
                // uint8_t* and indexes `aux8[l]` for l in 0..4 (bytes 0..3). Build it flat.
                let mut aux8 = [0u8; 8];
                aux8[0..4].copy_from_slice(&aux32[0].to_le_bytes());
                aux8[4..8].copy_from_slice(&aux32[1].to_le_bytes());
                let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[aux8[l] as usize];
                    let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        let g = gbyte64(grid, j) as f32;
                        let s = if (signs & KMASK_IQ2XS[j]) != 0 { -1.0 } else { 1.0 };
                        o[oi] = db * g * s;
                        oi += 1;
                    }
                }
            }
        }
    }
}

fn dequant_iq2_xs(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 74;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 74..i * 74 + 74];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let qs = &blk[2..66];   // 64 bytes = 32 uint16
            let scales = &blk[66..74]; // 8 bytes
            let mut oi = 0;
            for ib32 in 0..QKK / 32 {
                let db = [
                    d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
                    d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                ];
                for l in 0..4 {
                    let off = 8 * ib32 + 2 * l;
                    let q = u16::from_le_bytes([qs[off], qs[off + 1]]);
                    let grid = IQ2XS_GRID[(q & 511) as usize];
                    let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                    for j in 0..8 {
                        let g = gbyte64(grid, j) as f32;
                        let s = if (signs & KMASK_IQ2XS[j]) != 0 { -1.0 } else { 1.0 };
                        o[oi] = db[l / 2] * g * s;
                        oi += 1;
                    }
                }
            }
        }
    }
}

fn dequant_iq2_s(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 82;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 82..i * 82 + 82];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let qs0 = &blk[2..66];     // 64
            let qh = &blk[66..74];     // 8
            let scales = &blk[74..82]; // 8
            // signs live right after qs in the on-disk struct: qs[QK_K/8 .. QK_K/4] (32 bytes)
            // The struct layout is d | qs[QK_K/4] | qh[QK_K/32] | scales[QK_K/32], but ggml's
            // quantizer writes signs into qs[QK_K/8..] (the second half of the qs field). dequant
            // reads `signs = qs + QK_K/8`.
            let signs = &blk[2 + QKK / 8..2 + QKK / 4];
            let mut qsi = 0usize;
            let mut sgni = 0usize;
            let mut oi = 0;
            for ib32 in 0..QKK / 32 {
                let db = [
                    d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
                    d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                ];
                for l in 0..4 {
                    let dl = db[l / 2];
                    let idx = (qs0[qsi + l] as usize) | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                    let grid = IQ2S_GRID[idx];
                    let signs_b = signs[sgni + l];
                    for j in 0..8 {
                        let g = gbyte64(grid, j) as f32;
                        let s = if (signs_b & KMASK_IQ2XS[j]) != 0 { -1.0 } else { 1.0 };
                        o[oi] = dl * g * s;
                        oi += 1;
                    }
                }
                qsi += 4;
                sgni += 4;
            }
        }
    }
}

fn dequant_iq3_xxs(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 98;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 98..i * 98 + 98];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let qs = &blk[2..66];            // grid indices (64)
            let scales_and_signs = &blk[66..98]; // 8 x uint32 (32 bytes)
            let mut qi = 0usize;
            let mut oi = 0;
            for ib32 in 0..QKK / 32 {
                let aux32 = u32::from_le_bytes([
                    scales_and_signs[4 * ib32],
                    scales_and_signs[4 * ib32 + 1],
                    scales_and_signs[4 * ib32 + 2],
                    scales_and_signs[4 * ib32 + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let grid1 = IQ3XXS_GRID[qs[qi + 2 * l] as usize];
                    let grid2 = IQ3XXS_GRID[qs[qi + 2 * l + 1] as usize];
                    for j in 0..4 {
                        let g1 = gbyte32(grid1, j) as f32;
                        let s1 = if (signs & KMASK_IQ2XS[j]) != 0 { -1.0 } else { 1.0 };
                        o[oi + j] = db * g1 * s1;
                        let g2 = gbyte32(grid2, j) as f32;
                        let s2 = if (signs & KMASK_IQ2XS[j + 4]) != 0 { -1.0 } else { 1.0 };
                        o[oi + j + 4] = db * g2 * s2;
                    }
                    oi += 8;
                }
                qi += 8;
            }
        }
    }
}

fn dequant_iq3_s(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 110;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 110..i * 110 + 110];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let mut bp = 2usize;
            let qs_base = bp; bp += QKK / 4;          // 64
            let qh_base = bp; bp += QKK / 32;         // 8
            let signs_base = bp; bp += QKK / 8;       // 32
            let scales_base = bp;                    // 4
            let _ = scales_base;
            let mut oi = 0;
            for ib32 in (0..QKK / 32).step_by(2) {
                let db1 = d * (1.0 + 2.0 * (blk[scales_base + ib32 / 2] & 0xf) as f32);
                let db2 = d * (1.0 + 2.0 * (blk[scales_base + ib32 / 2] >> 4) as f32);
                for half in 0..2 {
                    let db = if half == 0 { db1 } else { db2 };
                    let qh = blk[qh_base + ib32 + half];
                    let mut qi = qs_base + 8 * (ib32 + half);
                    let mut si = signs_base + 4 * (ib32 + half);
                    for l in 0..4 {
                        let idx1 = (blk[qi + 2 * l] as usize) | ((((qh as u32) << (8 - 2 * l)) & 256) as usize);
                        let idx2 = (blk[qi + 2 * l + 1] as usize) | ((((qh as u32) << (7 - 2 * l)) & 256) as usize);
                        let grid1 = IQ3S_GRID[idx1];
                        let grid2 = IQ3S_GRID[idx2];
                        let signs_b = blk[si + l];
                        for j in 0..4 {
                            let g1 = gbyte32(grid1, j) as f32;
                            let s1 = if (signs_b & KMASK_IQ2XS[j]) != 0 { -1.0 } else { 1.0 };
                            o[oi + j] = db * g1 * s1;
                            let g2 = gbyte32(grid2, j) as f32;
                            let s2 = if (signs_b & KMASK_IQ2XS[j + 4]) != 0 { -1.0 } else { 1.0 };
                            o[oi + j + 4] = db * g2 * s2;
                        }
                        oi += 8;
                    }
                    qi += 8;
                    si += 4;
                }
            }
        }
    }
}

fn dequant_iq1_s(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 50;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 50..i * 50 + 50];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let qs = &blk[2..34];     // 32 bytes
            let qh = &blk[34..50];     // 16 bytes = 8 uint16
            let mut oi = 0;
            let mut qsi = 0usize;
            for ib in 0..QKK / 32 {
                let h = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
                let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
                let delta = if h & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for l in 0..4 {
                    let idx = (qs[qsi + l] as usize) | ((((h >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[idx];
                    for j in 0..8 {
                        let g = gbyte64(grid, j) as f32;
                        o[oi + j] = dl * (g + delta);
                    }
                    oi += 8;
                }
                qsi += 4;
            }
        }
    }
}

fn dequant_iq1_m(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 56;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 56..i * 56 + 56];
            // block_iq1_m layout: qs[QK_K/8] | qh[QK_K/16] | scales[QK_K/32] (no d; scale packed in scales).
            let qs = &blk[0..32];   // 32
            let qh = &blk[32..48];  // 16
            // scales: 8 bytes reinterpreted as 4 uint16
            let sc = [
                u16::from_le_bytes([blk[48], blk[49]]),
                u16::from_le_bytes([blk[50], blk[51]]),
                u16::from_le_bytes([blk[52], blk[53]]),
                u16::from_le_bytes([blk[54], blk[55]]),
            ];
            let scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            let d = f16_to(f16::from_bits(scale_u16));
            let mut oi = 0;
            let mut qsi = 0usize;
            let mut qhi = 0usize;
            for ib in 0..QKK / 32 {
                let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) as f32 + 1.0);
                let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
                let qh0 = qh[qhi];
                let qh1 = qh[qhi + 1];
                let idx = [
                    (qs[qsi] as usize) | ((qh0 as usize) << 8 & 0x700),
                    (qs[qsi + 1] as usize) | ((qh0 as usize) << 4 & 0x700),
                    (qs[qsi + 2] as usize) | ((qh1 as usize) << 8 & 0x700),
                    (qs[qsi + 3] as usize) | ((qh1 as usize) << 4 & 0x700),
                ];
                let delta = [
                    if qh0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                ];
                for l in 0..2 {
                    let grid = IQ1S_GRID[idx[l]];
                    for j in 0..8 {
                        let g = gbyte64(grid, j) as f32;
                        o[oi + j] = dl1 * (g + delta[l]);
                    }
                    oi += 8;
                }
                for l in 2..4 {
                    let grid = IQ1S_GRID[idx[l]];
                    for j in 0..8 {
                        let g = gbyte64(grid, j) as f32;
                        o[oi + j] = dl2 * (g + delta[l]);
                    }
                    oi += 8;
                }
                qsi += 4;
                qhi += 2;
            }
        }
    }
}

fn dequant_iq4_nl(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QK4_NL;
    let row_bytes = nb * 18;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 18..i * 18 + 18];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let qs = &blk[2..18];
            let off = i * QK4_NL;
            for j in 0..QK4_NL / 2 {
                o[off + j] = d * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
                o[off + j + QK4_NL / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
            }
        }
    }
}

fn dequant_iq4_xs(bytes: &[u8], out: &mut [f32], rows: usize, cols: usize) {
    let nb = cols / QKK;
    let row_bytes = nb * 136;
    for r in 0..rows {
        let y = &bytes[r * row_bytes..];
        let o = &mut out[r * cols..];
        for i in 0..nb {
            let blk = &y[i * 136..i * 136 + 136];
            let d = f16_to(f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let scales_l = &blk[4..8]; // QK_K/64 = 4
            let qs = &blk[8..136];    // QK_K/2 = 128
            let mut oi = 0;
            for ib in 0..QKK / 32 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                    | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                for j in 0..16 {
                    o[oi + j] = dl * KVALUES_IQ4NL[(qs[16 * ib + j] & 0xf) as usize] as f32;
                    o[oi + j + 16] = dl * KVALUES_IQ4NL[(qs[16 * ib + j] >> 4) as usize] as f32;
                }
                oi += 32;
            }
        }
    }
}

/// ggml's `iq1_find_best_neighbour2`. `xg` is the x_p/x_m 3-float table.
fn iq1_find_best_neighbour2(
    neighbours: &[u16],
    grid: &[u64],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    xg: &[f32; 3],
    l_out: &mut [i8],
) -> usize {
    let num = neighbours[0] as usize;
    let mut best = f32::INFINITY;
    let mut grid_index = 0usize;
    for j in 1..=num {
        let gi = neighbours[j] as usize;
        let g = grid[gi];
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = xg[((gbyte64(g, i) - 1) / 2) as usize];
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best {
            best = d2;
            grid_index = gi;
        }
    }
    let g = grid[grid_index];
    for i in 0..8 {
        l_out[i] = (gbyte64(g, i) - 1) / 2;
    }
    grid_index
}

// ============================== quantize: IQ4_NL / IQ4_XS ==============================

#[inline]
fn write_f16(dh: &mut [u8], v: f32) {
    let b = f16_from(v).to_bits();
    dh[0] = b as u8;
    dh[1] = (b >> 8) as u8;
}

/// ggml's `quantize_row_iq4_nl_impl`. Shared by IQ4_NL (block 32, single scale) and IQ4_XS
/// (super-block 256, 8 sub-scales). `ntry` is always 7 on the dispatcher path (the only path
/// `llama-quantize` uses); the `_ref` kernel's ntry=-1 is not used here.
fn quantize_row_iq4_nl_impl(
    super_block_size: usize,
    block_size: usize,
    x: &[f32],
    dh: &mut [u8],
    q4: &mut [u8],
    scales_h: Option<&mut u16>,
    scales_l: Option<&mut [u8]>,
    scales: &mut [f32],
    weight: &mut [f32],
    l_arr: &mut [u8],
    values: &[i8],
    quant_weights: Option<&[f32]>,
    ntry: i32,
) {
    let mut sigma2 = 0.0f32;
    for j in 0..super_block_size {
        sigma2 += x[j] * x[j];
    }
    sigma2 *= 2.0 / super_block_size as f32;
    for b in 0..super_block_size / 2 {
        q4[b] = 0;
    }
    dh[0] = 0;
    dh[1] = 0;
    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;
    let nb = super_block_size / block_size;
    for ib in 0..nb {
        let xb = &x[ib * block_size..ib * block_size + block_size];
        let lb = &mut l_arr[ib * block_size..ib * block_size + block_size];
        if let Some(qw) = quant_weights {
            for j in 0..block_size {
                weight[j] = qw[ib * block_size + j] * (sigma2 + xb[j] * xb[j]).sqrt();
            }
        } else {
            for j in 0..block_size {
                weight[j] = xb[j] * xb[j];
            }
        }
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..block_size {
            let ax = xb[j].abs();
            if ax > amax {
                amax = ax;
                max = xb[j];
            }
        }
        if amax < crate::kquant::GROUP_MAX_EPS {
            scales[ib] = 0.0;
            for j in 0..block_size {
                lb[j] = 0;
            }
            continue;
        }
        let mut d = if ntry > 0 { -max / values[0] as f32 } else { max / values[0] as f32 };
        let mut id = 1.0 / d;
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for j in 0..block_size {
            let al = id * xb[j];
            let l = best_index_int8(16, values, al);
            lb[j] = l as u8;
            let q = values[l] as f32;
            let w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumqx / sumq2;
        let mut best = d * sumqx;
        for itry in -ntry..=ntry {
            id = (itry as f32 + values[0] as f32) / max;
            sumqx = 0.0;
            sumq2 = 0.0;
            for j in 0..block_size {
                let al = id * xb[j];
                let l = best_index_int8(16, values, al);
                let q = values[l] as f32;
                let w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }
    if nb > 1 {
        let scales_h = scales_h.unwrap();
        let scales_l = scales_l.unwrap();
        *scales_h = 0;
        let d = -max_scale / 32.0;
        write_f16(dh, d);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..nb {
            let mut l = nearest_int(id * scales[ib]);
            l = clamp_i32(l, -32, 31);
            let dl = d * l as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let lb = &mut l_arr[ib * block_size..ib * block_size + block_size];
            let xb = &x[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                lb[j] = best_index_int8(16, values, idl * xb[j]) as u8;
            }
            l += 32;
            let l_l = (l & 0xf) as u8;
            let l_h = (l >> 4) as u8;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            *scales_h |= (l_h as u16) << (2 * (ib % 8));
        }
    } else {
        write_f16(dh, scales[0]);
        if ntry > 0 {
            let id = if scales[0] != 0.0 { 1.0 / scales[0] } else { 0.0 };
            for j in 0..super_block_size {
                l_arr[j] = best_index_int8(16, values, id * x[j]) as u8;
            }
        }
    }
    for i in 0..super_block_size / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l_arr[32 * i + j] | (l_arr[32 * i + 16 + j] << 4);
        }
    }
}

fn quant_row_iq4_nl(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let nblock = cols / QK4_NL;
    let mut l_arr = vec![0u8; QK4_NL];
    let mut weight = vec![0f32; QK4_NL];
    let mut scales = vec![0f32; 1];
    for ibl in 0..nblock {
        let blk = &mut out[ibl * 18..ibl * 18 + 18];
        let (dh, q4) = blk.split_at_mut(2);
        let qw = im.map(|m| &m[QK4_NL * ibl..QK4_NL * ibl + QK4_NL]);
        quantize_row_iq4_nl_impl(
            QK4_NL, 32, &x[QK4_NL * ibl..], dh, q4, None, None, &mut scales, &mut weight,
            &mut l_arr, &KVALUES_IQ4NL, qw, 7,
        );
    }
}

fn quant_row_iq4_xs(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let nblock = cols / QKK;
    let mut l_arr = vec![0u8; QKK];
    let mut weight = vec![0f32; 32];
    let mut scales = vec![0f32; QKK / 32];
    for ibl in 0..nblock {
        let blk = &mut out[ibl * 136..ibl * 136 + 136];
        // Layout: d[0..2], scales_h[2..4] (u16), scales_l[4..8], qs[8..136].
        let (dh, rest) = blk.split_at_mut(2);
        let (sh_bytes, rest) = rest.split_at_mut(2);
        let (scales_l, q4) = rest.split_at_mut(4);
        let mut sh = 0u16;
        let qw = im.map(|m| &m[QKK * ibl..QKK * ibl + QKK]);
        quantize_row_iq4_nl_impl(
            QKK, 32, &x[QKK * ibl..], dh, q4, Some(&mut sh), Some(scales_l), &mut scales,
            &mut weight, &mut l_arr, &KVALUES_IQ4NL, qw, 7,
        );
        sh_bytes[0] = sh as u8;
        sh_bytes[1] = (sh >> 8) as u8;
    }
}

// ============================== public dispatch ==============================

/// Bytes per super-block (or block for IQ4_NL) for an i-quant type. 0 if unknown.
pub fn bytes_per_block(t: u32) -> usize {
    match t {
        16 => 66,  // IQ2_XXS
        17 => 74,  // IQ2_XS
        18 => 98,  // IQ3_XXS
        19 => 50,  // IQ1_S
        20 => 18,  // IQ4_NL (block 32)
        21 => 110, // IQ3_S
        22 => 82,  // IQ2_S
        23 => 136, // IQ4_XS
        29 => 56,  // IQ1_M
        _ => 0,
    }
}

/// Block size (elements per block) for an i-quant type.
pub fn block_size_of(t: u32) -> usize {
    if t == 20 { QK4_NL } else { QKK }
}

/// True if `t` is an i-quant type handled by this module.
pub fn handles(t: u32) -> bool {
    matches!(t, 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 29)
}

/// Whether `t` requires an imatrix (ggml aborts without one).
fn requires_imatrix(t: u32) -> bool {
    matches!(t, 16 | 17 | 19) // IQ2_XXS, IQ2_XS, IQ1_S
}

pub fn quantize_rows(
    ggml_type: u32,
    x: &[f32],
    out: &mut [u8],
    rows: usize,
    cols: usize,
    imatrix: Option<&[f32]>,
) -> Result<()> {
    let bs = block_size_of(ggml_type);
    if cols % bs != 0 {
        bail!("i-quant type {ggml_type}: cols {cols} not divisible by block {bs}");
    }
    if let Some(im) = imatrix {
        if im.len() < cols {
            bail!("imatrix length {} < cols {cols} for i-quant type {ggml_type}", im.len());
        }
    }
    if requires_imatrix(ggml_type) && imatrix.is_none() {
        bail!(
            "i-quant type {} requires an imatrix (ggml aborts without quant_weights); \
             pass --imatrix or pick a format that supports RTN",
            requant_io::ggml_type_name(ggml_type)
        );
    }
    let row_bytes = (cols / bs) * bytes_per_block(ggml_type);
    if out.len() < rows * row_bytes {
        bail!("i-quant: out buffer {} < needed {}", out.len(), rows * row_bytes);
    }
    for r in 0..rows {
        let xrow = &x[r * cols..r * cols + cols];
        let orow = &mut out[r * row_bytes..r * row_bytes + row_bytes];
        let imrow = imatrix; // per-input-channel, reused across rows (length == cols)
        match ggml_type {
            20 => quant_row_iq4_nl(xrow, orow, cols, imrow),
            23 => quant_row_iq4_xs(xrow, orow, cols, imrow),
            16 => quant_row_iq2_xxs(xrow, orow, cols, imrow.unwrap()),
            17 => quant_row_iq2_xs(xrow, orow, cols, imrow),
            22 => quant_row_iq2_s(xrow, orow, cols, imrow),
            18 => quant_row_iq3_xxs(xrow, orow, cols, imrow),
            21 => quant_row_iq3_s(xrow, orow, cols, imrow),
            19 => quant_row_iq1_s(xrow, orow, cols, imrow.unwrap()),
            29 => quant_row_iq1_m(xrow, orow, cols, imrow),
            _ => bail!("i-quant type {ggml_type}: kernel not implemented"),
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
    let bs = block_size_of(ggml_type);
    if cols % bs != 0 {
        bail!("i-quant dequant type {ggml_type}: cols {cols} not divisible by block {bs}");
    }
    let row_bytes = (cols / bs) * bytes_per_block(ggml_type);
    if bytes.len() < rows * row_bytes {
        bail!("i-quant dequant: {} bytes < needed {}", bytes.len(), rows * row_bytes);
    }
    if out.len() < rows * cols {
        bail!("i-quant dequant: out {} < needed {}", out.len(), rows * cols);
    }
    match ggml_type {
        16 => dequant_iq2_xxs(bytes, out, rows, cols),
        17 => dequant_iq2_xs(bytes, out, rows, cols),
        18 => dequant_iq3_xxs(bytes, out, rows, cols),
        19 => dequant_iq1_s(bytes, out, rows, cols),
        20 => dequant_iq4_nl(bytes, out, rows, cols),
        21 => dequant_iq3_s(bytes, out, rows, cols),
        22 => dequant_iq2_s(bytes, out, rows, cols),
        23 => dequant_iq4_xs(bytes, out, rows, cols),
        29 => dequant_iq1_m(bytes, out, rows, cols),
        _ => bail!("i-quant dequant type {ggml_type}: not implemented"),
    }
    Ok(())
}

// ============================== codebook grid generation ==============================
//
// ggml builds, at runtime (`ggml_quantize_init` / `iq2xs_init_impl` / `iq3xs_init_impl`), a
// hash map (`kmap`) plus a precomputed nearest-neighbour table (`kneighbors`) from the static
// packed grids. The quantize codebook-search kernels consume these. Generation is deterministic
// (depends only on the static grids), so we compute it lazily on first use and cache it in a
// `OnceLock`, mirroring ggml's one-shot init. The C `qsort` uses the comparator
// `(d2, grid_index)` ascending; since grid_index is unique this is a total order, so Rust's
// stable sort with the same key yields an identical ordering — required for bit-exactness.

struct Iq2Grid {
    grid: Box<[u64]>,      // grid_size entries, each = 8 bytes (levels 2*l+1, little-endian)
    map: Box<[i32]>,      // kmap_size = 43692; >=0 = grid index, <0 = -(kneighbors_slot+1)
    neighbours: Box<[u16]>,
}
struct Iq3Grid {
    grid: Box<[u32]>,      // grid_size entries, each = 4 bytes (levels 2*l+1, little-endian)
    map: Box<[i32]>,      // kmap_size = 4096
    neighbours: Box<[u16]>,
}

const KMAP_SIZE_Q2: usize = 43692;
const KMAP_SIZE_Q3: usize = 4096;

fn gen_q2(kgrid: &[u16], grid_size: usize, nwant: usize) -> Iq2Grid {
    // Expand the packed 2-bit grid into 8-byte u64 entries (level = 2*l+1 per byte).
    let mut grid = vec![0u64; grid_size];
    for k in 0..grid_size {
        let mut bytes = [0u8; 8];
        for i in 0..8 {
            let l = ((kgrid[k] >> (2 * i)) & 3) as u8;
            bytes[i] = 2 * l + 1;
        }
        grid[k] = u64::from_le_bytes(bytes);
    }
    // Build the hash map: index = sum_k l_k << 2k, mapping to grid position.
    let mut map = vec![-1i32; KMAP_SIZE_Q2];
    for i in 0..grid_size {
        let aux8 = grid[i].to_le_bytes();
        let mut index: u16 = 0;
        for k in 0..8 {
            let q = (aux8[k] - 1) / 2;
            index |= (q as u16) << (2 * k);
        }
        map[index as usize] = i as i32;
    }
    // First pass: count neighbours needed (one slot per miss for the count, plus n indices).
    let mut pos = [0i8; 8];
    let mut dist2: Vec<(i32, i32)> = vec![(0, 0); grid_size];
    let mut num_neighbors = 0usize;
    let mut num_not_in_map = 0usize;
    for i in 0..KMAP_SIZE_Q2 {
        if map[i] >= 0 {
            continue;
        }
        num_not_in_map += 1;
        for k in 0..8 {
            pos[k] = 2 * ((i >> (2 * k)) & 3) as i8 + 1;
        }
        for j in 0..grid_size {
            let pg = grid[j].to_le_bytes();
            let mut d2 = 0i32;
            for k in 0..8 {
                let diff = (pg[k] as i8 - pos[k]) as i32;
                d2 += diff * diff;
            }
            dist2[j] = (d2, j as i32);
        }
        dist2.sort_by(|a, b| a.cmp(b));
        let mut n = 0usize;
        let mut d2 = dist2[0].0;
        let mut nhave = 1usize;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            n += 1;
        }
        num_neighbors += n;
    }
    // Second pass: fill the neighbour table.
    let mut neighbours = vec![0u16; num_neighbors + num_not_in_map];
    let mut counter = 0usize;
    for i in 0..KMAP_SIZE_Q2 {
        if map[i] >= 0 {
            continue;
        }
        for k in 0..8 {
            pos[k] = 2 * ((i >> (2 * k)) & 3) as i8 + 1;
        }
        for j in 0..grid_size {
            let pg = grid[j].to_le_bytes();
            let mut d2 = 0i32;
            for k in 0..8 {
                let diff = (pg[k] as i8 - pos[k]) as i32;
                d2 += diff * diff;
            }
            dist2[j] = (d2, j as i32);
        }
        dist2.sort_by(|a, b| a.cmp(b));
        map[i] = -((counter + 1) as i32);
        let start = counter;
        counter += 1; // count slot
        let mut d2 = dist2[0].0;
        let mut n = 0usize;
        let mut nhave = 1usize;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            neighbours[counter] = dist2[j].1 as u16;
            counter += 1;
            n += 1;
        }
        neighbours[start] = n as u16;
    }
    Iq2Grid {
        grid: grid.into_boxed_slice(),
        map: map.into_boxed_slice(),
        neighbours: neighbours.into_boxed_slice(),
    }
}

fn gen_q3(kgrid: &[u16], grid_size: usize, nwant: usize) -> Iq3Grid {
    // Expand the packed 3-bit grid into 4-byte u32 entries (level = 2*l+1 per byte).
    let mut grid = vec![0u32; grid_size];
    for k in 0..grid_size {
        let mut bytes = [0u8; 4];
        for i in 0..4 {
            let l = ((kgrid[k] >> (3 * i)) & 7) as u8;
            bytes[i] = 2 * l + 1;
        }
        grid[k] = u32::from_le_bytes(bytes);
    }
    let mut map = vec![-1i32; KMAP_SIZE_Q3];
    for i in 0..grid_size {
        let aux8 = grid[i].to_le_bytes();
        let mut index: u16 = 0;
        for k in 0..4 {
            let q = (aux8[k] - 1) / 2;
            index |= (q as u16) << (3 * k);
        }
        map[index as usize] = i as i32;
    }
    let mut pos = [0i8; 4];
    let mut dist2: Vec<(i32, i32)> = vec![(0, 0); grid_size];
    let mut num_neighbors = 0usize;
    let mut num_not_in_map = 0usize;
    for i in 0..KMAP_SIZE_Q3 {
        if map[i] >= 0 {
            continue;
        }
        num_not_in_map += 1;
        for k in 0..4 {
            pos[k] = 2 * ((i >> (3 * k)) & 7) as i8 + 1;
        }
        for j in 0..grid_size {
            let pg = grid[j].to_le_bytes();
            let mut d2 = 0i32;
            for k in 0..4 {
                let diff = (pg[k] as i8 - pos[k]) as i32;
                d2 += diff * diff;
            }
            dist2[j] = (d2, j as i32);
        }
        dist2.sort_by(|a, b| a.cmp(b));
        let mut n = 0usize;
        let mut d2 = dist2[0].0;
        let mut nhave = 1usize;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            n += 1;
        }
        num_neighbors += n;
    }
    let mut neighbours = vec![0u16; num_neighbors + num_not_in_map];
    let mut counter = 0usize;
    for i in 0..KMAP_SIZE_Q3 {
        if map[i] >= 0 {
            continue;
        }
        for k in 0..4 {
            pos[k] = 2 * ((i >> (3 * k)) & 7) as i8 + 1;
        }
        for j in 0..grid_size {
            let pg = grid[j].to_le_bytes();
            let mut d2 = 0i32;
            for k in 0..4 {
                let diff = (pg[k] as i8 - pos[k]) as i32;
                d2 += diff * diff;
            }
            dist2[j] = (d2, j as i32);
        }
        dist2.sort_by(|a, b| a.cmp(b));
        map[i] = -((counter + 1) as i32);
        let start = counter;
        counter += 1;
        let mut d2 = dist2[0].0;
        let mut n = 0usize;
        let mut nhave = 1usize;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            neighbours[counter] = dist2[j].1 as u16;
            counter += 1;
            n += 1;
        }
        neighbours[start] = n as u16;
    }
    Iq3Grid {
        grid: grid.into_boxed_slice(),
        map: map.into_boxed_slice(),
        neighbours: neighbours.into_boxed_slice(),
    }
}

// gindex: 0=IQ2_XXS(256), 1=IQ2_XS(512), 2=IQ1_S/IQ1_M(2048), 3=IQ2_S(1024)
fn iq2_grid(gindex: usize) -> &'static Iq2Grid {
    static GRIDS: [OnceLock<Iq2Grid>; 4] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    GRIDS[gindex].get_or_init(|| match gindex {
        0 => gen_q2(&KGRID_2BIT_256, 256, 2),
        1 => gen_q2(&KGRID_2BIT_512, 512, 2),
        2 => gen_q2(&KGRID_1BIT_2048, NGRID_IQ1S, 3),
        3 => gen_q2(&KGRID_2BIT_1024, 1024, 1),
        _ => unreachable!(),
    })
}
// gindex: 0=IQ3_XXS(256), 1=IQ3_S(512)
fn iq3_grid(gindex: usize) -> &'static Iq3Grid {
    static GRIDS: [OnceLock<Iq3Grid>; 2] = [OnceLock::new(), OnceLock::new()];
    GRIDS[gindex].get_or_init(|| match gindex {
        0 => gen_q3(&KGRID_3BIT_256, 256, 2),
        1 => gen_q3(&KGRID_3BIT_512, 512, 3),
        _ => unreachable!(),
    })
}

/// ggml `iq2_find_best_neighbour`: among the precomputed neighbours of a miss, pick the grid
/// entry minimising weighted squared error. Writes `L[i] = (pg[i]-1)/2`.
fn iq2_find_best_neighbour(
    neighbours: &[u16],
    grid: &[u64],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> usize {
    let num_neighbors = neighbours[0] as usize;
    let mut best_d2 = f32::INFINITY;
    let mut grid_index: usize = 0;
    for j in 1..=num_neighbors {
        let pg = grid[neighbours[j] as usize].to_le_bytes();
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = pg[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neighbours[j] as usize;
        }
    }
    let pg = grid[grid_index].to_le_bytes();
    for i in 0..8 {
        l[i] = ((pg[i] - 1) / 2) as i8;
    }
    grid_index
}

/// ggml `iq3_find_best_neighbour` (4-byte / 4-element variant).
fn iq3_find_best_neighbour(
    neighbours: &[u16],
    grid: &[u32],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> usize {
    let num_neighbors = neighbours[0] as usize;
    let mut best_d2 = f32::INFINITY;
    let mut grid_index: usize = 0;
    for j in 1..=num_neighbors {
        let pg = grid[neighbours[j] as usize].to_le_bytes();
        let mut d2 = 0.0f32;
        for i in 0..4 {
            let q = pg[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neighbours[j] as usize;
        }
    }
    let pg = grid[grid_index].to_le_bytes();
    for i in 0..4 {
        l[i] = ((pg[i] - 1) / 2) as i8;
    }
    grid_index
}

// ---- IQ2_XXS quantize ----
fn quant_row_iq2_xxs(x: &[f32], out: &mut [u8], cols: usize, im: &[f32]) {
    let g = iq2_grid(0);
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const KMAXQ: i32 = 3;
    let nbl = cols / QKK;
    // block_iq2_xxs = d(2) + qs(QK_K/4 = 64)
    for ibl in 0..nbl {
        let y = &mut out[ibl * 66..ibl * 66 + 66];
        // d
        y[0] = 0;
        y[1] = 0;
        let mut q2 = [0u32; 2 * (QKK / 32)]; // 16 u32 = 64 bytes
        let mut max_scale = 0.0f32;
        let mut scales = [0f32; QKK / 32];
        let xbl = &x[QKK * ibl..];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK {
            sumx2 += xbl[i] * xbl[i];
        }
        let sigma2 = sumx2 / QKK as f32;
        let mut weight = [0f32; 32];
        let mut waux = [0f32; 32];
        let mut xval = [0f32; 32];
        let mut l_arr = [0i8; 32];
        let mut laux = [0i8; 32];
        let mut block_signs = [0u8; 4];
        for ib in 0..QKK / 32 {
            let xb = &xbl[32 * ib..];
            let qw = &im[QKK * ibl + 32 * ib..];
            for i in 0..32 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }
            for k in 0..4 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0;
                    let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }
            let mut max = xval[0];
            for i in 1..32 {
                if xval[i] > max {
                    max = xval[i];
                }
            }
            if max < crate::kquant::GROUP_MAX_EPS {
                scales[ib] = 0.0;
                for z in 0..32 {
                    l_arr[z] = 0;
                }
                continue;
            }
            let mut scale = make_qp_quants(32, KMAXQ + 1, &xval, by_u8_mut(&mut l_arr), &weight);
            let eff_max = scale * KMAXQ as f32;
            let mut best = 0.0f32;
            for is in -6..=6 {
                let id = (2.0 * KMAXQ as f32 - 1.0 + is as f32 * 0.1) / eff_max;
                let this_scale = 1.0 / id;
                for k in 0..4 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = clamp_i32(l, 0, KMAXQ - 1) as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as u16) << (2 * i);
                    }
                    let gi = kmap[u as usize];
                    if gi < 0 {
                        let start = (-gi - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq2_find_best_neighbour(nb, kgrid, &xval[8 * k..], &waux[8 * k..], this_scale, &mut laux[8 * k..8 * k + 8]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l_arr.copy_from_slice(&laux);
                }
            }
            if scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..4 {
                    let mut u = 0u16;
                    for i in 0..8 {
                        let mut l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        l = clamp_i32(l, 0, KMAXQ - 1);
                        u |= (l as u16) << (2 * i);
                    }
                    let gi = kmap[u as usize];
                    if gi < 0 {
                        let start = (-gi - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq2_find_best_neighbour(nb, kgrid, &xval[8 * k..], &waux[8 * k..], scale, &mut l_arr[8 * k..8 * k + 8]);
                    } else {
                        let pg = kgrid[gi as usize].to_le_bytes();
                        for i in 0..8 {
                            l_arr[8 * k + i] = ((pg[i] - 1) / 2) as i8;
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l_arr[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..4 {
                let mut u = 0u16;
                for i in 0..8 {
                    u |= (l_arr[8 * k + i] as u16) << (2 * i);
                }
                let gi = kmap[u as usize];
                // On-grid by construction (the final L came from the grid search).
                let grid_index = gi as usize;
                q2[2 * ib] |= (grid_index as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
            }
            scales[ib] = scale;
            if scale > max_scale {
                max_scale = scale;
            }
        }
        if max_scale == 0.0 {
            for z in 0..QKK / 4 {
                y[2 + z] = 0;
            }
            continue;
        }
        let d = max_scale / 31.0;
        let dh = f16_from(d);
        let dbits = dh.to_bits();
        y[0] = dbits as u8;
        y[1] = (dbits >> 8) as u8;
        let id = 1.0 / d;
        for ib in 0..QKK / 32 {
            let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l = clamp_i32(l, 0, 15);
            q2[2 * ib + 1] |= (l as u32) << 28;
        }
        let qs_bytes = by_u8_slice(&mut q2);
        y[2..66].copy_from_slice(qs_bytes);
    }
}

#[inline(always)]
fn by_u8_mut(arr: &mut [i8; 32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut u8, 32) }
}
#[inline(always)]
fn by_u8_slice(arr: &mut [u32; 16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u8, 64) }
}

fn quant_row_iq2_xs(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let im = im.unwrap(); // IQ2_XS requires an imatrix (ggml asserts non-NULL)
    let g = iq2_grid(1); // kgrid_2bit_512, nwant = 2
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const KMAXQ: i32 = 3;
    let nbl = cols / QKK;
    for ibl in 0..nbl {
        let y = &mut out[ibl * 74..ibl * 74 + 74];
        let (d_bytes, rest) = y.split_at_mut(2);
        let (qs, scales_b) = rest.split_at_mut(64);
        d_bytes[0] = 0; d_bytes[1] = 0;
        let mut q2 = [0u16; 32];
        for s in scales_b.iter_mut() { *s = 0; }
        let mut max_scale = 0.0f32;
        let xbl = &x[QKK * ibl..QKK * ibl + QKK];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK { sumx2 += xbl[i] * xbl[i]; }
        let sigma2 = sumx2 / QKK as f32;
        let mut weight = [0f32; 16];
        let mut waux = [0f32; 16];
        let mut xval = [0f32; 16];
        let mut l_arr = [0i8; 16];
        let mut laux = [0i8; 16];
        let mut block_signs = [0u8; 2];
        let mut is_on_grid = [false; 2];
        let mut is_on_grid_aux = [false; 2];
        let mut scales = [0f32; 16];
        for ib in 0..QKK / 16 {
            let xb = &xbl[16 * ib..16 * ib + 16];
            let qw = &im[QKK * ibl + 16 * ib..QKK * ibl + 16 * ib + 16];
            for i in 0..16 { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            for i in 0..16 { waux[i] = weight[i].sqrt(); }
            for k in 0..2 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 { xval[8 * k + i] = xb[8 * k + i]; }
                    else { xval[8 * k + i] = -xb[8 * k + i]; nflip += 1; s |= 1 << i; }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0;
                    let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min { min = ax; imin = i; }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }
            let mut max = xval[0];
            for i in 1..16 { if xval[i] > max { max = xval[i]; } }
            if max < crate::kquant::GROUP_MAX_EPS {
                scales[ib] = 0.0;
                for i in 0..16 { l_arr[i] = 0; }
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * KMAXQ as f32 - 1.0);
            is_on_grid[0] = true; is_on_grid[1] = true;
            for is in -9..=9 {
                let id = (2.0 * KMAXQ as f32 - 1.0 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                for k in 0..2 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = clamp_i32(l, 0, KMAXQ - 1) as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..8 { u |= (laux[8 * k + i] as u16) << (2 * i); }
                    let grid_index = kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq2_find_best_neighbour(nb, kgrid, &xval[8 * k..8 * k + 8], &waux[8 * k..8 * k + 8], this_scale, &mut laux[8 * k..8 * k + 8]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..16 { l_arr[i] = laux[i]; }
                    for k in 0..2 { is_on_grid[k] = is_on_grid_aux[k]; }
                }
            }
            let mut n_not_ongrid = 0;
            for k in 0..2 { if !is_on_grid[k] { n_not_ongrid += 1; } }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] { continue; }
                    let mut u = 0u16;
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        let l = clamp_i32(l, 0, KMAXQ - 1);
                        u |= (l as u16) << (2 * i);
                        l_arr[8 * k + i] = l as i8;
                    }
                    let grid_index = kmap[u as usize];
                    if grid_index < 0 {
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq2_find_best_neighbour(nb, kgrid, &xval[8 * k..8 * k + 8], &waux[8 * k..8 * k + 8], scale, &mut l_arr[8 * k..8 * k + 8]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * l_arr[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 { scale = sumqx / sumq2; }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 { block_signs[k] = (!block_signs[k]) & 127; }
            }
            for k in 0..2 {
                let mut u = 0u16;
                for i in 0..8 { u |= (l_arr[8 * k + i] as u16) << (2 * i); }
                let grid_index = kmap[u as usize];
                // ggml aborts if off-grid here; the search above guarantees on-grid.
                q2[2 * ib + k] = (grid_index as u16) | ((block_signs[k] as u16) << 9);
            }
            scales[ib] = scale;
            if scale > max_scale { max_scale = scale; }
        }
        if max_scale == 0.0 {
            for s in qs.iter_mut() { *s = 0; }
            continue;
        }
        let d = max_scale / 31.0;
        let b = f16_from(d).to_bits();
        d_bytes[0] = b as u8; d_bytes[1] = (b >> 8) as u8;
        let id = 1.0 / d;
        for ib in 0..QKK / 16 {
            let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l = clamp_i32(l, 0, 15);
            if ib % 2 == 0 { scales_b[ib / 2] = l as u8; }
            else { scales_b[ib / 2] |= (l as u8) << 4; }
        }
        for i in 0..32 {
            let b = q2[i].to_le_bytes();
            qs[2 * i] = b[0]; qs[2 * i + 1] = b[1];
        }
    }
}
fn quant_row_iq2_s(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let g = iq2_grid(3); // kgrid_2bit_1024, nwant = 1
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const KMAXQ: i32 = 3;
    let nbl = cols / QKK;
    for ibl in 0..nbl {
        let y = &mut out[ibl * 82..ibl * 82 + 82];
        for b in y.iter_mut() { *b = 0; }
        let (d_bytes, rest) = y.split_at_mut(2);
        let (qs, rest) = rest.split_at_mut(64);
        let (qh, scales_b) = rest.split_at_mut(8);
        let mut max_scale = 0.0f32;
        let xbl = &x[QKK * ibl..QKK * ibl + QKK];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK { sumx2 += xbl[i] * xbl[i]; }
        let sigma2 = 2.0 * sumx2 / QKK as f32;
        let mut weight = [0f32; 16];
        let mut waux = [0f32; 16];
        let mut xval = [0f32; 16];
        let mut l_arr = [0i8; 16];
        let mut laux = [0i8; 16];
        let mut block_signs = [0u8; 2];
        let mut is_on_grid = [false; 2];
        let mut is_on_grid_aux = [false; 2];
        let mut scales = [0f32; 16];
        for ib in 0..QKK / 16 {
            let xb = &xbl[16 * ib..16 * ib + 16];
            if let Some(im) = im {
                let qw = &im[QKK * ibl + 16 * ib..QKK * ibl + 16 * ib + 16];
                for i in 0..16 { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            } else {
                for i in 0..16 { weight[i] = 0.25 * sigma2 + xb[i] * xb[i]; }
            }
            for i in 0..16 { waux[i] = weight[i].sqrt(); }
            // IQ2_S records signs verbatim (no parity flip, unlike IQ2_XXS/XS).
            for k in 0..2 {
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 { xval[8 * k + i] = xb[8 * k + i]; }
                    else { xval[8 * k + i] = -xb[8 * k + i]; s |= 1 << i; }
                }
                block_signs[k] = s;
            }
            let mut max = xval[0];
            for i in 1..16 { if xval[i] > max { max = xval[i]; } }
            if max < GROUP_MAX_EPS_IQ2_S {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * KMAXQ as f32 - 1.0);
            is_on_grid[0] = true; is_on_grid[1] = true;
            for is in -9..=9 {
                let id = (2.0 * KMAXQ as f32 - 1.0 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                for k in 0..2 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = clamp_i32(l, 0, KMAXQ - 1) as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..8 { u |= (laux[8 * k + i] as u16) << (2 * i); }
                    let grid_index = kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq2_find_best_neighbour(nb, kgrid, &xval[8 * k..8 * k + 8], &waux[8 * k..8 * k + 8], this_scale, &mut laux[8 * k..8 * k + 8]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..16 { l_arr[i] = laux[i]; }
                    for k in 0..2 { is_on_grid[k] = is_on_grid_aux[k]; }
                }
            }
            let mut n_not_ongrid = 0;
            for k in 0..2 { if !is_on_grid[k] { n_not_ongrid += 1; } }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] { continue; }
                    let mut u = 0u16;
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        let l = clamp_i32(l, 0, KMAXQ - 1);
                        u |= (l as u16) << (2 * i);
                        l_arr[8 * k + i] = l as i8;
                    }
                    let grid_index = kmap[u as usize];
                    if grid_index < 0 {
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq2_find_best_neighbour(nb, kgrid, &xval[8 * k..8 * k + 8], &waux[8 * k..8 * k + 8], scale, &mut l_arr[8 * k..8 * k + 8]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * l_arr[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 { scale = sumqx / sumq2; }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 { block_signs[k] = !block_signs[k]; }
            }
            for k in 0..2 {
                let mut u = 0u16;
                for i in 0..8 { u |= (l_arr[8 * k + i] as u16) << (2 * i); }
                let grid_index = kmap[u as usize] as usize;
                let i8 = 2 * ib + k;
                qs[i8] = (grid_index & 255) as u8;
                qh[i8 / 4] |= ((grid_index >> 8) as u8) << (2 * (i8 % 4));
                qs[QKK / 8 + i8] = block_signs[k];
            }
            scales[ib] = scale;
            if scale > max_scale { max_scale = scale; }
        }
        if max_scale == 0.0 { continue; }
        let d = max_scale / 31.0;
        let b = f16_from(d * 0.9875).to_bits();
        d_bytes[0] = b as u8; d_bytes[1] = (b >> 8) as u8;
        let id = 1.0 / d;
        for ib in 0..QKK / 16 {
            let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l = clamp_i32(l, 0, 15);
            if ib % 2 == 0 { scales_b[ib / 2] = l as u8; }
            else { scales_b[ib / 2] |= (l as u8) << 4; }
        }
    }
}
fn quant_row_iq3_xxs(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let g = iq3_grid(0); // kgrid_3bit_256, nwant = 2 (IQ3_XXS)
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const KMAXQ: i32 = 8;
    let nbl = cols / QKK;
    for ibl in 0..nbl {
        let y = &mut out[ibl * 98..ibl * 98 + 98];
        let (d_bytes, qs) = y.split_at_mut(2);
        let mut q3 = [0u8; 96];
        let mut max_scale = 0.0f32;
        let xbl = &x[QKK * ibl..QKK * ibl + QKK];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK { sumx2 += xbl[i] * xbl[i]; }
        let sigma2 = 2.0 * sumx2 / QKK as f32;
        let mut weight = [0f32; 32];
        let mut waux = [0f32; 32];
        let mut xval = [0f32; 32];
        let mut l_arr = [0i8; 32];
        let mut laux = [0i8; 32];
        let mut block_signs = [0u8; 8];
        let mut is_on_grid = [false; 8];
        let mut is_on_grid_aux = [false; 8];
        let mut scales = [0f32; 8];
        let mut scales_ss_signonly = [0u32; 8];
        for ib in 0..QKK / 32 {
            let xb = &xbl[32 * ib..32 * ib + 32];
            if let Some(im) = im {
                let qw = &im[QKK * ibl + 32 * ib..QKK * ibl + 32 * ib + 32];
                for i in 0..32 { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            } else {
                for i in 0..32 { weight[i] = xb[i] * xb[i]; }
            }
            for i in 0..32 { waux[i] = weight[i].sqrt(); }
            // 4 sign-groups of 8 elements (nflip parity flip, 7-bit mask).
            for k in 0..4 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 { xval[8 * k + i] = xb[8 * k + i]; }
                    else { xval[8 * k + i] = -xb[8 * k + i]; nflip += 1; s |= 1 << i; }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0;
                    let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min { min = ax; imin = i; }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }
            let mut max = xval[0];
            for i in 1..32 { if xval[i] > max { max = xval[i]; } }
            if max < GROUP_MAX_EPS_IQ3_XXS {
                scales[ib] = 0.0;
                for i in 0..32 { l_arr[i] = 0; }
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * KMAXQ as f32 - 1.0);
            for k in 0..8 { is_on_grid[k] = true; }
            for is in -15..=15 {
                let id = (2.0 * KMAXQ as f32 - 1.0 + is as f32 * 0.2) / max;
                let this_scale = 1.0 / id;
                for k in 0..8 {
                    for i in 0..4 {
                        let l = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        laux[4 * k + i] = clamp_i32(l, 0, KMAXQ - 1) as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..4 { u |= (laux[4 * k + i] as u16) << (3 * i); }
                    let grid_index = kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq3_find_best_neighbour(nb, kgrid, &xval[4 * k..4 * k + 4], &waux[4 * k..4 * k + 4], this_scale, &mut laux[4 * k..4 * k + 4]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..32 { l_arr[i] = laux[i]; }
                    for k in 0..8 { is_on_grid[k] = is_on_grid_aux[k]; }
                }
            }
            let mut n_not_ongrid = 0;
            for k in 0..8 { if !is_on_grid[k] { n_not_ongrid += 1; } }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..8 {
                    if is_on_grid[k] { continue; }
                    let mut u = 0u16;
                    for i in 0..4 {
                        let l = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        let l = clamp_i32(l, 0, KMAXQ - 1);
                        u |= (l as u16) << (3 * i);
                    }
                    let grid_index = kmap[u as usize];
                    if grid_index < 0 {
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        let gi = iq3_find_best_neighbour(nb, kgrid, &xval[4 * k..4 * k + 4], &waux[4 * k..4 * k + 4], scale, &mut l_arr[4 * k..4 * k + 4]);
                        let pg = kgrid[gi].to_le_bytes();
                        for i in 0..4 { l_arr[4 * k + i] = ((pg[i] - 1) / 2) as i8; }
                    } else {
                        let pg = kgrid[grid_index as usize].to_le_bytes();
                        for i in 0..4 { l_arr[4 * k + i] = ((pg[i] - 1) / 2) as i8; }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l_arr[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 { scale = sumqx / sumq2; }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 { block_signs[k] = (!block_signs[k]) & 127; }
            }
            for k in 0..8 {
                let mut u = 0u16;
                for i in 0..4 { u |= (l_arr[4 * k + i] as u16) << (3 * i); }
                let grid_index = kmap[u as usize] as usize;
                q3[8 * ib + k] = grid_index as u8; // 256 grid fits in 8 bits
            }
            let mut ss = (block_signs[0] as u32)
                | ((block_signs[1] as u32) << 7)
                | ((block_signs[2] as u32) << 14)
                | ((block_signs[3] as u32) << 21);
            scales[ib] = scale;
            if scale > max_scale { max_scale = scale; }
            // The 4-bit per-sub-block scale is OR'd into the top nibble after d is known below.
            // Stash the sign part now; merge the scale bits post-loop (ggml does it in one pass).
            let ss_bytes = ss.to_le_bytes();
            q3[64 + 4 * ib..64 + 4 * ib + 4].copy_from_slice(&ss_bytes);
            // remember sign-only word to merge scale bits later
            scales_ss_signonly[ib] = ss;
        }
        // merge scale bits: ggml does `scales_and_signs[ib] |= (l << 28)` after computing l from d.
        // We stored the sign-only word; recompute l and OR it in.
        if max_scale == 0.0 {
            for b in qs.iter_mut() { *b = 0; }
            d_bytes[0] = 0; d_bytes[1] = 0;
            continue;
        }
        let d = max_scale / 31.0;
        let b = f16_from(d * 1.0125).to_bits();
        d_bytes[0] = b as u8; d_bytes[1] = (b >> 8) as u8;
        let id = 1.0 / d;
        for ib in 0..QKK / 32 {
            let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l = clamp_i32(l, 0, 15);
            let ss = scales_ss_signonly[ib] | ((l as u32) << 28);
            q3[64 + 4 * ib..64 + 4 * ib + 4].copy_from_slice(&ss.to_le_bytes());
        }
        qs.copy_from_slice(&q3);
    }
}
fn quant_row_iq3_s(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let g = iq3_grid(1); // kgrid_3bit_512, nwant = 3 (IQ3_S)
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const KMAXQ: i32 = 8;
    const BS: usize = 32;
    const BS4: usize = BS / 4; // 8 grid-groups
    const BS8: usize = BS / 8; // 4 sign-groups
    let nbl = cols / QKK;
    for ibl in 0..nbl {
        let y = &mut out[ibl * 110..ibl * 110 + 110];
        for b in y.iter_mut() { *b = 0; }
        let (d_bytes, rest) = y.split_at_mut(2);
        let (qs, rest) = rest.split_at_mut(QKK / 4);       // 64
        let (qh, rest) = rest.split_at_mut(QKK / 32);      // 8
        let (signs, scales_b) = rest.split_at_mut(QKK / 8); // 32 ; scales_b = 4
        let mut max_scale = 0.0f32;
        let xbl = &x[QKK * ibl..QKK * ibl + QKK];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK { sumx2 += xbl[i] * xbl[i]; }
        let sigma2 = 2.0 * sumx2 / QKK as f32;
        let mut weight = [0f32; BS];
        let mut waux = [0f32; BS];
        let mut xval = [0f32; BS];
        let mut l_arr = [0i8; BS];
        let mut laux = [0i8; BS];
        let mut block_signs = [0u8; BS8];
        let mut is_on_grid = [false; BS4];
        let mut is_on_grid_aux = [false; BS4];
        let mut scales = [0f32; QKK / BS];
        for ib in 0..QKK / BS {
            let xb = &xbl[BS * ib..BS * ib + BS];
            if let Some(im) = im {
                let qw = &im[QKK * ibl + BS * ib..QKK * ibl + BS * ib + BS];
                for i in 0..BS { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            } else {
                for i in 0..BS { weight[i] = xb[i] * xb[i]; }
            }
            for i in 0..BS { waux[i] = weight[i].sqrt(); }
            // 4 sign-groups of 8 (no parity flip, no & 127).
            for k in 0..BS8 {
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 { xval[8 * k + i] = xb[8 * k + i]; }
                    else { xval[8 * k + i] = -xb[8 * k + i]; s |= 1 << i; }
                }
                block_signs[k] = s;
            }
            let mut max = xval[0];
            for i in 1..BS { if xval[i] > max { max = xval[i]; } }
            if max == 0.0 { scales[ib] = 0.0; continue; }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * KMAXQ as f32 - 1.0);
            for k in 0..BS4 { is_on_grid[k] = false; }
            for is in -9..=9 {
                let id = (2.0 * KMAXQ as f32 - 1.0 + is as f32 * 0.2) / max;
                let this_scale = 1.0 / id;
                for k in 0..BS4 {
                    for i in 0..4 {
                        let l = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        laux[4 * k + i] = clamp_i32(l, 0, KMAXQ - 1) as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..4 { u |= (laux[4 * k + i] as u16) << (3 * i); }
                    let grid_index = kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let start = (-grid_index - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq3_find_best_neighbour(nb, kgrid, &xval[4 * k..4 * k + 4], &waux[4 * k..4 * k + 4], this_scale, &mut laux[4 * k..4 * k + 4]);
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..BS {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..BS { l_arr[i] = laux[i]; }
                    for k in 0..BS4 { is_on_grid[k] = is_on_grid_aux[k]; }
                }
            }
            let mut n_not_ongrid = 0;
            for k in 0..BS4 { if !is_on_grid[k] { n_not_ongrid += 1; } }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..BS4 {
                    // NB: ggml's `if (is_on_grid[k]) continue;` is commented out — re-quant all k.
                    let mut u = 0u16;
                    for i in 0..4 {
                        let l = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        let l = clamp_i32(l, 0, KMAXQ - 1);
                        u |= (l as u16) << (3 * i);
                    }
                    let gi = if kmap[u as usize] < 0 {
                        let start = (-(kmap[u as usize]) - 1) as usize;
                        let nb = &kneighbors[start..];
                        iq3_find_best_neighbour(nb, kgrid, &xval[4 * k..4 * k + 4], &waux[4 * k..4 * k + 4], scale, &mut l_arr[4 * k..4 * k + 4])
                    } else {
                        kmap[u as usize] as usize
                    };
                    let pg = kgrid[gi].to_le_bytes();
                    for i in 0..4 { l_arr[4 * k + i] = ((pg[i] - 1) / 2) as i8; }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..BS {
                    let w = weight[i];
                    let q = 2.0 * l_arr[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 { scale = sumqx / sumq2; }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..BS8 { block_signs[k] = !block_signs[k]; }
            }
            for k in 0..BS4 {
                let mut u = 0u16;
                for i in 0..4 { u |= (l_arr[4 * k + i] as u16) << (3 * i); }
                let grid_index = kmap[u as usize] as usize;
                let i8 = BS4 * ib + k;
                qs[i8] = (grid_index & 255) as u8;
                qh[i8 / 8] |= ((grid_index >> 8) as u8) << (i8 % 8);
            }
            for k in 0..BS8 { signs[BS8 * ib + k] = block_signs[k]; }
            scales[ib] = scale;
            if scale > max_scale { max_scale = scale; }
        }
        if max_scale == 0.0 { continue; }
        let d = max_scale / 31.0;
        let b = f16_from(d * 1.033).to_bits();
        d_bytes[0] = b as u8; d_bytes[1] = (b >> 8) as u8;
        let id = 1.0 / d;
        let mut ib = 0;
        while ib < QKK / BS {
            let mut l1 = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l1 = clamp_i32(l1, 0, 15);
            let mut l2 = nearest_int(0.5 * (id * scales[ib + 1] - 1.0));
            l2 = clamp_i32(l2, 0, 15);
            scales_b[ib / 2] = (l1 as u8) | ((l2 as u8) << 4);
            ib += 2;
        }
    }
}
fn quant_row_iq1_s(x: &[f32], out: &mut [u8], cols: usize, im: &[f32]) {
    let g = iq2_grid(2); // KGRID_1BIT_2048, nwant = 3 (IQ1_S)
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const BS: usize = IQ1S_BLOCK_SIZE; // 32
    let x_p: [f32; 3] = [-1.0 + IQ1S_DELTA, IQ1S_DELTA, 1.0 + IQ1S_DELTA];
    let x_m: [f32; 3] = [-1.0 - IQ1S_DELTA, -IQ1S_DELTA, 1.0 - IQ1S_DELTA];
    let nbl = cols / QKK;
    for ibl in 0..nbl {
        let y = &mut out[ibl * 50..ibl * 50 + 50];
        let (d_bytes, rest) = y.split_at_mut(2);
        let (qs, qh) = rest.split_at_mut(QKK / 8); // qs=32, qh=16
        for b in qs.iter_mut() { *b = 0; }
        for b in qh.iter_mut() { *b = 0; }
        let mut max_scale = 0.0f32;
        let xbl = &x[QKK * ibl..QKK * ibl + QKK];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK { sumx2 += xbl[i] * xbl[i]; }
        let sigma2 = 2.0 * sumx2 / QKK as f32;
        let mut weight = [0f32; BS];
        let mut l_arr = [0i8; BS];
        let mut index = [0u16; BS / 8];
        let mut scales = [0f32; QKK / BS];
        let mut shifts = [0i8; QKK / BS];
        for ib in 0..QKK / BS {
            let xb = &xbl[BS * ib..BS * ib + BS];
            let qw = &im[QKK * ibl + BS * ib..QKK * ibl + BS * ib + BS];
            for i in 0..BS { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            let mut max = xb[0].abs();
            for i in 1..BS { if xb[i].abs() > max { max = xb[i].abs(); } }
            if max < GROUP_MAX_EPS_IQ1_S {
                scales[ib] = 0.0;
                for i in 0..BS { l_arr[i] = 1; }
                continue;
            }
            // Sort indices by xb value ascending (mirrors ggml's qsort on interleaved pairs).
            let mut order: [usize; BS] = [0; BS];
            for j in 0..BS { order[j] = j; }
            order.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).unwrap());
            // Prefix sums over sorted order.
            let mut sumx = [0f32; BS + 1];
            let mut sumw = [0f32; BS + 1];
            for j in 0..BS {
                let i = order[j];
                sumx[j + 1] = sumx[j] + weight[i] * xb[i];
                sumw[j + 1] = sumw[j] + weight[i];
            }
            let mut best_score = f32::MIN;
            let mut scale = max;
            let mut besti1 = 0isize;
            let mut besti2 = 0isize;
            let mut best_shift = 0i8;
            for i1 in 0..=BS as isize {
                for i2 in i1..=BS as isize {
                    // shift +1 (x_p)
                    let sumqx = (sumx[i1 as usize] - sumx[0]) * x_p[0]
                        + (sumx[i2 as usize] - sumx[i1 as usize]) * x_p[1]
                        + (sumx[BS] - sumx[i2 as usize]) * x_p[2];
                    let sumq2 = (sumw[i1 as usize] - sumw[0]) * x_p[0] * x_p[0]
                        + (sumw[i2 as usize] - sumw[i1 as usize]) * x_p[1] * x_p[1]
                        + (sumw[BS] - sumw[i2 as usize]) * x_p[2] * x_p[2];
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        besti1 = i1; besti2 = i2; best_shift = 1;
                    }
                    // shift -1 (x_m)
                    let sumqx = (sumx[i1 as usize] - sumx[0]) * x_m[0]
                        + (sumx[i2 as usize] - sumx[i1 as usize]) * x_m[1]
                        + (sumx[BS] - sumx[i2 as usize]) * x_m[2];
                    let sumq2 = (sumw[i1 as usize] - sumw[0]) * x_m[0] * x_m[0]
                        + (sumw[i2 as usize] - sumw[i1 as usize]) * x_m[1] * x_m[1]
                        + (sumw[BS] - sumw[i2 as usize]) * x_m[2] * x_m[2];
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        besti1 = i1; besti2 = i2; best_shift = -1;
                    }
                }
            }
            // Assign levels by sorted position: 0 for j<besti1, 1 for besti1<=j<besti2, 2 for j>=besti2.
            for j in 0..BS { l_arr[order[j]] = if (j as isize) < besti1 { 0 } else if (j as isize) < besti2 { 1 } else { 2 }; }
            if scale < 0.0 {
                for j in 0..BS { l_arr[j] = 2 - l_arr[j]; }
                scale = -scale;
                best_shift = -best_shift;
            }
            let xx: [f32; 3] = if best_shift == 1 { x_p } else { x_m };
            let mut all_on_grid = true;
            for k in 0..BS / 8 {
                let mut u = 0u16;
                for j in 0..8 { u |= (l_arr[8 * k + j] as u16) << (2 * j); }
                let gi = kmap[u as usize];
                if gi < 0 {
                    all_on_grid = false;
                    let start = (-gi - 1) as usize;
                    let nb = &kneighbors[start..];
                    let gidx = iq1_find_best_neighbour2(nb, kgrid, &xb[8 * k..8 * k + 8], &weight[8 * k..8 * k + 8], scale, &xx, &mut l_arr[8 * k..8 * k + 8]);
                    index[k] = gidx as u16;
                } else {
                    index[k] = gi as u16;
                }
            }
            if !all_on_grid {
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for k in 0..BS / 8 {
                    let pg = kgrid[index[k] as usize].to_le_bytes();
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((pg[j] - 1) / 2) as usize];
                        sumqx += w * q * xb[8 * k + j];
                        sumq2 += w * q * q;
                    }
                }
                if sumqx > 0.0 && sumq2 > 0.0 { scale = sumqx / sumq2; }
            }
            let mut h = 0u16;
            for k in 0..BS / 8 {
                qs[(BS / 8) * ib + k] = (index[k] & 255) as u8;
                h |= (index[k] >> 8) << (3 * k);
            }
            qh[2 * ib] = (h & 255) as u8;
            qh[2 * ib + 1] = (h >> 8) as u8;
            scales[ib] = scale;
            shifts[ib] = best_shift;
            if scale > max_scale { max_scale = scale; }
        }
        if max_scale == 0.0 { continue; }
        let d = max_scale / 15.0;
        let b = f16_from(d * 1.125).to_bits();
        d_bytes[0] = b as u8; d_bytes[1] = (b >> 8) as u8;
        let id = 1.0 / d;
        for ib in 0..QKK / BS {
            let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l = clamp_i32(l, 0, 7);
            if shifts[ib] == -1 { l |= 8; }
            // qh[ib] is a uint16; OR the scale/shift nibble into bits 12..15.
            let mut h = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
            h |= (l as u16) << 12;
            qh[2 * ib] = (h & 255) as u8;
            qh[2 * ib + 1] = (h >> 8) as u8;
        }
    }
}
fn quant_row_iq1_m(x: &[f32], out: &mut [u8], cols: usize, im: Option<&[f32]>) {
    let g = iq2_grid(2); // same grid as IQ1_S (KGRID_1BIT_2048, nwant = 3)
    let kgrid = g.grid.as_ref();
    let kmap = g.map.as_ref();
    let kneighbors = g.neighbours.as_ref();
    const BS: usize = IQ1M_BLOCK_SIZE; // 16
    let x_p: [f32; 3] = [-1.0 + IQ1M_DELTA, IQ1M_DELTA, 1.0 + IQ1M_DELTA];
    let x_m: [f32; 3] = [-1.0 - IQ1M_DELTA, -IQ1M_DELTA, 1.0 - IQ1M_DELTA];
    const MASKS: [u8; 4] = [0x00, 0x80, 0x08, 0x88];
    let nbl = cols / QKK;
    for ibl in 0..nbl {
        let y = &mut out[ibl * 56..ibl * 56 + 56];
        let (qs, rest) = y.split_at_mut(QKK / 8);       // qs = 32
        let (qh, scales_b) = rest.split_at_mut(QKK / 16); // qh = 16, scales_b = 8
        for b in qs.iter_mut() { *b = 0; }
        for b in qh.iter_mut() { *b = 0; }
        for b in scales_b.iter_mut() { *b = 0; }
        let mut max_scale = 0.0f32;
        let xbl = &x[QKK * ibl..QKK * ibl + QKK];
        let mut sumx2 = 0.0f32;
        for i in 0..QKK { sumx2 += xbl[i] * xbl[i]; }
        let sigma2 = 2.0 * sumx2 / QKK as f32;
        let mut weight = [0f32; BS];
        let mut l_arr = [0i8; BS];
        let mut index = [0u16; BS / 8];
        let mut scales = [0f32; QKK / BS];
        let mut shifts = [0i8; QKK / BS];
        for ib in 0..QKK / BS {
            let xb = &xbl[BS * ib..BS * ib + BS];
            if let Some(im) = im {
                let qw = &im[QKK * ibl + BS * ib..QKK * ibl + BS * ib + BS];
                for i in 0..BS { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            } else {
                for i in 0..BS { weight[i] = xb[i] * xb[i]; }
            }
            let mut max = xb[0].abs();
            for i in 1..BS { if xb[i].abs() > max { max = xb[i].abs(); } }
            if max < GROUP_MAX_EPS_IQ1_M {
                scales[ib] = 0.0;
                for i in 0..BS { l_arr[i] = 1; }
                continue;
            }
            let mut order: [usize; BS] = [0; BS];
            for j in 0..BS { order[j] = j; }
            order.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).unwrap());
            let mut best_score = f32::MIN;
            let mut scale = max;
            let mut besti1 = 0isize;
            let mut besti2 = 0isize;
            let mut best_k = 0i8;
            for i1 in 0..=BS as isize {
                for i2 in i1..=BS as isize {
                    let mut sumqx = [0f32; 4];
                    let mut sumq2 = [0f32; 4];
                    for j in 0..BS {
                        let i = order[j];
                        let grp = if (j as isize) < i1 { 0 } else if (j as isize) < i2 { 1 } else { 2 };
                        let half = if i < BS / 2 { 0 } else { 1 };
                        for k in 0..4 {
                            let use_p = if half == 0 { k < 2 } else { k % 2 == 0 };
                            let xv = if use_p { x_p[grp] } else { x_m[grp] };
                            sumqx[k] += weight[i] * xv * xb[i];
                            sumq2[k] += weight[i] * xv * xv;
                        }
                    }
                    for k in 0..4 {
                        if sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = i1; besti2 = i2; best_k = k as i8;
                        }
                    }
                }
            }
            for j in 0..BS { l_arr[order[j]] = if (j as isize) < besti1 { 0 } else if (j as isize) < besti2 { 1 } else { 2 }; }
            if scale < 0.0 {
                for j in 0..BS { l_arr[j] = 2 - l_arr[j]; }
                scale = -scale;
                best_k = 3 - best_k;
            }
            let mut all_on_grid = true;
            for k in 0..BS / 8 {
                let xx: [f32; 3] = if k == 0 { if best_k < 2 { x_p } else { x_m } } else if best_k % 2 == 0 { x_p } else { x_m };
                let mut u = 0u16;
                for j in 0..8 { u |= (l_arr[8 * k + j] as u16) << (2 * j); }
                let gi = kmap[u as usize];
                if gi < 0 {
                    all_on_grid = false;
                    let start = (-gi - 1) as usize;
                    let nb = &kneighbors[start..];
                    let gidx = iq1_find_best_neighbour2(nb, kgrid, &xb[8 * k..8 * k + 8], &weight[8 * k..8 * k + 8], scale, &xx, &mut l_arr[8 * k..8 * k + 8]);
                    index[k] = gidx as u16;
                } else {
                    index[k] = gi as u16;
                }
            }
            if !all_on_grid {
                let mut sumqx_f = 0.0f32;
                let mut sumq2_f = 0.0f32;
                for k in 0..BS / 8 {
                    let xx: [f32; 3] = if k == 0 { if best_k < 2 { x_p } else { x_m } } else if best_k % 2 == 0 { x_p } else { x_m };
                    let pg = kgrid[index[k] as usize].to_le_bytes();
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((pg[j] - 1) / 2) as usize];
                        sumqx_f += w * q * xb[8 * k + j];
                        sumq2_f += w * q * q;
                    }
                }
                if sumqx_f > 0.0 && sumq2_f > 0.0 { scale = sumqx_f / sumq2_f; }
            }
            qs[2 * ib] = (index[0] & 255) as u8;
            qs[2 * ib + 1] = (index[1] & 255) as u8;
            qh[ib] = ((index[0] >> 8) | ((index[1] >> 8) << 4)) as u8;
            scales[ib] = scale;
            shifts[ib] = best_k;
            if scale > max_scale { max_scale = scale; }
        }
        if max_scale == 0.0 { continue; }
        // d refinement pass: write 3-bit scales + shift mask bits, then refine d from reconstruction.
        let mut d = max_scale / 15.0;
        let id = 1.0 / d;
        let mut sumqx_f = 0.0f32;
        let mut sumq2_f = 0.0f32;
        let mut sc = [0u16; 4];
        for ib in 0..QKK / BS {
            let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l = clamp_i32(l, 0, 7);
            sc[ib / 4] |= (l as u16) << (3 * (ib % 4));
            qh[ib] |= MASKS[shifts[ib] as usize];
            let xb = &xbl[BS * ib..BS * ib + BS];
            if let Some(im) = im {
                let qw = &im[QKK * ibl + BS * ib..QKK * ibl + BS * ib + BS];
                for i in 0..BS { weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt(); }
            } else {
                for i in 0..BS { weight[i] = xb[i] * xb[i]; }
            }
            for k in 0..BS / 8 {
                let xx: [f32; 3] = if k == 0 { if shifts[ib] < 2 { x_p } else { x_m } } else if shifts[ib] % 2 == 0 { x_p } else { x_m };
                let grid_index = (qs[2 * ib + k] as usize) | (((qh[ib] as usize) << (8 - 4 * k)) & 0x700);
                let pg = kgrid[grid_index].to_le_bytes();
                for j in 0..8 {
                    let w = weight[8 * k + j];
                    let q = xx[((pg[j] - 1) / 2) as usize] * (2.0 * l as f32 + 1.0);
                    sumqx_f += w * q * xb[8 * k + j];
                    sumq2_f += w * q * q;
                }
            }
        }
        if sumq2_f > 0.0 { d = sumqx_f / sumq2_f; }
        let s_u16 = f16_from(d * 1.1125).to_bits();
        sc[0] |= (s_u16 & 0x000f) << 12;
        sc[1] |= (s_u16 & 0x00f0) << 8;
        sc[2] |= (s_u16 & 0x0f00) << 4;
        sc[3] |= (s_u16 & 0xf000) << 0;
        for i in 0..4 {
            scales_b[2 * i..2 * i + 2].copy_from_slice(&sc[i].to_le_bytes());
        }
    }
}
