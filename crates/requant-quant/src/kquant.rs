//! k-quant family (Q2_K..Q6_K) — super-block (256) quants with quantized sub-block scales
//! and imatrix-weighted scale search. Bit-exact port of ggml's `quantize_row_*_ref` /
//! `dequantize_row_*` (ggml-quants.c).
//!
//! Block geometry (elements-per-block, bytes-per-block) is owned by `requant-io::block_layout`;
//! this module owns the kernel math. Each row is processed one super-block at a time.
//!
//! ## Layout convention
//! We write the on-disk ggml struct field order directly into the byte buffer:
//!   - Q6_K: `ql[128] | qh[64] | scales[16] | d[2]`  (210 bytes / 256 elems)

use anyhow::{bail, Result};
use half::f16;

pub const QKK: usize = 256;
pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

#[inline(always)]
fn f32_to_f16(v: f32) -> f16 {
    f16::from_f32(v)
}
#[inline(always)]
fn f16_to_f32(v: f16) -> f32 {
    v.to_f32()
}

/// ggml's `nearest_int`: round-to-nearest-even via the float-bit magic, exact for |f| <= 2^22.
#[inline(always)]
pub(crate) fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// ggml's `make_qx_quants` (rmse_type path). Returns the optimal scale and writes the integer
/// levels `L` (already shifted by `nmax`, i.e. in `[0, 2*nmax-1]`) for the non-rmse-type-0 path.
///
/// `rmse_type`: 0 = pure RTN (no search); 1 = weight = x²; 2 = weight = 1; 3 = |x|; 4 = sqrt(|x|).
/// `qw`: optional explicit per-element weights (imatrix); when present they override rmse_type.
fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            l[i] = (nmax + clamp_i32(li, -nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut return_early = false;
    let mut rmse_type = rmse_type;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let weight = |i: usize| -> f32 {
        if let Some(qw) = qw {
            qw[i]
        } else {
            match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            }
        }
    };
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        li = clamp_i32(li, -nmax, nmax - 1);
        l[i] = (li + nmax) as i8;
        let w = weight(i);
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li as f32) * (li as f32);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -((nmax as f32) + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let mut li = nearest_int(iscale * x[i]);
            li = clamp_i32(li, -nmax, nmax - 1);
            let w = weight(i);
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li as f32) * (li as f32);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]);
                l[i] = (nmax + clamp_i32(li, -nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

#[inline(always)]
fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// ggml's `make_qkx2_quants`: asymmetric (scale+min) weighted least-squares scale search.
/// Returns the optimal `scale`; writes the integer levels `L` (uint8, in `[0,nmax]`) and the
/// fitted `the_min = -min` (a non-negative magnitude). `Laux` is scratch of length `n`.
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = weights[0] * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for i in 0..n {
            l[i] = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = (nmax as f32) / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = clamp_i32(nearest_int(iscale * (x[i] - min)), 0, nmax);
        l[i] = li as u8;
        let mut diff = scale * (li as f32) + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li = clamp_i32(nearest_int(iscale * (x[i] - min)), 0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            sum_l += w * (li as f32);
            sum_l2 += w * (li as f32) * (li as f32);
            sum_xl += w * (li as f32) * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = if sum_l2 != 0.0 { sum_xl / sum_l2 } else { 0.0 };
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * (laux[i] as f32) + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                for i in 0..n {
                    l[i] = laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// ggml's `make_qp_quants` (ggml-quants.c): weighted super-block scale search used by the
/// imatrix path of Q4_K/Q5_K/Q6_K to quantize the per-sub-block scales (`scales`/`mins`,
/// length `n = QK_K/32 = 8`) down to 6-bit `Ls`/`Lm` against the super-block `d`/`dmin`.
///
/// Returns the fitted super-block scale `sumlx/suml2` (ggml writes this straight to the f16
/// `d`/`dmin` field — note this differs from the no-imatrix path's `max/63`).
pub(crate) fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], qw: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..n {
        if x[i] > max {
            max = x[i];
        }
    }
    if max < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * (l[i] as f32);
        let w = qw[i];
        best_mse += w * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            // ggml: `int l = nearest_int(iscale_is*x[i]); l = MIN(nmax, l);` — signed, clamped only
            // from above. Used signed in the mse (unlike best_mse above, which uses the wrapped
            // `L[i]`). For the non-negative inputs the k-quants pass here, signed == wrapped.
            let li = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * (li as f32);
            let w = qw[i];
            mse += w * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        // ggml: signed `l` for sumlx/suml2; `L[i] = l` stores the uint8 (wrapped) value the itry
        // loop reads back. Signed vs wrapped only diverges for negative l (iq-quant callers that
        // pass xval with a flipped sign); k-quant callers pass non-negative x so the two agree.
        let li = nearest_int(iscale * x[i]).min(nmax);
        l[i] = li as u8;
        let w = qw[i];
        sumlx += w * x[i] * (li as f32);
        suml2 += w * (li as f32) * (li as f32);
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = qw[i];
            // ggml itry: subtracts using `L[i]` (uint8, wrapped), not the signed l.
            let li = l[i] as f32;
            let slx = sumlx - w * x[i] * li;
            let sl2 = suml2 - w * li * li;
            if slx > 0.0 && sl2 > 0.0 {
                // ggml: `int new_l = nearest_int(...); new_l = MIN(nmax, new_l);` — signed; the
                // `new_l != L[i]` compare promotes the uint8 L[i] to int.
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != l[i] as i32 {
                    let slx = slx + w * x[i] * (new_l as f32);
                    let sl2 = sl2 + w * (new_l as f32) * (new_l as f32);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// ggml's `make_q3_quants` (ggml-quants.c): symmetric 3-bit sub-block scale search used by the
/// Q3_K no-imatrix (`_ref`) path. Returns the fitted scale; writes integer levels `L` in
/// `[0, 2*nmax-1]` (i.e. already shifted by `nmax`). With `do_rmse` it runs the iterative
/// weighted (w=x²) least-squares refinement ggml uses for Q3_K; without it, plain RTN.
fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], do_rmse: bool) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let mut li = nearest_int(iscale * x[i]);
            li = clamp_i32(li, -nmax, nmax - 1);
            l[i] = li as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * (li as f32);
            suml2 += w * (li as f32) * (li as f32);
        }
        for _itry in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let slx = sumlx - w * x[i] * (l[i] as f32);
                if slx > 0.0 {
                    let sl2 = suml2 - w * (l[i] as f32) * (l[i] as f32);
                    let new_l = clamp_i32(nearest_int(x[i] * sl2 / slx), -nmax, nmax - 1) as i8;
                    if new_l != l[i] {
                        let slx = slx + w * x[i] * (new_l as f32);
                        let sl2 = sl2 + w * (new_l as f32) * (new_l as f32);
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for i in 0..n {
            l[i] += nmax as i8;
        }
        if suml2 > 0.0 {
            sumlx / suml2
        } else {
            0.0
        }
    } else {
        for i in 0..n {
            let li = clamp_i32(nearest_int(iscale * x[i]), -nmax, nmax - 1);
            l[i] = (li + nmax) as i8;
        }
        1.0 / iscale
    }
}

/// ggml's `get_scale_min_k4`: unpack the 6-bit scale (`d`) and min (`m`) for sub-block `j`
/// from the 12-byte packed scales array. Inverse of the Q4_K/Q5_K scale-packing step.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Pack 8 sub-block scales (`ls`) and mins (`lm`), each in `[0,63]`, into the 12-byte ggml
/// layout. Must process `j < 4` (full assignments) before `j >= 4` (OR-into-high-bits), so
/// the buffer is written in order exactly as ggml does.
fn pack_scales_k4(scales: &mut [u8; 12], ls: &[u8; 8], lm: &[u8; 8]) {
    for j in 0..4 {
        scales[j] = ls[j];
        scales[j + 4] = lm[j];
    }
    for j in 4..8 {
        scales[j + 4] = (ls[j] & 0x0f) | ((lm[j] & 0x0f) << 4);
        scales[j - 4] |= (ls[j] >> 4) << 6;
        scales[j] |= (lm[j] >> 4) << 6;
    }
}

// ============================== Q6_K ==============================
// 16 sub-blocks of 16. Per sub-block: symmetric 6-bit, scale searched with make_qx_quants(16,32,rmse=1).
// The 16 sub-scales are quantized to int8 against the super-block d (f16). Quants packed 6-bit:
// low 4 bits in ql, high 2 bits in qh, interleaved per the ggml layout.
const Q6K_QL: usize = QKK / 2; // 128
const Q6K_QH: usize = QKK / 4; // 64
const Q6K_SC: usize = QKK / 16; // 16
const Q6K_BYTES: usize = Q6K_QL + Q6K_QH + Q6K_SC + 2; // 210

fn quant_q6_k_row(x: &[f32], out: &mut [u8], im: Option<&[f32]>) {
    // x length must be a multiple of 256; one super-block per 210 out bytes.
    let nb = x.len() / QKK;
    let mut l = [0i8; QKK];
    let mut scales = [0f32; QKK / 16];
    for i in 0..nb {
        let xb = &x[i * QKK..i * QKK + QKK];
        let off = i * Q6K_BYTES;

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for ib in 0..QKK / 16 {
            // imatrix path (ggml `quantize_row_q6_K_impl`): pass the per-channel importance as the
            // weight override to make_qx_quants (sub-block size 16). The super-block d still uses
            // -128/max_scale (no make_qp_quants here).
            let qw = im.map(|im| &im[i * QKK + 16 * ib..i * QKK + 16 * ib + 16]);
            let scale = make_qx_quants(16, 32, &xb[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16], 1, qw);
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        let d_off = off + Q6K_QL + Q6K_QH + Q6K_SC;
        if max_abs_scale < GROUP_MAX_EPS {
            for b in out[off..off + Q6K_BYTES].iter_mut() {
                *b = 0;
            }
            out[d_off..d_off + 2].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
            continue;
        }

        let iscale = -128.0 / max_scale;
        out[d_off..d_off + 2].copy_from_slice(&f32_to_f16(1.0 / iscale).to_le_bytes());
        let d = f16_to_f32(f16::from_le_bytes([out[d_off], out[d_off + 1]]));
        let sc = &mut out[off + Q6K_QL + Q6K_QH..off + Q6K_QL + Q6K_QH + Q6K_SC];
        for ib in 0..QKK / 16 {
            // ggml stores into int8_t with only an upper clamp (MIN(127, ...)); negative scales
            // are preserved as signed bytes. Match by capping the i32 then reinterpreting as i8.
            sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8 as u8;
        }

        for j in 0..QKK / 16 {
            let djs = d * (sc[j] as i8) as f32;
            if djs == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let li = nearest_int(xb[16 * j + ii] / djs);
                l[16 * j + ii] = (clamp_i32(li, -32, 31) + 32) as i8;
            }
        }

        // Offset-based packing (avoids mutable borrow churn):
        let ql_base = off;
        let qh_base = off + Q6K_QL;
        for blk in 0..QKK / 128 {
            let j = blk * 128;
            let ql0 = ql_base + blk * 64;
            let qh0 = qh_base + blk * 32;
            for li in 0..32 {
                let q1 = (l[j + li] as u8) & 0x0f;
                let q2 = (l[j + li + 32] as u8) & 0x0f;
                let q3 = (l[j + li + 64] as u8) & 0x0f;
                let q4 = (l[j + li + 96] as u8) & 0x0f;
                out[ql0 + li] = q1 | (q3 << 4);
                out[ql0 + li + 32] = q2 | (q4 << 4);
                out[qh0 + li] = ((l[j + li] as u8) >> 4)
                    | (((l[j + li + 32] as u8) >> 4) << 2)
                    | (((l[j + li + 64] as u8) >> 4) << 4)
                    | (((l[j + li + 96] as u8) >> 4) << 6);
            }
        }
    }
}

fn dequant_q6_k_row(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QKK;
    for i in 0..nb {
        let off = i * Q6K_BYTES;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off + Q6K_QL + Q6K_QH + Q6K_SC], bytes[off + Q6K_QL + Q6K_QH + Q6K_SC + 1]]));
        let sc = &bytes[off + Q6K_QL + Q6K_QH..off + Q6K_QL + Q6K_QH + Q6K_SC];
        let mut y = 0usize;
        let mut qlp = off;
        let mut qhp = off + Q6K_QL;
        let mut scp = 0usize;
        while y < QKK {
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((bytes[qlp + l] & 0x0f) | (((bytes[qhp + l] >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((bytes[qlp + l + 32] & 0x0f) | (((bytes[qhp + l] >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((bytes[qlp + l] >> 4) | (((bytes[qhp + l] >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((bytes[qlp + l + 32] >> 4) | (((bytes[qhp + l] >> 6) & 3) << 4)) as i8) - 32;
                out[i * QKK + y + l] = d * (sc[scp + is + 0] as i8) as f32 * q1 as f32;
                out[i * QKK + y + l + 32] = d * (sc[scp + is + 2] as i8) as f32 * q2 as f32;
                out[i * QKK + y + l + 64] = d * (sc[scp + is + 4] as i8) as f32 * q3 as f32;
                out[i * QKK + y + l + 96] = d * (sc[scp + is + 6] as i8) as f32 * q4 as f32;
            }
            y += 128;
            qlp += 64;
            qhp += 32;
            scp += 8;
        }
    }
}

// ============================== Q4_K ==============================
// 8 sub-blocks of 32. Per sub-block: asymmetric 4-bit (scale+min), scale+min searched with
// make_qkx2_quants(32,15, weights=av_x+|x|). The 8 scales+mins pack into 12 bytes (6-bit each),
// then super-block d/dmin (f16) scale them. Quants are 4-bit, two per byte.
const Q4K_BYTES: usize = 2 + 2 + 12 + 128; // d + dmin + scales + qs = 144

fn quant_q4_k_row(x: &[f32], out: &mut [u8], im: Option<&[f32]>) {
    let nb = x.len() / QKK;
    let mut l = [0u8; QKK];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut mins = [0f32; 8];
    let mut scales = [0f32; 8];
    let mut sw = [0f32; 8];
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for i in 0..nb {
        let xb = &x[i * QKK..i * QKK + QKK];
        let off = i * Q4K_BYTES;
        let mut packed = [0u8; 12];

        if let Some(im) = im {
            // imatrix path (ggml `quantize_row_q4_K_impl`): sigma2 over the whole 256-block,
            // weights = qw * sqrt(sigma2 + x^2), sub-block search via make_qkx2_quants with the
            // qkx3 finer grid (-0.9, 0.05, 36), and super-block d/dmin via the weighted
            // make_qp_quants (returns sumlx/suml2, not max/63).
            let sum_x2: f32 = (0..QKK).map(|k| xb[k] * xb[k]).sum();
            let sigma2 = 2.0 * sum_x2 / QKK as f32;
            for j in 0..8 {
                let sub = &xb[32 * j..32 * j + 32];
                let qw = &im[i * QKK + 32 * j..i * QKK + 32 * j + 32];
                let mut sumw = 0.0f32;
                for k in 0..32 {
                    weights[k] = qw[k] * (sigma2 + sub[k] * sub[k]).sqrt();
                    sumw += weights[k];
                }
                sw[j] = sumw;
                let mut mn = 0f32;
                scales[j] = make_qkx2_quants(32, 15, sub, &weights, &mut l[32 * j..32 * j + 32], &mut mn, &mut laux, -0.9, 0.05, 36, false);
                mins[j] = mn;
            }
            let d_block = make_qp_quants(8, 63, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(8, 63, &mins, &mut lm, &sw);
            pack_scales_k4(&mut packed, &ls, &lm);
            out[off + 4..off + 16].copy_from_slice(&packed);
            out[off..off + 2].copy_from_slice(&f32_to_f16(d_block).to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&f32_to_f16(m_block).to_le_bytes());
        } else {
            // no-imatrix path (ggml `quantize_row_q4_K_ref`): av_x = sqrt(sum_x2/32), weights =
            // av_x + |x|, sub-block search with the qkx2 grid, super-block d/dmin = max/63.
            let mut max_scale = 0.0f32;
            let mut max_min = 0.0f32;
            for j in 0..8 {
                let sub = &xb[32 * j..32 * j + 32];
                let sum_x2: f32 = sub.iter().map(|v| v * v).sum();
                let av_x = (sum_x2 / 32.0).sqrt();
                for k in 0..32 {
                    weights[k] = av_x + sub[k].abs();
                }
                let mut mn = 0f32;
                scales[j] = make_qkx2_quants(32, 15, sub, &weights, &mut l[32 * j..32 * j + 32], &mut mn, &mut laux, -1.0, 0.1, 20, false);
                mins[j] = mn;
                if scales[j] > max_scale {
                    max_scale = scales[j];
                }
                if mins[j] > max_min {
                    max_min = mins[j];
                }
            }
            let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            for j in 0..8 {
                ls[j] = (nearest_int(inv_scale * scales[j]) as u8).min(63);
                lm[j] = (nearest_int(inv_min * mins[j]) as u8).min(63);
            }
            pack_scales_k4(&mut packed, &ls, &lm);
            out[off + 4..off + 16].copy_from_slice(&packed);
            let d_f = max_scale / 63.0;
            let dmin_f = max_min / 63.0;
            out[off..off + 2].copy_from_slice(&f32_to_f16(d_f).to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&f32_to_f16(dmin_f).to_le_bytes());
        }

        // Shared tail: re-read d/dmin (round-tripped through f16), recompute the 4-bit levels
        // from the packed sub-block scales, and pack the quants.
        let d = f16_to_f32(f16::from_le_bytes([out[off], out[off + 1]]));
        let dmin = f16_to_f32(f16::from_le_bytes([out[off + 2], out[off + 3]]));
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &packed);
            let dj = d * (sc as f32);
            if dj == 0.0 {
                continue;
            }
            let dm = dmin * (m as f32);
            for ii in 0..32 {
                let li = clamp_i32(nearest_int((xb[32 * j + ii] + dm) / dj), 0, 15);
                l[32 * j + ii] = li as u8;
            }
        }

        let qs = &mut out[off + 16..off + 16 + 128];
        let mut j = 0;
        while j < QKK {
            for li in 0..32 {
                qs[(j / 64) * 32 + li] = l[j + li] | (l[j + li + 32] << 4);
            }
            j += 64;
        }
    }
}

fn dequant_q4_k_row(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QKK;
    for i in 0..nb {
        let off = i * Q4K_BYTES;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let min = f16_to_f32(f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]));
        let scales = &bytes[off + 4..off + 16];
        let q = &bytes[off + 16..off + 16 + 128];
        let mut is = 0usize;
        let mut qp = 0usize;
        let mut yo = i * QKK;
        while is < 8 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * (sc1 as f32);
            let m1 = min * (m1 as f32);
            let d2 = d * (sc2 as f32);
            let m2 = min * (m2 as f32);
            for li in 0..32 {
                out[yo + li] = d1 * ((q[qp + li] & 0x0f) as f32) - m1;
            }
            for li in 0..32 {
                out[yo + 32 + li] = d2 * ((q[qp + li] >> 4) as f32) - m2;
            }
            yo += 64;
            qp += 32;
            is += 2;
        }
    }
}

// ============================== Q5_K ==============================
// Like Q4_K but 5-bit levels: low 4 bits in qs, high bit in qh (one bit per element, packed
// per-64 with a 2-bit stride so each qh byte holds 4 high-bits).
const Q5K_BYTES: usize = 2 + 2 + 12 + 32 + 128; // d + dmin + scales + qh + qs = 176

fn quant_q5_k_row(x: &[f32], out: &mut [u8], im: Option<&[f32]>) {
    let nb = x.len() / QKK;
    let mut l = [0u8; QKK];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut mins = [0f32; 8];
    let mut scales = [0f32; 8];
    let mut sw = [0f32; 8];
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for i in 0..nb {
        let xb = &x[i * QKK..i * QKK + QKK];
        let off = i * Q5K_BYTES;
        let mut packed = [0u8; 12];

        if let Some(im) = im {
            let sum_x2: f32 = (0..QKK).map(|k| xb[k] * xb[k]).sum();
            let sigma2 = 2.0 * sum_x2 / QKK as f32;
            for j in 0..8 {
                let sub = &xb[32 * j..32 * j + 32];
                let qw = &im[i * QKK + 32 * j..i * QKK + 32 * j + 32];
                let mut sumw = 0.0f32;
                for k in 0..32 {
                    weights[k] = qw[k] * (sigma2 + sub[k] * sub[k]).sqrt();
                    sumw += weights[k];
                }
                sw[j] = sumw;
                let mut mn = 0f32;
                scales[j] = make_qkx2_quants(32, 31, sub, &weights, &mut l[32 * j..32 * j + 32], &mut mn, &mut laux, -0.9, 0.05, 36, false);
                mins[j] = mn;
            }
            let d_block = make_qp_quants(8, 63, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(8, 63, &mins, &mut lm, &sw);
            for j in 0..8 {
                ls[j] = ls[j].min(63);
                lm[j] = lm[j].min(63);
            }
            pack_scales_k4(&mut packed, &ls, &lm);
            out[off + 4..off + 16].copy_from_slice(&packed);
            out[off..off + 2].copy_from_slice(&f32_to_f16(d_block).to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&f32_to_f16(m_block).to_le_bytes());
        } else {
            let mut max_scale = 0.0f32;
            let mut max_min = 0.0f32;
            for j in 0..8 {
                let sub = &xb[32 * j..32 * j + 32];
                let sum_x2: f32 = sub.iter().map(|v| v * v).sum();
                let av_x = (sum_x2 / 32.0).sqrt();
                for k in 0..32 {
                    weights[k] = av_x + sub[k].abs();
                }
                let mut mn = 0f32;
                scales[j] = make_qkx2_quants(32, 31, sub, &weights, &mut l[32 * j..32 * j + 32], &mut mn, &mut laux, -0.5, 0.1, 15, false);
                mins[j] = mn;
                if scales[j] > max_scale {
                    max_scale = scales[j];
                }
                if mins[j] > max_min {
                    max_min = mins[j];
                }
            }
            let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            for j in 0..8 {
                ls[j] = (nearest_int(inv_scale * scales[j]) as u8).min(63);
                lm[j] = (nearest_int(inv_min * mins[j]) as u8).min(63);
            }
            pack_scales_k4(&mut packed, &ls, &lm);
            out[off + 4..off + 16].copy_from_slice(&packed);
            out[off..off + 2].copy_from_slice(&f32_to_f16(max_scale / 63.0).to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&f32_to_f16(max_min / 63.0).to_le_bytes());
        }

        let d = f16_to_f32(f16::from_le_bytes([out[off], out[off + 1]]));
        let dmin = f16_to_f32(f16::from_le_bytes([out[off + 2], out[off + 3]]));
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &packed);
            let dj = d * (sc as f32);
            if dj == 0.0 {
                continue;
            }
            let dm = dmin * (m as f32);
            for ii in 0..32 {
                let li = clamp_i32(nearest_int((xb[32 * j + ii] + dm) / dj), 0, 31);
                l[32 * j + ii] = li as u8;
            }
        }

        // qh at [off+16 .. off+48], qs at [off+48 .. off+176].
        let (qh, qs) = out[off + 16..off + 176].split_at_mut(32);
        for b in qh.iter_mut() {
            *b = 0;
        }
        let mut m1: u8 = 1;
        let mut m2: u8 = 2;
        for n in (0..QKK).step_by(64) {
            let chunk = n / 64;
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[chunk * 32 + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
    }
}

fn dequant_q5_k_row(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QKK;
    for i in 0..nb {
        let off = i * Q5K_BYTES;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let min = f16_to_f32(f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]));
        let scales = &bytes[off + 4..off + 16];
        let qh = &bytes[off + 16..off + 48];
        let ql = &bytes[off + 48..off + 176];
        let mut is = 0usize;
        let mut qp = 0usize;
        let mut yo = i * QKK;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        while is < 8 {
            let (sc1, m1v) = get_scale_min_k4(is, scales);
            let (sc2, m2v) = get_scale_min_k4(is + 1, scales);
            let d1 = d * (sc1 as f32);
            let m1 = min * (m1v as f32);
            let d2 = d * (sc2 as f32);
            let m2 = min * (m2v as f32);
            for li in 0..32 {
                let h = if qh[li] & u1 != 0 { 16.0 } else { 0.0 };
                out[yo + li] = d1 * (((ql[qp + li] & 0x0f) as f32) + h) - m1;
            }
            for li in 0..32 {
                let h = if qh[li] & u2 != 0 { 16.0 } else { 0.0 };
                out[yo + 32 + li] = d2 * (((ql[qp + li] >> 4) as f32) + h) - m2;
            }
            yo += 64;
            qp += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ============================== dispatch ==============================

// ============================== Q3_K ==============================
// 16 sub-blocks of 16. Symmetric 3-bit: 2 bits in `qs` + 1 bit in `hmask`. Per-sub-block scale
// searched with `make_q3_quants(16,4,do_rmse=true)` (no-imatrix `_ref`) or `make_qx_quants(16,4,
// rmse=1,weight)` (imatrix `_impl`); the 16 sub-scales (6-bit) pack into `scales[12]` against the
// super-block `d` (f16). Struct field order (ggml block_q3_K): hmask[32] | qs[64] | scales[12] | d(2).
const Q3K_HMASK: usize = 0;
const Q3K_QS: usize = 32;
const Q3K_SCALES: usize = 96;
const Q3K_D: usize = 108;
const Q3K_BYTES: usize = 110;

/// Pack 16 six-bit sub-scales (`l6`, each in `[0,63]`) into the 12-byte ggml Q3_K scale layout.
/// `s` (12 bytes) must be pre-zeroed. Mirrors ggml's `_ref`/`_impl` scale-packing step.
fn pack_q3_scales(s: &mut [u8], l6: &[u8; 16]) {
    for j in 0..16 {
        let mut v = l6[j];
        if j < 8 {
            s[j] |= v & 0x0f;
        } else {
            s[j - 8] |= (v & 0x0f) << 4;
        }
        v >>= 4;
        s[(j % 4) + 8] |= v << (2 * (j / 4));
    }
}

/// Reconstruct the signed 8-bit sub-scale for sub-block `j` from the 12 packed bytes (offset base).
/// `sc = (low4 | (high2 << 4)) - 32`, matching ggml's `_ref`/`_impl` re-quant read-back.
#[inline(always)]
fn get_q3_scale(s: &[u8], j: usize) -> i8 {
    let low4: u8 = if j < 8 { s[j] & 0x0f } else { s[j - 8] >> 4 };
    let high2: u8 = (s[8 + (j % 4)] >> (2 * (j / 4))) & 3;
    (low4 | (high2 << 4)) as i8 - 32
}

fn quant_q3_k_row(x: &[f32], out: &mut [u8], im: Option<&[f32]>) {
    let nb = x.len() / QKK;
    let mut l = [0i8; QKK];
    let mut scales = [0f32; QKK / 16];
    let mut weight = [0f32; 16];
    let mut sw = [0f32; QKK / 16];
    let mut ls = [0i8; QKK / 16];
    for i in 0..nb {
        let xb = &x[i * QKK..i * QKK + QKK];
        let off = i * Q3K_BYTES;
        let s = &mut out[off + Q3K_SCALES..off + Q3K_SCALES + 12];
        for b in s.iter_mut() {
            *b = 0;
        }

        if let Some(im) = im {
            // imatrix (`_impl`) path: weight = qw * sqrt(sigma2 + x²), sub-block scale via
            // make_qx_quants(rmse=1, weight); super-block d via make_qx_quants(16, 32, sw).
            let sum_x2: f32 = (0..QKK).map(|k| xb[k] * xb[k]).sum();
            let sigma2 = 2.0 * sum_x2 / QKK as f32;
            for j in 0..QKK / 16 {
                let qw = &im[i * QKK + 16 * j..i * QKK + 16 * j + 16];
                for k in 0..16 {
                    weight[k] = qw[k] * (sigma2 + xb[16 * j + k] * xb[16 * j + k]).sqrt();
                }
                let mut sumw = 0.0f32;
                for k in 0..16 {
                    sumw += weight[k];
                }
                sw[j] = sumw;
                scales[j] = make_qx_quants(
                    16,
                    4,
                    &xb[16 * j..16 * j + 16],
                    &mut l[16 * j..16 * j + 16],
                    1,
                    Some(&weight),
                );
            }
            let d_block = make_qx_quants(QKK / 16, 32, &scales, &mut ls, 1, Some(&sw));
            let mut l6 = [0u8; 16];
            for j in 0..16 {
                l6[j] = ls[j] as u8; // make_qx_quants nmax=32 → L in [0,63]
            }
            pack_q3_scales(s, &l6);
            out[off + Q3K_D..off + Q3K_D + 2].copy_from_slice(&f32_to_f16(d_block).to_le_bytes());
        } else {
            // no-imatrix (`_ref`) path: sub-block scale via make_q3_quants(do_rmse=true);
            // super-block d = 1/iscale, iscale = -32/max_scale.
            let mut max_scale = 0.0f32;
            let mut amax = 0.0f32;
            for j in 0..QKK / 16 {
                scales[j] = make_q3_quants(16, 4, &xb[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16], true);
                let scale = scales[j].abs();
                if scale > amax {
                    amax = scale;
                    max_scale = scales[j];
                }
            }
            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                let mut l6 = [0u8; 16];
                for j in 0..16 {
                    let lj = clamp_i32(nearest_int(iscale * scales[j]), -32, 31) + 32;
                    l6[j] = lj as u8; // [0,63]
                }
                pack_q3_scales(s, &l6);
                out[off + Q3K_D..off + Q3K_D + 2].copy_from_slice(&f32_to_f16(1.0 / iscale).to_le_bytes());
            } else {
                out[off + Q3K_D..off + Q3K_D + 2].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
            }
        }

        // Re-quant the levels from the packed (f16-rounded) d and packed sub-scales (both paths).
        let d = f16_to_f32(f16::from_le_bytes([out[off + Q3K_D], out[off + Q3K_D + 1]]));
        let s_ro = &out[off + Q3K_SCALES..off + Q3K_SCALES + 12];
        for j in 0..QKK / 16 {
            let djs = d * get_q3_scale(s_ro, j) as f32;
            if djs == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let li = nearest_int(xb[16 * j + ii] / djs);
                l[16 * j + ii] = (clamp_i32(li, -4, 3) + 4) as i8; // [0,7]
            }
        }

        // hmask: high bit for values > 3, then subtract 4 → 2-bit residual in [0,3].
        for b in 0..QKK / 8 {
            out[off + Q3K_HMASK + b] = 0;
        }
        let mut m = 0usize;
        let mut hm = 1u8;
        for j in 0..QKK {
            if l[j] > 3 {
                out[off + Q3K_HMASK + m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QKK / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        // qs: pack 4 consecutive 2-bit residuals per byte.
        for j in (0..QKK).step_by(128) {
            for k in 0..32 {
                out[off + Q3K_QS + j / 4 + k] =
                    (l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6)) as u8;
            }
        }
    }
}

fn dequant_q3_k_row(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QKK;
    let kmask1 = 0x03030303u32;
    let kmask2 = 0x0f0f0f0fu32;
    for i in 0..nb {
        let off = i * Q3K_BYTES;
        let d_all = f16_to_f32(f16::from_le_bytes([bytes[off + Q3K_D], bytes[off + Q3K_D + 1]]));
        let q = &bytes[off + Q3K_QS..off + Q3K_QS + 64];
        let hm = &bytes[off + Q3K_HMASK..off + Q3K_HMASK + 32];
        let s = &bytes[off + Q3K_SCALES..off + Q3K_SCALES + 12];

        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        aux[1] = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
        aux[2] = u32::from_le_bytes([s[8], s[9], s[10], s[11]]);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let mut sb = [0u8; 16];
        for k in 0..4 {
            sb[4 * k..4 * k + 4].copy_from_slice(&aux[k].to_le_bytes());
        }

        let mut is = 0usize;
        let mut m = 1u8;
        let mut yw = i * QKK;
        let mut qpos = 0usize;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl = d_all * (sb[is] as i8 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = ((q[qpos + l] >> shift) & 3) as i8;
                    let hv: i8 = if (hm[l] & m) != 0 { 0 } else { 4 };
                    out[yw] = dl * (qv - hv) as f32;
                    yw += 1;
                }
                let dl = d_all * (sb[is] as i8 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = ((q[qpos + l + 16] >> shift) & 3) as i8;
                    let hv: i8 = if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                    out[yw] = dl * (qv - hv) as f32;
                    yw += 1;
                }
                shift += 2;
                m <<= 1;
            }
            qpos += 32;
        }
    }
}

// ============================== Q2_K ==============================
// 16 sub-blocks of 16. Asymmetric 2-bit (scale+min): levels 0..3 in `qs` (2 bits each). Per-sub-block
// scale+min searched with `make_qkx2_quants(16,3,|x|,...)` (no-imatrix `_ref`) or `make_qkx3_quants`
// + `make_qp_quants(16,15,...)` (imatrix `_impl`); the 16 sub-scales+mins pack 4 bits each into
// `scales[16]` against the super-block `d`/`dmin` (f16). Struct: scales[16] | qs[64] | d(2) | dmin(2).
const Q2K_SCALES: usize = 0;
const Q2K_QS: usize = 16;
const Q2K_D: usize = 80;
const Q2K_DMIN: usize = 82;
const Q2K_BYTES: usize = 84;

fn quant_q2_k_row(x: &[f32], out: &mut [u8], im: Option<&[f32]>) {
    let nb = x.len() / QKK;
    let mut l = [0u8; QKK];
    let mut laux = [0u8; 16];
    let mut weight = [0f32; 16];
    let mut mins = [0f32; QKK / 16];
    let mut scales = [0f32; QKK / 16];
    let mut sw = [0f32; QKK / 16];
    let mut ls = [0u8; QKK / 16];
    let mut lm = [0u8; QKK / 16];
    for i in 0..nb {
        let xb = &x[i * QKK..i * QKK + QKK];
        let off = i * Q2K_BYTES;
        let sc = off + Q2K_SCALES; // base index into the 16 scale bytes (avoids a held borrow across d/dmin writes)

        if let Some(im) = im {
            // imatrix (`_impl`) path: sigma2 = sum_x2/QK_K (no factor of 2), weight = qw*sqrt(sigma2+x²),
            // sub-block via make_qkx3_quants; super-block d/dmin via make_qp_quants(16,15,sw).
            let sum_x2: f32 = (0..QKK).map(|k| xb[k] * xb[k]).sum();
            let sigma2 = sum_x2 / QKK as f32;
            for j in 0..QKK / 16 {
                let qw = &im[i * QKK + 16 * j..i * QKK + 16 * j + 16];
                for k in 0..16 {
                    weight[k] = qw[k] * (sigma2 + xb[16 * j + k] * xb[16 * j + k]).sqrt();
                }
                let mut sumw = 0.0f32;
                for k in 0..16 {
                    sumw += weight[k];
                }
                sw[j] = sumw;
                scales[j] = make_qkx2_quants(
                    16,
                    3,
                    &xb[16 * j..16 * j + 16],
                    &weight,
                    &mut l[16 * j..16 * j + 16],
                    &mut mins[j],
                    &mut laux,
                    -0.9,
                    0.05,
                    36,
                    false,
                );
            }
            let dm = make_qp_quants(QKK / 16, 15, &scales, &mut ls, &sw);
            let mm = make_qp_quants(QKK / 16, 15, &mins, &mut lm, &sw);
            out[off + Q2K_D..off + Q2K_D + 2].copy_from_slice(&f32_to_f16(dm).to_le_bytes());
            out[off + Q2K_DMIN..off + Q2K_DMIN + 2].copy_from_slice(&f32_to_f16(mm).to_le_bytes());
            let dm = f16_to_f32(f16::from_le_bytes([out[off + Q2K_D], out[off + Q2K_D + 1]]));
            let mm = f16_to_f32(f16::from_le_bytes([out[off + Q2K_DMIN], out[off + Q2K_DMIN + 1]]));
            for j in 0..QKK / 16 {
                out[sc + j] = ls[j] | (lm[j] << 4);
            }
            for j in 0..QKK / 16 {
                let d = dm * (out[sc + j] & 0x0f) as f32;
                if d == 0.0 {
                    continue;
                }
                let m = mm * (out[sc + j] >> 4) as f32;
                for ii in 0..16 {
                    let li = nearest_int((xb[16 * j + ii] + m) / d);
                    l[16 * j + ii] = clamp_i32(li, 0, 3) as u8;
                }
            }
        } else {
            // no-imatrix (`_ref`) path: weight = |x|, sub-block via make_qkx2_quants(16,3);
            // super-block d = max_scale/15, dmin = max_min/15.
            let q4scale = 15.0f32;
            let mut max_scale = 0.0f32;
            let mut max_min = 0.0f32;
            for j in 0..QKK / 16 {
                for k in 0..16 {
                    weight[k] = xb[16 * j + k].abs();
                }
                scales[j] = make_qkx2_quants(
                    16,
                    3,
                    &xb[16 * j..16 * j + 16],
                    &weight,
                    &mut l[16 * j..16 * j + 16],
                    &mut mins[j],
                    &mut laux,
                    -0.5,
                    0.1,
                    15,
                    true,
                );
                if scales[j] > max_scale {
                    max_scale = scales[j];
                }
                if mins[j] > max_min {
                    max_min = mins[j];
                }
            }
            if max_scale > 0.0 {
                let iscale = q4scale / max_scale;
                for j in 0..QKK / 16 {
                    out[sc + j] = nearest_int(iscale * scales[j]) as u8;
                }
                out[off + Q2K_D..off + Q2K_D + 2].copy_from_slice(&f32_to_f16(max_scale / q4scale).to_le_bytes());
            } else {
                for j in 0..QKK / 16 {
                    out[sc + j] = 0;
                }
                out[off + Q2K_D..off + Q2K_D + 2].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
            }
            if max_min > 0.0 {
                let iscale = q4scale / max_min;
                for j in 0..QKK / 16 {
                    out[sc + j] |= (nearest_int(iscale * mins[j]) as u8) << 4;
                }
                out[off + Q2K_DMIN..off + Q2K_DMIN + 2].copy_from_slice(&f32_to_f16(max_min / q4scale).to_le_bytes());
            } else {
                out[off + Q2K_DMIN..off + Q2K_DMIN + 2].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
            }
            let d = f16_to_f32(f16::from_le_bytes([out[off + Q2K_D], out[off + Q2K_D + 1]]));
            let dm = f16_to_f32(f16::from_le_bytes([out[off + Q2K_DMIN], out[off + Q2K_DMIN + 1]]));
            for j in 0..QKK / 16 {
                let dd = d * (out[sc + j] & 0x0f) as f32;
                if dd == 0.0 {
                    continue;
                }
                let dm = dm * (out[sc + j] >> 4) as f32;
                for ii in 0..16 {
                    let li = nearest_int((xb[16 * j + ii] + dm) / dd);
                    l[16 * j + ii] = clamp_i32(li, 0, 3) as u8;
                }
            }
        }

        // qs: pack 4 consecutive 2-bit levels per byte.
        for j in (0..QKK).step_by(128) {
            for k in 0..32 {
                out[off + Q2K_QS + j / 4 + k] =
                    l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6);
            }
        }
    }
}

fn dequant_q2_k_row(bytes: &[u8], out: &mut [f32]) {
    let nb = out.len() / QKK;
    for i in 0..nb {
        let off = i * Q2K_BYTES;
        let d = f16_to_f32(f16::from_le_bytes([bytes[off + Q2K_D], bytes[off + Q2K_D + 1]]));
        let min = f16_to_f32(f16::from_le_bytes([bytes[off + Q2K_DMIN], bytes[off + Q2K_DMIN + 1]]));
        let q = &bytes[off + Q2K_QS..off + Q2K_QS + 64];
        let sc = &bytes[off + Q2K_SCALES..off + Q2K_SCALES + 16];
        let mut is = 0usize;
        let mut yw = i * QKK;
        let mut qpos = 0usize;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let s = sc[is];
                is += 1;
                let dl = d * (s & 0x0f) as f32;
                let ml = min * (s >> 4) as f32;
                for l in 0..16 {
                    out[yw] = dl * ((q[qpos + l] >> shift) & 3) as i8 as f32 - ml;
                    yw += 1;
                }
                let s = sc[is];
                is += 1;
                let dl = d * (s & 0x0f) as f32;
                let ml = min * (s >> 4) as f32;
                for l in 0..16 {
                    out[yw] = dl * ((q[qpos + l + 16] >> shift) & 3) as i8 as f32 - ml;
                    yw += 1;
                }
                shift += 2;
            }
            qpos += 32;
        }
    }
}

/// Quantize `rows`×`cols` f32 into `out`. `cols` must be divisible by 256.
pub fn quantize_rows(
    ggml_type: u32,
    x: &[f32],
    out: &mut [u8],
    rows: usize,
    cols: usize,
    imatrix: Option<&[f32]>,
) -> Result<()> {
    if cols % QKK != 0 {
        bail!("k-quant type {ggml_type}: cols {cols} not divisible by 256");
    }
    if let Some(im) = imatrix {
        if im.len() < cols {
            bail!("imatrix length {} < cols {} for k-quant type {ggml_type}", im.len(), cols);
        }
    }
    let row_bytes = (cols / QKK) * bytes_per_block(ggml_type);
    if out.len() < rows * row_bytes {
        bail!("k-quant: out buffer too small");
    }
    // The imatrix is per-input-channel (length `cols`) and reused across rows, exactly as
    // ggml's `quantize_qX_K(src, dst, nrow, n_per_row, quant_weights)` reuses `quant_weights`
    // for every row. Pass the same `cols`-length slice to each row.
    match ggml_type {
        12 => {
            for r in 0..rows {
                quant_q4_k_row(&x[r * cols..r * cols + cols], &mut out[r * row_bytes..r * row_bytes + row_bytes], imatrix);
            }
            Ok(())
        }
        13 => {
            for r in 0..rows {
                quant_q5_k_row(&x[r * cols..r * cols + cols], &mut out[r * row_bytes..r * row_bytes + row_bytes], imatrix);
            }
            Ok(())
        }
        14 => {
            for r in 0..rows {
                quant_q6_k_row(&x[r * cols..r * cols + cols], &mut out[r * row_bytes..r * row_bytes + row_bytes], imatrix);
            }
            Ok(())
        }
        11 => {
            for r in 0..rows {
                quant_q3_k_row(&x[r * cols..r * cols + cols], &mut out[r * row_bytes..r * row_bytes + row_bytes], imatrix);
            }
            Ok(())
        }
        10 => {
            for r in 0..rows {
                quant_q2_k_row(&x[r * cols..r * cols + cols], &mut out[r * row_bytes..r * row_bytes + row_bytes], imatrix);
            }
            Ok(())
        }
        _ => bail!("k-quant type {ggml_type}: kernel not implemented yet"),
    }
}

pub fn dequantize_rows(
    ggml_type: u32,
    bytes: &[u8],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if cols % QKK != 0 {
        bail!("k-quant type {ggml_type}: cols {cols} not divisible by 256");
    }
    let row_bytes = (cols / QKK) * bytes_per_block(ggml_type);
    match ggml_type {
        12 => {
            for r in 0..rows {
                dequant_q4_k_row(&bytes[r * row_bytes..r * row_bytes + row_bytes], &mut out[r * cols..r * cols + cols]);
            }
            Ok(())
        }
        13 => {
            for r in 0..rows {
                dequant_q5_k_row(&bytes[r * row_bytes..r * row_bytes + row_bytes], &mut out[r * cols..r * cols + cols]);
            }
            Ok(())
        }
        14 => {
            for r in 0..rows {
                dequant_q6_k_row(&bytes[r * row_bytes..r * row_bytes + row_bytes], &mut out[r * cols..r * cols + cols]);
            }
            Ok(())
        }
        11 => {
            for r in 0..rows {
                dequant_q3_k_row(&bytes[r * row_bytes..r * row_bytes + row_bytes], &mut out[r * cols..r * cols + cols]);
            }
            Ok(())
        }
        10 => {
            for r in 0..rows {
                dequant_q2_k_row(&bytes[r * row_bytes..r * row_bytes + row_bytes], &mut out[r * cols..r * cols + cols]);
            }
            Ok(())
        }
        _ => bail!("k-quant type {ggml_type}: dequant not implemented yet"),
    }
}

pub fn bytes_per_block(ggml_type: u32) -> usize {
    match ggml_type {
        12 => Q4K_BYTES,
        13 => Q5K_BYTES,
        14 => Q6K_BYTES,
        11 => Q3K_BYTES,
        10 => Q2K_BYTES,
        _ => 0,
    }
}
