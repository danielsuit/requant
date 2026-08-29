//! Bit-exact oracle: compares our legacy quantize/dequantize against ggml's canonical
//! `quantize_row_*_ref` / `dequantize_row_*` kernels (linked from libggml-base via the
//! `ggml-oracle` feature). If our bytes match the reference, they match `llama-quantize`.
#![cfg(feature = "ggml-oracle")]

use requant_quant::legacy;

// FFI to ggml reference kernels. `k` is the element count; output buffers are sized by the caller.
extern "C" {
    fn quantize_row_q4_0_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_q4_1_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_q5_0_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_q5_1_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_q8_0_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_q8_1_ref(x: *const f32, y: *mut u8, k: i64);

    fn dequantize_row_q4_0(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q4_1(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q5_0(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q5_1(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q8_0(x: *const u8, y: *mut f32, k: i64);

    fn quantize_row_q6_K_ref(x: *const f32, y: *mut u8, k: i64);
    fn dequantize_row_q6_K(x: *const u8, y: *mut f32, k: i64);

    fn quantize_row_q4_K_ref(x: *const f32, y: *mut u8, k: i64);
    fn dequantize_row_q4_K(x: *const u8, y: *mut f32, k: i64);
    fn quantize_row_q5_K_ref(x: *const f32, y: *mut u8, k: i64);
    fn dequantize_row_q5_K(x: *const u8, y: *mut f32, k: i64);
    fn quantize_row_q3_K_ref(x: *const f32, y: *mut u8, k: i64);
    fn dequantize_row_q3_K(x: *const u8, y: *mut f32, k: i64);
    fn quantize_row_q2_K_ref(x: *const f32, y: *mut u8, k: i64);
    fn dequantize_row_q2_K(x: *const u8, y: *mut f32, k: i64);

    // Dispatchers that take a `quant_weights` (imatrix) pointer. With NULL they call the `_ref`
    // (no-imatrix) path; with non-NULL they call the `_impl` (imatrix) path. Returns bytes written.
    fn quantize_q4_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, quant_weights: *const f32) -> usize;
    fn quantize_q5_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, quant_weights: *const f32) -> usize;
    fn quantize_q6_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, quant_weights: *const f32) -> usize;
    fn quantize_q3_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, quant_weights: *const f32) -> usize;
    fn quantize_q2_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, quant_weights: *const f32) -> usize;
}

/// Deterministic pseudo-random vector (no Math.random dependency): LCG over a fixed seed.
fn lcg_vec(n: usize) -> Vec<f32> {
    let mut s: u32 = 0x12345678;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift32 for decent spread
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let f = (s as f32) / (u32::MAX as f32) * 4.0 - 2.0; // range [-2, 2]
        v.push(f);
    }
    v
}

/// Deterministic non-negative imatrix-like weights in [0.1, 2.0] (imatrix entries are mean
/// of squared activations, hence >= 0). Reused across rows exactly as ggml reuses quant_weights.
fn imatrix_vec(n: usize) -> Vec<f32> {
    let mut s: u32 = 0xC0FFEE42;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push(0.1 + (s as f32) / (u32::MAX as f32) * 1.9);
    }
    v
}

const K: usize = 256; // 8 blocks of 32

fn bytes_per_block(t: u32) -> usize { legacy::bytes_per_block(t) }

fn check_quant(t: u32, qref: unsafe extern "C" fn(*const f32, *mut u8, i64)) {
    let x = lcg_vec(K);
    let nbytes = K / 32 * bytes_per_block(t);
    let mut mine = vec![0u8; nbytes];
    let mut ref_ = vec![0u8; nbytes];
    legacy::quantize_row(t, &x, &mut mine);
    unsafe { qref(x.as_ptr(), ref_.as_mut_ptr(), K as i64) };
    assert_eq!(mine, ref_, "quant bytes differ for type {t}");
}

fn check_dequant(t: u32, dref: unsafe extern "C" fn(*const u8, *mut f32, i64)) {
    let x = lcg_vec(K);
    let nbytes = K / 32 * bytes_per_block(t);
    let mut bytes = vec![0u8; nbytes];
    legacy::quantize_row(t, &x, &mut bytes);
    let mut mine = vec![0f32; K];
    let mut ref_ = vec![0f32; K];
    legacy::dequantize_row(t, &bytes, &mut mine);
    unsafe { dref(bytes.as_ptr(), ref_.as_mut_ptr(), K as i64) };
    assert_eq!(mine, ref_, "dequant differs for type {t}");
}

#[test]
fn q4_0_matches_ref() { check_quant(2, quantize_row_q4_0_ref); check_dequant(2, dequantize_row_q4_0); }
#[test]
fn q4_1_matches_ref() { check_quant(3, quantize_row_q4_1_ref); check_dequant(3, dequantize_row_q4_1); }
#[test]
fn q5_0_matches_ref() { check_quant(6, quantize_row_q5_0_ref); check_dequant(6, dequantize_row_q5_0); }
#[test]
fn q5_1_matches_ref() { check_quant(7, quantize_row_q5_1_ref); check_dequant(7, dequantize_row_q5_1); }
#[test]
fn q8_0_matches_ref() { check_quant(8, quantize_row_q8_0_ref); check_dequant(8, dequantize_row_q8_0); }
#[test]
fn q8_1_matches_ref() { check_quant(9, quantize_row_q8_1_ref); }

// ---- k-quants ----
fn check_kquant(t: u32, bytes_per_256: usize, qref: unsafe extern "C" fn(*const f32, *mut u8, i64)) {
    let x = lcg_vec(256);
    let nbytes = bytes_per_256;
    let mut mine = vec![0u8; nbytes];
    let mut ref_ = vec![0u8; nbytes];
    requant_quant::kquant::quantize_rows(t, &x, &mut mine, 1, 256, None).unwrap();
    unsafe { qref(x.as_ptr(), ref_.as_mut_ptr(), 256) };
    assert_eq!(mine, ref_, "k-quant bytes differ for type {t}");
}

fn check_kdequant(t: u32, bytes_per_256: usize, dref: unsafe extern "C" fn(*const u8, *mut f32, i64)) {
    let x = lcg_vec(256);
    let nbytes = bytes_per_256;
    let mut bytes = vec![0u8; nbytes];
    requant_quant::kquant::quantize_rows(t, &x, &mut bytes, 1, 256, None).unwrap();
    let mut mine = vec![0f32; 256];
    let mut ref_ = vec![0f32; 256];
    requant_quant::kquant::dequantize_rows(t, &bytes, &mut mine, 1, 256).unwrap();
    unsafe { dref(bytes.as_ptr(), ref_.as_mut_ptr(), 256) };
    assert_eq!(mine, ref_, "k-quant dequant differs for type {t}");
}

#[test]
fn q6_k_matches_ref() {
    check_kquant(14, 210, quantize_row_q6_K_ref);
    check_kdequant(14, 210, dequantize_row_q6_K);
}

#[test]
fn q4_k_matches_ref() {
    check_kquant(12, 144, quantize_row_q4_K_ref);
    check_kdequant(12, 144, dequantize_row_q4_K);
}

#[test]
fn q5_k_matches_ref() {
    check_kquant(13, 176, quantize_row_q5_K_ref);
    check_kdequant(13, 176, dequantize_row_q5_K);
}

#[test]
fn q3_k_matches_ref() {
    check_kquant(11, 110, quantize_row_q3_K_ref);
    check_kdequant(11, 110, dequantize_row_q3_K);
}

#[test]
fn q2_k_matches_ref() {
    check_kquant(10, 84, quantize_row_q2_K_ref);
    check_kdequant(10, 84, dequantize_row_q2_K);
}

#[test]
fn k_quants_multi_block() {
    // 4 super-blocks (1024 elems) to exercise the 128-element packing loop beyond blk 0.
    let x = lcg_vec(1024);
    for (t, bpb, qref) in [
        (14usize, 210usize, quantize_row_q6_K_ref as unsafe extern "C" fn(*const f32, *mut u8, i64)),
        (12, 144, quantize_row_q4_K_ref),
        (13, 176, quantize_row_q5_K_ref),
        (11, 110, quantize_row_q3_K_ref),
        (10, 84, quantize_row_q2_K_ref),
    ] {
        let nbytes = 4 * bpb;
        let mut mine = vec![0u8; nbytes];
        let mut ref_ = vec![0u8; nbytes];
        requant_quant::kquant::quantize_rows(t as u32, &x, &mut mine, 1, 1024, None).unwrap();
        unsafe { qref(x.as_ptr(), ref_.as_mut_ptr(), 1024) };
        assert_eq!(mine, ref_, "type {t} multi-block mismatch");
    }
}
/// Edge cases: all-zero block, constant block, and a block with a single large outlier.
#[test]
fn edge_cases_q4_0_q8_0() {    for &t in &[2u32, 8] {
        // all zeros
        let mut x = vec![0f32; K];
        let nb = K / 32 * bytes_per_block(t);
        let mut mine = vec![0u8; nb]; let mut rf = vec![0u8; nb];
        legacy::quantize_row(t, &x, &mut mine);
        unsafe { (if t == 2 { quantize_row_q4_0_ref } else { quantize_row_q8_0_ref })(x.as_ptr(), rf.as_mut_ptr(), K as i64) }
        assert_eq!(mine, rf, "zeros type {t}");
        // single outlier at position 10
        x[10] = 1.5;
        legacy::quantize_row(t, &x, &mut mine);
        unsafe { (if t == 2 { quantize_row_q4_0_ref } else { quantize_row_q8_0_ref })(x.as_ptr(), rf.as_mut_ptr(), K as i64) }
        assert_eq!(mine, rf, "outlier type {t}");
    }
}

// ---- imatrix-weighted k-quants (ggml `quantize_qX_K` with non-NULL quant_weights) ----
//
// This is the bit-exactness gate for the imatrix path: our `quantize_rows(..., Some(im))` must
// match ggml's `quantize_qX_K(src, dst, nrow, n_per_row, quant_weights)`, which routes to the
// `_impl` kernels (make_qkx3_quants + make_qp_quants for Q4_K/Q5_K; make_qx_quants with the qw
// override for Q6_K). The no-imatrix path (`_ref`) is already covered above.

fn check_kquant_imatrix(t: u32, bytes_per_256: usize, qdisp: unsafe extern "C" fn(*const f32, *mut u8, i64, i64, *const f32) -> usize) {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    let nbytes = bytes_per_256;
    let mut mine = vec![0u8; nbytes];
    let mut ref_ = vec![0u8; nbytes];
    requant_quant::kquant::quantize_rows(t, &x, &mut mine, 1, cols, Some(&im)).unwrap();
    let wrote = unsafe { qdisp(x.as_ptr(), ref_.as_mut_ptr(), 1, cols as i64, im.as_ptr()) };
    assert_eq!(wrote, nbytes, "ggml wrote {wrote} bytes, expected {nbytes} for type {t}");
    assert_eq!(mine, ref_, "imatrix k-quant bytes differ for type {t}");
}

#[test]
fn q4_k_imatrix_matches_ref() { check_kquant_imatrix(12, 144, quantize_q4_K); }
#[test]
fn q5_k_imatrix_matches_ref() { check_kquant_imatrix(13, 176, quantize_q5_K); }
#[test]
fn q6_k_imatrix_matches_ref() { check_kquant_imatrix(14, 210, quantize_q6_K); }
#[test]
fn q3_k_imatrix_matches_ref() { check_kquant_imatrix(11, 110, quantize_q3_K); }
#[test]
fn q2_k_imatrix_matches_ref() { check_kquant_imatrix(10, 84, quantize_q2_K); }

#[test]
fn k_quants_imatrix_multi_block_and_row() {
    // 4 super-blocks (1024 elems), 3 rows: exercises multi-block packing AND the per-row reuse
    // of the (cols-length) imatrix, which is how ggml applies one quant_weights across nrow.
    let cols = 1024;
    let nrow = 3;
    let x = lcg_vec(cols * nrow);
    let im = imatrix_vec(cols); // length == cols, reused per row
    for (t, bpb, qdisp) in [
        (12usize, 144usize, quantize_q4_K as unsafe extern "C" fn(*const f32, *mut u8, i64, i64, *const f32) -> usize),
        (13, 176, quantize_q5_K),
        (14, 210, quantize_q6_K),
        (11, 110, quantize_q3_K),
        (10, 84, quantize_q2_K),
    ] {
        let nbytes = 4 * bpb * nrow;
        let mut mine = vec![0u8; nbytes];
        let mut ref_ = vec![0u8; nbytes];
        requant_quant::kquant::quantize_rows(t as u32, &x, &mut mine, nrow, cols, Some(&im)).unwrap();
        let wrote = unsafe { qdisp(x.as_ptr(), ref_.as_mut_ptr(), nrow as i64, cols as i64, im.as_ptr()) };
        assert_eq!(wrote, nbytes, "type {t}: ggml wrote {wrote}, expected {nbytes}");
        assert_eq!(mine, ref_, "type {t} imatrix multi-block/row mismatch");
    }
}

#[test]
fn k_quants_null_weights_match_ref_path() {
    // Sanity: quantize_qX_K with NULL quant_weights must equal our None path (the _ref kernel).
    let cols = 256;
    let x = lcg_vec(cols);
    for (t, bpb, qdisp) in [
        (12usize, 144usize, quantize_q4_K as unsafe extern "C" fn(*const f32, *mut u8, i64, i64, *const f32) -> usize),
        (13, 176, quantize_q5_K),
        (14, 210, quantize_q6_K),
        (11, 110, quantize_q3_K),
        (10, 84, quantize_q2_K),
    ] {
        let nbytes = bpb;
        let mut mine = vec![0u8; nbytes];
        let mut ref_ = vec![0u8; nbytes];
        requant_quant::kquant::quantize_rows(t as u32, &x, &mut mine, 1, cols, None).unwrap();
        unsafe { qdisp(x.as_ptr(), ref_.as_mut_ptr(), 1, cols as i64, std::ptr::null()) };
        assert_eq!(mine, ref_, "type {t}: NULL-weights dispatcher != our None path");
    }
}

// ============================== i-quants (codebook family) ==============================
//
// i-quants are imatrix-driven. Five have a no-imatrix `quantize_row_*_ref` kernel; four
// (IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M) are imatrix-only via the `quantize_iqX` dispatchers. With a
// NULL imatrix the dispatcher uses uniform weights. Dequantize needs no imatrix.

extern "C" {
    // Dispatchers (imatrix path). Returns bytes written.
    fn quantize_iq2_xxs(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq2_xs (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq2_s  (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq3_xxs(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq3_s  (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq1_s  (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq1_m  (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq4_nl (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn quantize_iq4_xs (src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, imatrix: *const f32) -> usize;

    // No-imatrix single-row ref kernels (5 of the 9).
    fn quantize_row_iq3_xxs_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_iq4_nl_ref (x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_iq4_xs_ref (x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_iq3_s_ref  (x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_iq2_s_ref  (x: *const f32, y: *mut u8, k: i64);

    // Dequantize (no imatrix needed).
    fn dequantize_row_iq2_xxs(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq2_xs (x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq2_s  (x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq3_xxs(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq3_s  (x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq1_s  (x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq1_m  (x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq4_nl (x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq4_xs (x: *const u8, y: *mut f32, k: i64);

    // Initialise the codebook grids the i-quant dispatchers rely on. Must run before any
    // quantize_iqX / dequantize_row_iqX call (the dispatchers assert the grids are non-NULL).
    fn ggml_quantize_init(type_: u32);
}

/// Bytes per 256-element super-block for an i-quant (IQ4_NL uses block 32, handled separately).
fn iq_bytes_per_256(t: u32) -> usize {
    match t {
        16 => 66, 17 => 74, 18 => 98, 19 => 50, 21 => 110, 22 => 82, 23 => 136, 29 => 56,
        _ => 0,
    }
}

/// Produce ggml-quantized reference bytes for one row of an i-quant via its dispatcher, using a
/// synthetic non-negative imatrix (the imatrix-only types require a non-NULL pointer).
fn iq_ref_bytes(t: u32, x: &[f32], cols: usize, im: &[f32]) -> Vec<u8> {
    let qdisp: unsafe extern "C" fn(*const f32, *mut u8, i64, i64, *const f32) -> usize = match t {
        16 => quantize_iq2_xxs, 17 => quantize_iq2_xs, 18 => quantize_iq3_xxs, 19 => quantize_iq1_s,
        21 => quantize_iq3_s, 22 => quantize_iq2_s, 23 => quantize_iq4_xs, 29 => quantize_iq1_m,
        20 => quantize_iq4_nl,
        _ => unreachable!(),
    };
    let nbytes = if t == 20 { cols / 32 * 18 } else { cols / 256 * iq_bytes_per_256(t) };
    let mut bytes = vec![0u8; nbytes];
    unsafe { ggml_quantize_init(t) };
    let wrote = unsafe { qdisp(x.as_ptr(), bytes.as_mut_ptr(), 1, cols as i64, im.as_ptr()) };
    assert_eq!(wrote, nbytes, "ggml dispatcher wrote {wrote}, expected {nbytes} for iq type {t}");
    bytes
}

/// Dequant-only oracle: produce bytes with ggml's dispatcher, then compare our dequant to ggml's.
/// This validates the dequant kernels independently of the quantize kernels.
fn check_iq_dequant(t: u32, dref: unsafe extern "C" fn(*const u8, *mut f32, i64)) {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    let bytes = iq_ref_bytes(t, &x, cols, &im);
    let mut mine = vec![0f32; cols];
    let mut ref_ = vec![0f32; cols];
    requant_quant::iquant::dequantize_rows(t, &bytes, &mut mine, 1, cols).unwrap();
    unsafe { dref(bytes.as_ptr(), ref_.as_mut_ptr(), cols as i64) };
    assert_eq!(mine, ref_, "iq type {t} dequant differs from ggml");
}

#[test]
fn iq2_xxs_dequant_matches_ref() { check_iq_dequant(16, dequantize_row_iq2_xxs); }
#[test]
fn iq2_xs_dequant_matches_ref()  { check_iq_dequant(17, dequantize_row_iq2_xs); }
#[test]
fn iq2_s_dequant_matches_ref()   { check_iq_dequant(22, dequantize_row_iq2_s); }
#[test]
fn iq3_xxs_dequant_matches_ref() { check_iq_dequant(18, dequantize_row_iq3_xxs); }
#[test]
fn iq3_s_dequant_matches_ref()   { check_iq_dequant(21, dequantize_row_iq3_s); }
#[test]
fn iq1_s_dequant_matches_ref()   { check_iq_dequant(19, dequantize_row_iq1_s); }
#[test]
fn iq1_m_dequant_matches_ref()   { check_iq_dequant(29, dequantize_row_iq1_m); }
#[test]
fn iq4_nl_dequant_matches_ref() { check_iq_dequant(20, dequantize_row_iq4_nl); }
#[test]
fn iq4_xs_dequant_matches_ref() { check_iq_dequant(23, dequantize_row_iq4_xs); }

// ---- i-quant quantize oracle (imatrix path) ----
//
// Compares our `iquant::quantize_rows(..., Some(im))` to ggml's `quantize_iqX` dispatcher with the
// same imatrix, and our `(..., None)` to the dispatcher with NULL (uniform weights). Only the
// kernels we have implemented (IQ4_NL, IQ4_XS so far) are exercised here; the codebook-search
// types are gated on their kernels being ported.

/// Compare our imatrix quantize to ggml's dispatcher for one row.
fn check_iq_quant_imatrix(t: u32, x: &[f32], im: &[f32], cols: usize) {
    let ref_bytes = iq_ref_bytes(t, x, cols, im);
    let nbytes = ref_bytes.len();
    let mut mine = vec![0u8; nbytes];
    requant_quant::iquant::quantize_rows(t, x, &mut mine, 1, cols, Some(im)).unwrap();
    assert_eq!(mine, ref_bytes, "iq type {t} imatrix quant bytes differ from ggml");
}

/// Compare our no-imatrix quantize to ggml's dispatcher with NULL quant_weights.
fn check_iq_quant_null(t: u32, x: &[f32], cols: usize) {
    let qdisp: unsafe extern "C" fn(*const f32, *mut u8, i64, i64, *const f32) -> usize = match t {
        20 => quantize_iq4_nl, 23 => quantize_iq4_xs,
        18 => quantize_iq3_xxs, 21 => quantize_iq3_s, 22 => quantize_iq2_s,
        29 => quantize_iq1_m,
        _ => unreachable!(),
    };
    let nbytes = if t == 20 { cols / 32 * 18 } else { cols / 256 * iq_bytes_per_256(t) };
    let mut mine = vec![0u8; nbytes];
    let mut ref_ = vec![0u8; nbytes];
    requant_quant::iquant::quantize_rows(t, x, &mut mine, 1, cols, None).unwrap();
    unsafe { ggml_quantize_init(t) };
    let wrote = unsafe { qdisp(x.as_ptr(), ref_.as_mut_ptr(), 1, cols as i64, std::ptr::null()) };
    assert_eq!(wrote, nbytes, "ggml NULL dispatcher wrote {wrote}, expected {nbytes} for iq type {t}");
    assert_eq!(mine, ref_, "iq type {t} NULL-weights quant != our None path");
}

#[test]
fn iq4_nl_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(20, &x, &im, cols);
}

#[test]
fn iq4_xs_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(23, &x, &im, cols);
}

#[test]
fn iq4_quant_null_weights_match_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    check_iq_quant_null(20, &x, cols);
    check_iq_quant_null(23, &x, cols);
}

#[test]
fn iq2_xxs_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(16, &x, &im, cols);
}

#[test]
fn iq2_xs_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(17, &x, &im, cols);
}

#[test]
fn iq2_s_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(22, &x, &im, cols);
}

#[test]
fn iq2_s_quant_null_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    check_iq_quant_null(22, &x, cols);
}

#[test]
fn iq3_xxs_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(18, &x, &im, cols);
}

#[test]
fn iq3_xxs_quant_null_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    check_iq_quant_null(18, &x, cols);
}

#[test]
fn iq3_s_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(21, &x, &im, cols);
}

#[test]
fn iq3_s_quant_null_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    check_iq_quant_null(21, &x, cols);
}

#[test]
fn iq1_s_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(19, &x, &im, cols);
}

#[test]
fn iq1_m_quant_imatrix_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    let im = imatrix_vec(cols);
    check_iq_quant_imatrix(29, &x, &im, cols);
}

#[test]
fn iq1_m_quant_null_matches_ref() {
    let cols = 256;
    let x = lcg_vec(cols);
    check_iq_quant_null(29, &x, cols);
}

#[test]
fn iq2_xxs_quant_multi_block_and_row() {
    let cols = 1024;
    let nrow = 3;
    let x = lcg_vec(cols * nrow);
    let im = imatrix_vec(cols);
    let t = 16u32;
    let nbytes = cols / 256 * iq_bytes_per_256(t) * nrow;
    let mut mine = vec![0u8; nbytes];
    let mut ref_ = vec![0u8; nbytes];
    requant_quant::iquant::quantize_rows(t, &x, &mut mine, nrow, cols, Some(&im)).unwrap();
    unsafe { ggml_quantize_init(t) };
    let wrote = unsafe { quantize_iq2_xxs(x.as_ptr(), ref_.as_mut_ptr(), nrow as i64, cols as i64, im.as_ptr()) };
    assert_eq!(wrote, nbytes, "iq type {t}: ggml wrote {wrote}, expected {nbytes}");
    assert_eq!(mine, ref_, "iq type {t} imatrix multi-block/row mismatch");
}

#[test]
fn iq4_quant_multi_block_and_row() {
    // 4 super-blocks (1024 elems), 3 rows: exercises multi-block packing and the per-row reuse of
    // the cols-length imatrix, exactly as ggml's dispatcher reuses quant_weights across nrow.
    let cols = 1024;
    let nrow = 3;
    let x = lcg_vec(cols * nrow);
    let im = imatrix_vec(cols);
    for t in [20u32, 23] {
        let qdisp: unsafe extern "C" fn(*const f32, *mut u8, i64, i64, *const f32) -> usize = match t {
            20 => quantize_iq4_nl, 23 => quantize_iq4_xs, _ => unreachable!(),
        };
        let nbytes = if t == 20 { cols / 32 * 18 * nrow } else { cols / 256 * iq_bytes_per_256(t) * nrow };
        let mut mine = vec![0u8; nbytes];
        let mut ref_ = vec![0u8; nbytes];
        requant_quant::iquant::quantize_rows(t, &x, &mut mine, nrow, cols, Some(&im)).unwrap();
        unsafe { ggml_quantize_init(t) };
        let wrote = unsafe { qdisp(x.as_ptr(), ref_.as_mut_ptr(), nrow as i64, cols as i64, im.as_ptr()) };
        assert_eq!(wrote, nbytes, "iq type {t}: ggml wrote {wrote}, expected {nbytes}");
        assert_eq!(mine, ref_, "iq type {t} imatrix multi-block/row mismatch");
    }
}
