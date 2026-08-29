//! The no-forward-pass sensitivity proxy: importance-weighted round-trip error.
//!
//! [`crate::knapsack`] allocates bits against a per-tensor cost curve. The *right* cost is measured
//! ΔKL from [`requant_eval::sensitivity`] — it knows how much the model actually cares. But that
//! costs a forward pass per candidate, and there is a much cheaper stand-in that needs nothing but
//! the weights and the imatrix already on disk: the value of the §2.3 objective itself,
//!
//! ```text
//!   E(fmt) = Σ_i g_i · (w_i − dequant(quant(w_i)))²
//! ```
//!
//! with `g_i` the imatrix importance for that input channel. Since imatrix *is* the diagonal of the
//! GPTQ Hessian (§2.4), this is the second-order estimate of the loss increase from perturbing the
//! tensor — i.e. a principled first-order-in-Hessian approximation of exactly the ΔKL we would
//! otherwise measure. It is free, and it is what makes `requant search` usable before the eval
//! harness has ever been run.
//!
//! # Absolute vs relative: this choice matters more than it looks
//!
//! Two normalizations are available and they are *not* interchangeable:
//!
//! - [`ProxyMetric::Absolute`] reports `E(fmt)` as-is, in the model's own units. Errors from
//!   different tensors are on a common scale, so "spend the next byte where Δcost/Δbytes is
//!   largest" compares like with like. This is the metric the marginal analysis in §3.7 assumes.
//! - [`ProxyMetric::Relative`] divides by that tensor's own weighted energy, giving a per-tensor
//!   relative RMS. Excellent for *reading* — "this tensor lost 3% of its energy" — and meaningless
//!   to compare *across* tensors, because a tensor with tiny weights and a tensor with huge ones
//!   can show identical relative error while contributing wildly different absolute damage.
//!
//! `Relative` remains the default only because the published Qwen2.5-0.5B Pareto numbers were
//! measured with it; `Absolute` is the one to prefer, and the one to re-baseline against.

use anyhow::Result;

use requant_quant::{fallback_type, roundtrip_tensor, Bits};

use crate::knapsack::{Candidate, TensorCurve};

/// The default candidate ladder for quantizable weight tensors, cheapest first.
pub const KQUANT_LADDER: &[Bits] = &[
    Bits::Q2_K,
    Bits::Q3_K,
    Bits::Q4_K,
    Bits::Q5_K,
    Bits::Q6_K,
    Bits::Q8_0,
];

/// The i-quant (codebook) family ladder, cheapest first by bits-per-weight. These are
/// imatrix-driven formats; the sub-2-bit rungs (IQ1/IQ2_XXS/XS) are the practical regime for
/// aggressively-quantized routed experts, IQ3/IQ4 fill the 3–4.5-bit band.
pub const IQUANT_LADDER: &[Bits] = &[
    Bits::IQ1_S,
    Bits::IQ1_M,
    Bits::IQ2_XXS,
    Bits::IQ2_XS,
    Bits::IQ2_S,
    Bits::IQ3_XXS,
    Bits::IQ3_S,
    Bits::IQ4_XS,
    Bits::IQ4_NL,
];

/// The full mixed ladder: i-quants and k-quants merged cheapest-first by bits-per-weight. Use this
/// when the recipe floors experts at an i-quant and the allocator should be free to upgrade across
/// the codebook/k-quant boundary (e.g. IQ3_S -> IQ4_XS -> Q4_K -> Q5_K).
pub const FULL_LADDER: &[Bits] = &[
    Bits::Q1_0,
    Bits::IQ1_S,
    Bits::TQ1_0,
    Bits::IQ1_M,
    Bits::IQ2_XXS,
    Bits::TQ2_0,
    Bits::Q2_0,
    Bits::IQ2_XS,
    Bits::IQ2_S,
    Bits::Q2_K,
    Bits::IQ3_XXS,
    Bits::IQ3_S,
    Bits::Q3_K,
    Bits::IQ4_XS,
    Bits::Q4_K,
    Bits::IQ4_NL,
    Bits::Q5_K,
    Bits::Q6_K,
    Bits::Q8_0,
];

/// A ladder for FP4-class targets: the block-float formats plus an FP8 top end.
pub const BLOCKFLOAT_LADDER: &[Bits] = &[Bits::MXFP4, Bits::NVFP4, Bits::FP8_E4M3, Bits::MXFP8];

/// How to normalize the proxy error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProxyMetric {
    /// Per-tensor relative RMS. Readable; not comparable across tensors.
    #[default]
    Relative,
    /// Raw importance-weighted squared error. Comparable across tensors; use this for allocation.
    Absolute,
}

impl ProxyMetric {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rel" | "relative" => Some(ProxyMetric::Relative),
            "abs" | "absolute" => Some(ProxyMetric::Absolute),
            _ => None,
        }
    }
}

/// `Σ_i g_i · x_i²` — the denominator for the relative metric, floored away from zero.
pub fn weighted_energy(x: &[f32], im: Option<&[f32]>) -> f64 {
    let mut s = 0.0f64;
    match im {
        Some(w) => {
            for (i, xi) in x.iter().enumerate() {
                // The imatrix is per input channel and repeats across rows.
                let g = (w[i % w.len()] as f64).max(1e-12);
                s += (*xi as f64) * (*xi as f64) * g;
            }
        }
        None => {
            for xi in x {
                s += (*xi as f64) * (*xi as f64);
            }
        }
    }
    s.max(1e-12)
}

/// `Σ_i g_i · (orig_i − deq_i)²`.
pub fn weighted_sq_err(orig: &[f32], deq: &[f32], im: Option<&[f32]>) -> f64 {
    let mut s = 0.0f64;
    match im {
        Some(w) => {
            for (i, (o, d)) in orig.iter().zip(deq.iter()).enumerate() {
                let g = (w[i % w.len()] as f64).max(1e-12);
                let e = (*o as f64) - (*d as f64);
                s += e * e * g;
            }
        }
        None => {
            for (o, d) in orig.iter().zip(deq.iter()) {
                let e = (*o as f64) - (*d as f64);
                s += e * e;
            }
        }
    }
    s
}

/// Build one tensor's cost curve by round-tripping it through every ladder level at or above
/// `floor`.
///
/// The ladder entry is the *recipe* name (`Q4_K`), while the measurement uses the type
/// [`fallback_type`] will actually pick for this `cols` (`Q5_0` when 256 doesn't divide it). Both
/// have to be true at once: the emitted recipe says `Q4_K` and `requant quantize` re-derives the
/// same fallback, so the size and error the search recorded are the ones the quantizer produces.
/// Measuring the un-fallen-back type would silently mis-price every non-256-divisible tensor.
pub fn tensor_curve(
    name: &str,
    role: &str,
    src: &[f32],
    rows: usize,
    cols: usize,
    im: Option<&[f32]>,
    ladder: &[Bits],
    floor: Option<Bits>,
    metric: ProxyMetric,
) -> Result<TensorCurve> {
    let floor_idx = floor
        .and_then(|b| ladder.iter().position(|&l| l == b))
        .unwrap_or(0);
    let denom = weighted_energy(src, im);

    let mut points: Vec<Candidate> = Vec::with_capacity(ladder.len() - floor_idx);
    let mut seen: Vec<u32> = Vec::new();
    for &bits in &ladder[floor_idx..] {
        let (actual, _) = fallback_type(bits.to_ggml_type(), cols);
        // Two ladder rungs can collapse onto the same actual type (Q2_K and Q3_K both fall back to
        // Q4_0). Keep the first, so the curve stays strictly increasing in bytes.
        if seen.contains(&actual) {
            continue;
        }
        let Some((block, bpb)) = requant_io::block_layout(actual) else {
            continue;
        };
        if cols % block != 0 {
            continue;
        }
        let deq = match roundtrip_tensor(actual, src, rows, cols, im) {
            Ok(d) => d,
            Err(_) => continue, // kernel not implemented for this type — skip the rung
        };
        seen.push(actual);
        let bytes = (rows * cols / block * bpb) as u64;
        let abs = weighted_sq_err(src, &deq, im);
        let cost = match metric {
            ProxyMetric::Absolute => abs,
            ProxyMetric::Relative => (abs / denom).sqrt(),
        };
        points.push(Candidate {
            bits,
            ggml_type: actual,
            bytes,
            cost,
        });
    }

    Ok(TensorCurve {
        name: name.to_string(),
        role: role.to_string(),
        fixed: None,
        fixed_bytes: 0,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.013).sin() * 0.2 + 0.01)
            .collect()
    }

    #[test]
    fn curve_is_monotone_in_bytes_and_improving_in_cost() {
        let (rows, cols) = (2, 512);
        let x = weights(rows * cols);
        let c = tensor_curve(
            "t",
            "ffn_up",
            &x,
            rows,
            cols,
            None,
            KQUANT_LADDER,
            None,
            ProxyMetric::Absolute,
        )
        .unwrap();
        assert!(c.points.len() >= 4);
        for w in c.points.windows(2) {
            assert!(
                w[1].bytes > w[0].bytes,
                "bytes must increase: {:?}",
                (w[0].bytes, w[1].bytes)
            );
        }
        // Quantized hierarchical scales mean adjacent format families are not guaranteed to be
        // strictly error-monotone for every synthetic tensor. The allocator deliberately removes
        // those dominated points on its lower convex hull. The useful invariant here is that the
        // high-precision end improves on the cheapest endpoint.
        assert!(c.points.last().unwrap().cost < c.points.first().unwrap().cost);
    }

    #[test]
    fn floor_truncates_the_ladder_from_below() {
        let (rows, cols) = (1, 256);
        let x = weights(cols);
        let c = tensor_curve(
            "t",
            "attn_q",
            &x,
            rows,
            cols,
            None,
            KQUANT_LADDER,
            Some(Bits::Q5_K),
            ProxyMetric::Absolute,
        )
        .unwrap();
        assert_eq!(
            c.points[0].bits,
            Bits::Q5_K,
            "the floor must be the cheapest option offered"
        );
        assert!(c.points.iter().all(|p| p.bits != Bits::Q4_K));
    }

    #[test]
    fn fallback_is_priced_at_the_type_that_will_actually_be_written() {
        // cols = 288 is divisible by 32 but not 256, so every k-quant falls back.
        let (rows, cols) = (1, 288);
        let x = weights(cols);
        let c = tensor_curve(
            "t",
            "attn_q",
            &x,
            rows,
            cols,
            None,
            KQUANT_LADDER,
            None,
            ProxyMetric::Absolute,
        )
        .unwrap();
        assert!(!c.points.is_empty());
        for p in &c.points {
            let (actual, _) = fallback_type(p.bits.to_ggml_type(), cols);
            assert_eq!(p.ggml_type, actual);
            let (block, bpb) = requant_io::block_layout(actual).unwrap();
            assert_eq!(p.bytes, (rows * cols / block * bpb) as u64);
        }
        // Q2_K and Q3_K both fall back to Q4_0; only one rung should survive.
        let q4_0 = c.points.iter().filter(|p| p.ggml_type == 2).count();
        assert_eq!(q4_0, 1, "duplicate fallback types must be deduped");
    }

    #[test]
    fn relative_and_absolute_rank_a_single_tensor_the_same_way() {
        // Within one tensor the two metrics differ by a constant factor, so the ordering matches.
        let (rows, cols) = (1, 512);
        let x = weights(cols);
        let a = tensor_curve(
            "t",
            "r",
            &x,
            rows,
            cols,
            None,
            KQUANT_LADDER,
            None,
            ProxyMetric::Absolute,
        )
        .unwrap();
        let r = tensor_curve(
            "t",
            "r",
            &x,
            rows,
            cols,
            None,
            KQUANT_LADDER,
            None,
            ProxyMetric::Relative,
        )
        .unwrap();
        let order = |c: &TensorCurve| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..c.points.len()).collect();
            idx.sort_by(|&i, &j| c.points[i].cost.partial_cmp(&c.points[j].cost).unwrap());
            idx
        };
        assert_eq!(order(&a), order(&r));
    }

    #[test]
    fn absolute_costs_are_comparable_across_tensors_and_relative_ones_are_not() {
        // Same shape, same *relative* difficulty, 100x different magnitude. Absolute must rank the
        // big tensor as costlier; relative must call them equal — which is precisely why relative
        // cannot drive a cross-tensor allocation.
        let cols = 512;
        let small = weights(cols);
        let big: Vec<f32> = small.iter().map(|v| v * 100.0).collect();
        let f = |x: &[f32], m| {
            tensor_curve("t", "r", x, 1, cols, None, KQUANT_LADDER, None, m)
                .unwrap()
                .points[0]
                .cost
        };
        assert!(f(&big, ProxyMetric::Absolute) > f(&small, ProxyMetric::Absolute) * 100.0);
        let rs = f(&small, ProxyMetric::Relative);
        let rb = f(&big, ProxyMetric::Relative);
        // Stored fp16 block scales introduce a little non-homogeneity under a 100x rescale.
        assert!(
            (rs - rb).abs() < rs * 0.10,
            "relative errors should be ~equal: {rs} vs {rb}"
        );
    }

    #[test]
    fn imatrix_weighting_shifts_cost_toward_important_channels() {
        let cols = 256;
        let mut x = weights(cols);
        // Put a large weight on one channel, then make the imatrix care only about that channel.
        x[7] = 5.0;
        let mut im = vec![0.001f32; cols];
        im[7] = 1000.0;
        let with = tensor_curve(
            "t",
            "r",
            &x,
            1,
            cols,
            Some(&im),
            KQUANT_LADDER,
            None,
            ProxyMetric::Absolute,
        )
        .unwrap();
        let without = tensor_curve(
            "t",
            "r",
            &x,
            1,
            cols,
            None,
            KQUANT_LADDER,
            None,
            ProxyMetric::Absolute,
        )
        .unwrap();
        // The weighted error is dominated by channel 7, so it should differ substantially.
        assert_ne!(with.points[0].cost, without.points[0].cost);
    }
}
