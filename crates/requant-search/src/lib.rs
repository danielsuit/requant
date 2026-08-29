//! requant-search: turn per-tensor cost curves into a recipe under a byte budget.
//!
//! Two halves:
//!
//! - [`proxy`] builds cost curves for free, from the weights and the imatrix, using the §2.3
//!   objective as a stand-in for "how much does the model care".
//! - [`knapsack`] allocates against whatever curves it is given — greedy-marginal over convex
//!   hulls (§3.7), with the recipe's precision floors as hard constraints.
//!
//! The seam between them is deliberate. [`apply_sensitivity`] swaps the proxy's costs for measured
//! ΔKL from `requant-eval` without the allocator knowing anything changed, because the allocator
//! never cared what the numbers meant — only that they were comparable across tensors and
//! decreasing in bytes. That is what makes the eval harness an upgrade rather than a rewrite.

pub mod knapsack;
pub mod proxy;

pub use knapsack::{allocate, emit_recipe, pareto_front, AllocOpts, Allocation, Candidate, TensorCurve};
pub use proxy::{tensor_curve, ProxyMetric, BLOCKFLOAT_LADDER, FULL_LADDER, IQUANT_LADDER, KQUANT_LADDER};

use requant_eval::SensitivityTable;

/// Replace proxy costs with measured ΔKL where the sensitivity table has an entry.
///
/// Returns `(matched, unmatched)` tensor counts. Matching goes tensor name → group membership →
/// role, so a table captured at role granularity still informs a per-tensor allocation: every
/// tensor of a role inherits that role's measured curve. That inheritance is an assumption, and it
/// is the same one [`requant_eval::Grouping::PerRole`] made when it captured the table.
///
/// Points the table doesn't cover keep their proxy cost. Mixing the two would be incoherent — the
/// units differ by orders of magnitude — so when *any* tensor is matched the unmatched ones are
/// rescaled onto the measured scale by the ratio of the two costs over the tensors that do match.
/// That is a crude bridge, and the honest fix is to capture a table that covers everything;
/// `unmatched > 0` in the CLI output is a prompt to do exactly that.
pub fn apply_sensitivity(curves: &mut [TensorCurve], table: &SensitivityTable) -> (usize, usize) {
    // First pass: collect the scale relationship on tensors present in both.
    let mut ratio_num = 0.0f64;
    let mut ratio_den = 0.0f64;
    for c in curves.iter() {
        if c.points.is_empty() {
            continue;
        }
        let Some(entry) = table.lookup(&c.name, &c.role) else { continue };
        for p in &c.points {
            if let Some(m) = entry.points.iter().find(|m| m.bits == p.bits.name()) {
                ratio_num += m.kl;
                ratio_den += p.cost;
            }
        }
    }
    let scale = if ratio_den > 0.0 && ratio_num > 0.0 { ratio_num / ratio_den } else { 1.0 };

    let mut matched = 0usize;
    let mut unmatched = 0usize;
    for c in curves.iter_mut() {
        if c.points.is_empty() {
            continue;
        }
        match table.lookup(&c.name, &c.role) {
            Some(entry) => {
                let mut hit = false;
                for p in c.points.iter_mut() {
                    if let Some(m) = entry.points.iter().find(|m| m.bits == p.bits.name()) {
                        // The measured ΔKL covers the whole group. Charge this tensor its share of
                        // it, by bytes at the same precision — otherwise a 40-tensor role would
                        // bill each member the full group cost and the allocator would treat that
                        // role as 40× more sensitive than it is.
                        let share = if m.bytes > 0 {
                            (p.bytes as f64 / m.bytes as f64).min(1.0)
                        } else {
                            1.0
                        };
                        p.cost = m.kl * share;
                        hit = true;
                    } else {
                        p.cost *= scale;
                    }
                }
                if hit {
                    matched += 1;
                } else {
                    unmatched += 1;
                }
            }
            None => {
                for p in c.points.iter_mut() {
                    p.cost *= scale;
                }
                unmatched += 1;
            }
        }
        // Costs changed, so the previously-normalized ordering may no longer hold.
        c.normalize();
    }
    (matched, unmatched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use requant_eval::sensitivity::{SensitivityEntry, SensitivityPoint};
    use requant_eval::{Grouping, SENSITIVITY_SCHEMA};
    use requant_quant::Bits;

    fn cand(bits: Bits, bytes: u64, cost: f64) -> Candidate {
        Candidate { bits, ggml_type: bits.to_ggml_type(), bytes, cost }
    }

    fn table() -> SensitivityTable {
        SensitivityTable {
            schema: SENSITIVITY_SCHEMA,
            source: "m.gguf".into(),
            corpus: "c".into(),
            evaluator: "test".into(),
            grouping: Grouping::PerRole,
            reference_ppl: None,
            entries: vec![SensitivityEntry {
                key: "attn_q".into(),
                role: "attn_q".into(),
                depth: 0.5,
                members: vec!["blk.0.attn_q.weight".into()],
                source_bytes: 400,
                points: vec![
                    SensitivityPoint {
                        bits: "Q4_K".into(),
                        ggml_type: 12,
                        type_name: "Q4_K".into(),
                        bytes: 100,
                        kl: 0.02,
                        kl_p99: None,
                        top1_agreement: None,
                        ppl: None,
                    },
                    SensitivityPoint {
                        bits: "Q6_K".into(),
                        ggml_type: 14,
                        type_name: "Q6_K".into(),
                        bytes: 200,
                        kl: 0.004,
                        kl_p99: None,
                        top1_agreement: None,
                        ppl: None,
                    },
                ],
            }],
        }
    }

    #[test]
    fn measured_kl_replaces_proxy_cost_for_matching_tensors() {
        let mut curves = vec![TensorCurve {
            name: "blk.0.attn_q.weight".into(),
            role: "attn_q".into(),
            fixed: None,
            fixed_bytes: 0,
            points: vec![cand(Bits::Q4_K, 100, 7.5), cand(Bits::Q6_K, 200, 1.25)],
        }];
        curves[0].normalize();
        let (m, u) = apply_sensitivity(&mut curves, &table());
        assert_eq!((m, u), (1, 0));
        // Single-member group, so this tensor carries the whole measured ΔKL.
        assert!((curves[0].points[0].cost - 0.02).abs() < 1e-12);
        assert!((curves[0].points[1].cost - 0.004).abs() < 1e-12);
    }

    #[test]
    fn a_multi_tensor_group_splits_its_measured_cost_by_share() {
        let mut t = table();
        // Pretend the role covers four identical tensors: group bytes are 4x a member's.
        for p in t.entries[0].points.iter_mut() {
            p.bytes *= 4;
        }
        t.entries[0].members = (0..4).map(|i| format!("blk.{i}.attn_q.weight")).collect();
        let mut curves = vec![TensorCurve {
            name: "blk.2.attn_q.weight".into(),
            role: "attn_q".into(),
            fixed: None,
            fixed_bytes: 0,
            points: vec![cand(Bits::Q4_K, 100, 7.5), cand(Bits::Q6_K, 200, 1.25)],
        }];
        curves[0].normalize();
        apply_sensitivity(&mut curves, &t);
        assert!((curves[0].points[0].cost - 0.02 / 4.0).abs() < 1e-12);
        assert!((curves[0].points[1].cost - 0.004 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn a_tensor_of_the_same_role_inherits_the_role_curve() {
        let mut curves = vec![TensorCurve {
            name: "blk.9.attn_q.weight".into(), // not a listed member
            role: "attn_q".into(),
            fixed: None,
            fixed_bytes: 0,
            points: vec![cand(Bits::Q4_K, 100, 7.5), cand(Bits::Q6_K, 200, 1.25)],
        }];
        curves[0].normalize();
        let (m, u) = apply_sensitivity(&mut curves, &table());
        assert_eq!((m, u), (1, 0), "role fallback should match");
    }

    #[test]
    fn unmatched_tensors_are_rescaled_not_left_on_a_different_scale() {
        let mut curves = vec![
            TensorCurve {
                name: "blk.0.attn_q.weight".into(),
                role: "attn_q".into(),
                fixed: None,
                fixed_bytes: 0,
                points: vec![cand(Bits::Q4_K, 100, 10.0), cand(Bits::Q6_K, 200, 2.0)],
            },
            TensorCurve {
                name: "blk.0.ffn_up.weight".into(),
                role: "ffn_up".into(),
                fixed: None,
                fixed_bytes: 0,
                points: vec![cand(Bits::Q4_K, 100, 10.0), cand(Bits::Q6_K, 200, 2.0)],
            },
        ];
        for c in curves.iter_mut() {
            c.normalize();
        }
        let (m, u) = apply_sensitivity(&mut curves, &table());
        assert_eq!((m, u), (1, 1));
        // The unmatched curve must end up on the same order of magnitude as the matched one, or
        // the allocator would spend every byte on whichever scale happened to be larger.
        let matched_max = curves[0].points[0].cost;
        let unmatched_max = curves[1].points[0].cost;
        assert!(
            unmatched_max < matched_max * 100.0 && unmatched_max > matched_max / 100.0,
            "scales diverged: {matched_max} vs {unmatched_max}"
        );
    }
}
