//! KL divergence and the statistics that go with it (DESIGN §3.6).
//!
//! Why KL and not perplexity: perplexity is a *scalar summary of the reference's own likelihood*
//! under the candidate, so it moves only when the quantized model gets worse at the specific
//! tokens the corpus happens to contain. KL(fp16 ‖ quant) compares the whole next-token
//! distribution at every position, which makes it (a) far more sensitive per token — you need
//! hundreds of tokens rather than hundreds of thousands to separate two recipes — and (b) *signed
//! the right way*: it is zero exactly when the distributions agree and grows with any divergence,
//! including the ones that leave perplexity flat because they reshuffle mass among tokens the
//! corpus never picks.
//!
//! Everything here is pure math over logit rows. Where those rows come from — llama.cpp,
//! an in-process forward pass, or a dump from a vLLM server — is [`crate::evaluator`]'s problem.

use serde::{Deserialize, Serialize};

/// Numerically stable `log(Σ exp(x))`.
pub fn log_sum_exp(x: &[f32]) -> f64 {
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return f64::NEG_INFINITY;
    }
    let m = max as f64;
    let s: f64 = x.iter().map(|&v| ((v as f64) - m).exp()).sum();
    m + s.ln()
}

/// Write `log_softmax(logits)` into `out`.
pub fn log_softmax_into(logits: &[f32], out: &mut [f64]) {
    let lse = log_sum_exp(logits);
    for (o, &v) in out.iter_mut().zip(logits) {
        *o = (v as f64) - lse;
    }
}

/// `KL(P ‖ Q)` in nats between two full-vocabulary logit rows, `P` being the reference.
///
/// The argument order matters and is not symmetric: `KL(ref ‖ cand)` weights each term by the
/// *reference's* probability, so it asks "where the fp16 model is confident, does the quant agree".
/// That is the question a quantization recipe is judged on. The reverse would let a candidate hide
/// errors by simply being less confident.
pub fn kl_divergence(reference: &[f32], candidate: &[f32]) -> f64 {
    debug_assert_eq!(reference.len(), candidate.len());
    let lse_p = log_sum_exp(reference);
    let lse_q = log_sum_exp(candidate);
    let mut acc = 0.0f64;
    for (&pl, &ql) in reference.iter().zip(candidate) {
        let lp = (pl as f64) - lse_p;
        let lq = (ql as f64) - lse_q;
        let p = lp.exp();
        if p > 0.0 {
            acc += p * (lp - lq);
        }
    }
    // Tiny negatives are floating-point noise on a quantity that is mathematically ≥ 0.
    acc.max(0.0)
}

/// A sparse (top-k) logit row: ids, their logits, and the log-sum-exp of the *whole* row so the
/// unstored tail mass is recoverable.
#[derive(Debug, Clone)]
pub struct SparseRow<'a> {
    pub ids: &'a [u32],
    pub logits: &'a [f32],
    /// `log Σ_v exp(logit_v)` over the full vocabulary.
    pub full_lse: f64,
    pub vocab: usize,
}

impl SparseRow<'_> {
    /// Probability mass held by the stored ids.
    fn stored_mass(&self) -> f64 {
        self.logits
            .iter()
            .map(|&l| ((l as f64) - self.full_lse).exp())
            .sum()
    }
}

/// `KL(P ‖ Q)` when both rows are stored top-k truncated.
///
/// This is an *approximation with a known direction of error*, and it exists because dumping full
/// logits for a 150k-vocab model over a few thousand tokens is gigabytes per candidate. The terms
/// for reference ids that the candidate also stored are exact. For a reference id outside the
/// candidate's top-k, `q_i` is unknown; we substitute the candidate's tail mass spread uniformly
/// over the unstored vocabulary, which is the maximum-entropy choice given what was kept. Since
/// such an id is by definition below the candidate's k-th largest, that substitution *overstates*
/// `q_i` when the candidate's tail is peaked, so the reported KL is a mild under-estimate on
/// exactly the rows where the two models disagree most.
///
/// Consequence for the caller: top-k KL is fine for *ranking* recipes (the bias is common-mode),
/// but do not quote it as an absolute fidelity number. Capture dense rows if you need that.
pub fn kl_divergence_sparse(reference: &SparseRow<'_>, candidate: &SparseRow<'_>) -> f64 {
    let cand_stored = candidate.stored_mass().clamp(0.0, 1.0);
    let cand_unstored_ids = candidate.vocab.saturating_sub(candidate.ids.len()).max(1);
    // Floor the per-id tail probability so a candidate that stored ~all the mass can't produce an
    // infinite KL from one reference id it happened to omit.
    let tail_q = ((1.0 - cand_stored) / cand_unstored_ids as f64).max(f64::MIN_POSITIVE);

    let mut acc = 0.0f64;
    for (i, &id) in reference.ids.iter().enumerate() {
        let lp = (reference.logits[i] as f64) - reference.full_lse;
        let p = lp.exp();
        if p <= 0.0 {
            continue;
        }
        let lq = match candidate.ids.iter().position(|&c| c == id) {
            Some(j) => (candidate.logits[j] as f64) - candidate.full_lse,
            None => tail_q.ln(),
        };
        acc += p * (lp - lq);
    }
    acc.max(0.0)
}

/// Top-1 agreement: does the candidate's argmax match the reference's?
pub fn top1_agrees(reference: &[f32], candidate: &[f32]) -> bool {
    argmax(reference) == argmax(candidate)
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Aggregated KL statistics over a corpus.
///
/// The distribution is heavy-tailed — most positions are trivially easy and a handful carry the
/// damage — so the mean alone hides regressions. `p99` and `max` are what catch a recipe that is
/// fine on average and catastrophic on the tokens that matter.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KlStats {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    /// Root-mean-square KL — more tail-sensitive than the mean, less brittle than the max.
    pub rms: f64,
    /// Fraction of positions where the argmax token agrees.
    pub top1_agreement: f64,
    /// Perplexity of the reference's targets under each model, when targets were available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_ppl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_ppl: Option<f64>,
}

impl KlStats {
    /// The single number the bit-allocation search consumes as "cost".
    ///
    /// Mean KL is the right default: it is additive across positions, so summing per-tensor
    /// sensitivities approximates the joint effect — which is the assumption the greedy-marginal
    /// allocator rests on. Tail metrics are for humans reading the report, not for the optimizer.
    pub fn cost(&self) -> f64 {
        self.mean
    }
}

/// Streaming accumulator for [`KlStats`].
#[derive(Debug, Default)]
pub struct KlAccumulator {
    values: Vec<f64>,
    top1_hits: usize,
    ref_logprob_sum: f64,
    cand_logprob_sum: f64,
    target_count: usize,
}

impl KlAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one position's KL and whether the argmax agreed.
    pub fn push(&mut self, kl: f64, top1: bool) {
        self.values.push(kl);
        if top1 {
            self.top1_hits += 1;
        }
    }

    /// Record the log-probability both models assign to the actual next token, for perplexity.
    pub fn push_target(&mut self, ref_logprob: f64, cand_logprob: f64) {
        self.ref_logprob_sum += ref_logprob;
        self.cand_logprob_sum += cand_logprob;
        self.target_count += 1;
    }

    /// Convenience: consume a matched pair of dense rows and a target id.
    pub fn push_rows(&mut self, reference: &[f32], candidate: &[f32], target: Option<u32>) {
        self.push(
            kl_divergence(reference, candidate),
            top1_agrees(reference, candidate),
        );
        if let Some(t) = target {
            let t = t as usize;
            if t < reference.len() && t < candidate.len() {
                let lp = (reference[t] as f64) - log_sum_exp(reference);
                let lq = (candidate[t] as f64) - log_sum_exp(candidate);
                self.push_target(lp, lq);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn finish(mut self) -> KlStats {
        let n = self.values.len();
        if n == 0 {
            return KlStats::default();
        }
        let sum: f64 = self.values.iter().sum();
        let sq: f64 = self.values.iter().map(|v| v * v).sum();
        self.values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |p: f64| -> f64 {
            let idx = ((n as f64 - 1.0) * p).round() as usize;
            self.values[idx.min(n - 1)]
        };
        let ppl = |sum_lp: f64| -> Option<f64> {
            if self.target_count == 0 {
                None
            } else {
                Some((-sum_lp / self.target_count as f64).exp())
            }
        };
        KlStats {
            n,
            mean: sum / n as f64,
            median: q(0.5),
            p90: q(0.90),
            p99: q(0.99),
            max: *self.values.last().unwrap(),
            rms: (sq / n as f64).sqrt(),
            top1_agreement: self.top1_hits as f64 / n as f64,
            reference_ppl: ppl(self.ref_logprob_sum),
            candidate_ppl: ppl(self.cand_logprob_sum),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rows_have_zero_kl() {
        let a = [1.0f32, 2.0, 3.0, -1.0];
        assert_eq!(kl_divergence(&a, &a), 0.0);
        assert!(top1_agrees(&a, &a));
    }

    #[test]
    fn kl_is_shift_invariant_in_the_logits() {
        // Adding a constant to every logit is the same distribution.
        let a = [1.0f32, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|v| v + 7.5).collect();
        assert!(kl_divergence(&a, &b) < 1e-12);
    }

    #[test]
    fn kl_matches_a_hand_computed_two_way_case() {
        // P = (0.5, 0.5), Q = (0.25, 0.75): KL = .5·ln2 + .5·ln(2/3) = 0.5·ln(4/3).
        let p = [0.0f32, 0.0];
        let q = [0.0f32, (3.0f64).ln() as f32];
        let expect = 0.5 * (4.0f64 / 3.0).ln();
        assert!(
            (kl_divergence(&p, &q) - expect).abs() < 1e-6,
            "{}",
            kl_divergence(&p, &q)
        );
    }

    #[test]
    fn kl_is_asymmetric() {
        // Do not use complementary Bernoulli distributions here: that special case has equal
        // forward/reverse KL even though KL is not symmetric in general.
        let p = [0.0f32, 0.0, 0.0];
        let q = [0.0f32, 1.0, 4.0];
        assert!((kl_divergence(&p, &q) - kl_divergence(&q, &p)).abs() > 1e-3);
    }

    #[test]
    fn sparse_kl_approximates_dense_when_k_covers_the_mass() {
        // A peaked distribution: top-3 of 8 holds almost everything, so truncation is cheap.
        let refr = [6.0f32, 5.0, 4.0, -4.0, -4.0, -4.0, -4.0, -4.0];
        let cand = [6.0f32, 4.6, 4.2, -4.0, -4.0, -4.0, -4.0, -4.0];
        let dense = kl_divergence(&refr, &cand);

        let ids = [0u32, 1, 2];
        let r = SparseRow {
            ids: &ids,
            logits: &refr[..3],
            full_lse: log_sum_exp(&refr),
            vocab: 8,
        };
        let c = SparseRow {
            ids: &ids,
            logits: &cand[..3],
            full_lse: log_sum_exp(&cand),
            vocab: 8,
        };
        let sparse = kl_divergence_sparse(&r, &c);
        assert!(
            (dense - sparse).abs() < 1e-3,
            "dense {dense} vs sparse {sparse}"
        );
    }

    #[test]
    fn sparse_kl_stays_finite_when_the_candidate_omits_a_reference_id() {
        let refr = [5.0f32, 4.0, 3.0];
        let cand = [5.0f32, 4.0, 3.0];
        let rid = [0u32, 2];
        let cid = [0u32, 1];
        let r = SparseRow {
            ids: &rid,
            logits: &[5.0, 3.0],
            full_lse: log_sum_exp(&refr),
            vocab: 3,
        };
        let c = SparseRow {
            ids: &cid,
            logits: &[5.0, 4.0],
            full_lse: log_sum_exp(&cand),
            vocab: 3,
        };
        let kl = kl_divergence_sparse(&r, &c);
        assert!(kl.is_finite() && kl >= 0.0, "{kl}");
    }

    #[test]
    fn accumulator_reports_quantiles_and_agreement() {
        let mut acc = KlAccumulator::new();
        for i in 0..100 {
            acc.push(i as f64 / 100.0, i % 4 != 0);
        }
        let s = acc.finish();
        assert_eq!(s.n, 100);
        assert!((s.mean - 0.495).abs() < 1e-9);
        assert!((s.median - 0.50).abs() < 0.02);
        assert!((s.p99 - 0.99).abs() < 0.02);
        assert!((s.max - 0.99).abs() < 1e-9);
        assert!((s.top1_agreement - 0.75).abs() < 1e-9);
    }

    #[test]
    fn accumulator_derives_perplexity_from_target_logprobs() {
        let mut acc = KlAccumulator::new();
        // Uniform over 4 tokens: log p = -ln 4, so PPL = 4.
        for _ in 0..10 {
            acc.push(0.0, true);
            acc.push_target(-(4.0f64.ln()), -(4.0f64.ln()));
        }
        let s = acc.finish();
        assert!((s.reference_ppl.unwrap() - 4.0).abs() < 1e-9);
        assert!((s.candidate_ppl.unwrap() - 4.0).abs() < 1e-9);
    }
}
