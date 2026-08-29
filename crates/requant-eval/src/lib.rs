//! requant-eval: the fidelity harness (DESIGN §3.6).
//!
//! Three layers, deliberately separable:
//!
//! - [`kl`] — the math. KL divergence and its aggregate statistics over logit rows. No I/O.
//! - [`logits`] / [`perplexity`] / [`evaluator`] — where logits come from. Either llama.cpp runs
//!   the model for us, or something else did and we diff its dumps. The [`evaluator::Evaluator`]
//!   trait is the seam, and picking the right side of it is the single most consequential decision
//!   in this crate — see that module's docs.
//! - [`sensitivity`] — the ablation loop that turns "quantize one thing at a time" into the
//!   per-tensor `(bits → ΔKL)` curves `requant-search` allocates against.
//!
//! The whole point of the split: whole-model perplexity tells you *whether* a recipe is good;
//! per-tensor KL tells you *which tensor to spend the next byte on*. Only the second closes the
//! optimizer loop, and only the second needs machinery beyond shelling out to llama.cpp.

pub mod evaluator;
pub mod kl;
pub mod logits;
pub mod perplexity;
pub mod sensitivity;

pub use evaluator::{
    compare_perplexity, Evaluator, LlamaCppEvaluator, LogitDumpEvaluator, Reference, Score,
};
pub use kl::{kl_divergence, kl_divergence_sparse, KlAccumulator, KlStats, SparseRow};
pub use logits::{LogitRow, LogitStore, NO_TARGET};
pub use perplexity::{find_llama_perplexity, parse_first_f64, parse_kl_report, parse_ppl, PplResult};
pub use sensitivity::{
    emit_candidates, run_sensitivity, score_candidates, CandidateManifest, Grouping,
    SensitivityEntry, SensitivityPlan, SensitivityPoint, SensitivityTable, SENSITIVITY_SCHEMA,
};
