//! The `Evaluator` seam: "score this candidate model against the fp16 reference."
//!
//! # Read this before committing to an eval path
//!
//! The sensitivity harness needs to score O(tensors × ladder-levels) candidate models. *How* a
//! candidate gets scored is the single decision that determines whether step 1 is a weekend or a
//! quarter, and it is decided by one question: **does anything you can run locally implement this
//! model's architecture?**
//!
//! - **Yes (llama.cpp supports it).** Use [`LlamaCppEvaluator`]. It shells out to
//!   `llama-perplexity --kl-divergence-base` / `--kl-divergence`, which is the same code path that
//!   will serve the GGUF. Nothing to port, numbers you can defend. This is the path for
//!   GLM-4.5-Air, Qwen3-30B-A3B, and everything else already in the GGUF ecosystem — which is
//!   exactly why those are the models to prove the optimizer loop on.
//!
//! - **No.** A model with novel compressed attention (V4-Flash-class CSA/HCA) is not going to be
//!   in llama.cpp or candle on day one, and writing that attention twice — once to eval, once
//!   because you needed to eval — is a model-porting project wearing an eval project's clothes.
//!   Use [`LogitDumpEvaluator`]: run each candidate through the stack that *does* implement it
//!   (vLLM on the Blackwell box), dump logits, diff here. Slower per candidate, zero architecture
//!   work, and the reference logits come from the real serving path.
//!
//! Check which case you are in *before* building the ablation loop, because it changes the shape
//! of the loop: the llama.cpp path can materialise candidates locally and score them in-process,
//! while the dump path has to ship each candidate to the serving box, so it wants far fewer, far
//! coarser candidates (role-level grouping, not per-tensor).
//!
//! There is no third option hiding here. An in-process candle forward pass is the same
//! architecture-porting cost as the llama.cpp one, just paid to a different library; it is worth
//! it only when you want per-*layer* activations rather than logits, which the sensitivity metric
//! does not need.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::kl::KlStats;
use crate::logits::LogitStore;
use crate::perplexity::{
    find_llama_perplexity, parse_ppl, run_kl_divergence, run_perplexity, write_kl_base, PplResult,
};

/// A prepared reference artifact. Built once from the fp16 model, reused for every candidate.
pub enum Reference {
    /// A `llama-perplexity --kl-divergence-base` file. Opaque to us by design.
    LlamaBase { path: PathBuf, ppl: Option<PplResult> },
    /// Captured logits we can diff in-process.
    Logits { store: Box<LogitStore>, ppl: Option<f64> },
}

impl Reference {
    pub fn reference_ppl(&self) -> Option<f64> {
        match self {
            Reference::LlamaBase { ppl, .. } => ppl.as_ref().map(|p| p.ppl),
            Reference::Logits { ppl, .. } => *ppl,
        }
    }
}

/// One candidate's score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub kl: KlStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ppl: Option<f64>,
}

/// Scores candidate models against a reference.
pub trait Evaluator {
    fn name(&self) -> &str;

    /// Build the reference artifact from the full-precision model.
    fn prepare_reference(&self, reference_model: &Path, corpus: &Path) -> Result<Reference>;

    /// Score one candidate. `candidate` is a path whose meaning is backend-specific: a GGUF for
    /// [`LlamaCppEvaluator`], an identifier that names a dump file for [`LogitDumpEvaluator`].
    fn score(&self, candidate: &Path, reference: &Reference, corpus: &Path) -> Result<Score>;

    /// Whether this backend can score arbitrary locally-materialised GGUF candidates. When false,
    /// the sensitivity driver refuses to write candidates it has no way to evaluate rather than
    /// burning hours producing files nothing will read.
    fn scores_local_candidates(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// llama.cpp
// ---------------------------------------------------------------------------

/// Scores GGUF candidates by shelling out to `llama-perplexity`.
pub struct LlamaCppEvaluator {
    pub bin: String,
    /// Extra flags passed to every invocation (`-c 512`, `-ngl 99`, `-t 8`, …). Whatever you put
    /// here must be *identical* between the reference capture and every candidate, or the KL
    /// comparison is between different context lengths rather than different quantizations.
    pub extra_args: Vec<String>,
    pub work_dir: PathBuf,
}

impl LlamaCppEvaluator {
    pub fn new(work_dir: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self { bin: find_llama_perplexity()?, extra_args: Vec::new(), work_dir: work_dir.into() })
    }

    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }
}

impl Evaluator for LlamaCppEvaluator {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn prepare_reference(&self, reference_model: &Path, corpus: &Path) -> Result<Reference> {
        std::fs::create_dir_all(&self.work_dir)
            .with_context(|| format!("creating work dir {}", self.work_dir.display()))?;
        let base = self.work_dir.join("reference.kldbase");
        eprintln!("eval: capturing reference logits from {} …", reference_model.display());
        let ppl = write_kl_base(&self.bin, reference_model, corpus, &base, &self.extra_args)?;
        Ok(Reference::LlamaBase { path: base, ppl: Some(ppl) })
    }

    fn score(&self, candidate: &Path, reference: &Reference, _corpus: &Path) -> Result<Score> {
        let Reference::LlamaBase { path, .. } = reference else {
            bail!("LlamaCppEvaluator needs a LlamaBase reference; got a logit capture");
        };
        let (kl, ppl) = run_kl_divergence(&self.bin, candidate, path, &self.extra_args)?;
        Ok(Score { kl, ppl: ppl.map(|p| p.ppl) })
    }
}

/// Standalone whole-model PPL comparison, for `requant eval`.
pub fn compare_perplexity(
    quant: &Path,
    reference: &Path,
    corpus: &Path,
    extra: &[String],
) -> Result<(PplResult, PplResult)> {
    let bin = find_llama_perplexity()?;
    eprintln!("eval: reference ppl over `{}` …", corpus.display());
    let r = run_perplexity(&bin, reference, corpus, extra)?;
    eprintln!("eval: quant ppl over `{}` …", corpus.display());
    let q = run_perplexity(&bin, quant, corpus, extra)?;
    Ok((r, q))
}

/// Parse a perplexity report captured elsewhere (used by tests and by `--from-log`).
pub fn ppl_from_log(text: &str) -> Result<PplResult> {
    parse_ppl(text)
}

// ---------------------------------------------------------------------------
// external logit dumps
// ---------------------------------------------------------------------------

/// Scores candidates by diffing pre-captured logit dumps.
///
/// The contract is a directory of `<id>.rqlg` files (or raw fp32 with `vocab` set): one for the
/// reference, one per candidate, all produced over the *same* token sequence. The candidate path
/// handed to [`Evaluator::score`] is interpreted as an id — its file stem — so the sensitivity
/// driver's candidate naming lines up with the dump names without either side knowing about the
/// other.
///
/// This is the path for architectures no local runtime implements. It trades wall-clock and a
/// manual capture step for not having to port an attention implementation.
pub struct LogitDumpEvaluator {
    pub dir: PathBuf,
    pub reference_id: String,
    /// Set when the dumps are bare fp32 rather than `RQLG`.
    pub raw_vocab: Option<usize>,
}

impl LogitDumpEvaluator {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), reference_id: "reference".to_string(), raw_vocab: None }
    }

    fn load(&self, id: &str) -> Result<LogitStore> {
        match self.raw_vocab {
            Some(vocab) => {
                let p = self.dir.join(format!("{id}.f32"));
                LogitStore::read_raw_f32(&p, vocab)
                    .with_context(|| format!("loading raw logit dump {}", p.display()))
            }
            None => {
                let p = self.dir.join(format!("{id}.rqlg"));
                LogitStore::read(&p)
                    .with_context(|| format!("loading logit dump {}", p.display()))
            }
        }
    }

    /// Id for a candidate path — the file stem, so `.../blk.0.attn_q@Q4_K.gguf` looks for
    /// `blk.0.attn_q@Q4_K.rqlg`.
    fn id_of(candidate: &Path) -> String {
        candidate
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| candidate.display().to_string())
    }
}

impl Evaluator for LogitDumpEvaluator {
    fn name(&self) -> &str {
        "logit-dump"
    }

    fn prepare_reference(&self, _reference_model: &Path, _corpus: &Path) -> Result<Reference> {
        let store = self.load(&self.reference_id)?;
        Ok(Reference::Logits { store: Box::new(store), ppl: None })
    }

    fn score(&self, candidate: &Path, reference: &Reference, _corpus: &Path) -> Result<Score> {
        let Reference::Logits { store, .. } = reference else {
            bail!("LogitDumpEvaluator needs a captured-logits reference");
        };
        let cand = self.load(&Self::id_of(candidate))?;
        let kl = store.compare(&cand)?;
        let ppl = kl.candidate_ppl;
        Ok(Score { kl, ppl })
    }

    fn scores_local_candidates(&self) -> bool {
        // Candidates must be run through the external stack first; there is nothing this process
        // can do with a freshly written GGUF except write it.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logits::{LogitRow, LogitStore};

    fn store(vocab: usize, n: usize, jitter: f32) -> LogitStore {
        let mut s = LogitStore::new(vocab as u32, 0);
        for i in 0..n {
            let row: Vec<f32> = (0..vocab)
                .map(|v| (((i * 7 + v * 3) % 11) as f32) * 0.3 + (v as f32) * jitter)
                .collect();
            s.push(LogitRow::Dense(row), (i % vocab) as u32);
        }
        s
    }

    #[test]
    fn logit_dump_evaluator_scores_from_files() {
        let dir = tempfile::tempdir().unwrap();
        store(32, 10, 0.0).write_to(dir.path().join("reference.rqlg")).unwrap();
        store(32, 10, 0.0).write_to(dir.path().join("cand_same.rqlg")).unwrap();
        store(32, 10, 0.05).write_to(dir.path().join("cand_off.rqlg")).unwrap();

        let ev = LogitDumpEvaluator::new(dir.path());
        assert!(!ev.scores_local_candidates());
        let reference = ev.prepare_reference(Path::new("unused"), Path::new("unused")).unwrap();

        let same = ev.score(Path::new("cand_same.gguf"), &reference, Path::new("c")).unwrap();
        assert!(same.kl.mean < 1e-12, "identical dumps must score 0, got {}", same.kl.mean);

        let off = ev.score(Path::new("cand_off.gguf"), &reference, Path::new("c")).unwrap();
        assert!(off.kl.mean > same.kl.mean, "a perturbed dump must score worse");
    }

    #[test]
    fn missing_dump_is_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        store(8, 2, 0.0).write_to(dir.path().join("reference.rqlg")).unwrap();
        let ev = LogitDumpEvaluator::new(dir.path());
        let reference = ev.prepare_reference(Path::new("x"), Path::new("y")).unwrap();
        let err = ev
            .score(Path::new("nope.gguf"), &reference, Path::new("c"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope.rqlg"), "{err}");
    }
}
