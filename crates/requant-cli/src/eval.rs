//! `requant eval`: whole-model fidelity of a quantized GGUF against its reference.
//!
//! Two modes:
//!
//! - **Perplexity** (default) — the headline number, comparable to anything else quoted in the
//!   ecosystem. Also the *insensitive* one: a PPL delta smaller than the reported `±` is noise.
//! - **`--kl`** — KL(reference ‖ quant) over the same corpus, which separates recipes that
//!   perplexity calls identical. Prefer this when comparing two candidates rather than reporting
//!   an absolute.
//!
//! Both delegate to `llama-perplexity`; see `requant_eval::perplexity` for why.

use std::path::Path;

use anyhow::Result;

use requant_eval::evaluator::{compare_perplexity, Evaluator, LlamaCppEvaluator};

pub fn run_eval(quant: &str, reference: &str, calib: Option<&str>) -> Result<()> {
    run_eval_ex(quant, reference, calib, false, &[])
}

pub fn run_eval_ex(
    quant: &str,
    reference: &str,
    calib: Option<&str>,
    kl: bool,
    extra_args: &[String],
) -> Result<()> {
    let corpus = Path::new(calib.unwrap_or("wikitext-2-raw/wiki.train.raw"));

    if kl {
        let work = std::env::temp_dir().join("requant-eval");
        let ev = LlamaCppEvaluator::new(&work)?.with_args(extra_args.to_vec());
        let reference_art = ev.prepare_reference(Path::new(reference), corpus)?;
        let score = ev.score(Path::new(quant), &reference_art, corpus)?;
        println!();
        println!("eval results (KL divergence, reference ‖ quant)");
        println!("  reference : {reference}");
        println!("  quant     : {quant}");
        println!("  corpus    : {}", corpus.display());
        println!();
        println!("  mean KLD   : {:.6}", score.kl.mean);
        println!("  median KLD : {:.6}", score.kl.median);
        println!("  p99 KLD    : {:.6}", score.kl.p99);
        println!("  max KLD    : {:.6}", score.kl.max);
        println!("  top-1 agree: {:.2}%", score.kl.top1_agreement * 100.0);
        if let Some(p) = score.ppl {
            println!("  quant PPL  : {p:.4}");
        }
        if let Some(r) = reference_art.reference_ppl() {
            println!("  ref PPL    : {r:.4}");
        }
        return Ok(());
    }

    let (r, q) = compare_perplexity(Path::new(quant), Path::new(reference), corpus, extra_args)?;
    println!();
    println!("eval results");
    println!(
        "  reference ({reference}): {:.4}{}",
        r.ppl,
        fmt_err(r.stderr)
    );
    println!(
        "  quant     ({quant}):     {:.4}{}",
        q.ppl,
        fmt_err(q.stderr)
    );
    let deg = q.ppl - r.ppl;
    let rel = if r.ppl > 0.0 {
        deg / r.ppl * 100.0
    } else {
        0.0
    };
    println!("  delta: {deg:+.4} ppl  ({rel:+.2}% relative)");
    // A delta inside the measurement uncertainty is not a result. Say so rather than letting a
    // reader treat 0.01 as an improvement.
    if let (Some(re), Some(qe)) = (r.stderr, q.stderr) {
        let noise = (re * re + qe * qe).sqrt();
        if deg.abs() < noise {
            println!(
                "  note: |delta| {:.4} is within the combined uncertainty ±{noise:.4} — these two \
                 models are indistinguishable on this corpus. Use `--kl` to separate them.",
                deg.abs()
            );
        }
    }
    Ok(())
}

fn fmt_err(e: Option<f64>) -> String {
    e.map(|v| format!(" ± {v:.4}")).unwrap_or_default()
}
