//! `requant sensitivity`: measure per-tensor `(bits → ΔKL)` curves and write them for `search`.
//!
//! This is the expensive command and the one that makes the rest of the tool an optimizer rather
//! than a quantizer. It writes one candidate model per (group × precision), scores each against
//! the fp16 reference, and emits a JSON table `requant search --sensitivity` consumes.
//!
//! Budget it deliberately: `--grouping role` (the default) is ~10 groups; `--grouping tensor` on a
//! large model is hundreds. The `--max-candidates` guard exists to make an accidental thousand-model
//! run fail immediately rather than four hours in.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use requant_calib::load_imatrix;
use requant_eval::evaluator::{Evaluator, LlamaCppEvaluator, LogitDumpEvaluator};
use requant_eval::sensitivity::{run_sensitivity, Grouping, SensitivityPlan};
use requant_quant::Bits;

/// Options for `requant sensitivity`.
#[derive(Debug, Clone, Default)]
pub struct SensitivityOpts {
    pub input: String,
    pub corpus: String,
    pub output: String,
    pub work_dir: Option<String>,
    pub imatrix: Option<String>,
    pub grouping: Option<String>,
    pub ladder: Option<String>,
    pub roles: Option<String>,
    pub max_candidates: Option<usize>,
    pub keep_candidates: bool,
    /// Directory of pre-captured logit dumps — the path for architectures llama.cpp can't run.
    pub logits_dir: Option<String>,
    /// Vocabulary size, required when the dumps are bare fp32 rather than `RQLG`.
    pub raw_vocab: Option<usize>,
    /// Extra flags forwarded to every `llama-perplexity` invocation.
    pub llama_args: Vec<String>,
}

fn parse_grouping(s: Option<&str>) -> Result<Grouping> {
    let Some(s) = s else {
        return Ok(Grouping::PerRole);
    };
    let s = s.trim().to_ascii_lowercase();
    if let Some(rest) = s.strip_prefix("role-depth") {
        let n: usize = rest
            .trim_start_matches([':', '='])
            .trim()
            .parse()
            .unwrap_or(4);
        return Ok(Grouping::PerRoleDepth { buckets: n.max(1) });
    }
    Ok(match s.as_str() {
        "tensor" | "per-tensor" => Grouping::PerTensor,
        "role" | "per-role" => Grouping::PerRole,
        other => bail!("unknown --grouping `{other}` (try `role`, `tensor`, or `role-depth:4`)"),
    })
}

fn parse_ladder(s: Option<&str>) -> Result<Vec<Bits>> {
    let Some(s) = s else {
        return Ok(vec![
            Bits::Q2_K,
            Bits::Q3_K,
            Bits::Q4_K,
            Bits::Q5_K,
            Bits::Q6_K,
            Bits::Q8_0,
        ]);
    };
    let mut out = Vec::new();
    for tok in s.split(',') {
        out.push(
            Bits::from_name(tok)
                .ok_or_else(|| anyhow::anyhow!("unknown format `{}` in --ladder", tok.trim()))?,
        );
    }
    if out.is_empty() {
        bail!("--ladder resolved to an empty list");
    }
    Ok(out)
}

pub fn run_sensitivity_cmd(opts: &SensitivityOpts) -> Result<()> {
    let work_dir = PathBuf::from(opts.work_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir()
            .join("requant-sensitivity")
            .display()
            .to_string()
    }));

    let mut plan = SensitivityPlan::new(&opts.input, &opts.corpus, &work_dir);
    plan.grouping = parse_grouping(opts.grouping.as_deref())?;
    plan.ladder = parse_ladder(opts.ladder.as_deref())?;
    plan.keep_candidates = opts.keep_candidates || opts.logits_dir.is_some();
    plan.max_candidates = opts.max_candidates.or(Some(256));
    plan.roles = opts.roles.as_ref().map(|r| {
        r.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });

    if let Some(p) = &opts.imatrix {
        let im = load_imatrix(p).with_context(|| format!("loading imatrix `{p}`"))?;
        eprintln!(
            "sensitivity: loaded imatrix ({} tensors) from `{p}`",
            im.len()
        );
        plan.imatrix = Some(im);
    }

    // Picking the evaluator is the decision described in `requant_eval::evaluator`: local runtime
    // if one exists, external dumps if not.
    let evaluator: Box<dyn Evaluator> = match &opts.logits_dir {
        Some(dir) => {
            let mut ev = LogitDumpEvaluator::new(dir);
            ev.raw_vocab = opts.raw_vocab;
            eprintln!(
                "sensitivity: using pre-captured logit dumps in `{dir}`. Candidates will be \
                 written to {} and kept; run each through your serving stack and drop the dumps \
                 alongside `reference` before scoring.",
                work_dir.display()
            );
            Box::new(ev)
        }
        None => Box::new(LlamaCppEvaluator::new(&work_dir)?.with_args(opts.llama_args.clone())),
    };

    let table = run_sensitivity(&plan, evaluator.as_ref())?;
    table.write_json(&opts.output)?;

    println!("sensitivity: {} -> {}", opts.input, opts.output);
    println!("  evaluator : {}", table.evaluator);
    println!("  grouping  : {:?}", table.grouping);
    if let Some(p) = table.reference_ppl {
        println!("  ref PPL   : {p:.4}");
    }
    println!();
    println!(
        "  {:<28} {:>8} {:>10} {:>14}",
        "group", "members", "bits", "mean ΔKL"
    );
    for e in &table.entries {
        for p in &e.points {
            println!(
                "  {:<28} {:>8} {:>10} {:>14.6e}",
                e.key,
                e.members.len(),
                p.bits,
                p.kl
            );
        }
    }
    println!();
    println!("Feed this to the allocator:");
    println!(
        "  requant search --input {} --budget <size> --sensitivity {}",
        opts.input, opts.output
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_spellings_parse() {
        assert_eq!(parse_grouping(None).unwrap(), Grouping::PerRole);
        assert_eq!(parse_grouping(Some("tensor")).unwrap(), Grouping::PerTensor);
        assert_eq!(parse_grouping(Some("Role")).unwrap(), Grouping::PerRole);
        assert_eq!(
            parse_grouping(Some("role-depth:8")).unwrap(),
            Grouping::PerRoleDepth { buckets: 8 }
        );
        assert_eq!(
            parse_grouping(Some("role-depth")).unwrap(),
            Grouping::PerRoleDepth { buckets: 4 }
        );
        assert!(parse_grouping(Some("nonsense")).is_err());
    }

    #[test]
    fn ladder_parses_and_rejects_junk() {
        assert_eq!(
            parse_ladder(Some("Q4_K,Q6_K")).unwrap(),
            vec![Bits::Q4_K, Bits::Q6_K]
        );
        assert_eq!(
            parse_ladder(Some("nvfp4,fp8")).unwrap(),
            vec![Bits::NVFP4, Bits::FP8_E4M3]
        );
        assert!(parse_ladder(Some("Q4_K,NOPE")).is_err());
    }
}
