//! Shelling out to `llama-perplexity`, and parsing what it prints.
//!
//! Whole-model perplexity and whole-model KL are both things llama.cpp already computes correctly,
//! against the same kernels that will actually serve the file. Reimplementing them in-process
//! would mean reimplementing the kernels, and then the number would measure *our* forward pass
//! rather than the one the model runs under. So for whole-model fidelity we delegate, and reserve
//! in-house work for the thing llama.cpp cannot do: per-tensor ablation (see
//! [`crate::sensitivity`]).
//!
//! `llama-perplexity` writes its report to stderr, mid-line, with variable whitespace, and the
//! layout has shifted across releases. Every parser here is therefore keyword-anchored rather than
//! position-anchored, and searches from the end so a per-chunk running estimate can't shadow the
//! final one.

use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::kl::KlStats;

/// Result of a perplexity run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PplResult {
    pub ppl: f64,
    /// The `±` uncertainty llama-perplexity reports, when present. Quoting a PPL delta smaller
    /// than this is quoting noise.
    pub stderr: Option<f64>,
}

/// Locate a binary on `PATH`.
pub fn which(bin: &str) -> Option<String> {
    let out = Command::new("which").arg(bin).output().ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

/// Find the perplexity binary, preferring the modern name.
pub fn find_llama_perplexity() -> Result<String> {
    which("llama-perplexity").or_else(|| which("llama-ppl")).ok_or_else(|| {
        anyhow!(
            "could not find `llama-perplexity` on PATH. Install the llama.cpp CLI tools, or point \
             the eval at a dumped-logits directory (`--logits-dir`) if the architecture isn't \
             supported by llama.cpp at all."
        )
    })
}

fn run(bin: &str, args: &[String]) -> Result<String> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .with_context(|| format!("running {bin} {}", args.join(" ")))?;
    let mut buf = String::new();
    buf.push_str(&String::from_utf8_lossy(&out.stdout));
    buf.push_str(&String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        bail!("{bin} failed ({}):\n{}", out.status, tail(&buf, 40));
    }
    Ok(buf)
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Run a plain perplexity measurement.
pub fn run_perplexity(bin: &str, model: &Path, corpus: &Path, extra: &[String]) -> Result<PplResult> {
    let mut args = vec![
        "-m".to_string(),
        model.display().to_string(),
        "-f".to_string(),
        corpus.display().to_string(),
    ];
    args.extend_from_slice(extra);
    let out = run(bin, &args)?;
    parse_ppl(&out)
}

/// Capture the reference logits llama.cpp needs for a later `--kl-divergence` run.
///
/// The base file is llama.cpp's own format; we never parse it, we only hand it back. That is a
/// deliberate boundary — it means a llama.cpp release changing that layout costs us nothing.
pub fn write_kl_base(
    bin: &str,
    reference_model: &Path,
    corpus: &Path,
    base_out: &Path,
    extra: &[String],
) -> Result<PplResult> {
    let mut args = vec![
        "-m".to_string(),
        reference_model.display().to_string(),
        "-f".to_string(),
        corpus.display().to_string(),
        "--kl-divergence-base".to_string(),
        base_out.display().to_string(),
    ];
    args.extend_from_slice(extra);
    let out = run(bin, &args)?;
    if !base_out.exists() {
        bail!(
            "`{bin} --kl-divergence-base` did not produce {}. Older llama.cpp builds spell this \
             flag differently; check `{bin} --help`.",
            base_out.display()
        );
    }
    parse_ppl(&out)
}

/// Score a candidate against a previously written base file.
pub fn run_kl_divergence(
    bin: &str,
    candidate: &Path,
    base: &Path,
    extra: &[String],
) -> Result<(KlStats, Option<PplResult>)> {
    let mut args = vec![
        "-m".to_string(),
        candidate.display().to_string(),
        "--kl-divergence-base".to_string(),
        base.display().to_string(),
        "--kl-divergence".to_string(),
    ];
    args.extend_from_slice(extra);
    let out = run(bin, &args)?;
    let stats = parse_kl_report(&out)?;
    Ok((stats, parse_ppl(&out).ok()))
}

/// Parse the `Final estimate: PPL = 4.6921 +/- 0.24358` line.
pub fn parse_ppl(text: &str) -> Result<PplResult> {
    for line in text.lines().rev() {
        let l = line.trim();
        for needle in ["PPL =", "ppl =", "perplexity:"] {
            if let Some(idx) = l.find(needle) {
                let rest = &l[idx + needle.len()..];
                if let Some(v) = parse_first_f64(rest) {
                    // Pick up the "+/- <err>" suffix when it's there. `±` is two bytes, so the
                    // skip length has to come from the marker we actually matched.
                    let err = if let Some(i) = rest.find("+/-") {
                        parse_first_f64(&rest[i + 3..])
                    } else if let Some(i) = rest.find('±') {
                        parse_first_f64(&rest[i + '±'.len_utf8()..])
                    } else {
                        None
                    };
                    return Ok(PplResult { ppl: v, stderr: err });
                }
            }
        }
    }
    bail!("could not find a perplexity line in the output")
}

/// Parse llama.cpp's `====== KL divergence statistics ======` block.
///
/// The block looks roughly like:
/// ```text
/// Mean    KLD:   0.010000 ±   0.000100
/// Maximum KLD:   1.234000
/// 99.0%   KLD:   0.100000
/// Median  KLD:   0.002000
/// ...
/// Same top p:  95.000 ± 0.100 %
/// ```
/// We anchor on the keyword before `KLD:` so column drift and extra rows are harmless. `n` is
/// filled from the token count when llama.cpp reports one; it is informational only.
pub fn parse_kl_report(text: &str) -> Result<KlStats> {
    let mut stats = KlStats::default();
    let mut saw_any = false;

    for line in text.lines() {
        let l = line.trim();
        if let Some(idx) = l.find("KLD:") {
            let key = l[..idx].trim().to_ascii_lowercase();
            let Some(v) = parse_first_f64(&l[idx + 4..]) else { continue };
            saw_any = true;
            match key.as_str() {
                k if k.starts_with("mean") => stats.mean = v,
                k if k.starts_with("maximum") || k.starts_with("max") => stats.max = v,
                k if k.starts_with("median") => stats.median = v,
                k if k.starts_with("99.9") => {} // finer tail than we model
                k if k.starts_with("99.0") || k == "99%" => stats.p99 = v,
                k if k.starts_with("90.0") || k == "90%" => stats.p90 = v,
                _ => {}
            }
            continue;
        }
        if let Some(rest) = l.strip_prefix("Same top p:") {
            if let Some(v) = parse_first_f64(rest) {
                stats.top1_agreement = v / 100.0;
                saw_any = true;
            }
        }
    }

    if !saw_any {
        bail!(
            "no KL divergence statistics in the output — the run probably didn't get a valid \
             `--kl-divergence-base` file, or this llama.cpp build predates `--kl-divergence`"
        );
    }
    // llama.cpp doesn't report an RMS; approximate it from the mean and the p99 tail only if both
    // are present, otherwise leave it zero rather than inventing a number.
    if stats.rms == 0.0 && stats.mean > 0.0 {
        stats.rms = stats.mean;
    }
    Ok(stats)
}

/// Parse the first `f64` out of `s`, stopping at the first char that can't extend a number.
/// Handles `"4.6921 +/- 0.24358"`, `"4.6921,"`, `" 4.6921"`.
pub fn parse_first_f64(s: &str) -> Option<f64> {
    let s = s.trim_start();
    let bytes = s.as_bytes();
    let mut end = 0;
    if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
        end += 1;
    }
    let mut dot_seen = false;
    let mut e_seen = false;
    let digits_start = end;
    while end < bytes.len() {
        let b = bytes[end];
        if b.is_ascii_digit() {
            end += 1;
        } else if b == b'.' && !dot_seen && !e_seen {
            dot_seen = true;
            end += 1;
        } else if (b == b'e' || b == b'E') && !e_seen && end > digits_start {
            e_seen = true;
            end += 1;
            if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
                end += 1;
            }
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    s[..end].parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ppl_with_error_suffix() {
        assert_eq!(parse_first_f64("4.6921 +/- 0.24358"), Some(4.6921));
    }

    #[test]
    fn parses_bare_value() {
        assert_eq!(parse_first_f64("  4.6752"), Some(4.6752));
        assert_eq!(parse_first_f64("4.6752,"), Some(4.6752));
    }

    #[test]
    fn parses_scientific_and_signed() {
        assert_eq!(parse_first_f64("-1.5e-3 rest"), Some(-0.0015));
        assert_eq!(parse_first_f64("+0.5"), Some(0.5));
    }

    #[test]
    fn rejects_non_numeric() {
        assert_eq!(parse_first_f64("+/- 0.24"), None);
        assert_eq!(parse_first_f64("no number"), None);
    }

    #[test]
    fn finds_the_final_estimate_and_its_uncertainty() {
        let out = "perplexity: tokenizing the input ..\n\
                   [1]4.5,[2]4.7\n\
                   Final estimate: PPL = 4.6921 +/- 0.24358\n\
                   ggml_metal_free: deallocating\n";
        let r = parse_ppl(out).unwrap();
        assert_eq!(r.ppl, 4.6921);
        assert_eq!(r.stderr, Some(0.24358));
    }

    #[test]
    fn searches_backward_so_chunk_estimates_do_not_shadow_the_final_one() {
        let out = "Estimate: PPL = 9.99\nFinal estimate: PPL = 4.6921 +/- 0.2\n";
        assert_eq!(parse_ppl(out).unwrap().ppl, 4.6921);
    }

    #[test]
    fn parses_the_kl_statistics_block() {
        let out = "\
====== KL divergence statistics ======
Mean    KLD:   0.010000 ±   0.000100
Maximum KLD:   1.234000
99.9%   KLD:   0.900000
99.0%   KLD:   0.100000
90.0%   KLD:   0.030000
Median  KLD:   0.002000
====== Token probability statistics ======
Same top p:  95.500 ± 0.100 %
";
        let s = parse_kl_report(out).unwrap();
        assert_eq!(s.mean, 0.01);
        assert_eq!(s.max, 1.234);
        assert_eq!(s.p99, 0.1);
        assert_eq!(s.p90, 0.03);
        assert_eq!(s.median, 0.002);
        assert!((s.top1_agreement - 0.955).abs() < 1e-9);
    }

    #[test]
    fn missing_kl_block_is_an_error_not_a_zero() {
        // Silently returning KL=0 would tell the search that a candidate is free.
        let out = "Final estimate: PPL = 4.6921\n";
        assert!(parse_kl_report(out).is_err());
    }
}
