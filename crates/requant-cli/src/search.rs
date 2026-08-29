//! `requant search`: find a size/quality-optimal recipe under a byte budget.
//!
//! This command is thin on purpose. Building per-tensor cost curves lives in
//! [`requant_search::proxy`], allocating against them lives in [`requant_search::knapsack`], and
//! measured sensitivity comes from `requant-eval`. What's left here is I/O and reporting.
//!
//! Two cost sources, same allocator:
//!
//! - **Default** — the imatrix-weighted round-trip error proxy. Free, no forward pass, available
//!   the moment you have weights and an imatrix.
//! - **`--sensitivity <table.json>`** — measured ΔKL from `requant sensitivity`. Strictly better
//!   information, because it knows how much the *model* cares rather than how much the *numbers*
//!   moved, at the cost of a forward pass per candidate.

use anyhow::{bail, Context, Result};

use requant_calib::load_imatrix;
use requant_quant::{dequantize_tensor, Bits, Recipe, DEFAULT_RECIPE_TOML};
use requant_search::{
    allocate, apply_sensitivity, emit_recipe, pareto_front, proxy, AllocOpts, Allocation,
    ProxyMetric, TensorCurve, BLOCKFLOAT_LADDER, FULL_LADDER, IQUANT_LADDER, KQUANT_LADDER,
};

use crate::common::{fmt_bytes, open_model, parse_bytes, tag_all};

/// Everything `requant search` needs. A struct rather than a dozen positional arguments.
#[derive(Debug, Clone, Default)]
pub struct SearchOpts {
    pub input: String,
    /// Byte budget. Optional when `--pareto` is set, since the front is budget-free.
    pub budget: Option<String>,
    pub imatrix: Option<String>,
    pub recipe_base: Option<String>,
    pub output: Option<String>,
    pub validate: bool,
    pub calib: Option<String>,
    /// Measured sensitivity table from `requant sensitivity`.
    pub sensitivity: Option<String>,
    /// `rel` (default, matches the published numbers) or `abs` (cross-tensor comparable).
    pub metric: Option<String>,
    /// Print the full size↔cost Pareto front.
    pub pareto: bool,
    /// `kquant`, `blockfloat`, or a comma-separated list of format names.
    pub ladder: Option<String>,
}

fn resolve_ladder(spec: Option<&str>) -> Result<Vec<Bits>> {
    let Some(s) = spec else {
        return Ok(KQUANT_LADDER.to_vec());
    };
    match s.trim().to_ascii_lowercase().as_str() {
        "kquant" | "k" => return Ok(KQUANT_LADDER.to_vec()),
        "iquant" | "iq" => return Ok(IQUANT_LADDER.to_vec()),
        "full" | "all" => return Ok(FULL_LADDER.to_vec()),
        "blockfloat" | "mx" | "fp4" => return Ok(BLOCKFLOAT_LADDER.to_vec()),
        _ => {}
    }
    let mut out = Vec::new();
    for tok in s.split(',') {
        let b = Bits::from_name(tok)
            .ok_or_else(|| anyhow::anyhow!("unknown format `{}` in --ladder", tok.trim()))?;
        out.push(b);
    }
    if out.is_empty() {
        bail!("--ladder resolved to an empty list");
    }
    // The allocator needs the ladder cheapest-first; sort by bits-per-weight so callers can pass
    // any order.
    out.sort_by(|a, b| {
        a.bpw()
            .unwrap_or(f64::MAX)
            .partial_cmp(&b.bpw().unwrap_or(f64::MAX))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

pub fn run_search_opts(opts: &SearchOpts) -> Result<()> {
    let metric = match opts.metric.as_deref() {
        Some(m) => ProxyMetric::parse(m)
            .ok_or_else(|| anyhow::anyhow!("--proxy-metric must be `rel` or `abs`, got `{m}`"))?,
        None => ProxyMetric::Relative,
    };
    let ladder = resolve_ladder(opts.ladder.as_deref())?;
    let budget = match &opts.budget {
        Some(b) => Some(parse_bytes(b)?),
        None if opts.pareto => None,
        None => bail!("--budget is required (or use --pareto to print the whole front)"),
    };

    let (reader, layout) = open_model(&opts.input)?;
    let base = match &opts.recipe_base {
        Some(p) => {
            let text =
                std::fs::read_to_string(p).with_context(|| format!("reading recipe `{p}`"))?;
            Recipe::parse(&text)?
        }
        None => Recipe::parse(DEFAULT_RECIPE_TOML)?,
    };

    let imatrix = match &opts.imatrix {
        Some(p) => {
            let im = load_imatrix(p).with_context(|| format!("loading imatrix `{p}`"))?;
            eprintln!("search: loaded imatrix ({} tensors) from `{p}`", im.len());
            Some(im)
        }
        None => None,
    };

    let (tags, _) = tag_all(&reader, &layout)?;
    let mut curves: Vec<TensorCurve> = Vec::with_capacity(reader.tensors.len());
    let mut n_searchable = 0usize;

    for (i, t) in reader.tensors.iter().enumerate() {
        let tag = &tags[i];
        let role = tag.role.label();
        let policy = base
            .resolve_named(tag, &layout, &t.name)
            .with_context(|| format!("resolving base policy for `{}`", t.name))?;
        let fixed_bits = Bits::from_ggml_type(policy.ggml_type);
        let target_is_float = requant_io::is_float_type(policy.ggml_type);
        let (rows, cols) = t.rows_cols();

        let fixed_curve =
            |bytes: u64| TensorCurve::fixed_at(t.name.clone(), role, fixed_bits, bytes);

        if target_is_float || !tag.quantizable || cols == 0 {
            let bytes = if policy.copy_unchanged {
                requant_io::tensor_nbytes(t).unwrap_or(0)
            } else {
                requant_io::packed_nbytes(policy.ggml_type, t.n_elems(), &t.name).unwrap_or(0)
            };
            curves.push(fixed_curve(bytes));
            continue;
        }

        let src = if requant_io::is_float_type(t.ggml_type) {
            reader.tensor_to_f32(i)?
        } else {
            reader
                .tensor_bytes(i)
                .ok()
                .and_then(|b| dequantize_tensor(t.ggml_type, b, rows, cols).ok())
                .unwrap_or_default()
        };
        if src.is_empty() {
            curves.push(fixed_curve(requant_io::tensor_nbytes(t).unwrap_or(0)));
            continue;
        }

        let im = if policy.use_imatrix {
            imatrix.as_ref().and_then(|im| im.get(&t.name))
        } else {
            None
        };
        // The base recipe's per-tensor bits is a HARD FLOOR. Without it the allocator would
        // happily buy bytes by taking a protected embedding from Q8_0 down to Q2_K — exactly the
        // silent quality trap the recipe exists to prevent. To hit a tighter budget, lower the
        // floors in the recipe rather than letting the optimizer do it behind your back.
        let mut curve = proxy::tensor_curve(
            &t.name, role, &src, rows, cols, im, &ladder, fixed_bits, metric,
        )?;
        if curve.points.is_empty() {
            curves.push(fixed_curve(requant_io::tensor_nbytes(t).unwrap_or(0)));
            continue;
        }
        curve.normalize();
        n_searchable += 1;
        curves.push(curve);
    }

    if n_searchable == 0 {
        bail!(
            "no searchable tensors found — every tensor is protected, non-quantizable, or already \
             quantized. A search needs a full-precision source with quantizable weights."
        );
    }

    // Swap proxy costs for measured ΔKL where we have it.
    let mut cost_source = format!("proxy ({:?})", metric);
    if let Some(p) = &opts.sensitivity {
        let table = requant_eval::SensitivityTable::read_json(p)
            .with_context(|| format!("reading sensitivity table `{p}`"))?;
        let (matched, unmatched) = apply_sensitivity(&mut curves, &table);
        eprintln!(
            "search: applied measured ΔKL from `{p}` ({} evaluator, grouping {:?}): {matched} tensors matched, {unmatched} fell back to the rescaled proxy",
            table.evaluator, table.grouping
        );
        if unmatched > 0 {
            eprintln!(
                "search: warning — {unmatched} tensors had no measured curve. Their costs are the \
                 proxy rescaled onto the measured scale, which is a bridge, not a measurement. \
                 Re-run `requant sensitivity` with a grouping that covers them."
            );
        }
        cost_source = format!("measured ΔKL ({})", table.evaluator);
    }

    let alloc_opts = AllocOpts::default();

    if opts.pareto {
        let front = pareto_front(&curves, &alloc_opts);
        println!("pareto front: {} ({cost_source})", opts.input);
        println!("  {:>14}  {:>16}  {:>10}", "size", "cost", "upgrades");
        for a in &front {
            println!(
                "  {:>14}  {:>16.6e}  {:>10}",
                fmt_bytes(a.total_bytes),
                a.total_cost,
                a.upgrades
            );
        }
        println!();
        if budget.is_none() {
            return Ok(());
        }
    }

    let budget = budget.expect("budget presence checked above");
    let alloc = allocate(&curves, budget, &alloc_opts);
    report(&opts.input, &curves, &alloc, n_searchable, &cost_source);

    let header = format!(
        "Auto-searched recipe from `requant search`\n\
         source     : {}\n\
         budget     : {}\n\
         allocated  : {}\n\
         fixed      : {}\n\
         searchable : {n_searchable} tensors, {} upgraded above floor\n\
         cost source: {cost_source}\n\
         total cost : {:.6e}",
        opts.input,
        fmt_bytes(budget),
        fmt_bytes(alloc.total_bytes),
        fmt_bytes(alloc.fixed_bytes),
        alloc.upgrades,
        alloc.total_cost,
    );
    let default_bits = match &base.defaults.bits {
        requant_quant::recipe::BitsOrStr::Named(b) => *b,
    };
    let recipe = emit_recipe(&curves, &alloc, &header, default_bits);

    if let Some(path) = &opts.output {
        std::fs::write(path, &recipe).with_context(|| format!("writing recipe `{path}`"))?;
        println!("recipe written to {path}");
    } else {
        print!("{recipe}");
    }

    // `--validate` closes the loop: quantize with the just-searched recipe and run a real
    // perplexity check. A search driven by a proxy is a hypothesis until this passes.
    if opts.validate {
        let recipe_path = match &opts.output {
            Some(p) => p.clone(),
            None => {
                let tmp = std::env::temp_dir().join("requant-search-recipe.toml");
                std::fs::write(&tmp, &recipe)?;
                tmp.to_string_lossy().into_owned()
            }
        };
        let quant_path = std::env::temp_dir().join("requant-search-validate.gguf");
        let quant_str = quant_path.to_string_lossy().into_owned();
        eprintln!(
            "search: --validate: quantizing `{}` with the searched recipe …",
            opts.input
        );
        crate::quantize::run_quantize(
            &opts.input,
            &quant_str,
            Some(&recipe_path),
            opts.imatrix.as_deref(),
        )?;
        eprintln!("search: --validate: running perplexity vs the source …");
        crate::eval::run_eval(&quant_str, &opts.input, opts.calib.as_deref())?;
    }

    Ok(())
}

fn report(
    input: &str,
    curves: &[TensorCurve],
    alloc: &Allocation,
    n_searchable: usize,
    cost_source: &str,
) {
    println!("search: {input}");
    println!("  budget             : {}", fmt_bytes(alloc.budget));
    println!("  cost source        : {cost_source}");
    println!("  searchable tensors : {n_searchable}");
    println!("  fixed (protected)  : {}", fmt_bytes(alloc.fixed_bytes));
    if alloc.over_budget {
        println!(
            "  allocated total    : {} (OVER BUDGET by {} — every tensor is already at its \
             recipe floor)",
            fmt_bytes(alloc.total_bytes),
            fmt_bytes(alloc.shortfall)
        );
        println!(
            "  → The floors in the base recipe cannot fit this budget. Lower them deliberately \
             (e.g. embedding Q8_0 -> Q6_K) rather than expecting the search to violate them."
        );
    } else {
        println!("  allocated total    : {}", fmt_bytes(alloc.total_bytes));
        println!(
            "  upgrades above floor: {} (slack left: {})",
            alloc.upgrades,
            fmt_bytes(alloc.slack())
        );
    }
    println!("  total cost         : {:.6e}", alloc.total_cost);

    // Where the bytes went, by role.
    let mut by_role: std::collections::BTreeMap<&str, (u64, usize)> =
        std::collections::BTreeMap::new();
    for (i, c) in curves.iter().enumerate() {
        let bytes = c
            .points
            .get(alloc.choice[i])
            .map_or(c.fixed_bytes, |p| p.bytes);
        let e = by_role.entry(c.role.as_str()).or_insert((0, 0));
        e.0 += bytes;
        e.1 += 1;
    }
    println!();
    println!("  bytes by role:");
    println!("    {:<22} {:>8} {:>14}", "role", "tensors", "bytes");
    for (role, (bytes, n)) in &by_role {
        println!("    {:<22} {:>8} {:>14}", role, n, fmt_bytes(*bytes));
    }
    println!();
}

/// Positional-argument form, kept because it reads better at call sites that only need the basics.
#[allow(clippy::too_many_arguments)]
pub fn run_search(
    input: &str,
    budget: &str,
    imatrix: Option<&str>,
    recipe_base: Option<&str>,
    out: Option<&str>,
    validate: bool,
    calib: Option<&str>,
) -> Result<()> {
    run_search_opts(&SearchOpts {
        input: input.to_string(),
        budget: Some(budget.to_string()),
        imatrix: imatrix.map(String::from),
        recipe_base: recipe_base.map(String::from),
        output: out.map(String::from),
        validate,
        calib: calib.map(String::from),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_presets_and_explicit_lists_both_work() {
        assert_eq!(resolve_ladder(None).unwrap(), KQUANT_LADDER.to_vec());
        assert_eq!(
            resolve_ladder(Some("kquant")).unwrap(),
            KQUANT_LADDER.to_vec()
        );
        assert_eq!(
            resolve_ladder(Some("iquant")).unwrap(),
            IQUANT_LADDER.to_vec()
        );
        assert_eq!(resolve_ladder(Some("full")).unwrap(), FULL_LADDER.to_vec());
        assert_eq!(
            resolve_ladder(Some("blockfloat")).unwrap().first(),
            Some(&Bits::MXFP4)
        );
        // The full ladder is cheapest-first by bits-per-weight, so the first rung is the sub-2-bit
        // IQ1_S and a later rung is the 8.5-bpw Q8_0.
        let full = resolve_ladder(Some("full")).unwrap();
        assert_eq!(full.first(), Some(&Bits::IQ1_S));
        assert_eq!(full.last(), Some(&Bits::Q8_0));
        // Explicit lists are sorted cheapest-first regardless of the order given.
        let l = resolve_ladder(Some("Q8_0,Q4_K,Q6_K")).unwrap();
        assert_eq!(l, vec![Bits::Q4_K, Bits::Q6_K, Bits::Q8_0]);
        // An explicit i-quant list resolves and sorts too.
        let l = resolve_ladder(Some("IQ4_XS,IQ1_S,IQ3_S")).unwrap();
        assert_eq!(l, vec![Bits::IQ1_S, Bits::IQ3_S, Bits::IQ4_XS]);
        assert!(resolve_ladder(Some("Q4_K,NOPE")).is_err());
    }
}
