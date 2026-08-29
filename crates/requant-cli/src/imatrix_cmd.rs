//! `requant imatrix import`: load a llama.cpp imatrix and report its contents.
//!
//! (The content-addressed on-disk cache lands in a later phase; for now this validates the
//! file and prints a per-tensor summary so you can sanity-check a calibration run.)

use anyhow::{Context, Result};

use requant_calib::load_imatrix;

pub fn run_imatrix_import(imatrix: &str, model: &str, _cache_dir: &str) -> Result<()> {
    let im = load_imatrix(imatrix).with_context(|| format!("loading imatrix `{imatrix}`"))?;

    println!("imatrix: {imatrix}");
    println!("  tensors: {}", im.len());
    let bad = requant_calib::count_nonfinite(&im);
    if bad > 0 {
        eprintln!("  warning: {bad} non-finite / non-positive values");
    }
    println!();
    println!(
        "  {:<34} {:>8} {:>10} {:>10} {:>10}",
        "tensor", "ncall", "channels", "min", "max"
    );
    println!("{}", "-".repeat(80));
    // Sort by name for stable output.
    let mut names: Vec<&String> = im.entries.keys().collect();
    names.sort();
    for name in names {
        let e = &im.entries[name];
        let (min, max) = e
            .values
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        println!(
            "  {:<34} {:>8} {:>10} {:>10.3} {:>10.3}",
            truncate(name, 34),
            e.ncall,
            e.values.len(),
            min,
            max,
        );
    }

    if !model.is_empty() {
        // Cross-check: open the model and report which tensors are missing an imatrix entry.
        if let Ok((reader, _layout)) = crate::common::open_model(model) {
            let mut missing = 0;
            let mut matched = 0;
            for t in &reader.tensors {
                if im.get(&t.name).is_some() {
                    matched += 1;
                } else {
                    missing += 1;
                }
            }
            println!();
            println!("vs model `{model}`: {matched} tensors matched, {missing} missing an imatrix entry.");
        } else {
            eprintln!("note: could not open model `{model}` for cross-check (skipping)");
        }
    }

    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut t = s.to_string();
        t.truncate(n);
        t
    }
}
