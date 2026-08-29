//! `requant inspect`: open a GGUF, detect the MoE layout, and print a per-tensor role table.

use anyhow::Result;

use requant_io::{bpw, ggml_type_name, tensor_nbytes, GgufValue};

use crate::common::{fmt_bytes, open_model, tag_all};

pub fn run_inspect(input: &str) -> Result<()> {
    let (reader, layout) = open_model(input)?;

    println!("GGUF: {input}");
    println!("  version   : {}", reader.version);
    println!("  alignment : {}", reader.alignment);
    println!("  tensors   : {}", reader.tensors.len());
    println!("  kv entries: {}", reader.kv.len());
    println!();
    println!("Model layout");
    println!("  architecture   : {}", layout.arch);
    println!("  layers         : {}", layout.n_layers);
    println!(
        "  MoE            : {}",
        if layout.is_moe { "yes" } else { "no" }
    );
    if layout.is_moe {
        println!("    routed experts : {}", layout.expert_count);
        println!("    experts/token  : {}", layout.expert_used);
        println!("    shared experts : {}", layout.shared_count);
    }
    // Tokenizer / context hints if present.
    if let Some(v) = reader.get(&format!("{}.context_length", layout.arch)) {
        let ctx = match v {
            GgufValue::U32(x) => *x as u64,
            GgufValue::U64(x) => *x,
            GgufValue::I32(x) => *x as u64,
            GgufValue::I64(x) => *x as u64,
            _ => 0,
        };
        println!("  context_length : {}", ctx);
    }
    println!();

    let (tags, n_unknown) = tag_all(&reader, &layout)?;

    // Per-tensor table.
    let mut total_bytes: u64 = 0;
    let mut total_elems: u64 = 0;
    println!(
        "{:<34} {:<16} {:>9} {:>9} {:<18} {:>6} {:>4}",
        "tensor", "type", "bytes", "bpw", "role", "depth", "q?"
    );
    println!("{}", "-".repeat(102.max(34 + 16 + 9 + 9 + 18 + 6 + 4 + 6)));
    for (t, tag) in reader.tensors.iter().zip(tags.iter()) {
        let nbytes = tensor_nbytes(t).unwrap_or(0);
        total_bytes += nbytes;
        total_elems += t.n_elems();
        let bpw_s = bpw(t.ggml_type)
            .map(|b| format!("{b:.2}"))
            .unwrap_or_else(|| "?".into());
        let depth_s = if tag.place.expert.is_some() {
            format!(
                "{}#{}",
                format!("{:.2}", tag.place.depth),
                tag.place.expert.unwrap()
            )
        } else {
            format!("{:.2}", tag.place.depth)
        };
        let q = if tag.quantizable { "y" } else { "-" };
        println!(
            "{:<34} {:<16} {:>9} {:>9} {:<18} {:>6} {:>4}",
            truncate(&t.name, 34),
            ggml_type_name(t.ggml_type),
            fmt_bytes(nbytes),
            bpw_s,
            tag.role.label(),
            depth_s,
            q,
        );
    }

    println!();
    println!(
        "total: {} tensors, {} params, {} on disk ({:.2} bpw overall)",
        reader.tensors.len(),
        fmt_count(total_elems),
        fmt_bytes(total_bytes),
        if total_elems > 0 {
            (total_bytes as f64) * 8.0 / total_elems as f64
        } else {
            0.0
        }
    );

    // Role distribution summary.
    println!();
    println!("Role distribution");
    let mut by_role: std::collections::BTreeMap<&str, (u64, u64, u64)> =
        std::collections::BTreeMap::new();
    for (t, tag) in reader.tensors.iter().zip(tags.iter()) {
        let nbytes = tensor_nbytes(t).unwrap_or(0);
        let e = by_role.entry(tag.role.label()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += nbytes;
        e.2 += t.n_elems();
    }
    println!(
        "  {:<18} {:>8} {:>12} {:>14}",
        "role", "tensors", "bytes", "params"
    );
    for (role, (n, b, p)) in &by_role {
        println!(
            "  {:<18} {:>8} {:>12} {:>14}",
            role,
            n,
            fmt_bytes(*b),
            fmt_count(*p)
        );
    }

    if n_unknown > 0 {
        println!();
        eprintln!("warning: {n_unknown} tensor(s) could not be classified into a role.");
        eprintln!(
            "         They will use the recipe default. Inspect their names — if they are weight"
        );
        eprintln!(
            "         matrices the model needs a tagger update; if they are harmless extra tensors"
        );
        eprintln!("         (e.g. position embeddings, RoPE freqs), assign them explicitly in the recipe.");
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

fn fmt_count(n: u64) -> String {
    let mut v = n as f64;
    const UNITS: &[&str] = &["", "K", "M", "B", "T"];
    let mut i = 0;
    while v >= 1000.0 && i + 1 < UNITS.len() {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{n}")
    } else {
        format!("{v:.2}{}", UNITS[i])
    }
}
