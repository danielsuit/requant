//! Shared CLI helpers: GGUF opening, recipe loading, byte/size formatting, role tagging.

use anyhow::{bail, Context, Result};
use std::path::Path;

use requant_io::{ggml_type_name, GgufReader, ModelLayout, TensorTag};
use requant_quant::{Recipe, DEFAULT_RECIPE_TOML};

/// Open a GGUF and parse its model layout, returning (reader, layout).
pub fn open_model(path: &str) -> Result<(GgufReader, ModelLayout)> {
    let reader =
        GgufReader::open(Path::new(path)).with_context(|| format!("opening GGUF `{path}`"))?;
    let layout = ModelLayout::from_kv(&reader.kv)
        .with_context(|| format!("parsing model layout from `{path}`"))?;
    Ok((reader, layout))
}

/// Load a recipe from a path, or fall back to the built-in MoE-aware default.
pub fn load_recipe(recipe_path: Option<&str>) -> Result<Recipe> {
    match recipe_path {
        Some(p) => {
            let text =
                std::fs::read_to_string(p).with_context(|| format!("reading recipe `{p}`"))?;
            Recipe::parse(&text)
        }
        None => Recipe::parse(DEFAULT_RECIPE_TOML),
    }
}

/// Tag every tensor in the reader against the layout, returning (tags, n_unknown).
pub fn tag_all(reader: &GgufReader, layout: &ModelLayout) -> Result<(Vec<TensorTag>, usize)> {
    let mut tags = Vec::with_capacity(reader.tensors.len());
    let mut n_unknown = 0usize;
    for t in &reader.tensors {
        let tag = TensorTag::tag(&t.name, layout);
        if matches!(tag.role, requant_io::Role::Unknown) {
            n_unknown += 1;
        }
        tags.push(tag);
    }
    Ok((tags, n_unknown))
}

/// Human-readable byte size (binary units).
pub fn fmt_bytes(b: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}

/// Pretty `rows×cols` or scalar shape.
pub fn fmt_shape(dims: &[u64]) -> String {
    if dims.is_empty() {
        return "scalar".into();
    }
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("×")
}

/// Name for a ggml type id, for display.
pub fn type_name(t: u32) -> String {
    ggml_type_name(t)
}

/// Parse a size budget string like "12G", "500M", "4.5GiB", or a bare byte count.
pub fn parse_bytes(s: &str) -> Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty size budget");
    }
    let (num_part, unit_part) = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .map(|i| (&s[..i], &s[i..]))
        .unwrap_or((s, ""));
    let n: f64 = num_part
        .parse()
        .with_context(|| format!("parsing size `{s}`: `{num_part}` is not a number"))?;
    let mult: u64 = match unit_part.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024 * 1024,
        "G" | "GB" | "GIB" => 1024 * 1024 * 1024,
        "T" | "TB" | "TIB" => 1024u64 * 1024 * 1024 * 1024,
        other => bail!("unknown size unit `{other}` in `{s}`"),
    };
    Ok((n * mult as f64) as u64)
}
