//! Quantize a Qwen3.5-MoE checkpoint into a minglang inference pack.
//!
//! # Why this is not `requant quantize`
//!
//! `quantize` is GGUF-in/GGUF-out, and there is no GGUF converter for the Qwen3.5 hybrid
//! (gated DeltaNet + packed experts). Going through GGUF would mean writing one converter to
//! get in and teaching minglang a second container format to get out. This command instead
//! reads the sharded safetensors `moefy-qwen38` already produces and writes sharded
//! safetensors back, so minglang's existing mmap loader keeps working unchanged.
//!
//! # What the pack is
//!
//! Each quantized tensor becomes a `U8` safetensors entry holding the raw ggml block bytes,
//! shaped `[nbytes]`. The logical shape and block format live in a sidecar
//! `minglang_pack.json`, because safetensors has no dtype for "Q4_K". Tensors that are not
//! worth quantizing (norms, biases, the router, the vision tower) are copied through
//! byte-for-byte and are absent from the manifest — a loader that finds no manifest entry
//! reads the safetensors dtype and is correct by default.
//!
//! # The point: fitting in RAM
//!
//! Qwen3.8-27B is 53.8 GB of BF16 on a 23 GB box, so decode is disk-bound at ~125 MB/s and
//! the CPU idles at under 10 % of capacity. The default recipe lands at ~18 GB, which is the
//! threshold where the whole model stays in the page cache and decode becomes compute-bound
//! for the first time. See minglang's `ROADMAP.md` Phase Q35.4.
//!
//! # Recipe
//!
//! | role | type | why |
//! |------|------|-----|
//! | routed experts, `gate_up_proj` | Q4_K | 22.9 GB of the checkpoint; only 4 of 32 fire per token |
//! | routed experts, `down_proj` | Q5_0 | `cols = 544` is not 256-aligned, so Q4_K falls back |
//! | attention (all 64 layers) | Q6_K | dense path — every token pays for all of it |
//! | embeddings, LM head | Q6_K | dense path, and the LM head sets the output distribution |
//! | router, norms, biases, conv1d | BF16 | 6-bit routing logits would reorder expert selection |
//! | vision tower | BF16 | unused by minglang's text-only track; not worth a fallback |
//!
//! Fallbacks are not hand-rolled: [`requant_quant::fallback_type`] applies the same rule
//! `llama-quantize` uses (Q4_K→Q5_0, Q6_K→Q8_0, then F16), so a tensor whose `cols` is not
//! block-aligned degrades predictably rather than failing the run.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use half::bf16;
use rayon::prelude::*;
use requant_io::{ShardedSafeTensors, StDtype, TensorSource};
use serde_json::{json, Map, Value};

/// ggml type ids. Spelled out because the numbers appear in the manifest minglang reads.
const GGML_F16: u32 = 1;
const GGML_Q5_0: u32 = 6;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_Q6_K: u32 = 14;

/// Rows quantized per batch. 4096 rows of the widest tensor (`cols = 6144`) is ~100 MB of f32
/// staging — large enough to keep every core busy, small enough that the pack tool's own RSS
/// never competes with the page cache it is trying to make room for.
const ROW_CHUNK: usize = 4096;

pub struct PackOptions {
    pub input_dir: String,
    pub output_dir: String,
    pub expert_type: u32,
    pub attn_type: u32,
    pub embed_type: u32,
    pub max_shard_size: u64,
    pub dry_run: bool,
}

/// What happens to one source tensor.
enum Plan {
    /// Copy the source bytes unchanged; stays a normal typed safetensors entry.
    Passthrough,
    /// Quantize `[rows, cols]` to `ggml_type`, emitted as a `U8` blob.
    Quantize {
        ggml_type: u32,
        rows: usize,
        cols: usize,
        /// True when `ggml_type` is not what the recipe asked for.
        fell_back: bool,
    },
}

struct Planned {
    name: String,
    /// Logical shape, as in the source checkpoint.
    shape: Vec<u64>,
    src_dtype: StDtype,
    src_bytes: u64,
    out_bytes: u64,
    plan: Plan,
    role: &'static str,
}

pub fn run_pack_qwen35(opts: &PackOptions) -> Result<()> {
    let input = Path::new(&opts.input_dir);
    let output = Path::new(&opts.output_dir);
    if !input.is_dir() {
        bail!("input checkpoint directory does not exist: {}", input.display());
    }
    if !opts.dry_run && output.exists() {
        bail!(
            "output directory already exists: {} (refusing to overwrite a checkpoint)",
            output.display()
        );
    }

    let source = ShardedSafeTensors::open_dir(input)
        .with_context(|| format!("opening checkpoint at {}", input.display()))?;

    let mut names: Vec<String> = source.names().map(|s| s.to_string()).collect();
    names.sort();
    if names.is_empty() {
        bail!("no tensors found in {}", input.display());
    }

    let mut planned = Vec::with_capacity(names.len());
    for name in &names {
        planned.push(plan_tensor(&source, name, opts)?);
    }

    report(&planned);

    if opts.dry_run {
        println!("\ndry run: nothing written");
        return Ok(());
    }

    fs::create_dir_all(output)
        .with_context(|| format!("creating {}", output.display()))?;
    write_pack(&source, output, &planned, opts.max_shard_size)?;
    write_manifest(output, &planned, opts)?;
    copy_support_files(input, output)?;
    println!("\nwrote {}", output.display());
    Ok(())
}

// ===========================================================================
// Planning
// ===========================================================================

/// Assign a role and target type to one tensor by name and shape.
///
/// Name matching is on suffixes rather than a layer-indexed regex so it works for both
/// flavors `moefy-qwen38` emits (`model.language_model.layers.*` and `model.layers.*`).
fn plan_tensor(source: &ShardedSafeTensors, name: &str, opts: &PackOptions) -> Result<Planned> {
    let entry = source
        .entry(name)
        .with_context(|| format!("tensor `{name}` vanished from the index"))?;
    let shape = entry.shape.clone();
    let src_bytes: u64 = shape.iter().product::<u64>() * entry.dtype.size() as u64;

    let passthrough = |role: &'static str| Planned {
        name: name.to_string(),
        shape: shape.clone(),
        src_dtype: entry.dtype,
        src_bytes,
        out_bytes: src_bytes,
        plan: Plan::Passthrough,
        role,
    };

    // Rank < 2 has no `[out, in]` to quantize along, and these are all tiny anyway.
    if shape.len() < 2 {
        return Ok(passthrough("scalar/vector"));
    }
    // The vision tower is dead weight for minglang's text-only track. Its `cols` (1152, 4304)
    // are neither 256- nor 32-aligned, so quantizing it would mostly produce F16 fallbacks.
    if name.starts_with("model.visual.") || name.contains(".visual.") {
        return Ok(passthrough("vision (unused)"));
    }
    // The router decides which experts run. Quantization error here does not blur an output,
    // it selects a different expert — a discrete error that no downstream precision recovers.
    if name.ends_with("mlp.gate.weight") {
        return Ok(passthrough("router (protected)"));
    }
    // `shared_expert_intermediate_size` is 1 in this config; the tensors are 5120 floats.
    if name.contains(".shared_expert.") {
        return Ok(passthrough("shared expert (degenerate)"));
    }
    // Depthwise conv: `[channels, 1, 4]`. The quantized axis would be 4 elements long.
    if name.ends_with("conv1d.weight") {
        return Ok(passthrough("conv1d"));
    }

    let (role, want) = if name.ends_with("mlp.experts.gate_up_proj")
        || name.ends_with("mlp.experts.down_proj")
    {
        ("routed experts", opts.expert_type)
    } else if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") {
        ("embeddings / LM head", opts.embed_type)
    } else {
        ("attention", opts.attn_type)
    };

    // Rows run along every leading axis: the packed expert tensors are `[E, out, in]`, so
    // `E * out` rows of `in`. This is the same flattening the GEMM does when it slices one
    // expert out of the pack.
    let cols = *shape.last().unwrap() as usize;
    let rows: usize = shape[..shape.len() - 1].iter().product::<u64>() as usize;

    let (ggml_type, fell_back) = requant_quant::fallback_type(want, cols);
    let out_bytes = row_bytes(ggml_type, cols)? as u64 * rows as u64;

    Ok(Planned {
        name: name.to_string(),
        shape,
        src_dtype: entry.dtype,
        src_bytes,
        out_bytes,
        plan: Plan::Quantize { ggml_type, rows, cols, fell_back },
        role,
    })
}

fn row_bytes(ggml_type: u32, cols: usize) -> Result<usize> {
    if ggml_type == GGML_F16 {
        return Ok(cols * 2);
    }
    let (block, bpb) = requant_io::block_layout(ggml_type)
        .with_context(|| format!("unsupported ggml type {ggml_type}"))?;
    if cols % block != 0 {
        bail!("type {ggml_type}: cols {cols} not divisible by block {block}");
    }
    Ok(cols / block * bpb)
}

fn type_name(t: u32) -> &'static str {
    match t {
        GGML_F16 => "F16",
        GGML_Q5_0 => "Q5_0",
        GGML_Q8_0 => "Q8_0",
        GGML_Q4_K => "Q4_K",
        GGML_Q6_K => "Q6_K",
        _ => "?",
    }
}

/// Parse a recipe type name into a ggml id. Only the types minglang has kernels for.
pub fn parse_type(s: &str) -> Result<u32> {
    Ok(match s.to_ascii_uppercase().as_str() {
        "Q4_K" => GGML_Q4_K,
        "Q6_K" => GGML_Q6_K,
        "Q5_0" => GGML_Q5_0,
        "Q8_0" => GGML_Q8_0,
        "F16" => GGML_F16,
        other => bail!(
            "unknown quant type `{other}`; minglang has kernels for Q4_K, Q6_K, Q5_0, Q8_0, F16"
        ),
    })
}

// ===========================================================================
// Reporting
// ===========================================================================

fn report(planned: &[Planned]) {
    let mut by_role: BTreeMap<&str, (u64, u64, usize, BTreeMap<&str, usize>)> = BTreeMap::new();
    for p in planned {
        let e = by_role.entry(p.role).or_default();
        e.0 += p.src_bytes;
        e.1 += p.out_bytes;
        e.2 += 1;
        let t = match &p.plan {
            Plan::Passthrough => p.src_dtype.name(),
            Plan::Quantize { ggml_type, .. } => type_name(*ggml_type),
        };
        *e.3.entry(t).or_default() += 1;
    }
    let gb = |b: u64| b as f64 / 1e9;
    println!("{:<28} {:>7} {:>10} {:>10}  {}", "role", "tensors", "source", "packed", "types");
    println!("{}", "-".repeat(78));
    let (mut ts, mut to) = (0u64, 0u64);
    for (role, (s, o, n, types)) in &by_role {
        let t: Vec<String> = types.iter().map(|(k, v)| format!("{k}×{v}")).collect();
        println!("{role:<28} {n:>7} {:>9.2}G {:>9.2}G  {}", gb(*s), gb(*o), t.join(" "));
        ts += s;
        to += o;
    }
    println!("{}", "-".repeat(78));
    println!(
        "{:<28} {:>7} {:>9.2}G {:>9.2}G  {:.2}x smaller",
        "total",
        planned.len(),
        gb(ts),
        gb(to),
        ts as f64 / to as f64
    );

    // The size that decides whether decode is disk-bound is not the file size — it is the set
    // of weights the loader actually touches. Vision tensors sit in the pack and are never
    // paged in by the text-only track, so they cost disk, not RAM.
    let resident: u64 = planned
        .iter()
        .filter(|p| p.role != "vision (unused)")
        .map(|p| p.out_bytes)
        .sum();
    println!(
        "{:<28} {:>7} {:>10} {:>9.2}G  (excludes the vision tower)",
        "resident working set", "", "", gb(resident)
    );

    let fallbacks: Vec<&Planned> = planned
        .iter()
        .filter(|p| matches!(p.plan, Plan::Quantize { fell_back: true, .. }))
        .collect();
    if !fallbacks.is_empty() {
        // Not a warning: `down_proj` falling back to Q5_0 is the expected, correct outcome for
        // `cols = 544`. It is printed so an unexpected one (an F16 fallback blowing the size
        // budget) is visible before the multi-hour write starts.
        let mut counts: BTreeMap<(usize, &str), usize> = BTreeMap::new();
        for p in &fallbacks {
            if let Plan::Quantize { ggml_type, cols, .. } = p.plan {
                *counts.entry((cols, type_name(ggml_type))).or_default() += 1;
            }
        }
        println!("\nblock-alignment fallbacks:");
        for ((cols, t), n) in counts {
            println!("  cols={cols} not block-aligned -> {t} ({n} tensors)");
        }
    }
}

// ===========================================================================
// Writing
// ===========================================================================

fn write_pack(
    source: &ShardedSafeTensors,
    output: &Path,
    planned: &[Planned],
    max_shard_bytes: u64,
) -> Result<()> {
    let mut shards: Vec<&[Planned]> = Vec::new();
    let mut start = 0usize;
    let mut size = 0u64;
    for (i, p) in planned.iter().enumerate() {
        if i > start && size.saturating_add(p.out_bytes) > max_shard_bytes {
            shards.push(&planned[start..i]);
            start = i;
            size = 0;
        }
        size = size.saturating_add(p.out_bytes);
    }
    if start < planned.len() {
        shards.push(&planned[start..]);
    }

    let count = shards.len();
    let mut weight_map = Map::new();
    for (i, shard) in shards.iter().enumerate() {
        let filename = format!("model-{:05}-of-{:05}.safetensors", i + 1, count);
        println!("shard {}/{count}: {} tensors", i + 1, shard.len());
        write_shard(source, &output.join(&filename), shard)?;
        for p in *shard {
            weight_map.insert(p.name.clone(), json!(filename));
        }
    }
    let total_size: u64 = planned.iter().map(|p| p.out_bytes).sum();
    write_json(
        output.join("model.safetensors.index.json"),
        &json!({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
    )
}

fn write_shard(source: &ShardedSafeTensors, path: &Path, planned: &[Planned]) -> Result<()> {
    let mut header = Map::new();
    header.insert("__metadata__".into(), json!({"format": "pt"}));
    let mut offset = 0u64;
    for p in planned {
        // A quantized tensor is a flat byte blob: its logical shape moves to the manifest,
        // because a `[6144, 5120]` entry with 4.5-bit elements would be a lie any standard
        // safetensors reader would act on.
        let (dtype, shape) = match &p.plan {
            Plan::Passthrough => (p.src_dtype.name(), p.shape.clone()),
            Plan::Quantize { .. } => ("U8", vec![p.out_bytes]),
        };
        header.insert(
            p.name.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + p.out_bytes],
            }),
        );
        offset += p.out_bytes;
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header))?;
    let padding = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat(b' ').take(padding));

    // Written to `.partial` and renamed so an interrupted multi-hour run never leaves a shard
    // that looks complete to the index.
    let partial = path.with_extension("safetensors.partial");
    let file = File::create(&partial).with_context(|| format!("creating {}", partial.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);
    writer.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    for p in planned {
        write_payload(source, &mut writer, p)
            .with_context(|| format!("writing tensor `{}`", p.name))?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(&partial, path)
        .with_context(|| format!("renaming {} to {}", partial.display(), path.display()))
}

fn write_payload<W: Write>(
    source: &ShardedSafeTensors,
    writer: &mut W,
    p: &Planned,
) -> Result<()> {
    let src = source.st_bytes(&p.name)?;
    match &p.plan {
        Plan::Passthrough => {
            writer.write_all(src)?;
            Ok(())
        }
        Plan::Quantize { ggml_type, rows, cols, .. } => {
            let rb = row_bytes(*ggml_type, *cols)?;
            let esz = p.src_dtype.size();
            let mut done = 0usize;
            while done < *rows {
                let n = ROW_CHUNK.min(rows - done);
                // Rows are quantized independently, so the chunk parallelizes cleanly and
                // each thread touches only its own row of the mmap.
                let packed: Vec<Vec<u8>> = (0..n)
                    .into_par_iter()
                    .map(|r| {
                        let off = (done + r) * cols * esz;
                        let raw = &src[off..off + cols * esz];
                        let x = widen(p.src_dtype, raw);
                        quantize_row(*ggml_type, &x, *cols)
                    })
                    .collect::<Result<Vec<_>>>()?;
                for row in packed {
                    debug_assert_eq!(row.len(), rb);
                    writer.write_all(&row)?;
                }
                done += n;
            }
            Ok(())
        }
    }
}

/// Widen one row of source elements to f32. BF16 in practice; F16/F32 accepted so the tool
/// does not care which precision the conversion stage happened to emit.
fn widen(dtype: StDtype, raw: &[u8]) -> Vec<f32> {
    match dtype {
        StDtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        StDtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        StDtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => unreachable!("widen: non-float dtype {} reached the quantize path", other.name()),
    }
}

fn quantize_row(ggml_type: u32, x: &[f32], cols: usize) -> Result<Vec<u8>> {
    if ggml_type == GGML_F16 {
        return requant_quant::pack_float(GGML_F16, x);
    }
    requant_quant::quantize_tensor(ggml_type, x, 1, cols, None)
}

/// The sidecar that tells minglang how to read each `U8` blob.
fn write_manifest(output: &Path, planned: &[Planned], opts: &PackOptions) -> Result<()> {
    let mut tensors = Map::new();
    for p in planned {
        if let Plan::Quantize { ggml_type, rows, cols, .. } = &p.plan {
            tensors.insert(
                p.name.clone(),
                json!({
                    "format": type_name(*ggml_type),
                    "ggml_type": ggml_type,
                    "shape": p.shape,
                    "rows": rows,
                    "cols": cols,
                    "row_bytes": row_bytes(*ggml_type, *cols)?,
                    "nbytes": p.out_bytes,
                }),
            );
        }
    }
    write_json(
        output.join("minglang_pack.json"),
        &json!({
            "format_version": 1,
            "producer": concat!("requant ", env!("CARGO_PKG_VERSION"), " pack-qwen35"),
            "source": opts.input_dir,
            "recipe": {
                "experts": type_name(opts.expert_type),
                "attention": type_name(opts.attn_type),
                "embeddings": type_name(opts.embed_type),
            },
            // Absent names are unquantized: read the safetensors dtype and shape directly.
            "tensors": tensors,
        }),
    )
}

fn copy_support_files(input: &Path, output: &Path) -> Result<()> {
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".safetensors") || name == "model.safetensors.index.json" {
            continue;
        }
        if entry.file_type()?.is_file() {
            fs::copy(entry.path(), output.join(name.as_ref()))
                .with_context(|| format!("copying {name}"))?;
        }
    }
    Ok(())
}

fn write_json(path: PathBuf, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

pub fn parse_size(s: &str) -> Result<u64> {
    let t = s.trim();
    let (num, mul) = match t.chars().last() {
        Some('G') | Some('g') => (&t[..t.len() - 1], 1u64 << 30),
        Some('M') | Some('m') => (&t[..t.len() - 1], 1u64 << 20),
        Some('K') | Some('k') => (&t[..t.len() - 1], 1u64 << 10),
        _ => (t, 1),
    };
    let n: u64 = num.trim().parse().with_context(|| format!("bad size `{s}`"))?;
    Ok(n * mul)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    /// A checkpoint with one tensor per interesting role, at the real model's row lengths so
    /// the alignment rules fire for the same reasons they do on the 27B: `cols = 5120` takes
    /// the k-quant path, `cols = 544` forces the Q5_0 fallback, `cols = 1` cannot be
    /// quantized at all.
    fn tiny_checkpoint(dir: &Path) {
        let mut header = Map::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut add = |header: &mut Map<String, Value>, blob: &mut Vec<u8>, name: &str, shape: Vec<u64>| {
            let n: u64 = shape.iter().product();
            let start = blob.len();
            for i in 0..n {
                // A deterministic spread with a few outliers, so the sub-block scale search
                // has something to do rather than quantizing a constant.
                let v = ((i % 61) as f32 - 30.0) * if i % 97 == 0 { 0.5 } else { 0.01 };
                blob.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            header.insert(
                name.into(),
                json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, blob.len()]}),
            );
        };
        add(&mut header, &mut blob, "model.layers.0.self_attn.q_proj.weight", vec![512, 5120]);
        add(&mut header, &mut blob, "model.layers.0.mlp.experts.gate_up_proj", vec![4, 32, 5120]);
        add(&mut header, &mut blob, "model.layers.0.mlp.experts.down_proj", vec![4, 32, 544]);
        add(&mut header, &mut blob, "model.layers.0.mlp.gate.weight", vec![4, 5120]);
        add(&mut header, &mut blob, "model.layers.0.input_layernorm.weight", vec![5120]);
        add(&mut header, &mut blob, "model.visual.blocks.0.attn.proj.weight", vec![64, 1152]);
        add(&mut header, &mut blob, "lm_head.weight", vec![256, 5120]);

        let h = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(blob);
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("model.safetensors"), out).unwrap();
        fs::write(dir.join("config.json"), br#"{"model_type":"qwen3_5_moe_text"}"#).unwrap();
    }

    fn pack(input: &Path, output: &Path) {
        run_pack_qwen35(&PackOptions {
            input_dir: input.display().to_string(),
            output_dir: output.display().to_string(),
            expert_type: GGML_Q4_K,
            attn_type: GGML_Q6_K,
            embed_type: GGML_Q6_K,
            max_shard_size: 1 << 30,
            dry_run: false,
        })
        .unwrap();
    }

    fn manifest(dir: &Path) -> Value {
        serde_json::from_str(&fs::read_to_string(dir.join("minglang_pack.json")).unwrap()).unwrap()
    }

    #[test]
    fn each_role_gets_the_type_the_recipe_assigns() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        pack(&src, &dst);
        let m = manifest(&dst);
        let t = &m["tensors"];

        assert_eq!(t["model.layers.0.self_attn.q_proj.weight"]["format"], "Q6_K");
        assert_eq!(t["lm_head.weight"]["format"], "Q6_K");
        assert_eq!(t["model.layers.0.mlp.experts.gate_up_proj"]["format"], "Q4_K");
        // cols = 544 is not a 256-element super-block, so Q4_K falls back the way
        // llama-quantize does. This is the case the whole fallback path exists for.
        assert_eq!(t["model.layers.0.mlp.experts.down_proj"]["format"], "Q5_0");

        // Protected and unquantizable roles must be absent, not present at some type: a
        // loader reads an absent name straight from the safetensors dtype.
        assert!(t.get("model.layers.0.mlp.gate.weight").is_none(), "router must stay BF16");
        assert!(t.get("model.layers.0.input_layernorm.weight").is_none(), "1-D must stay BF16");
        assert!(t.get("model.visual.blocks.0.attn.proj.weight").is_none(), "vision must stay BF16");
    }

    #[test]
    fn manifest_row_geometry_matches_the_bytes_on_disk() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        pack(&src, &dst);
        let m = manifest(&dst);
        let packed = ShardedSafeTensors::open_dir(&dst).unwrap();

        for (name, entry) in m["tensors"].as_object().unwrap() {
            let rows = entry["rows"].as_u64().unwrap() as usize;
            let rb = entry["row_bytes"].as_u64().unwrap() as usize;
            let cols = entry["cols"].as_u64().unwrap() as usize;
            let bytes = packed.st_bytes(name).unwrap();
            // The check the loader makes. If these disagree the loader reads a different
            // tensor than the manifest describes, and under Q4_K it is off by a factor of
            // ~3.5 rather than by a few bytes.
            assert_eq!(bytes.len(), rows * rb, "{name}: byte length");
            let want = row_bytes(entry["ggml_type"].as_u64().unwrap() as u32, cols).unwrap();
            assert_eq!(rb, want, "{name}: row_bytes disagrees with its own type");
            // The header must say `U8` with a flat shape — a quantized tensor with a logical
            // shape would be a lie a standard safetensors reader would act on.
            assert_eq!(packed.entry(name).unwrap().dtype.name(), "U8", "{name}: header dtype");
        }
    }

    #[test]
    fn quantized_tensors_round_trip_to_close_to_the_source() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        pack(&src, &dst);
        let m = manifest(&dst);
        let source = ShardedSafeTensors::open_dir(&src).unwrap();
        let packed = ShardedSafeTensors::open_dir(&dst).unwrap();

        let name = "model.layers.0.self_attn.q_proj.weight";
        let want = source.st_f32(name).unwrap();
        let e = &m["tensors"][name];
        let got = requant_quant::dequantize_tensor(
            e["ggml_type"].as_u64().unwrap() as u32,
            packed.st_bytes(name).unwrap(),
            e["rows"].as_u64().unwrap() as usize,
            e["cols"].as_u64().unwrap() as usize,
        )
        .unwrap();

        assert_eq!(got.len(), want.len());
        let se: f32 = want.iter().zip(&got).map(|(a, b)| (a - b) * (a - b)).sum();
        let sig: f32 = want.iter().map(|v| v * v).sum();
        // Measured 4.4 % on this fixture. High for Q6_K because the fixture puts a 50x
        // outlier inside a 16-element sub-block on purpose; real weights are far kinder. The
        // bound is here to catch a wrong row stride or a swapped tensor, which land near 1.0.
        assert!((se / sig).sqrt() < 0.10, "Q6_K round-trip error {:.4}", (se / sig).sqrt());
    }

    #[test]
    fn passthrough_tensors_are_copied_byte_for_byte() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        pack(&src, &dst);
        let source = ShardedSafeTensors::open_dir(&src).unwrap();
        let packed = ShardedSafeTensors::open_dir(&dst).unwrap();
        for name in ["model.layers.0.mlp.gate.weight", "model.layers.0.input_layernorm.weight"] {
            assert_eq!(source.st_bytes(name).unwrap(), packed.st_bytes(name).unwrap(), "{name}");
            assert_eq!(packed.entry(name).unwrap().dtype.name(), "BF16", "{name}");
            assert_eq!(packed.entry(name).unwrap().shape, source.entry(name).unwrap().shape);
        }
    }

    #[test]
    fn support_files_come_along_and_shards_are_reindexed() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        pack(&src, &dst);
        assert!(dst.join("config.json").exists(), "config.json must be copied");
        assert!(dst.join("model.safetensors.index.json").exists());
        // The source index must not be copied over the one just written.
        let idx: Value =
            serde_json::from_str(&fs::read_to_string(dst.join("model.safetensors.index.json")).unwrap())
                .unwrap();
        assert!(idx["weight_map"].as_object().unwrap().len() == 7);
        assert!(!dst.join("model.safetensors").exists(), "output is sharded, not single-file");
    }

    #[test]
    fn an_existing_output_directory_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        fs::create_dir_all(&dst).unwrap();
        let err = run_pack_qwen35(&PackOptions {
            input_dir: src.display().to_string(),
            output_dir: dst.display().to_string(),
            expert_type: GGML_Q4_K,
            attn_type: GGML_Q6_K,
            embed_type: GGML_Q6_K,
            max_shard_size: 1 << 30,
            dry_run: false,
        })
        .unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
    }

    #[test]
    fn a_dry_run_writes_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let (src, dst) = (temp.path().join("in"), temp.path().join("out"));
        tiny_checkpoint(&src);
        run_pack_qwen35(&PackOptions {
            input_dir: src.display().to_string(),
            output_dir: dst.display().to_string(),
            expert_type: GGML_Q4_K,
            attn_type: GGML_Q6_K,
            embed_type: GGML_Q6_K,
            max_shard_size: 1 << 30,
            dry_run: true,
        })
        .unwrap();
        assert!(!dst.exists(), "dry run must not create the output directory");
    }

    #[test]
    fn an_unknown_type_name_is_rejected_with_the_supported_list() {
        let err = parse_type("IQ2_XXS").unwrap_err().to_string();
        assert!(err.contains("Q4_K") && err.contains("Q6_K"), "{err}");
        assert_eq!(parse_type("q4_k").unwrap(), GGML_Q4_K);
    }

    #[test]
    fn size_suffixes_parse() {
        assert_eq!(parse_size("5G").unwrap(), 5 << 30);
        assert_eq!(parse_size("750M").unwrap(), 750 << 20);
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert!(parse_size("big").is_err());
    }
}
