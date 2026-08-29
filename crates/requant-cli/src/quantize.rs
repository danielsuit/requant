//! `requant quantize`: load a GGUF, resolve a per-tensor recipe, and write a requantized GGUF.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};

use requant_calib::load_imatrix;
use requant_io::{
    ggml_type_name, is_float_type, tensor_nbytes, GgufReader, GgufValue, GgufWriter, TensorSpec,
};
use requant_quant::{dequantize_tensor, fallback_type, pack_float, quantize_tensor, Recipe};

use crate::common::{fmt_bytes, load_recipe, open_model, tag_all};

/// A per-tensor outcome, for the summary.
struct Outcome {
    src_type: u32,
    dst_type: u32,
    src_bytes: u64,
    dst_bytes: u64,
    action: Action,
}

#[derive(Clone, Copy)]
enum Action {
    Copy,     // target == source, bytes copied unchanged
    Convert,  // float -> float (F16/BF16/F32)
    Quantize, // float -> block quant
    Requant,  // quant -> (dequant) -> quant (lossy, warned)
}

pub fn run_quantize(
    input: &str,
    output: &str,
    recipe_path: Option<&str>,
    imatrix_path: Option<&str>,
) -> Result<()> {
    let (reader, layout) = open_model(input)?;
    let recipe = load_recipe(recipe_path)?;

    let imatrix = match imatrix_path {
        Some(p) => {
            let im = load_imatrix(p).with_context(|| format!("loading imatrix `{p}`"))?;
            let bad = requant_calib::count_nonfinite(&im);
            if bad > 0 {
                eprintln!("warning: imatrix contains {bad} non-finite / non-positive entries — results may degrade.");
            }
            eprintln!("loaded imatrix: {} tensors from `{p}`", im.len());
            Some(im)
        }
        None => None,
    };

    let (tags, _n_unknown) = tag_all(&reader, &layout)?;
    let default_type = recipe.defaults.bits.clone();
    let default_ggml = match &default_type {
        requant_quant::recipe::BitsOrStr::Named(b) => b.to_ggml_type(),
    };

    let mut writer = GgufWriter::new();
    writer.set_alignment(reader.alignment);

    // Copy KV verbatim except general.alignment (writer-managed) and general.file_type (we set it).
    for (k, v) in &reader.kv {
        if k == "general.alignment" || k == "general.file_type" {
            continue;
        }
        writer.add_kv(k.clone(), v.clone());
    }
    // Set general.file_type to the recipe default's ggml type. For mixed-precision files this is
    // the conventional "dominant" type; downstream loaders do not gate tensor data on it.
    writer.add_kv(
        "general.file_type".to_string(),
        GgufValue::U32(default_ggml),
    );

    let mut outcomes: Vec<Outcome> = Vec::with_capacity(reader.tensors.len());
    let mut src_total: u64 = 0;
    let mut dst_total: u64 = 0;
    let mut warned_requant = false;
    let mut n_fallback: usize = 0;

    for (i, t) in reader.tensors.iter().enumerate() {
        let tag = &tags[i];
        let policy = recipe
            .resolve_named(tag, &layout, &t.name)
            .with_context(|| {
                format!(
                    "resolving policy for tensor `{}` (role {})",
                    t.name,
                    tag.role.label()
                )
            })?;

        let src_type = t.ggml_type;
        let mut dst_type = policy.ggml_type;
        // Norms and 1-D biases are never block-quantized — keep the source bytes verbatim
        // (matches llama-quantize, which copies these tensors unchanged).
        if policy.copy_unchanged {
            dst_type = src_type;
        }
        // A GGUF tensor header can only name a real ggml type. NVFP4 and dense FP8 keep their
        // block scales in sibling tensors, so writing them here would produce a file that parses
        // and reconstructs garbage — the worst possible failure mode. Refuse instead.
        if !requant_io::is_gguf_type(dst_type) {
            bail!(
                "recipe assigns `{}` to {}, which has no ggml type and cannot go in a GGUF (its \
                 block scales are separate tensors). Target a safetensors checkpoint for that \
                 format, or pick a GGUF-representable one (`MXFP4` or `NVFP4_GGUF`).",
                t.name,
                ggml_type_name(dst_type)
            );
        }

        let src_bytes = tensor_nbytes(t)?;
        src_total += src_bytes;

        let (data, action) = if dst_type == src_type {
            (reader.tensor_bytes(i)?.to_vec(), Action::Copy)
        } else if is_float_type(dst_type) {
            let f32 = source_to_f32(&reader, i, &mut warned_requant, &t.name)?;
            let data = pack_float(dst_type, &f32)
                .with_context(|| format!("packing `{}` to {}", t.name, ggml_type_name(dst_type)))?;
            (data, Action::Convert)
        } else {
            // target is a block quant
            let src_was_quant = !is_float_type(src_type);
            let f32 = source_to_f32(&reader, i, &mut warned_requant, &t.name)?;
            let (rows, cols) = t.rows_cols();
            let im = if policy.use_imatrix {
                imatrix.as_ref().and_then(|im| im.get(&t.name))
            } else {
                None
            };
            if cols == 0 {
                bail!(
                    "tensor `{}` has zero columns — cannot block-quantize a scalar",
                    t.name
                );
            }
            // k-quants need cols divisible by their super-block (256); when a tensor's
            // in-features aren't (e.g. Qwen hidden=896), ggml falls back to a compatible
            // legacy quant. Match that so the output stays loadable and size-equivalent.
            let (actual_type, fell_back) = fallback_type(dst_type, cols);
            if fell_back && actual_type != dst_type {
                n_fallback += 1;
                if n_fallback <= 8 || n_fallback % 32 == 0 {
                    eprintln!(
                        "note: `{}` cols {} not divisible by {} block; falling back {} -> {}",
                        t.name,
                        cols,
                        ggml_type_name(dst_type),
                        ggml_type_name(dst_type),
                        ggml_type_name(actual_type),
                    );
                }
            }
            let data = quantize_tensor(actual_type, &f32, rows, cols, im).with_context(|| {
                format!(
                    "quantizing `{}` ({}×{}, role {}) to {}{}",
                    t.name,
                    rows,
                    cols,
                    tag.role.label(),
                    ggml_type_name(actual_type),
                    if im.is_some() { " [imatrix]" } else { "" }
                )
            })?;
            dst_type = actual_type;
            (
                data,
                if src_was_quant {
                    Action::Requant
                } else {
                    Action::Quantize
                },
            )
        };

        let dst_bytes = data.len() as u64;
        dst_total += dst_bytes;

        let spec = TensorSpec {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: dst_type,
            data,
        };
        writer.add_tensor(spec);

        outcomes.push(Outcome {
            src_type,
            dst_type,
            src_bytes,
            dst_bytes,
            action,
        });
    }

    writer
        .write_to(output)
        .with_context(|| format!("writing output GGUF `{output}`"))?;

    // Summary.
    println!("quantize: {input} -> {output}");
    println!(
        "  recipe   : {}",
        recipe_path.unwrap_or("(built-in MoE-aware default)")
    );
    println!("  imatrix  : {}", imatrix_path.unwrap_or("(none — RTN)"));
    println!("  tensors  : {}", outcomes.len());
    println!();
    print_type_breakdown(&outcomes);
    println!();
    println!("  source size : {}", fmt_bytes(src_total));
    println!("  output size : {}", fmt_bytes(dst_total));
    let ratio = if dst_total > 0 {
        src_total as f64 / dst_total as f64
    } else {
        0.0
    };
    println!(
        "  compression : {:.2}× ({} -> {})",
        ratio,
        ggml_type_name(dominant_src(&outcomes)),
        ggml_type_name(default_ggml)
    );

    // Surface any quant->quant (lossy) conversions.
    let n_requant = outcomes
        .iter()
        .filter(|o| matches!(o.action, Action::Requant))
        .count();
    if n_requant > 0 {
        eprintln!();
        eprintln!(
            "warning: {n_requant} tensor(s) were requantized from an already-quantized source."
        );
        eprintln!(
            "         Information was already lost in the source; this output is strictly worse"
        );
        eprintln!("         than quantizing from fp16. Keep an fp16 source when available.");
    }

    // DESIGN §4's quant→quant caveat has a severe case that deserves more than a shared line: a
    // source that is *already* in the FP4 family. Going FP4 → FP4-with-some-tensors-lower means
    // there is no higher-precision master anywhere in the chain. The allocator will still place
    // bits intelligently — it just cannot recover signal that was never stored. Nothing downstream
    // can detect this from the output file, so it has to be said here, loudly.
    let n_fp4_src = outcomes
        .iter()
        .filter(|o| requant_io::is_fp4_family(o.src_type))
        .count();
    if n_fp4_src > 0 {
        eprintln!();
        eprintln!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║ NO FULL-PRECISION MASTER                                                     ║"
        );
        eprintln!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
        eprintln!("  {n_fp4_src} tensor(s) were read from an FP4-family source and re-quantized.");
        eprintln!("  This is quant→quant *inside* the 4-bit regime: the fp16 weights these came");
        eprintln!("  from do not exist in this pipeline, so the error you are adding compounds on");
        eprintln!("  top of error that is already there and cannot be undone.");
        eprintln!();
        eprintln!("  Bit allocation still helps — it decides where the remaining error lands. It");
        eprintln!("  cannot restore precision. Whether the result holds task quality is an");
        eprintln!("  empirical question, and the only thing that answers it is measurement:");
        eprintln!("    requant sensitivity --input <src> --corpus <corpus> -o sens.json");
        eprintln!("    requant eval --quant {output} --reference <src> --kl");
    }

    Ok(())
}

/// Get a tensor's data as f32, dequantizing if the source is itself a block quant (with a warning).
fn source_to_f32(reader: &GgufReader, i: usize, warned: &mut bool, name: &str) -> Result<Vec<f32>> {
    let info = &reader.tensors[i];
    if is_float_type(info.ggml_type) {
        reader.tensor_to_f32(i)
    } else {
        if !*warned {
            eprintln!(
                "warning: dequantizing source `{name}` (type {}) to requantize — lossy path",
                ggml_type_name(info.ggml_type)
            );
            *warned = true;
        }
        let bytes = reader.tensor_bytes(i)?;
        let (rows, cols) = info.rows_cols();
        dequantize_tensor(info.ggml_type, bytes, rows, cols).with_context(|| {
            format!(
                "dequantizing source `{name}` (type {})",
                ggml_type_name(info.ggml_type)
            )
        })
    }
}

fn dominant_src(outcomes: &[Outcome]) -> u32 {
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    for o in outcomes {
        *counts.entry(o.src_type).or_default() += o.src_bytes;
    }
    counts
        .into_iter()
        .max_by_key(|(_, b)| *b)
        .map(|(t, _)| t)
        .unwrap_or(0)
}

fn print_type_breakdown(outcomes: &[Outcome]) {
    let mut by_dst: BTreeMap<u32, (u64, usize)> = BTreeMap::new();
    for o in outcomes {
        let e = by_dst.entry(o.dst_type).or_insert((0, 0));
        e.0 += o.dst_bytes;
        e.1 += 1;
    }
    println!("  output by type:");
    println!("    {:<10} {:>8} {:>10}", "type", "tensors", "bytes");
    for (t, (b, n)) in &by_dst {
        println!(
            "    {:<10} {:>8} {:>10}",
            ggml_type_name(*t),
            n,
            fmt_bytes(*b)
        );
    }
}

#[allow(dead_code)]
fn _unused(_: &Recipe) {}
