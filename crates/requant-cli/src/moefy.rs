//! Dense Qwen3.8 -> Qwen3.5-MoE checkpoint structural conversion.
//!
//! This is deliberately a *warm-start builder*, not a claim that an untrained router turns a
//! dense model into a good sparse model. Each dense SwiGLU intermediate dimension is assigned to
//! exactly one routed expert, preserving the total FFN parameter count. The output has the native
//! packed expert layout consumed by Transformers' `Qwen3_5MoeExperts`. Router and shared-expert
//! tensors are zero-initialized, and the manifest marks the checkpoint as requiring router/expert
//! training before inference.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use requant_io::{ShardedSafeTensors, StDtype, TensorSource};
use serde_json::{json, Map, Value};

const TEXT_LAYER_PREFIX: &str = "model.language_model.layers.";

#[derive(Debug, Clone)]
pub struct MoefyOptions {
    pub input_dir: String,
    pub output_dir: String,
    pub experts: usize,
    pub top_k: usize,
    pub max_shard_size: String,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
struct DenseLayer {
    prefix: String,
    gate: String,
    up: String,
    down: String,
    dtype: StDtype,
    hidden: usize,
    intermediate: usize,
    expert_intermediate: usize,
}

#[derive(Debug, Clone)]
enum Payload {
    Copy {
        source: String,
    },
    GateUp {
        gate: String,
        up: String,
        experts: usize,
        expert_intermediate: usize,
        hidden: usize,
    },
    Down {
        source: String,
        experts: usize,
        hidden: usize,
        intermediate: usize,
        expert_intermediate: usize,
        scale: f32,
    },
    Zeros,
    RandomRouter {
        seed: u64,
        amplitude: f32,
    },
}

#[derive(Debug, Clone)]
struct OutputTensor {
    name: String,
    dtype: StDtype,
    shape: Vec<u64>,
    nbytes: u64,
    payload: Payload,
}

pub fn run_moefy_qwen38(opts: &MoefyOptions) -> Result<()> {
    let input = Path::new(&opts.input_dir);
    let output = Path::new(&opts.output_dir);
    if !input.is_dir() {
        bail!(
            "input checkpoint directory does not exist: {}",
            input.display()
        );
    }
    if opts.experts < 2 {
        bail!("--experts must be at least 2");
    }
    if opts.top_k == 0 || opts.top_k > opts.experts {
        bail!(
            "--top-k must be in 1..={} (got {})",
            opts.experts,
            opts.top_k
        );
    }
    let max_shard_bytes = parse_size(&opts.max_shard_size)?;
    if max_shard_bytes == 0 {
        bail!("--max-shard-size must be positive");
    }

    let config_path = input.join("config.json");
    let config_text = fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: Value = serde_json::from_str(&config_text)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    let (new_config, n_layers, mtp_layers, hidden, intermediate) =
        convert_config(config, opts.experts, opts.top_k)?;

    let source = ShardedSafeTensors::open_dir(input)?;
    let layers = discover_layers(
        &source,
        n_layers,
        mtp_layers,
        hidden,
        intermediate,
        opts.experts,
    )?;
    let tensors = build_output_plan(&source, &layers, opts.experts)?;
    let input_bytes: u64 = source
        .names()
        .map(|n| {
            source
                .entry(n)
                .map(|e| (e.end - e.start) as u64)
                .unwrap_or(0)
        })
        .sum();
    let output_bytes: u64 = tensors.iter().map(|t| t.nbytes).sum();

    println!("Qwen3.8 dense -> Qwen3.5-MoE warm-start plan");
    println!("  text layers       : {n_layers}");
    println!("  MTP layers        : {mtp_layers}");
    println!("  hidden/intermediate: {hidden}/{intermediate}");
    println!("  experts / top-k   : {} / {}", opts.experts, opts.top_k);
    println!("  expert width      : {}", layers[0].expert_intermediate);
    println!("  tensor bytes      : {input_bytes} -> {output_bytes}");
    println!("  initialization    : neuron partition; seeded router; zero shared expert");
    println!("  training required : yes (do not use as an inference checkpoint yet)");

    if opts.dry_run {
        return Ok(());
    }
    if output.exists() {
        bail!(
            "output directory already exists: {} (refusing to overwrite)",
            output.display()
        );
    }
    if output.starts_with(input) {
        bail!("output directory may not be nested inside the input checkpoint directory");
    }

    fs::create_dir_all(output)
        .with_context(|| format!("creating output directory {}", output.display()))?;
    let result = (|| {
        copy_support_files(input, output)?;
        write_json(output.join("config.json"), &new_config)?;
        write_checkpoint(&source, output, &tensors, max_shard_bytes)?;
        let manifest = json!({
            "format": "requant-qwen38-dense-to-qwen35-moe-v1",
            "source": input.display().to_string(),
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "strategy": "contiguous-neuron-partition",
            "num_experts": opts.experts,
            "num_experts_per_tok": opts.top_k,
            "dense_intermediate_size": intermediate,
            "moe_intermediate_size": layers[0].expert_intermediate,
            "shared_expert_intermediate_size": 1,
        "router_initialization": "deterministic uniform random in [-0.02, 0.02]",
        "shared_expert_initialization": "zeros",
        "down_projection_scale": opts.experts,
            "requires_training": true,
            "inference_ready": false,
            "notes": [
                "Attention, vision, linear-attention, embeddings, lm_head, and MTP non-MLP tensors are copied byte-for-byte.",
                "The MTP MLP is converted to the same packed expert layout as the backbone MLPs.",
                "Dense SwiGLU neurons are partitioned across routed experts with no duplication.",
                "Expert down projections are multiplied by num_experts to preserve the dense FFN output in expectation under balanced routing.",
                "The seeded random router avoids top-k tie collapse, but the checkpoint is not functionally equivalent before training.",
                "Train the router and experts (with load balancing and dense-teacher distillation) before reducing or evaluating quality."
            ]
        });
        write_json(output.join("requant_moe_conversion.json"), &manifest)
    })();
    if result.is_err() {
        eprintln!(
            "conversion stopped; partial output was left at {} for inspection/removal",
            output.display()
        );
    }
    result?;

    println!("  wrote             : {}", output.display());
    Ok(())
}

fn convert_config(
    mut config: Value,
    experts: usize,
    top_k: usize,
) -> Result<(Value, usize, usize, usize, usize)> {
    let root = config
        .as_object_mut()
        .ok_or_else(|| anyhow!("config.json root must be an object"))?;
    let model_type = root.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "qwen3_5" {
        bail!("expected dense Qwen3.8/Qwen3.5 model_type `qwen3_5`, found `{model_type}`");
    }
    let text = root
        .get_mut("text_config")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("config.json is missing object `text_config`"))?;
    let text_type = text.get("model_type").and_then(Value::as_str).unwrap_or("");
    if text_type != "qwen3_5_text" {
        bail!("expected dense text_config.model_type `qwen3_5_text`, found `{text_type}`");
    }
    let n_layers = required_usize(text, "num_hidden_layers")?;
    let mtp_layers = text
        .get("mtp_num_hidden_layers")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(0);
    let hidden = required_usize(text, "hidden_size")?;
    let intermediate = required_usize(text, "intermediate_size")?;
    if intermediate % experts != 0 {
        bail!(
            "dense intermediate_size {intermediate} is not divisible by {experts} experts; choose a divisor so no neurons are dropped"
        );
    }

    root.insert(
        "architectures".into(),
        json!(["Qwen3_5MoeForConditionalGeneration"]),
    );
    root.insert("model_type".into(), json!("qwen3_5_moe"));
    let text = root
        .get_mut("text_config")
        .unwrap()
        .as_object_mut()
        .unwrap();
    text.insert("model_type".into(), json!("qwen3_5_moe_text"));
    text.remove("intermediate_size");
    text.insert(
        "moe_intermediate_size".into(),
        json!(intermediate / experts),
    );
    text.insert("shared_expert_intermediate_size".into(), json!(1));
    text.insert("num_experts".into(), json!(experts));
    text.insert("num_experts_per_tok".into(), json!(top_k));
    text.insert("output_router_logits".into(), json!(true));
    text.insert("router_aux_loss_coef".into(), json!(0.001));
    Ok((config, n_layers, mtp_layers, hidden, intermediate))
}

fn required_usize(obj: &Map<String, Value>, key: &str) -> Result<usize> {
    obj.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| anyhow!("text_config.{key} must be a positive integer"))
}

fn discover_layers(
    source: &ShardedSafeTensors,
    n_layers: usize,
    mtp_layers: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
) -> Result<Vec<DenseLayer>> {
    let mut prefixes: Vec<String> = (0..n_layers)
        .map(|layer| format!("{TEXT_LAYER_PREFIX}{layer}.mlp"))
        .collect();
    prefixes.extend((0..mtp_layers).map(|layer| format!("mtp.layers.{layer}.mlp")));
    let mut layers = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let gate = format!("{prefix}.gate_proj.weight");
        let up = format!("{prefix}.up_proj.weight");
        let down = format!("{prefix}.down_proj.weight");
        let ge = source
            .entry(&gate)
            .ok_or_else(|| anyhow!("missing dense layer tensor `{gate}`"))?;
        let ue = source
            .entry(&up)
            .ok_or_else(|| anyhow!("missing dense layer tensor `{up}`"))?;
        let de = source
            .entry(&down)
            .ok_or_else(|| anyhow!("missing dense layer tensor `{down}`"))?;
        if ge.dtype != ue.dtype || ge.dtype != de.dtype {
            bail!("MLP `{prefix}`: gate/up/down dtypes differ");
        }
        if !matches!(ge.dtype, StDtype::BF16 | StDtype::F16 | StDtype::F32) {
            bail!(
                "MLP `{prefix}`: dense dtype {:?} is unsupported; start from BF16/F16/F32, not an already-quantized checkpoint",
                ge.dtype
            );
        }
        check_shape(&gate, &ge.shape, &[intermediate, hidden])?;
        check_shape(&up, &ue.shape, &[intermediate, hidden])?;
        check_shape(&down, &de.shape, &[hidden, intermediate])?;
        layers.push(DenseLayer {
            prefix,
            gate,
            up,
            down,
            dtype: ge.dtype,
            hidden,
            intermediate,
            expert_intermediate: intermediate / experts,
        });
    }
    Ok(layers)
}

fn check_shape(name: &str, actual: &[u64], expected: &[usize]) -> Result<()> {
    let expected: Vec<u64> = expected.iter().map(|&x| x as u64).collect();
    if actual != expected {
        bail!("tensor `{name}` has shape {actual:?}, expected {expected:?}");
    }
    Ok(())
}

fn build_output_plan(
    source: &ShardedSafeTensors,
    layers: &[DenseLayer],
    experts: usize,
) -> Result<Vec<OutputTensor>> {
    let dense_names: BTreeSet<&str> = layers
        .iter()
        .flat_map(|l| [l.gate.as_str(), l.up.as_str(), l.down.as_str()])
        .collect();
    let mut out = Vec::new();
    for name in source.names() {
        if dense_names.contains(name) {
            continue;
        }
        let e = source.entry(name).unwrap();
        out.push(OutputTensor {
            name: name.to_string(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            nbytes: (e.end - e.start) as u64,
            payload: Payload::Copy {
                source: name.to_string(),
            },
        });
    }
    for l in layers {
        let elem = l.dtype.size() as u64;
        let gate_up_name = format!("{}.experts.gate_up_proj", l.prefix);
        let down_name = format!("{}.experts.down_proj", l.prefix);
        let router_name = format!("{}.gate.weight", l.prefix);
        let shared = format!("{}.shared_expert", l.prefix);
        let ei = l.expert_intermediate;
        out.push(OutputTensor {
            name: gate_up_name,
            dtype: l.dtype,
            shape: vec![experts as u64, (2 * ei) as u64, l.hidden as u64],
            nbytes: (experts * 2 * ei * l.hidden) as u64 * elem,
            payload: Payload::GateUp {
                gate: l.gate.clone(),
                up: l.up.clone(),
                experts,
                expert_intermediate: ei,
                hidden: l.hidden,
            },
        });
        out.push(OutputTensor {
            name: down_name,
            dtype: l.dtype,
            shape: vec![experts as u64, l.hidden as u64, ei as u64],
            nbytes: (experts * l.hidden * ei) as u64 * elem,
            payload: Payload::Down {
                source: l.down.clone(),
                experts,
                hidden: l.hidden,
                intermediate: l.intermediate,
                expert_intermediate: ei,
                scale: experts as f32,
            },
        });
        let router_shape = vec![experts as u64, l.hidden as u64];
        out.push(OutputTensor {
            name: router_name,
            dtype: l.dtype,
            nbytes: router_shape.iter().product::<u64>() * elem,
            shape: router_shape,
            payload: Payload::RandomRouter {
                seed: stable_seed(&l.prefix),
                amplitude: 0.02,
            },
        });
        for (name, shape) in [
            (
                format!("{shared}.gate_proj.weight"),
                vec![1, l.hidden as u64],
            ),
            (format!("{shared}.up_proj.weight"), vec![1, l.hidden as u64]),
            (
                format!("{shared}.down_proj.weight"),
                vec![l.hidden as u64, 1],
            ),
            (
                format!("{}.shared_expert_gate.weight", l.prefix),
                vec![1, l.hidden as u64],
            ),
        ] {
            let nbytes = shape.iter().product::<u64>() * elem;
            out.push(OutputTensor {
                name,
                dtype: l.dtype,
                shape,
                nbytes,
                payload: Payload::Zeros,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    let mut seen = BTreeSet::new();
    for tensor in &out {
        if !seen.insert(tensor.name.as_str()) {
            bail!("output tensor name collision: `{}`", tensor.name);
        }
    }
    Ok(out)
}

fn write_checkpoint(
    source: &ShardedSafeTensors,
    output: &Path,
    tensors: &[OutputTensor],
    max_shard_bytes: u64,
) -> Result<()> {
    let mut shards: Vec<&[OutputTensor]> = Vec::new();
    let mut start = 0usize;
    let mut size = 0u64;
    for (i, tensor) in tensors.iter().enumerate() {
        if i > start && size.saturating_add(tensor.nbytes) > max_shard_bytes {
            shards.push(&tensors[start..i]);
            start = i;
            size = 0;
        }
        size = size.saturating_add(tensor.nbytes);
    }
    if start < tensors.len() {
        shards.push(&tensors[start..]);
    }

    let count = shards.len();
    let mut weight_map = Map::new();
    for (i, shard) in shards.iter().enumerate() {
        let filename = format!("model-{:05}-of-{:05}.safetensors", i + 1, count);
        write_shard(source, &output.join(&filename), shard)?;
        for tensor in *shard {
            weight_map.insert(tensor.name.clone(), json!(filename));
        }
    }
    let total_size: u64 = tensors.iter().map(|t| t.nbytes).sum();
    let index = json!({
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    });
    write_json(output.join("model.safetensors.index.json"), &index)
}

fn write_shard(source: &ShardedSafeTensors, path: &Path, tensors: &[OutputTensor]) -> Result<()> {
    let mut header = Map::new();
    header.insert("__metadata__".into(), json!({"format": "pt"}));
    let mut offset = 0u64;
    for tensor in tensors {
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": tensor.dtype.name(),
                "shape": tensor.shape,
                "data_offsets": [offset, offset + tensor.nbytes],
            }),
        );
        offset += tensor.nbytes;
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header))?;
    let padding = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', padding));

    let partial = path.with_extension("safetensors.partial");
    let file = File::create(&partial).with_context(|| format!("creating {}", partial.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    for tensor in tensors {
        write_payload(source, &mut writer, tensor)
            .with_context(|| format!("writing tensor `{}`", tensor.name))?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(&partial, path)
        .with_context(|| format!("renaming {} to {}", partial.display(), path.display()))?;
    Ok(())
}

fn write_payload<W: Write>(
    source: &ShardedSafeTensors,
    writer: &mut W,
    tensor: &OutputTensor,
) -> Result<()> {
    match &tensor.payload {
        Payload::Copy { source: name } => writer.write_all(source.st_bytes(name)?)?,
        Payload::Zeros => write_zeros(writer, tensor.nbytes)?,
        Payload::RandomRouter { seed, amplitude } => {
            write_random_floats(writer, tensor.dtype, tensor.nbytes, *seed, *amplitude)?
        }
        Payload::GateUp {
            gate,
            up,
            experts,
            expert_intermediate,
            hidden,
        } => {
            let row_bytes = hidden * tensor.dtype.size();
            let group_bytes = expert_intermediate * row_bytes;
            let gate = source.st_bytes(gate)?;
            let up = source.st_bytes(up)?;
            for expert in 0..*experts {
                let start = expert * group_bytes;
                let end = start + group_bytes;
                writer.write_all(&gate[start..end])?;
                writer.write_all(&up[start..end])?;
            }
        }
        Payload::Down {
            source: name,
            experts,
            hidden,
            intermediate,
            expert_intermediate,
            scale,
        } => {
            let src = source.st_bytes(name)?;
            let elem = tensor.dtype.size();
            let src_row_bytes = intermediate * elem;
            let expert_row_bytes = expert_intermediate * elem;
            for expert in 0..*experts {
                let col_start = expert * expert_row_bytes;
                for row in 0..*hidden {
                    let start = row * src_row_bytes + col_start;
                    write_scaled_floats(
                        writer,
                        tensor.dtype,
                        &src[start..start + expert_row_bytes],
                        *scale,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn write_scaled_floats<W: Write>(
    writer: &mut W,
    dtype: StDtype,
    source: &[u8],
    scale: f32,
) -> Result<()> {
    match dtype {
        StDtype::BF16 => {
            for bytes in source.chunks_exact(2) {
                let value = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
                writer.write_all(&bf16::from_f32(value * scale).to_bits().to_le_bytes())?;
            }
        }
        StDtype::F16 => {
            for bytes in source.chunks_exact(2) {
                let value = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
                writer.write_all(&f16::from_f32(value * scale).to_bits().to_le_bytes())?;
            }
        }
        StDtype::F32 => {
            for bytes in source.chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                writer.write_all(&(value * scale).to_le_bytes())?;
            }
        }
        _ => bail!("cannot scale non-floating dtype {dtype:?}"),
    }
    Ok(())
}

fn write_random_floats<W: Write>(
    writer: &mut W,
    dtype: StDtype,
    nbytes: u64,
    mut state: u64,
    amplitude: f32,
) -> Result<()> {
    let count = nbytes / dtype.size() as u64;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 40) as f32 / ((1u32 << 24) - 1) as f32;
        let value = (unit * 2.0 - 1.0) * amplitude;
        match dtype {
            StDtype::BF16 => writer.write_all(&bf16::from_f32(value).to_bits().to_le_bytes())?,
            StDtype::F16 => writer.write_all(&f16::from_f32(value).to_bits().to_le_bytes())?,
            StDtype::F32 => writer.write_all(&value.to_le_bytes())?,
            _ => bail!("cannot initialize router with non-floating dtype {dtype:?}"),
        }
    }
    Ok(())
}

fn stable_seed(name: &str) -> u64 {
    // FNV-1a: stable across Rust versions and process invocations, unlike DefaultHasher.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash.max(1)
}

fn write_zeros<W: Write>(writer: &mut W, nbytes: u64) -> Result<()> {
    let zero = [0u8; 64 * 1024];
    let mut remaining = nbytes;
    while remaining > 0 {
        let n = remaining.min(zero.len() as u64) as usize;
        writer.write_all(&zero[..n])?;
        remaining -= n as u64;
    }
    Ok(())
}

fn copy_support_files(input: &Path, output: &Path) -> Result<()> {
    for entry in fs::read_dir(input).with_context(|| format!("listing {}", input.display()))? {
        let entry = entry?;
        // `Path::is_file` follows symlinks, which matters for Hugging Face cache snapshots.
        if !entry.path().is_file() {
            continue;
        }
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s == "config.json"
            || name_s == "model.safetensors.index.json"
            || name_s.ends_with(".safetensors")
            || name_s == "requant_moe_conversion.json"
        {
            continue;
        }
        fs::copy(entry.path(), output.join(&name))
            .with_context(|| format!("copying support file {name_s}"))?;
    }
    Ok(())
}

fn write_json(path: PathBuf, value: &Value) -> Result<()> {
    let mut data = serde_json::to_vec_pretty(value)?;
    data.push(b'\n');
    fs::write(&path, data).with_context(|| format!("writing {}", path.display()))
}

fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty size");
    }
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let value: u64 = s[..split]
        .parse()
        .with_context(|| format!("invalid size `{s}`"))?;
    let suffix = s[split..].trim().to_ascii_uppercase();
    let multiplier = match suffix.as_str() {
        "" | "B" => 1,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024u64.pow(2),
        "G" | "GB" | "GIB" => 1024u64.pow(3),
        _ => bail!("unknown size suffix in `{s}` (use B, K, M, or G)"),
    };
    value
        .checked_mul(multiplier)
        .ok_or_else(|| anyhow!("size `{s}` overflows u64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut header = Map::new();
        let mut blob = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let start = blob.len();
            blob.extend_from_slice(bytes);
            header.insert(
                (*name).into(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, blob.len()]}),
            );
        }
        let h = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(blob);
        out
    }

    fn bf16_seq(n: usize, base: u16) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (base + i as u16).to_le_bytes())
            .collect()
    }

    fn fixture(dir: &Path) {
        let config = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 2,
                "mtp_num_hidden_layers": 1,
                "hidden_size": 4,
                "intermediate_size": 8,
                "layer_types": ["linear_attention", "full_attention"]
            },
            "vision_config": {"model_type": "qwen3_5", "hidden_size": 2}
        });
        write_json(dir.join("config.json"), &config).unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}\n").unwrap();
        let mut owned = Vec::new();
        for layer in 0..2 {
            owned.push((
                format!("{TEXT_LAYER_PREFIX}{layer}.mlp.gate_proj.weight"),
                "BF16",
                vec![8, 4],
                bf16_seq(32, 1000 + layer * 100),
            ));
            owned.push((
                format!("{TEXT_LAYER_PREFIX}{layer}.mlp.up_proj.weight"),
                "BF16",
                vec![8, 4],
                bf16_seq(32, 2000 + layer * 100),
            ));
            owned.push((
                format!("{TEXT_LAYER_PREFIX}{layer}.mlp.down_proj.weight"),
                "BF16",
                vec![4, 8],
                bf16_seq(32, 3000 + layer * 100),
            ));
        }
        owned.push((
            "mtp.layers.0.mlp.gate_proj.weight".into(),
            "BF16",
            vec![8, 4],
            bf16_seq(32, 4000),
        ));
        owned.push((
            "mtp.layers.0.mlp.up_proj.weight".into(),
            "BF16",
            vec![8, 4],
            bf16_seq(32, 5000),
        ));
        owned.push((
            "mtp.layers.0.mlp.down_proj.weight".into(),
            "BF16",
            vec![4, 8],
            bf16_seq(32, 6000),
        ));
        owned.push((
            "model.language_model.embed_tokens.weight".into(),
            "BF16",
            vec![3, 4],
            bf16_seq(12, 50),
        ));
        let borrowed: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = owned
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
            .collect();
        fs::write(dir.join("model.safetensors"), safetensors(&borrowed)).unwrap();
    }

    #[test]
    fn converts_tiny_dense_checkpoint_without_model_weights() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("dense");
        let output = temp.path().join("moe");
        fs::create_dir(&input).unwrap();
        fixture(&input);
        run_moefy_qwen38(&MoefyOptions {
            input_dir: input.display().to_string(),
            output_dir: output.display().to_string(),
            experts: 2,
            top_k: 1,
            max_shard_size: "1K".into(),
            dry_run: false,
        })
        .unwrap();

        let cfg: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(cfg["model_type"], "qwen3_5_moe");
        assert_eq!(cfg["text_config"]["model_type"], "qwen3_5_moe_text");
        assert_eq!(cfg["text_config"]["num_experts"], 2);
        assert_eq!(cfg["text_config"]["moe_intermediate_size"], 4);
        assert!(cfg["text_config"].get("intermediate_size").is_none());

        let out = ShardedSafeTensors::open_dir(&output).unwrap();
        let p = format!("{TEXT_LAYER_PREFIX}0.mlp");
        let gu = out.entry(&format!("{p}.experts.gate_up_proj")).unwrap();
        assert_eq!(gu.shape, vec![2, 8, 4]);
        let down = out.entry(&format!("{p}.experts.down_proj")).unwrap();
        assert_eq!(down.shape, vec![2, 4, 4]);
        assert_eq!(
            out.entry(&format!("{p}.gate.weight")).unwrap().shape,
            vec![2, 4]
        );
        assert!(!out.contains(&format!("{p}.gate_proj.weight")));
        assert!(out.contains("model.language_model.embed_tokens.weight"));
        assert!(out.contains("mtp.layers.0.mlp.experts.gate_up_proj"));
        assert!(!out.contains("mtp.layers.0.mlp.gate_proj.weight"));
        assert_eq!(fs::read(output.join("tokenizer.json")).unwrap(), b"{}\n");
        assert_eq!(
            out.bytes("model.language_model.embed_tokens.weight")
                .unwrap(),
            bf16_seq(12, 50)
        );

        let gate = bf16_seq(32, 1000);
        let up = bf16_seq(32, 2000);
        let packed = out.bytes(&format!("{p}.experts.gate_up_proj")).unwrap();
        assert_eq!(&packed[0..32], &gate[0..32]);
        assert_eq!(&packed[32..64], &up[0..32]);
        assert_eq!(&packed[64..96], &gate[32..64]);
        assert_eq!(&packed[96..128], &up[32..64]);

        let dense_down = bf16_seq(32, 3000);
        let packed_down = out.bytes(&format!("{p}.experts.down_proj")).unwrap();
        let expected_first_unscaled: Vec<u8> = (0..4)
            .flat_map(|row| dense_down[row * 16..row * 16 + 8].to_vec())
            .collect();
        let mut expected_first = Vec::new();
        write_scaled_floats(
            &mut expected_first,
            StDtype::BF16,
            &expected_first_unscaled,
            2.0,
        )
        .unwrap();
        assert_eq!(&packed_down[..32], expected_first.as_slice());
        assert!(out
            .bytes(&format!("{p}.gate.weight"))
            .unwrap()
            .iter()
            .any(|&b| b != 0));
        assert_eq!(
            serde_json::from_slice::<Value>(
                &fs::read(output.join("requant_moe_conversion.json")).unwrap()
            )
            .unwrap()["inference_ready"],
            false
        );

        // When the reference Python package is available, validate every generated shard with an
        // implementation independent of requant's reader. Environments without it still run all
        // byte-layout assertions above.
        let python_has_safetensors = std::process::Command::new("python3")
            .args(["-c", "import safetensors"])
            .status()
            .is_ok_and(|s| s.success());
        if python_has_safetensors {
            let script = r#"
import pathlib, sys
from safetensors import safe_open
root = pathlib.Path(sys.argv[1])
count = 0
for shard in root.glob('*.safetensors'):
    with safe_open(shard, framework='numpy') as f:
        for key in f.keys():
            f.get_slice(key).get_shape()
            count += 1
assert count > 0
"#;
            let status = std::process::Command::new("python3")
                .args(["-c", script, output.to_str().unwrap()])
                .status()
                .unwrap();
            assert!(
                status.success(),
                "Python safetensors rejected generated shards"
            );
        }
    }

    #[test]
    fn dry_run_validates_without_writing() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("dense");
        let output = temp.path().join("unused");
        fs::create_dir(&input).unwrap();
        fixture(&input);
        run_moefy_qwen38(&MoefyOptions {
            input_dir: input.display().to_string(),
            output_dir: output.display().to_string(),
            experts: 2,
            top_k: 1,
            max_shard_size: "1M".into(),
            dry_run: true,
        })
        .unwrap();
        assert!(!output.exists());
    }

    #[test]
    fn rejects_non_divisible_expert_count() {
        let config = json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "intermediate_size": 8
            }
        });
        assert!(convert_config(config, 3, 1).is_err());
    }

    #[test]
    fn parses_binary_shard_sizes() {
        assert_eq!(parse_size("5G").unwrap(), 5 * 1024u64.pow(3));
        assert_eq!(parse_size("750MiB").unwrap(), 750 * 1024u64.pow(2));
        assert!(parse_size("1T").is_err());
    }
}
