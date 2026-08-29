//! Dense Qwen3.8 -> Qwen3.5-MoE checkpoint structural conversion.
//!
//! This is deliberately a *warm-start builder*, not a claim that an untrained router turns a
//! dense model into a good sparse model. Each dense SwiGLU intermediate dimension is assigned to
//! exactly one routed expert, preserving the total FFN parameter count. The output has the native
//! packed expert layout consumed by Transformers' `Qwen3_5MoeExperts`. Routers are initialized
//! deterministically, shared-expert tensors are zero-initialized, and the manifest marks the
//! checkpoint as requiring router/expert training before inference.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use requant_io::{ShardedSafeTensors, StDtype, TensorSource};
use serde_json::{json, Map, Value};

const VLM_LAYER_PREFIX: &str = "model.language_model.layers.";
const TEXT_LAYER_PREFIX: &str = "model.layers.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFlavor {
    VisionLanguage,
    TextOnly,
}

#[derive(Debug, Clone)]
struct ConversionProfile {
    config: Value,
    flavor: ModelFlavor,
    source_model_type: String,
    target_model_type: &'static str,
    target_architecture: &'static str,
    layer_prefix: &'static str,
    n_layers: usize,
    mtp_layers: usize,
    hidden: usize,
    intermediate: usize,
    initializer_range: f32,
    configured_layer_types: Option<Vec<String>>,
}

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
    let mut profile = convert_config(config, opts.experts, opts.top_k)?;

    let source = ShardedSafeTensors::open_dir(input)?;
    let attention = audit_attention_layout(&source, &profile)?;
    set_explicit_layer_types(&mut profile, &attention.kinds);
    let layers = discover_layers(&source, &profile, opts.experts)?;
    let tensors = build_output_plan(&source, &layers, opts.experts, profile.initializer_range)?;
    audit_passthrough_plan(&source, &layers, &tensors)?;
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

    println!("Qwen3.5-family dense -> Qwen3.5-MoE warm-start plan");
    println!("  source model type : {}", profile.source_model_type);
    println!("  target class      : {}", profile.target_architecture);
    println!("  text layers       : {}", profile.n_layers);
    println!("  attention layout  : {}", attention.summary());
    println!("  MTP layers        : {}", profile.mtp_layers);
    println!(
        "  hidden/intermediate: {}/{}",
        profile.hidden, profile.intermediate
    );
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
        write_json(output.join("config.json"), &profile.config)?;
        write_checkpoint(&source, output, &tensors, max_shard_bytes)?;
        let manifest = json!({
            "format": "requant-qwen35-family-dense-to-qwen35-moe-v2",
            "source": input.display().to_string(),
            "source_model_type": profile.source_model_type,
            "source_flavor": match profile.flavor {
                ModelFlavor::VisionLanguage => "vision-language",
                ModelFlavor::TextOnly => "text-only",
            },
            "target_model_type": profile.target_model_type,
            "architecture": profile.target_architecture,
            "attention_layout": attention.kinds,
            "strategy": "contiguous-neuron-partition",
            "num_experts": opts.experts,
            "num_experts_per_tok": opts.top_k,
            "dense_intermediate_size": profile.intermediate,
            "moe_intermediate_size": layers[0].expert_intermediate,
            "shared_expert_intermediate_size": 1,
            "router_initialization": format!(
                "deterministic uniform random in [-{}, {}]",
                profile.initializer_range, profile.initializer_range
            ),
            "shared_expert_initialization": "zeros",
            "down_projection_scale": opts.experts,
            "requires_training": true,
            "inference_ready": false,
            "notes": [
                "Every non-MLP source tensor is copied byte-for-byte with the same name, dtype, shape, and payload.",
                "Full attention, linear attention, hybrid layouts, vision attention, and MTP attention are preserved rather than rewritten.",
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

fn convert_config(mut config: Value, experts: usize, top_k: usize) -> Result<ConversionProfile> {
    let root = config
        .as_object_mut()
        .ok_or_else(|| anyhow!("config.json root must be an object"))?;
    let source_model_type = root
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if root.contains_key("quantization_config") {
        bail!(
            "quantized source checkpoints are unsupported for dense-to-MoE conversion; use the original BF16/F16/F32 checkpoint"
        );
    }
    let flavor = match source_model_type.as_str() {
        "qwen3_5" => ModelFlavor::VisionLanguage,
        "qwen3_5_text" => ModelFlavor::TextOnly,
        other => bail!(
            "unsupported dense model_type `{other}`. This converter supports the Qwen3.5-family runtime only: `qwen3_5` (multimodal) and `qwen3_5_text` (text-only). Other families need their own target MoE class and expert layout."
        ),
    };

    let text: &Map<String, Value> = match flavor {
        ModelFlavor::VisionLanguage => {
            let text = root
                .get("text_config")
                .and_then(Value::as_object)
                .ok_or_else(|| anyhow!("multimodal config.json is missing object `text_config`"))?;
            let text_type = text.get("model_type").and_then(Value::as_str).unwrap_or("");
            if text_type != "qwen3_5_text" {
                bail!("expected dense text_config.model_type `qwen3_5_text`, found `{text_type}`");
            }
            text
        }
        ModelFlavor::TextOnly => root,
    };
    let n_layers = required_positive_usize(text, "num_hidden_layers")?;
    let mtp_layers = text
        .get("mtp_num_hidden_layers")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(0);
    let hidden = required_positive_usize(text, "hidden_size")?;
    let intermediate = required_positive_usize(text, "intermediate_size")?;
    let initializer_range = text
        .get("initializer_range")
        .and_then(Value::as_f64)
        .unwrap_or(0.02) as f32;
    if !initializer_range.is_finite() || initializer_range <= 0.0 {
        bail!("initializer_range must be a positive finite number");
    }
    let configured_layer_types = parse_layer_types(text, n_layers)?;
    if intermediate % experts != 0 {
        bail!(
            "dense intermediate_size {intermediate} is not divisible by {experts} experts; choose a divisor so no neurons are dropped"
        );
    }

    let (target_model_type, target_architecture, layer_prefix) = match flavor {
        ModelFlavor::VisionLanguage => (
            "qwen3_5_moe",
            "Qwen3_5MoeForConditionalGeneration",
            VLM_LAYER_PREFIX,
        ),
        ModelFlavor::TextOnly => (
            "qwen3_5_moe_text",
            "Qwen3_5MoeForCausalLM",
            TEXT_LAYER_PREFIX,
        ),
    };
    root.insert("architectures".into(), json!([target_architecture]));
    let text = match flavor {
        ModelFlavor::VisionLanguage => {
            root.insert("model_type".into(), json!(target_model_type));
            let vision = root
                .get_mut("vision_config")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| {
                    anyhow!("multimodal config.json is missing object `vision_config`")
                })?;
            let vision_type = vision
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !matches!(vision_type, "qwen3_5" | "qwen3_5_vision") {
                bail!("unsupported dense vision_config.model_type `{vision_type}`");
            }
            // Match the official Qwen3.5-MoE checkpoint convention. Transformers' outer MoE
            // config then materializes this dictionary as Qwen3_5MoeVisionConfig.
            vision.insert("model_type".into(), json!("qwen3_5_moe"));
            root.get_mut("text_config")
                .unwrap()
                .as_object_mut()
                .unwrap()
        }
        ModelFlavor::TextOnly => root,
    };
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
    Ok(ConversionProfile {
        config,
        flavor,
        source_model_type,
        target_model_type,
        target_architecture,
        layer_prefix,
        n_layers,
        mtp_layers,
        hidden,
        intermediate,
        initializer_range,
        configured_layer_types,
    })
}

fn required_positive_usize(obj: &Map<String, Value>, key: &str) -> Result<usize> {
    let value = obj
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| anyhow!("text config `{key}` must be a positive integer"))?;
    if value == 0 {
        bail!("text config `{key}` must be positive");
    }
    Ok(value)
}

fn parse_layer_types(text: &Map<String, Value>, n_layers: usize) -> Result<Option<Vec<String>>> {
    let Some(value) = text.get("layer_types") else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| anyhow!("layer_types must be an array"))?;
    if values.len() != n_layers {
        bail!(
            "layer_types has {} entries but num_hidden_layers is {n_layers}",
            values.len()
        );
    }
    let mut out = Vec::with_capacity(n_layers);
    for (layer, value) in values.iter().enumerate() {
        let kind = value
            .as_str()
            .ok_or_else(|| anyhow!("layer_types[{layer}] must be a string"))?;
        if !matches!(kind, "full_attention" | "linear_attention") {
            bail!(
                "unsupported layer_types[{layer}] `{kind}`; the Qwen3.5-MoE runtime supports `full_attention` and `linear_attention`"
            );
        }
        out.push(kind.to_string());
    }
    Ok(Some(out))
}

fn set_explicit_layer_types(profile: &mut ConversionProfile, kinds: &[String]) {
    let root = profile.config.as_object_mut().unwrap();
    let text = match profile.flavor {
        ModelFlavor::VisionLanguage => root
            .get_mut("text_config")
            .unwrap()
            .as_object_mut()
            .unwrap(),
        ModelFlavor::TextOnly => root,
    };
    // Persist the audited list even when the source relied on an interval-derived default. This
    // prevents the target config class from deriving a different hybrid pattern in the future.
    text.insert("layer_types".into(), json!(kinds));
}

#[derive(Debug, Clone)]
struct AttentionAudit {
    kinds: Vec<String>,
    full: usize,
    linear: usize,
}

impl AttentionAudit {
    fn summary(&self) -> String {
        match (self.full, self.linear) {
            (full, 0) => format!("full attention ({full} layers)"),
            (0, linear) => format!("linear attention ({linear} layers)"),
            (full, linear) => {
                format!("hybrid: {full} full + {linear} linear attention layers")
            }
        }
    }
}

/// Verify the config's per-layer attention architecture against the checkpoint itself. Attention
/// is not rewritten by moefication, but silently changing the outer model class is safe only when
/// the target Qwen3.5-MoE class will instantiate the same block type at every layer.
fn audit_attention_layout(
    source: &ShardedSafeTensors,
    profile: &ConversionProfile,
) -> Result<AttentionAudit> {
    let names: Vec<&str> = source.names().collect();
    let mut kinds = Vec::with_capacity(profile.n_layers);
    let mut full = 0usize;
    let mut linear = 0usize;
    for layer in 0..profile.n_layers {
        let base = format!("{}{layer}.", profile.layer_prefix);
        let full_prefix = format!("{base}self_attn.");
        let linear_prefix = format!("{base}linear_attn.");
        let has_full = names.iter().any(|name| name.starts_with(&full_prefix));
        let has_linear = names.iter().any(|name| name.starts_with(&linear_prefix));
        let actual = match (has_full, has_linear) {
            (true, false) => "full_attention",
            (false, true) => "linear_attention",
            (true, true) => bail!(
                "layer {layer} contains both `self_attn` and `linear_attn` tensors; refusing an ambiguous architecture conversion"
            ),
            (false, false) => bail!(
                "layer {layer} has neither `self_attn` nor `linear_attn` tensors under `{base}`; the checkpoint does not match the supported Qwen3.5 architecture"
            ),
        };
        if let Some(configured) = &profile.configured_layer_types {
            if configured[layer] != actual {
                bail!(
                    "layer {layer}: config declares `{}`, but checkpoint tensors contain `{actual}`",
                    configured[layer]
                );
            }
        }
        match actual {
            "full_attention" => full += 1,
            "linear_attention" => linear += 1,
            _ => unreachable!(),
        }
        kinds.push(actual.to_string());
    }
    Ok(AttentionAudit {
        kinds,
        full,
        linear,
    })
}

fn discover_layers(
    source: &ShardedSafeTensors,
    profile: &ConversionProfile,
    experts: usize,
) -> Result<Vec<DenseLayer>> {
    let mut prefixes: Vec<String> = (0..profile.n_layers)
        .map(|layer| format!("{}{layer}.mlp", profile.layer_prefix))
        .collect();
    prefixes.extend((0..profile.mtp_layers).map(|layer| format!("mtp.layers.{layer}.mlp")));
    let mut layers = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let gate = format!("{prefix}.gate_proj.weight");
        let up = format!("{prefix}.up_proj.weight");
        let down = format!("{prefix}.down_proj.weight");
        for bias in [
            format!("{prefix}.gate_proj.bias"),
            format!("{prefix}.up_proj.bias"),
            format!("{prefix}.down_proj.bias"),
        ] {
            if source.contains(&bias) {
                bail!(
                    "MLP `{prefix}` contains bias tensor `{bias}`, but the Qwen3.5-MoE expert runtime is bias-free"
                );
            }
        }
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
        check_shape(&gate, &ge.shape, &[profile.intermediate, profile.hidden])?;
        check_shape(&up, &ue.shape, &[profile.intermediate, profile.hidden])?;
        check_shape(&down, &de.shape, &[profile.hidden, profile.intermediate])?;
        layers.push(DenseLayer {
            prefix,
            gate,
            up,
            down,
            dtype: ge.dtype,
            hidden: profile.hidden,
            intermediate: profile.intermediate,
            expert_intermediate: profile.intermediate / experts,
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
    initializer_range: f32,
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
                amplitude: initializer_range,
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

/// Prove that architecture-specific state outside the converted MLP triplets is opaque
/// passthrough. This covers attention variants without trying to maintain an ever-growing list of
/// Q/K/V/DeltaNet/vision tensor names.
fn audit_passthrough_plan(
    source: &ShardedSafeTensors,
    layers: &[DenseLayer],
    output: &[OutputTensor],
) -> Result<()> {
    let replaced: BTreeSet<&str> = layers
        .iter()
        .flat_map(|l| [l.gate.as_str(), l.up.as_str(), l.down.as_str()])
        .collect();
    let by_name: BTreeMap<&str, &OutputTensor> = output
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    for name in source.names() {
        if replaced.contains(name) {
            if by_name.contains_key(name) {
                bail!("dense MLP source tensor `{name}` was not removed from the output plan");
            }
            continue;
        }
        let input = source.entry(name).unwrap();
        let planned = by_name
            .get(name)
            .ok_or_else(|| anyhow!("non-MLP tensor `{name}` is missing from the output plan"))?;
        match &planned.payload {
            Payload::Copy { source } if source == name => {}
            _ => bail!("non-MLP tensor `{name}` is not planned as byte-for-byte passthrough"),
        }
        if planned.dtype != input.dtype
            || planned.shape != input.shape
            || planned.nbytes != (input.end - input.start) as u64
        {
            bail!("non-MLP tensor `{name}` changed dtype, shape, or byte length in the plan");
        }
    }
    Ok(())
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

    fn validate_config_with_transformers(output: &Path, expected_outer: &str, expected_text: &str) {
        let available = std::process::Command::new("python3")
            .args(["-c", "import transformers"])
            .status()
            .is_ok_and(|status| status.success());
        if !available {
            return;
        }
        let script = r#"
import sys
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
assert cfg.model_type == sys.argv[2], (cfg.model_type, sys.argv[2])
text = getattr(cfg, 'text_config', cfg)
assert text.model_type == sys.argv[3], (text.model_type, sys.argv[3])
"#;
        let status = std::process::Command::new("python3")
            .args([
                "-c",
                script,
                output.to_str().unwrap(),
                expected_outer,
                expected_text,
            ])
            .status()
            .unwrap();
        assert!(status.success(), "Transformers rejected converted config");
    }

    fn add_qwen_attention_tensors(
        owned: &mut Vec<(String, &str, Vec<u64>, Vec<u8>)>,
        layer_prefix: &str,
    ) {
        let mut base = 7000u16;
        let mut add = |name: String| {
            owned.push((name, "BF16", vec![2, 2], bf16_seq(4, base)));
            base += 10;
        };
        for suffix in [
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "conv1d.weight",
            "dt_bias",
            "A_log",
            "norm.weight",
            "out_proj.weight",
        ] {
            add(format!("{layer_prefix}0.linear_attn.{suffix}"));
        }
        for suffix in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "q_norm.weight",
            "k_norm.weight",
        ] {
            add(format!("{layer_prefix}1.self_attn.{suffix}"));
        }
        add("model.visual.blocks.0.attn.qkv.weight".into());
        add("model.visual.blocks.0.attn.proj.weight".into());
        add("mtp.layers.0.self_attn.q_proj.weight".into());
        add("mtp.layers.0.self_attn.k_proj.weight".into());
        add("mtp.layers.0.self_attn.v_proj.weight".into());
        add("mtp.layers.0.self_attn.o_proj.weight".into());
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
                format!("{VLM_LAYER_PREFIX}{layer}.mlp.gate_proj.weight"),
                "BF16",
                vec![8, 4],
                bf16_seq(32, 1000 + layer * 100),
            ));
            owned.push((
                format!("{VLM_LAYER_PREFIX}{layer}.mlp.up_proj.weight"),
                "BF16",
                vec![8, 4],
                bf16_seq(32, 2000 + layer * 100),
            ));
            owned.push((
                format!("{VLM_LAYER_PREFIX}{layer}.mlp.down_proj.weight"),
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
        add_qwen_attention_tensors(&mut owned, VLM_LAYER_PREFIX);
        let borrowed: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = owned
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
            .collect();
        fs::write(dir.join("model.safetensors"), safetensors(&borrowed)).unwrap();
    }

    fn text_fixture(dir: &Path) {
        let config = json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 2,
            "hidden_size": 4,
            "intermediate_size": 8,
            "initializer_range": 0.01,
            "layer_types": ["linear_attention", "full_attention"]
        });
        write_json(dir.join("config.json"), &config).unwrap();
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
        add_qwen_attention_tensors(&mut owned, TEXT_LAYER_PREFIX);
        // Text-only checkpoints do not carry vision or MTP; remove those helper entries.
        owned.retain(|(name, _, _, _)| {
            !name.starts_with("model.visual.") && !name.starts_with("mtp.")
        });
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
        assert_eq!(cfg["vision_config"]["model_type"], "qwen3_5_moe");
        assert_eq!(cfg["text_config"]["num_experts"], 2);
        assert_eq!(cfg["text_config"]["moe_intermediate_size"], 4);
        assert!(cfg["text_config"].get("intermediate_size").is_none());

        let out = ShardedSafeTensors::open_dir(&output).unwrap();
        let p = format!("{VLM_LAYER_PREFIX}0.mlp");
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
        for name in [
            format!("{VLM_LAYER_PREFIX}0.linear_attn.in_proj_qkv.weight"),
            format!("{VLM_LAYER_PREFIX}0.linear_attn.conv1d.weight"),
            format!("{VLM_LAYER_PREFIX}1.self_attn.q_proj.weight"),
            format!("{VLM_LAYER_PREFIX}1.self_attn.q_norm.weight"),
            "model.visual.blocks.0.attn.qkv.weight".into(),
            "mtp.layers.0.self_attn.o_proj.weight".into(),
        ] {
            assert_eq!(
                out.bytes(&name).unwrap(),
                ShardedSafeTensors::open_dir(&input)
                    .unwrap()
                    .bytes(&name)
                    .unwrap(),
                "{name} must pass through byte-for-byte"
            );
        }

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
        validate_config_with_transformers(&output, "qwen3_5_moe", "qwen3_5_moe_text");

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
    fn converts_text_only_full_and_linear_attention_architecture() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("dense-text");
        let output = temp.path().join("moe-text");
        fs::create_dir(&input).unwrap();
        text_fixture(&input);
        run_moefy_qwen38(&MoefyOptions {
            input_dir: input.display().to_string(),
            output_dir: output.display().to_string(),
            experts: 2,
            top_k: 1,
            max_shard_size: "1M".into(),
            dry_run: false,
        })
        .unwrap();

        let cfg: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(cfg["model_type"], "qwen3_5_moe_text");
        assert_eq!(cfg["architectures"][0], "Qwen3_5MoeForCausalLM");
        let out = ShardedSafeTensors::open_dir(&output).unwrap();
        assert!(out.contains("model.layers.0.mlp.experts.gate_up_proj"));
        assert!(out.contains("model.layers.0.linear_attn.in_proj_qkv.weight"));
        assert!(out.contains("model.layers.1.self_attn.q_proj.weight"));
        validate_config_with_transformers(&output, "qwen3_5_moe_text", "qwen3_5_moe_text");
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
    fn rejects_unsupported_model_families_instead_of_corrupting_attention() {
        let llama = json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 4,
            "intermediate_size": 8
        });
        let error = convert_config(llama, 2, 1).unwrap_err().to_string();
        assert!(
            error.contains("unsupported dense model_type `llama`"),
            "{error}"
        );
        assert!(error.contains("target MoE class"), "{error}");
    }

    #[test]
    fn rejects_quantized_source_config_before_rewriting() {
        let config = json!({
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 2,
            "hidden_size": 4,
            "intermediate_size": 8,
            "quantization_config": {"quant_method": "fp8"}
        });
        let error = convert_config(config, 2, 1).unwrap_err().to_string();
        assert!(error.contains("quantized source checkpoints"), "{error}");
    }

    #[test]
    fn rejects_attention_config_tensor_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("dense");
        fs::create_dir(&input).unwrap();
        fixture(&input);
        let mut config: Value =
            serde_json::from_slice(&fs::read(input.join("config.json")).unwrap()).unwrap();
        config["text_config"]["layer_types"] = json!(["full_attention", "full_attention"]);
        write_json(input.join("config.json"), &config).unwrap();

        let error = run_moefy_qwen38(&MoefyOptions {
            input_dir: input.display().to_string(),
            output_dir: temp.path().join("unused").display().to_string(),
            experts: 2,
            top_k: 1,
            max_shard_size: "1M".into(),
            dry_run: true,
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("config declares `full_attention`"),
            "{error}"
        );
        assert!(error.contains("linear_attention"), "{error}");
    }

    #[test]
    fn infers_and_persists_attention_layout_when_config_omits_it() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("dense-text");
        let output = temp.path().join("moe-text");
        fs::create_dir(&input).unwrap();
        text_fixture(&input);
        let mut config: Value =
            serde_json::from_slice(&fs::read(input.join("config.json")).unwrap()).unwrap();
        config.as_object_mut().unwrap().remove("layer_types");
        write_json(input.join("config.json"), &config).unwrap();

        run_moefy_qwen38(&MoefyOptions {
            input_dir: input.display().to_string(),
            output_dir: output.display().to_string(),
            experts: 2,
            top_k: 1,
            max_shard_size: "1M".into(),
            dry_run: false,
        })
        .unwrap();
        let converted: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(
            converted["layer_types"],
            json!(["linear_attention", "full_attention"])
        );
    }

    #[test]
    fn parses_binary_shard_sizes() {
        assert_eq!(parse_size("5G").unwrap(), 5 * 1024u64.pow(3));
        assert_eq!(parse_size("750MiB").unwrap(), 750 * 1024u64.pow(2));
        assert!(parse_size("1T").is_err());
    }

    #[test]
    fn transformers_dense_and_moe_non_mlp_state_dicts_match() {
        let available = std::process::Command::new("python3")
            .args(["-c", "import torch, transformers"])
            .status()
            .is_ok_and(|status| status.success());
        if !available {
            return;
        }
        // Instantiate tiny synthetic dense/MoE pairs from the installed Transformers runtime.
        // This checks full attention (including GQA), linear attention, vision, embeddings, norms,
        // and heads without downloading or loading any checkpoint weights.
        let script = r#"
from transformers import (
    Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig, Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

text_common = dict(
    hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=4, linear_num_key_heads=2,
    linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
    vocab_size=16, layer_types=['linear_attention', 'full_attention'],
)
dense_text = Qwen3_5TextConfig(intermediate_size=8, **text_common)
moe_text = Qwen3_5MoeTextConfig(
    moe_intermediate_size=4, shared_expert_intermediate_size=1,
    num_experts=2, num_experts_per_tok=1, **text_common,
)

def non_mlp(model):
    return {k: tuple(v.shape) for k, v in model.state_dict().items() if '.mlp.' not in k}

assert non_mlp(Qwen3_5ForCausalLM(dense_text)) == non_mlp(Qwen3_5MoeForCausalLM(moe_text))

vision_common = dict(
    hidden_size=4, intermediate_size=8, depth=1, num_heads=1,
    out_hidden_size=8, patch_size=2, spatial_merge_size=2,
    temporal_patch_size=1, num_position_embeddings=16,
)
dense_vlm = Qwen3_5Config(
    text_config=dense_text, vision_config=Qwen3_5VisionConfig(**vision_common),
    image_token_id=8, video_token_id=9, vision_start_token_id=10, vision_end_token_id=11,
)
moe_vlm = Qwen3_5MoeConfig(
    text_config=moe_text, vision_config=Qwen3_5MoeVisionConfig(**vision_common),
    image_token_id=8, video_token_id=9, vision_start_token_id=10, vision_end_token_id=11,
)
assert non_mlp(Qwen3_5ForConditionalGeneration(dense_vlm)) == non_mlp(
    Qwen3_5MoeForConditionalGeneration(moe_vlm)
)
"#;
        let status = std::process::Command::new("python3")
            .args(["-c", script])
            .status()
            .unwrap();
        assert!(
            status.success(),
            "Transformers dense/MoE non-MLP state dicts differ"
        );
    }
}
