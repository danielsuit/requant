//! Role-tagged tensor IR shared across crates.

use serde::{Deserialize, Serialize};

/// Functional role of a tensor within the model. Drives per-role quantization policy.
///
/// MoE-aware: `Router` (gate logits) is always protected hard; `SharedExpert` is treated like
/// attention (always-on), `RoutedExpert` tolerates aggressive quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Token embedding / input embedding weight.
    Embedding,
    /// LM head / output projection.
    LmHead,
    /// Layer norm / rms norm weights (and biases).
    Norm,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    /// Dense FFN gate (SiLU/SwiGLU up-projection).
    FfnGate,
    /// Dense FFN up.
    FfnUp,
    /// Dense FFN down.
    FfnDown,
    /// MoE router / gate logits. Always protected.
    Router,
    /// MoE shared expert (always-on) — FFN part.
    SharedExpert(FfnPart),
    /// MoE routed expert — FFN part + expert index.
    RoutedExpert { idx: u32, part: FfnPart },
    /// Fused QKV projection (Qwen2, Phi-3, GLM-4, InternLM2, …). A single tensor serving all of
    /// Q/K/V, so it gets its own role rather than being split across AttnQ/K/V. Always-on, on the
    /// dense path with attention policy.
    AttnQkv,
    /// Multi-head Latent Attention projections (DeepSeek-V2/V3). The query/KV are each factored
    /// into a down-projection (latent-compressing `_a`) and an up-projection (`_b`); `kv_a_mqa` is
    /// the multi-query-attention variant of the KV down-projection. All always-on dense-path
    /// attention — they must not be left at the default expert quantization.
    AttnMla(MlaPart),
    /// State-space model (Mamba/Jamba) weight matrices — `ssm_in`, `ssm_out`, `ssm_x`,
    /// `ssm_conv1d`, `ssm_f`, `ssm_g`, `ssm_b`, `ssm_c`. Always-on, dense-path.
    Ssm,
    /// Small/sensitive SSM parameters that must stay full-precision — `ssm_a` (log-space decay),
    /// `ssm_dt` (selective delta), `ssm_d`, `ssm_alpha`, `ssm_beta`, `ssm_ba`. These are 1-D
    /// log-space / selective parameters; block-quantizing them corrupts the SSM dynamics. Kept
    /// full-precision like norms (non-quantizable).
    SsmParam,
    /// Multi-token-prediction / "nextn" head (DeepSeek-V3, GLM-4.x, and the V4-class models).
    /// Always-on like attention and it writes directly into a logit distribution, so it belongs on
    /// the protected dense path, not with the routed experts.
    Mtp,
    /// Anything we could not classify. By default this is a hard error (fail-loud).
    Unknown,
}

/// Which MLA projection a tensor is (see [`Role::AttnMla`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MlaPart {
    /// `attn_q_a` — query down-projection (latent-compressing).
    QA,
    /// `attn_q_b` — query up-projection.
    QB,
    /// `attn_kv_a` — KV down-projection.
    KvA,
    /// `attn_kv_a_mqa` — KV down-projection (multi-query-attention variant).
    KvAMqa,
    /// `attn_kv_b` — KV up-projection.
    KvB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FfnPart {
    Gate,
    Up,
    Down,
}

impl Role {
    /// Human-readable stable label for tables/recipes.
    pub fn label(&self) -> &'static str {
        match self {
            Role::Embedding => "embedding",
            Role::LmHead => "lm_head",
            Role::Norm => "norm",
            Role::AttnQ => "attn_q",
            Role::AttnK => "attn_k",
            Role::AttnV => "attn_v",
            Role::AttnO => "attn_o",
            Role::FfnGate => "ffn_gate",
            Role::FfnUp => "ffn_up",
            Role::FfnDown => "ffn_down",
            Role::Router => "router",
            Role::AttnQkv => "attn_qkv",
            Role::AttnMla(p) => match p {
                MlaPart::QA => "attn_mla.q_a",
                MlaPart::QB => "attn_mla.q_b",
                MlaPart::KvA => "attn_mla.kv_a",
                MlaPart::KvAMqa => "attn_mla.kv_a_mqa",
                MlaPart::KvB => "attn_mla.kv_b",
            },
            Role::Ssm => "ssm",
            Role::SsmParam => "ssm_param",
            Role::SharedExpert(p) => match p {
                FfnPart::Gate => "shared_expert.gate",
                FfnPart::Up => "shared_expert.up",
                FfnPart::Down => "shared_expert.down",
            },
            Role::RoutedExpert { part, .. } => match part {
                FfnPart::Gate => "routed_expert.gate",
                FfnPart::Up => "routed_expert.up",
                FfnPart::Down => "routed_expert.down",
            },
            Role::Mtp => "mtp",
            Role::Unknown => "unknown",
        }
    }

    pub fn is_router(&self) -> bool {
        matches!(self, Role::Router)
    }

    /// True for tensors on the **dense path** — the ones every token flows through regardless of
    /// routing. Routed experts are the exception: each is touched by only `expert_used / expert_count`
    /// of the traffic, which is precisely why they tolerate aggressive quantization and the dense
    /// path does not (DESIGN §0). Recipes use this to apply a single "never below X" floor to
    /// everything that is always on.
    pub fn is_dense_path(&self) -> bool {
        match self {
            Role::Embedding
            | Role::LmHead
            | Role::Norm
            | Role::AttnQ
            | Role::AttnK
            | Role::AttnV
            | Role::AttnO
            | Role::FfnGate
            | Role::FfnUp
            | Role::FfnDown
            | Role::Router
            | Role::AttnQkv
            | Role::AttnMla(_)
            | Role::Ssm
            | Role::SharedExpert(_)
            | Role::Mtp => true,
            Role::RoutedExpert { .. } | Role::SsmParam | Role::Unknown => false,
        }
    }
}

/// True if `name` is a norm tensor (RMSNorm/LayerNorm weight or bias). These are 1-D and must
/// never be block-quantized.
///
/// Matches any name whose base (with `.weight`/`.bias` stripped) ends in `_norm`, plus the
/// numeric-suffixed norm variants some architectures emit — e.g. Gemma2/Gemma3's `attn_norm_2`,
/// which does *not* end in `_norm` and would otherwise be wrongly quantized.
fn is_norm_tensor(name: &str) -> bool {
    let base = name.strip_suffix(".weight").or_else(|| name.strip_suffix(".bias")).unwrap_or(name);
    if base.ends_with("_norm") {
        return true;
    }
    // `*_norm_<digits>` (attn_norm_2, …). `find` gives the first `_norm_`, which is the right one
    // for names like `attn_norm_2`; everything after it must be digits only.
    if let Some(i) = base.find("_norm_") {
        let after = &base[i + "_norm_".len()..];
        return !after.is_empty() && after.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

/// A tensor's resolved place in the model (layer depth in [0,1), expert id if any).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Place {
    /// Fractional layer depth, 0.0 = first block, 1.0 = last.
    pub depth: f32,
    /// Expert index for routed experts; None otherwise.
    pub expert: Option<u32>,
}

/// Metadata for one tensor in the IR.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub role: Role,
    pub place: Place,
}

/// A typed view over a tensor's element data.
pub struct TensorView<'a> {
    pub name: &'a str,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
    pub dtype: u32,
}

/// Model-level layout parsed from GGUF metadata. Drives role tagging and MoE detection.
///
/// `expert_used` is the top-k of routed experts activated per token (0 for dense models).
/// `shared_count` is the number of always-on shared experts (DeepSeek/GLM-style; 0 otherwise).
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub arch: String,
    pub n_layers: u32,
    pub expert_count: u32,
    pub expert_used: u32,
    pub shared_count: u32,
    pub is_moe: bool,
}

impl ModelLayout {
    /// Parse the layout from a GGUF KV table. Reads `general.architecture` then the
    /// arch-prefixed counts (`{arch}.block_count`, `{arch}.expert_count`, …).
    pub fn from_kv(kv: &[(String, crate::gguf::GgufValue)]) -> anyhow::Result<Self> {
        let arch = kv
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .and_then(|(_, v)| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("GGUF missing general.architecture"))?
            .to_string();

        let get = |suffix: &str| -> Option<u32> {
            let key = format!("{arch}.{suffix}");
            kv.iter()
                .find(|(k, _)| k == &key)
                .and_then(|(_, v)| v.as_u32())
                .or_else(|| {
                    // Some converters emit these without the arch prefix.
                    kv.iter()
                        .find(|(k, _)| k == suffix)
                        .and_then(|(_, v)| v.as_u32())
                })
        };

        let n_layers = get("block_count").ok_or_else(|| anyhow::anyhow!("missing block_count"))?;
        let expert_count = get("expert_count").unwrap_or(0);
        let expert_used = get("expert_used_count").unwrap_or(0);
        let shared_count = get("expert_shared_count").unwrap_or(0);
        let is_moe = expert_count > 0;

        Ok(Self { arch, n_layers, expert_count, expert_used, shared_count, is_moe })
    }
}

/// The result of tagging one tensor: its role, place, and whether it may be block-quantized.
///
/// `quantizable = false` for norms, biases, and other 1-D float tensors that should be kept
/// at full precision (F16/F32). The recipe must assign such tensors a float type; assigning a
/// block-quant type is a hard error.
#[derive(Debug, Clone)]
pub struct TensorTag {
    pub role: Role,
    pub place: Place,
    /// True for weight matrices that may be block-quantized; false for norms/biases.
    pub quantizable: bool,
}

impl TensorTag {
    /// Parse a `blk.{i}.<suffix>` tensor name into (layer_index, suffix). Returns None for
    /// non-block tensors (embeddings, output, etc.).
    fn split_blk(name: &str) -> Option<(u32, &str)> {
        let rest = name.strip_prefix("blk.")?;
        let dot = rest.find('.')?;
        let (idx_str, suffix) = rest.split_at(dot);
        let suffix = &suffix[1..]; // drop the dot
        let idx: u32 = idx_str.parse().ok()?;
        Some((idx, suffix))
    }

    /// Tag one tensor given the model layout. Fail-loud: unclassified weight tensors return
    /// `Role::Unknown` so the caller can decide whether to error or fall back.
    pub fn tag(name: &str, layout: &ModelLayout) -> Self {
        let n = layout.n_layers.max(1);
        // Norms and biases are never block-quantized.
        let is_norm = is_norm_tensor(name);
        let is_bias = name.ends_with(".bias");
        let quantizable = !(is_norm || is_bias);

        // Non-block tensors.
        if name == "token_embd.weight" {
            return Self { role: Role::Embedding, place: Place { depth: 0.0, expert: None }, quantizable: true };
        }
        if name == "output_norm.weight" || name == "output_norm.bias" {
            return Self { role: Role::Norm, place: Place { depth: 1.0, expert: None }, quantizable };
        }
        if name == "output.weight" || name == "output.bias" {
            return Self { role: Role::LmHead, place: Place { depth: 1.0, expert: None }, quantizable: !is_bias };
        }
        if name == "token_embd.bias" {
            return Self { role: Role::Embedding, place: Place { depth: 0.0, expert: None }, quantizable: false };
        }

        // Fixed (non-learnable) lookup tables: RoPE frequency schedules. These are deterministic
        // sin/cos tables, not weights — block-quantizing them corrupts positional encoding and
        // crashes/underperforms downstream (llama-quantize leaves them F32). Copy verbatim.
        if name.starts_with("rope_freqs") || name == "rope_freqs_v.weight" {
            return Self { role: Role::Unknown, place: Place { depth: 0.0, expert: None }, quantizable: false };
        }

        // Block tensors: blk.{i}.<suffix>.
        let Some((idx, suffix)) = Self::split_blk(name) else {
            return Self { role: Role::Unknown, place: Place { depth: 0.0, expert: None }, quantizable };
        };
        let depth = idx as f32 / n as f32;
        let place_of = |expert: Option<u32>| Place { depth, expert };
        let norm_tag = || Self { role: Role::Norm, place: place_of(None), quantizable };

        // Multi-token-prediction head. llama.cpp emits these as `blk.{i}.nextn.<part>`
        // (`eh_proj`, `embed_tokens`, `enorm`, `hnorm`, `shared_head_head`, `shared_head_norm`).
        // Note `enorm`/`hnorm` do *not* end in `_norm`, so the generic norm test above misses
        // them — check for "norm" anywhere in the part instead, or we would happily block-quantize
        // a 1-D norm vector.
        if let Some(part) = suffix.strip_prefix("nextn.") {
            let is_1d = part.contains("norm") || is_bias;
            return Self { role: Role::Mtp, place: place_of(None), quantizable: !is_1d };
        }

        // Attention.
        match suffix {
            "attn_norm.weight" | "attn_norm.bias" => return norm_tag(),
            "attn_q.weight" => return Self { role: Role::AttnQ, place: place_of(None), quantizable: true },
            "attn_k.weight" => return Self { role: Role::AttnK, place: place_of(None), quantizable: true },
            "attn_v.weight" => return Self { role: Role::AttnV, place: place_of(None), quantizable: true },
            // `attn_o` (Llama) and `attn_output` (Qwen) are the same op.
            "attn_o.weight" | "attn_output.weight" => {
                return Self { role: Role::AttnO, place: place_of(None), quantizable: true };
            }
            "attn_q.bias" | "attn_k.bias" | "attn_v.bias" | "attn_o.bias" | "attn_output.bias" => {
                let role = match suffix {
                    "attn_q.bias" => Role::AttnQ,
                    "attn_k.bias" => Role::AttnK,
                    "attn_v.bias" => Role::AttnV,
                    _ => Role::AttnO,
                };
                return Self { role, place: place_of(None), quantizable: false };
            }
            "attn_output_norm.weight" | "attn_output_norm.bias" => return norm_tag(),
            // Gemma2/Gemma3 emit a second attention norm as `attn_norm_2` (caught as a norm by
            // `is_norm_tensor` above, so `quantizable` is already false here).
            "attn_norm_2.weight" | "attn_norm_2.bias" => return norm_tag(),
            // Gemma2/Gemma3/Qwen3 per-head Q/K norms (1-D) — also norms.
            "attn_q_norm.weight" | "attn_q_norm.bias"
            | "attn_k_norm.weight" | "attn_k_norm.bias" => return norm_tag(),
            // Fused QKV (Qwen2, Phi-3, GLM-4, InternLM2, …): one tensor for all of Q/K/V.
            "attn_qkv.weight" => {
                return Self { role: Role::AttnQkv, place: place_of(None), quantizable: true };
            }
            "attn_qkv.bias" => {
                return Self { role: Role::AttnQkv, place: place_of(None), quantizable: false };
            }
            // MLA (DeepSeek-V2/V3). Query/KV each factored into a down (`_a`) and up (`_b`) proj.
            "attn_q_a.weight" => return Self { role: Role::AttnMla(MlaPart::QA), place: place_of(None), quantizable: true },
            "attn_q_b.weight" => return Self { role: Role::AttnMla(MlaPart::QB), place: place_of(None), quantizable: true },
            "attn_kv_a.weight" => return Self { role: Role::AttnMla(MlaPart::KvA), place: place_of(None), quantizable: true },
            "attn_kv_a_mqa.weight" => return Self { role: Role::AttnMla(MlaPart::KvAMqa), place: place_of(None), quantizable: true },
            "attn_kv_b.weight" => return Self { role: Role::AttnMla(MlaPart::KvB), place: place_of(None), quantizable: true },
            // MLA norms (`attn_q_a_norm`, `attn_kv_a_norm`) are already caught as norms by the
            // `_norm` suffix; MLA biases, if any converter emits them, stay full-precision.
            "attn_q_a.bias" | "attn_q_b.bias" | "attn_kv_a.bias" | "attn_kv_a_mqa.bias" | "attn_kv_b.bias" => {
                let part = match suffix {
                    "attn_q_a.bias" => MlaPart::QA,
                    "attn_q_b.bias" => MlaPart::QB,
                    "attn_kv_a.bias" => MlaPart::KvA,
                    "attn_kv_a_mqa.bias" => MlaPart::KvAMqa,
                    _ => MlaPart::KvB,
                };
                return Self { role: Role::AttnMla(part), place: place_of(None), quantizable: false };
            }
            _ => {}
        }

        // FFN: dense, router, shared expert, routed experts.
        match suffix {
            "ffn_norm.weight" | "ffn_norm.bias" => return norm_tag(),
            "ffn_gate_inp.weight" | "ffn_gate_inp.bias" => {
                return Self { role: Role::Router, place: place_of(None), quantizable: !is_bias };
            }
            // Dense FFN (non-MoE).
            "ffn_gate.weight" => return Self { role: Role::FfnGate, place: place_of(None), quantizable: true },
            "ffn_up.weight" => return Self { role: Role::FfnUp, place: place_of(None), quantizable: true },
            "ffn_down.weight" => return Self { role: Role::FfnDown, place: place_of(None), quantizable: true },
            "ffn_gate.bias" | "ffn_up.bias" | "ffn_down.bias" => {
                let part = match suffix {
                    "ffn_gate.bias" => FfnPart::Gate,
                    "ffn_up.bias" => FfnPart::Up,
                    _ => FfnPart::Down,
                };
                return Self { role: dense_ffn_role(part), place: place_of(None), quantizable: false };
            }
            // Shared expert (DeepSeek/GLM/Qwen2MoE).
            "ffn_gate_shexp.weight" => return Self { role: Role::SharedExpert(FfnPart::Gate), place: place_of(None), quantizable: true },
            "ffn_up_shexp.weight" => return Self { role: Role::SharedExpert(FfnPart::Up), place: place_of(None), quantizable: true },
            "ffn_down_shexp.weight" => return Self { role: Role::SharedExpert(FfnPart::Down), place: place_of(None), quantizable: true },
            // Packed routed experts (llama.cpp `_exps` tensors, shape [n_expert*out, in]).
            "ffn_gate_exps.weight" => return Self { role: Role::RoutedExpert { idx: u32::MAX, part: FfnPart::Gate }, place: place_of(None), quantizable: true },
            "ffn_up_exps.weight" => return Self { role: Role::RoutedExpert { idx: u32::MAX, part: FfnPart::Up }, place: place_of(None), quantizable: true },
            "ffn_down_exps.weight" => return Self { role: Role::RoutedExpert { idx: u32::MAX, part: FfnPart::Down }, place: place_of(None), quantizable: true },
            _ => {}
        }

        // State-space models (Mamba/Jamba). Split into quantizable weight matrices and small
        // log-space / selective params that must stay full-precision. `ssm_norm` is already caught
        // as a norm by `is_norm_tensor`. Names verified against llama.cpp `llama-arch.cpp`.
        if suffix.starts_with("ssm_") {
            let base = suffix.strip_suffix(".weight").or_else(|| suffix.strip_suffix(".bias"));
            if let Some(b) = base {
                let quant = !is_bias;
                // Small / sensitive 1-D params — keep full-precision like norms.
                const SSMPARAM: &[&str] = &[
                    "ssm_a", "ssm_dt", "ssm_d", "ssm_alpha", "ssm_beta", "ssm_ba",
                ];
                if SSMPARAM.contains(&b) {
                    return Self { role: Role::SsmParam, place: place_of(None), quantizable: false };
                }
                // Weight matrices of the SSM block.
                const SSM: &[&str] = &[
                    "ssm_in", "ssm_out", "ssm_x", "ssm_conv1d", "ssm_f", "ssm_g",
                    "ssm_b", "ssm_c",
                ];
                if SSM.contains(&b) {
                    return Self { role: Role::Ssm, place: place_of(None), quantizable: quant };
                }
                // Any other `ssm_*` we don't recognize (e.g. `ssm_norm` already handled, but a
                // novel variant): fall through to Unknown rather than guessing.
            }
        }

        // Split routed experts: `ffn_gate.{e}.weight` / `ffn_up.{e}.weight` / `ffn_down.{e}.weight`
        // (Mixtral-style when each expert is a separate tensor).
        for (part_name, part) in [("gate", FfnPart::Gate), ("up", FfnPart::Up), ("down", FfnPart::Down)] {
            let prefix = format!("ffn_{part_name}.");
            if let Some(rest) = suffix.strip_prefix(&prefix) {
                if let Some(expert_str) = rest.strip_suffix(".weight") {
                    if let Ok(e) = expert_str.parse::<u32>() {
                        return Self {
                            role: Role::RoutedExpert { idx: e, part },
                            place: place_of(Some(e)),
                            quantizable: true,
                        };
                    }
                }
            }
        }

        // Anything we didn't explicitly classify: a tensor we detected as a norm (e.g. MLA's
        // `attn_q_a_norm`, `ssm_norm`, or any future `_norm` variant without a dedicated arm) is
        // a Norm — copied verbatim, never block-quantized. Everything else is genuinely unknown.
        if is_norm {
            Self { role: Role::Norm, place: place_of(None), quantizable }
        } else {
            Self { role: Role::Unknown, place: place_of(None), quantizable }
        }
    }
}

pub fn dense_ffn_role(part: FfnPart) -> Role {
    match part {
        FfnPart::Gate => Role::FfnGate,
        FfnPart::Up => Role::FfnUp,
        FfnPart::Down => Role::FfnDown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> ModelLayout {
        ModelLayout { arch: "llama".into(), n_layers: 4, expert_count: 0, expert_used: 0, shared_count: 0, is_moe: false }
    }

    fn tag(name: &str) -> TensorTag {
        TensorTag::tag(name, &layout())
    }

    #[test]
    fn fused_qkv_tags_onto_dense_attention_path() {
        let w = tag("blk.0.attn_qkv.weight");
        assert_eq!(w.role, Role::AttnQkv);
        assert!(w.quantizable, "qkv weight must be quantizable");
        assert!(w.role.is_dense_path(), "fused qkv must be on the dense path (floor protection)");

        let b = tag("blk.0.attn_qkv.bias");
        assert_eq!(b.role, Role::AttnQkv);
        assert!(!b.quantizable, "qkv bias must not be quantized");
    }

    #[test]
    fn mla_projections_tag_onto_dense_attention_path() {
        for (name, part) in [
            ("blk.0.attn_q_a.weight", MlaPart::QA),
            ("blk.0.attn_q_b.weight", MlaPart::QB),
            ("blk.0.attn_kv_a.weight", MlaPart::KvA),
            ("blk.0.attn_kv_a_mqa.weight", MlaPart::KvAMqa),
            ("blk.0.attn_kv_b.weight", MlaPart::KvB),
        ] {
            let t = tag(name);
            assert_eq!(t.role, Role::AttnMla(part), "{name}");
            assert!(t.quantizable, "{name} must be quantizable");
            assert!(t.role.is_dense_path(), "{name} must be on the dense path (floor protection)");
        }
        // MLA norms are 1-D and must not be quantized.
        assert!(!tag("blk.0.attn_q_a_norm.weight").quantizable);
        assert!(!tag("blk.0.attn_kv_a_norm.weight").quantizable);
        assert_eq!(tag("blk.0.attn_q_a_norm.weight").role, Role::Norm);
    }

    #[test]
    fn ssm_weight_matrices_tag_onto_dense_path_params_protected() {
        for name in ["blk.0.ssm_in.weight", "blk.0.ssm_out.weight", "blk.0.ssm_x.weight",
                     "blk.0.ssm_conv1d.weight", "blk.0.ssm_f.weight", "blk.0.ssm_b.weight"] {
            let t = tag(name);
            assert_eq!(t.role, Role::Ssm, "{name}");
            assert!(t.quantizable, "{name} must be quantizable");
            assert!(t.role.is_dense_path(), "{name} must be on the dense path");
        }
        // Small / sensitive SSM params stay full-precision.
        for name in ["blk.0.ssm_a.weight", "blk.0.ssm_dt.weight", "blk.0.ssm_d.weight",
                     "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_ba.weight"] {
            let t = tag(name);
            assert_eq!(t.role, Role::SsmParam, "{name}");
            assert!(!t.quantizable, "{name} must not be quantized (log-space / selective param)");
            assert!(!t.role.is_dense_path(), "{name} is non-quantizable; not a dense-path weight");
        }
        // ssm_norm is a norm.
        assert_eq!(tag("blk.0.ssm_norm.weight").role, Role::Norm);
        assert!(!tag("blk.0.ssm_norm.weight").quantizable);
    }

    #[test]
    fn attn_norm_2_is_a_norm_not_quantizable() {
        // Gemma2/Gemma3's second attention norm does NOT end in `_norm`; without the generalized
        // norm detection it would be wrongly block-quantized.
        let w = tag("blk.0.attn_norm_2.weight");
        assert_eq!(w.role, Role::Norm, "attn_norm_2 is a norm");
        assert!(!w.quantizable, "attn_norm_2 must not be block-quantized");
        let b = tag("blk.0.attn_norm_2.bias");
        assert_eq!(b.role, Role::Norm);
        assert!(!b.quantizable);
        // Per-head Q/K norms (Gemma2/3, Qwen3) are norms too.
        assert_eq!(tag("blk.0.attn_q_norm.weight").role, Role::Norm);
        assert_eq!(tag("blk.0.attn_k_norm.weight").role, Role::Norm);
    }

    #[test]
    fn norm_detection_does_not_swallow_unrelated_suffixes() {
        // Sanity: a regular weight ending in `_norm` still classifies as Norm, and a non-norm
        // weight is not mis-flagged by the numeric-suffix rule.
        assert!(is_norm_tensor("blk.0.attn_norm.weight"));
        assert!(is_norm_tensor("blk.0.ffn_norm.bias"));
        assert!(is_norm_tensor("blk.0.attn_norm_2.weight"));
        assert!(!is_norm_tensor("blk.0.attn_q.weight"));
        assert!(!is_norm_tensor("blk.0.ffn_down.weight"));
        // `_norm_` followed by non-digits is NOT a norm (avoid false positives).
        assert!(!is_norm_tensor("blk.0.ssm_norm_thing.weight"));
    }
}
