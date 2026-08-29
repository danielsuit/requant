//! Recipe / policy language: a TOML config that assigns a target ggml type to each tensor by
//! role and (optionally) layer depth / expert range. Last-match-wins.
//!
//! This is where MoE-awareness lives: the router is protected by default, routed experts can
//! be quantized aggressively, deep down-projections kept a notch higher, etc. The CLI and the
//! search loop both consume resolved `Policy`s, so a hand-written recipe and an auto-searched
//! recipe are the same kind of object.

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use requant_io::{ModelLayout, Role, TensorTag};

/// Calibration statistic a method needs from the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatKind {
    /// No calibration (RTN).
    None,
    /// Per-input-channel importance = imatrix = diagonal of the GPTQ Hessian (§2.4).
    Diag,
    /// Full `XXᵀ` Hessian (GPTQ, later).
    Gram,
    /// Per-channel activation statistics (AWQ/SmoothQuant, later).
    ActScale,
}

/// Quantization method selecting an optimization algorithm. RTN/K-quant scale search are live;
/// GPTQ/AWQ remain reserved recipe vocabulary for future kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantMethod {
    Rtn,
    Kquant,
    Gptq,
    Awq,
}

/// A ggml type name accepted in recipes, mapped to its numeric id. Keeps recipes portable
/// across the raw integer ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[allow(non_camel_case_types)] // names mirror ggml format names (Q4_K, Q6_K, …)
pub enum Bits {
    F32,
    F16,
    BF16,
    Q1_0,
    Q2_0,
    Q8_0,
    Q8_1,
    Q6_K,
    Q5_K,
    Q5_1,
    Q5_0,
    Q4_K,
    Q4_1,
    Q4_0,
    Q3_K,
    Q2_K,
    Q8_K,
    TQ1_0,
    TQ2_0,
    /// i-quant codebook family (ggml types 16..23, 29). imatrix-driven; the formats that make
    /// sub-3-bit expert quantization practical.
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
    /// MXFP4 in ggml's `block_mxfp4` layout.
    MXFP4,
    /// NVFP4 (E2M1 × 16 + E4M3 block scale + fp32 tensor scale). safetensors only; this is the
    /// format Blackwell's FP4 tensor cores actually consume.
    NVFP4,
    /// Current ggml's self-contained NVFP4 block (GGUF type 40), without ModelOpt's tensor-level
    /// `weight_scale_2` sidecar.
    NVFP4_GGUF,
    /// MXFP8 (E4M3 × 32 + E8M0 block scale).
    MXFP8,
    /// Dense FP8 e4m3 with a per-output-channel scale.
    FP8_E4M3,
    /// Dense FP8 e5m2 with a per-output-channel scale.
    FP8_E5M2,
    /// Raw numeric ggml type, for formats not named above.
    Raw(u32),
}

impl Bits {
    pub fn to_ggml_type(self) -> u32 {
        match self {
            Bits::MXFP4 => requant_io::GGML_TYPE_MXFP4,
            Bits::NVFP4 => requant_io::RQ_TYPE_NVFP4,
            Bits::NVFP4_GGUF => requant_io::GGML_TYPE_NVFP4,
            Bits::MXFP8 => requant_io::RQ_TYPE_MXFP8_E4M3,
            Bits::FP8_E4M3 => requant_io::RQ_TYPE_FP8_E4M3,
            Bits::FP8_E5M2 => requant_io::RQ_TYPE_FP8_E5M2,
            Bits::F32 => 0,
            Bits::F16 => 1,
            Bits::BF16 => 30,
            Bits::Q1_0 => 41,
            Bits::Q2_0 => 42,
            Bits::Q8_0 => 8,
            Bits::Q8_1 => 9,
            Bits::Q6_K => 14,
            Bits::Q5_K => 13,
            Bits::Q5_1 => 7,
            Bits::Q5_0 => 6,
            Bits::Q4_K => 12,
            Bits::Q4_1 => 3,
            Bits::Q4_0 => 2,
            Bits::Q3_K => 11,
            Bits::Q2_K => 10,
            Bits::Q8_K => 15,
            Bits::TQ1_0 => 34,
            Bits::TQ2_0 => 35,
            Bits::IQ1_S => 19,
            Bits::IQ1_M => 29,
            Bits::IQ2_XXS => 16,
            Bits::IQ2_XS => 17,
            Bits::IQ2_S => 22,
            Bits::IQ3_XXS => 18,
            Bits::IQ3_S => 21,
            Bits::IQ4_NL => 20,
            Bits::IQ4_XS => 23,
            Bits::Raw(n) => n,
        }
    }

    /// True if this is a float type that should be copied/converted rather than block-quantized.
    /// FP8 is deliberately *not* included: despite being a float format it is produced by a
    /// quantizer with a scale tensor, not by `pack_float`.
    pub fn is_float(self) -> bool {
        matches!(self, Bits::F32 | Bits::F16 | Bits::BF16)
    }

    /// Bits per weight, including sidecar scales. `None` for a `Raw` id with no known geometry.
    pub fn bpw(self) -> Option<f64> {
        requant_io::bpw(self.to_ggml_type())
    }

    /// Stable display/recipe name.
    pub fn name(self) -> &'static str {
        match self {
            Bits::F32 => "F32",
            Bits::F16 => "F16",
            Bits::BF16 => "BF16",
            Bits::Q1_0 => "Q1_0",
            Bits::Q2_0 => "Q2_0",
            Bits::Q8_0 => "Q8_0",
            Bits::Q8_1 => "Q8_1",
            Bits::Q6_K => "Q6_K",
            Bits::Q5_K => "Q5_K",
            Bits::Q5_1 => "Q5_1",
            Bits::Q5_0 => "Q5_0",
            Bits::Q4_K => "Q4_K",
            Bits::Q4_1 => "Q4_1",
            Bits::Q4_0 => "Q4_0",
            Bits::Q3_K => "Q3_K",
            Bits::Q2_K => "Q2_K",
            Bits::Q8_K => "Q8_K",
            Bits::TQ1_0 => "TQ1_0",
            Bits::TQ2_0 => "TQ2_0",
            Bits::IQ1_S => "IQ1_S",
            Bits::IQ1_M => "IQ1_M",
            Bits::IQ2_XXS => "IQ2_XXS",
            Bits::IQ2_XS => "IQ2_XS",
            Bits::IQ2_S => "IQ2_S",
            Bits::IQ3_XXS => "IQ3_XXS",
            Bits::IQ3_S => "IQ3_S",
            Bits::IQ4_NL => "IQ4_NL",
            Bits::IQ4_XS => "IQ4_XS",
            Bits::MXFP4 => "MXFP4",
            Bits::NVFP4 => "NVFP4",
            Bits::NVFP4_GGUF => "NVFP4_GGUF",
            Bits::MXFP8 => "MXFP8",
            Bits::FP8_E4M3 => "FP8_E4M3",
            Bits::FP8_E5M2 => "FP8_E5M2",
            Bits::Raw(_) => "RAW",
        }
    }

    /// Parse a recipe/CLI spelling (case-insensitive). Inverse of [`name`](Self::name).
    pub fn from_name(s: &str) -> Option<Bits> {
        let u = s.trim().to_ascii_uppercase();
        Some(match u.as_str() {
            "F32" => Bits::F32,
            "F16" | "FP16" => Bits::F16,
            "BF16" => Bits::BF16,
            "Q1_0" => Bits::Q1_0,
            "Q2_0" => Bits::Q2_0,
            "Q8_0" => Bits::Q8_0,
            "Q8_1" => Bits::Q8_1,
            "Q6_K" => Bits::Q6_K,
            "Q5_K" => Bits::Q5_K,
            "Q5_1" => Bits::Q5_1,
            "Q5_0" => Bits::Q5_0,
            "Q4_K" => Bits::Q4_K,
            "Q4_1" => Bits::Q4_1,
            "Q4_0" => Bits::Q4_0,
            "Q3_K" => Bits::Q3_K,
            "Q2_K" => Bits::Q2_K,
            "Q8_K" => Bits::Q8_K,
            "TQ1_0" => Bits::TQ1_0,
            "TQ2_0" => Bits::TQ2_0,
            "IQ1_S" => Bits::IQ1_S,
            "IQ1_M" => Bits::IQ1_M,
            "IQ2_XXS" => Bits::IQ2_XXS,
            "IQ2_XS" => Bits::IQ2_XS,
            "IQ2_S" => Bits::IQ2_S,
            "IQ3_XXS" => Bits::IQ3_XXS,
            "IQ3_S" => Bits::IQ3_S,
            "IQ4_NL" => Bits::IQ4_NL,
            "IQ4_XS" => Bits::IQ4_XS,
            "MXFP4" => Bits::MXFP4,
            "NVFP4" => Bits::NVFP4,
            "NVFP4_GGUF" | "GGML_NVFP4" => Bits::NVFP4_GGUF,
            "MXFP8" => Bits::MXFP8,
            "FP8" | "FP8_E4M3" => Bits::FP8_E4M3,
            "FP8_E5M2" => Bits::FP8_E5M2,
            _ => return None,
        })
    }

    /// Inverse of `to_ggml_type` for the named variants. Returns `None` for scalar/integer or
    /// otherwise unsupported ids.
    pub fn from_ggml_type(ty: u32) -> Option<Bits> {
        Some(match ty {
            requant_io::GGML_TYPE_MXFP4 => Bits::MXFP4,
            requant_io::RQ_TYPE_NVFP4 => Bits::NVFP4,
            requant_io::GGML_TYPE_NVFP4 => Bits::NVFP4_GGUF,
            requant_io::RQ_TYPE_MXFP8_E4M3 => Bits::MXFP8,
            requant_io::RQ_TYPE_FP8_E4M3 => Bits::FP8_E4M3,
            requant_io::RQ_TYPE_FP8_E5M2 => Bits::FP8_E5M2,
            0 => Bits::F32,
            1 => Bits::F16,
            30 => Bits::BF16,
            41 => Bits::Q1_0,
            42 => Bits::Q2_0,
            8 => Bits::Q8_0,
            9 => Bits::Q8_1,
            14 => Bits::Q6_K,
            13 => Bits::Q5_K,
            7 => Bits::Q5_1,
            6 => Bits::Q5_0,
            12 => Bits::Q4_K,
            3 => Bits::Q4_1,
            2 => Bits::Q4_0,
            11 => Bits::Q3_K,
            10 => Bits::Q2_K,
            15 => Bits::Q8_K,
            34 => Bits::TQ1_0,
            35 => Bits::TQ2_0,
            16 => Bits::IQ2_XXS,
            17 => Bits::IQ2_XS,
            18 => Bits::IQ3_XXS,
            19 => Bits::IQ1_S,
            20 => Bits::IQ4_NL,
            21 => Bits::IQ3_S,
            22 => Bits::IQ2_S,
            23 => Bits::IQ4_XS,
            29 => Bits::IQ1_M,
            _ => return None,
        })
    }
}

/// One rule in a recipe. Matches a set of roles (by label) and optionally a layer-depth range.
/// Last-match-wins over the rule list.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Rule {
    /// A single role label, or a list of them. See `Role::label`. Optional: a rule with only
    /// `name` set matches by tensor name regardless of role. If both `role` and `name` are set,
    /// both must match. If neither is set, the rule matches every tensor (use sparingly).
    #[serde(default)]
    pub role: Option<RoleMatcher>,
    /// Optional exact tensor-name match (single name or list). Used by `requant search` to emit
    /// per-tensor allocations as real rules. A rule with `name` set applies only to those names.
    #[serde(default)]
    pub name: Option<NameMatcher>,
    /// Optional fractional depth bounds, e.g. ">= 0.75". A tensor matches if its depth is in range.
    #[serde(default)]
    pub layer: Option<String>,
    /// Optional expert-index filter (routed experts only), e.g. "0..=15".
    #[serde(default)]
    pub expert: Option<String>,
    /// Target bits for matched tensors.
    pub bits: BitsOrStr,
    /// If true (default), use imatrix-weighted scale search when an imatrix is available.
    #[serde(default = "default_true")]
    pub imatrix: bool,
}

fn default_true() -> bool {
    true
}

/// `bits` may be a bare variant (`Q4_K`) or, for future flexibility, a table. We accept both.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum BitsOrStr {
    /// A named/raw bits variant.
    Named(Bits),
}

/// A role matcher: either a single label string or a list of them.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RoleMatcher {
    One(String),
    Many(Vec<String>),
}

impl RoleMatcher {
    fn matches(&self, label: &str) -> bool {
        match self {
            RoleMatcher::One(s) => role_label_matches(s, label),
            RoleMatcher::Many(v) => v.iter().any(|s| role_label_matches(s, label)),
        }
    }
}

/// A name matcher: either a single tensor name or a list of them. Matching is exact (no glob) —
/// names are the GGUF tensor names verbatim (e.g. `blk.0.ffn_down.weight`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NameMatcher {
    One(String),
    Many(Vec<String>),
}

impl NameMatcher {
    fn matches(&self, name: &str) -> bool {
        match self {
            NameMatcher::One(s) => s == name,
            NameMatcher::Many(v) => v.iter().any(|s| s == name),
        }
    }
}

/// Match a recipe role string against a tensor's role label. Supports a dot-prefix wildcard:
/// `"routed_expert"` matches `routed_expert.gate`, `.up`, `.down`; `"routed_expert.down"` matches
/// only the down-projection of routed experts.
fn role_label_matches(pattern: &str, label: &str) -> bool {
    if pattern == label {
        return true;
    }
    // "routed_expert" matches "routed_expert.<part>" and "shared_expert" matches "shared_expert.<part>".
    if let Some(prefix) = pattern.strip_suffix(".*") {
        return label == prefix || label.starts_with(&format!("{prefix}."));
    }
    if !pattern.contains('.') {
        // bare family name: matches the family and any "<family>.<part>".
        return label == pattern || label.starts_with(&format!("{pattern}."));
    }
    false
}

/// A parsed depth range like ">= 0.75", "< 0.25", or "0.5..0.75".
#[derive(Debug, Clone)]
enum DepthFilter {
    Ge(f32),
    Gt(f32),
    Le(f32),
    Lt(f32),
    Range(f32, f32),
    All,
}

fn parse_depth(s: &str) -> Result<DepthFilter> {
    let s = s.trim();
    if s == "*" || s.is_empty() {
        return Ok(DepthFilter::All);
    }
    if let Some(rest) = s.strip_prefix(">=").map(str::trim) {
        return Ok(DepthFilter::Ge(rest.parse()?));
    }
    if let Some(rest) = s.strip_prefix("<=").map(str::trim) {
        return Ok(DepthFilter::Le(rest.parse()?));
    }
    if let Some(rest) = s.strip_prefix('>').map(str::trim) {
        return Ok(DepthFilter::Gt(rest.parse()?));
    }
    if let Some(rest) = s.strip_prefix('<').map(str::trim) {
        return Ok(DepthFilter::Lt(rest.parse()?));
    }
    if let Some(idx) = s.find("..") {
        let lo: f32 = s[..idx].trim().parse()?;
        let hi: f32 = s[idx + 2..].trim_start_matches('=').trim().parse()?;
        return Ok(DepthFilter::Range(lo, hi));
    }
    bail!("unparseable depth filter `{s}` (try `>= 0.75`, `< 0.25`, or `0.5..0.75`)");
}

impl DepthFilter {
    fn matches(&self, depth: f32) -> bool {
        match self {
            DepthFilter::Ge(t) => depth >= *t,
            DepthFilter::Gt(t) => depth > *t,
            DepthFilter::Le(t) => depth <= *t,
            DepthFilter::Lt(t) => depth < *t,
            DepthFilter::Range(lo, hi) => depth >= *lo && depth <= *hi,
            DepthFilter::All => true,
        }
    }
}

/// An expert-index filter like "0..=15" or ">= 12".
#[derive(Debug, Clone)]
enum ExpertFilter {
    All,
    Range(u32, u32),
    Ge(u32),
}

fn parse_expert(s: &str) -> Result<ExpertFilter> {
    let s = s.trim();
    if s == "*" || s.is_empty() {
        return Ok(ExpertFilter::All);
    }
    if let Some(rest) = s.strip_prefix(">=") {
        return Ok(ExpertFilter::Ge(rest.trim().parse()?));
    }
    if let Some(idx) = s.find("..") {
        let lo: u32 = s[..idx].trim().parse()?;
        let hi: u32 = s[idx + 2..].trim_start_matches('=').trim().parse()?;
        return Ok(ExpertFilter::Range(lo, hi));
    }
    bail!("unparseable expert filter `{s}` (try `0..=15` or `>= 12`)");
}

impl ExpertFilter {
    fn matches(&self, expert: Option<u32>) -> bool {
        match self {
            ExpertFilter::All => true,
            ExpertFilter::Range(lo, hi) => expert.map_or(false, |e| e >= *lo && e <= *hi),
            ExpertFilter::Ge(t) => expert.map_or(false, |e| e >= *t),
        }
    }
}

/// Hard lower bounds on precision that no rule — hand-written or search-emitted — may cross.
///
/// DESIGN §7.6 makes this concrete for the router: misdetect the MoE role, quantize the gate
/// logits, and routing silently degrades in a way perplexity barely registers. The same argument
/// generalises to everything on the **dense path** (§0): attention, the shared expert, the MTP
/// head, embeddings and the LM head are touched by *every* token, so a bit spent there buys far
/// more than a bit spent in an expert that sees `expert_used / expert_count` of the traffic.
///
/// `dense_path` defaults to `None` (opt-in) so existing k-quant recipes keep working; a
/// sub-4-bit-expert recipe should set it — for an FP4 target, `"FP8_E4M3"`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Floors {
    /// The router may never be assigned fewer bits per weight than this. Default `F16`.
    #[serde(default = "default_router_floor")]
    pub router: Bits,
    /// Optional floor for every always-on tensor (see [`requant_io::Role::is_dense_path`]).
    #[serde(default)]
    pub dense_path: Option<Bits>,
}

fn default_router_floor() -> Bits {
    Bits::F16
}

impl Default for Floors {
    fn default() -> Self {
        Self { router: default_router_floor(), dense_path: None }
    }
}

/// `Some((assigned_bpw, floor_bpw))` when `assigned` is strictly below `floor`.
fn floor_violation(assigned: Bits, floor: Bits) -> Option<(f64, f64)> {
    let a = assigned.bpw()?;
    let f = floor.bpw()?;
    // The 1e-9 slack keeps an exactly-equal comparison from tripping on float representation.
    if a + 1e-9 < f {
        Some((a, f))
    } else {
        None
    }
}

/// A complete recipe.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Recipe {
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(default)]
    pub floors: Floors,
    #[serde(default)]
    pub rule: Vec<Rule>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Defaults {
    /// Default target bits when no rule matches.
    #[serde(default = "default_bits")]
    pub bits: BitsOrStr,
    /// Default imatrix usage.
    #[serde(default = "default_true")]
    pub imatrix: bool,
}

impl Default for Defaults {
    fn default() -> Self {
        Self { bits: default_bits(), imatrix: true }
    }
}

fn default_bits() -> BitsOrStr {
    BitsOrStr::Named(Bits::Q4_K)
}

/// Per-tensor policy resolved from a recipe + tag.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Policy {
    /// Target ggml type (e.g. 12 = Q4_K).
    pub ggml_type: u32,
    /// If true and an imatrix is available, use imatrix-weighted scale search.
    pub use_imatrix: bool,
    /// If true, copy the tensor bytes unchanged (target == source type) rather than round-tripping.
    #[serde(default)]
    pub copy_unchanged: bool,
}

impl Recipe {
    pub fn parse(toml_str: &str) -> Result<Self> {
        let recipe: Recipe = toml::from_str(toml_str).map_err(|e| anyhow!("recipe parse error: {e}"))?;
        Ok(recipe)
    }

    /// Resolve a policy for one tensor given its tag and the model layout. Applies rules
    /// last-match-wins, then validates that non-quantizable tensors (norms/biases) are not
    /// assigned a block-quant type. This overload matches only `role`/`layer`/`expert` (no name);
    /// call [`resolve_named`](Self::resolve_named) to also match per-tensor `name` rules.
    pub fn resolve(&self, tag: &TensorTag, layout: &ModelLayout) -> Result<Policy> {
        self.resolve_named(tag, layout, "")
    }

    /// Like [`resolve`](Self::resolve) but also applies per-tensor `name` matchers. `name` is the
    /// GGUF tensor name verbatim (e.g. `blk.0.ffn_down.weight`). Pass `""` to skip name matching.
    pub fn resolve_named(&self, tag: &TensorTag, _layout: &ModelLayout, name: &str) -> Result<Policy> {
        let label = tag.role.label();
        let mut bits = self.defaults.bits.clone();
        let mut use_imatrix = self.defaults.imatrix;

        for rule in &self.rule {
            // A rule matches only if every present matcher (role, name, layer, expert) matches.
            // `role` and `name` default to None (match-all), so a search-emitted `name`-only rule
            // applies regardless of role, and a legacy `role`-only rule applies regardless of name.
            if let Some(role) = &rule.role {
                if !role.matches(label) {
                    continue;
                }
            }
            if let Some(names) = &rule.name {
                if !names.matches(name) {
                    continue;
                }
            }
            if let Some(depth_str) = &rule.layer {
                let filter = parse_depth(depth_str)?;
                if !filter.matches(tag.place.depth) {
                    continue;
                }
            }
            if let Some(exp_str) = &rule.expert {
                let filter = parse_expert(exp_str)?;
                let expert = match tag.role {
                    Role::RoutedExpert { idx, .. } if idx != u32::MAX => Some(idx),
                    // Packed `_exps` tensor covers all experts: match iff filter is All or covers all.
                    Role::RoutedExpert { .. } => Some(0),
                    _ => None,
                };
                if !filter.matches(expert) {
                    continue;
                }
            }
            bits = rule.bits.clone();
            use_imatrix = rule.imatrix;
        }

        let named = match bits {
            BitsOrStr::Named(b) => b,
        };
        let ggml_type = named.to_ggml_type();

        // Non-quantizable tensors (norms/biases) are never block-quantized: copy them verbatim
        // (matches llama-quantize). A role rule that assigns a block-quant to e.g. `attn_q` is
        // meant for the weight, not the bias, so we ignore the assigned bits here.
        let copy_unchanged = !tag.quantizable;

        // Precision floors (§7.6). Only meaningful for tensors we would actually quantize —
        // norms and biases are copied verbatim and keep whatever the source had.
        if tag.quantizable {
            // `resolve()` passes an empty name; fall back to the role so the diagnostic is still
            // actionable when the caller didn't have a tensor name to hand.
            let disp = if name.is_empty() { format!("<{label}>") } else { name.to_string() };
            let name = disp.as_str();
            if tag.role.is_router() {
                let floor = self.floors.router;
                // An unknown-geometry `Raw` type on the router can't be checked, and the router is
                // exactly the tensor where "probably fine" is not good enough.
                let Some(assigned_bpw) = named.bpw() else {
                    bail!(
                        "router tensor `{name}` assigned an unrecognised format (ggml type \
                         {ggml_type}); the router floor cannot be verified, and mis-quantizing it \
                         corrupts MoE routing silently. Name a known format or raise the floor."
                    );
                };
                if let Some((_, floor_bpw)) = floor_violation(named, floor) {
                    bail!(
                        "router tensor `{name}` would be quantized to {} ({assigned_bpw:.2} bpw), \
                         below the router floor {} ({floor_bpw:.2} bpw). Quantizing the gate \
                         logits corrupts MoE routing in a way perplexity barely shows — fix the \
                         recipe's router rule, or raise [floors].router deliberately.",
                        named.name(),
                        floor.name(),
                    );
                }
            }
            if let Some(floor) = self.floors.dense_path {
                if tag.role.is_dense_path() {
                    if let Some((a, f)) = floor_violation(named, floor) {
                        bail!(
                            "dense-path tensor `{name}` (role {label}) would be quantized to {} \
                             ({a:.2} bpw), below the [floors].dense_path bound {} ({f:.2} bpw). \
                             Every token flows through this tensor; routed experts are the only \
                             role that earns sub-floor precision.",
                            named.name(),
                            floor.name(),
                        );
                    }
                }
            }
        }

        Ok(Policy { ggml_type, use_imatrix, copy_unchanged })
    }
}

/// A sensible MoE-aware default recipe: protect the router + lm_head + norms at F16, keep the
/// embedding high-fidelity at Q8_0 (as `llama-quantize`'s k-quant presets do — F16 here is the
/// main size trap on large-vocab models), attention a notch above the experts at Q5_K (Q6_K would
/// fall back to Q8_0 on models whose head dim isn't a multiple of 256, bloating size for no
/// quality gain), routed experts at Q4_K, deep routed-expert down-projections at Q5_K.
pub const DEFAULT_RECIPE_TOML: &str = r#"
[defaults]
bits = "Q4_K"
imatrix = true

# Protect the routing gate everywhere — quantizing it silently wrecks MoE routing.
[[rule]]
role = "router"
bits = "F16"

# Norms are tiny and sensitive; keep them exact. The lm_head directly produces logits, so
# protect it too — the search loop can downgrade it if the sensitivity curve allows.
[[rule]]
role = ["lm_head", "norm"]
bits = "F16"

# Embeddings are sensitive but large on big-vocab models; Q8_0 is the standard
# "protected-but-quantized" choice (matches llama-quantize k-quant presets) — F16 here is the
# dominant size trap. Q8_0 also never needs a fallback (block 32 divides almost every dim).
[[rule]]
role = "embedding"
bits = "Q8_0"

# Attention is always-on and tolerance-limited; a notch above the experts. Q5_K (not Q6_K): on
# models whose head dim isn't a multiple of 256, Q6_K falls back to Q8_0 (8.5 bpw) while Q5_K
# falls back to Q5_1 (5.5 bpw) — same intent, no bloat. The fused-QKV tensor (Qwen2/Phi-3/GLM-4)
# and the MLA projections (DeepSeek-V2/V3) are attention too, so they ride this same rule.
[[rule]]
role = ["attn_q", "attn_k", "attn_v", "attn_o", "attn_qkv", "attn_mla"]
bits = "Q5_K"

# State-space model weights (Mamba/Jamba) are always-on like attention; same tolerance argument.
# The small/sensitive SSM params (ssm_a/dt/d/…) are non-quantizable and copied verbatim, so they
# need no rule here.
[[rule]]
role = "ssm"
bits = "Q5_K"

# The shared expert is always-on (like attention), so don't starve it.
[[rule]]
role = "shared_expert"
bits = "Q5_K"

# Dense FFN (non-MoE): treat down-projections a notch higher, they hurt most.
[[rule]]
role = "ffn_down"
bits = "Q5_K"

# Routed experts are sparsely activated and dominate parameter count: aggressive is fine.
[[rule]]
role = "routed_expert"
bits = "Q4_K"

# ...except deep-layer routed-expert down-projections, which compound errors the most.
[[rule]]
role       = "routed_expert.down"
layer      = ">= 0.75"
bits       = "Q5_K"
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use requant_io::{FfnPart, Place};

    fn layout_dense() -> ModelLayout {
        ModelLayout { arch: "llama".into(), n_layers: 4, expert_count: 0, expert_used: 0, shared_count: 0, is_moe: false }
    }

    fn layout_moe() -> ModelLayout {
        ModelLayout { arch: "qwen2moe".into(), n_layers: 4, expert_count: 16, expert_used: 2, shared_count: 1, is_moe: true }
    }

    #[test]
    fn parses_and_resolves_default() {
        let r = Recipe::parse(DEFAULT_RECIPE_TOML).unwrap();
        let lay = layout_moe();
        let router = Recipe::resolve(&r, &TensorTag { role: Role::Router, place: Place { depth: 0.5, expert: None }, quantizable: true }, &lay).unwrap();
        assert_eq!(router.ggml_type, 1, "router must be F16");

        let expert_down = Recipe::resolve(&r, &TensorTag { role: Role::RoutedExpert { idx: 0, part: FfnPart::Down }, place: Place { depth: 0.8, expert: Some(0) }, quantizable: true }, &lay).unwrap();
        assert_eq!(expert_down.ggml_type, 13, "deep routed-expert down -> Q5_K");

        let expert_down_early = Recipe::resolve(&r, &TensorTag { role: Role::RoutedExpert { idx: 0, part: FfnPart::Down }, place: Place { depth: 0.1, expert: Some(0) }, quantizable: true }, &lay).unwrap();
        assert_eq!(expert_down_early.ggml_type, 12, "early routed-expert down -> Q4_K");

        let attn = Recipe::resolve(&r, &TensorTag { role: Role::AttnQ, place: Place { depth: 0.0, expert: None }, quantizable: true }, &lay).unwrap();
        assert_eq!(attn.ggml_type, 13, "attention -> Q5_K (Q6_K would fall back to Q8_0 on non-256 dims)");

        let embed = Recipe::resolve(&r, &TensorTag { role: Role::Embedding, place: Place { depth: 0.0, expert: None }, quantizable: true }, &lay).unwrap();
        assert_eq!(embed.ggml_type, 8, "embedding -> Q8_0 (protected-but-quantized, matches llama-quantize)");

        let lm_head = Recipe::resolve(&r, &TensorTag { role: Role::LmHead, place: Place { depth: 1.0, expert: None }, quantizable: true }, &lay).unwrap();
        assert_eq!(lm_head.ggml_type, 1, "lm_head -> F16 (protected)");
    }

    #[test]
    fn rejects_quantized_router() {
        let bad = r#"
[defaults]
bits = "Q4_K"
[[rule]]
role = "router"
bits = "Q4_K"
"#;
        let r = Recipe::parse(bad).unwrap();
        let lay = layout_moe();
        let res = Recipe::resolve(&r, &TensorTag { role: Role::Router, place: Place { depth: 0.0, expert: None }, quantizable: true }, &lay);
        assert!(res.is_err(), "router quantization must be rejected");
    }

    #[test]
    fn norm_bias_copies_unchanged() {
        let r = Recipe::parse(r#"
[defaults]
bits = "Q4_K"
"#).unwrap();
        let lay = layout_dense();
        // A norm is non-quantizable; even though the default assigns Q4_K, the policy must
        // signal "copy unchanged" rather than block-quantize a 1-D tensor.
        let res = Recipe::resolve(&r, &TensorTag { role: Role::Norm, place: Place { depth: 0.0, expert: None }, quantizable: false }, &lay).unwrap();
        assert!(res.copy_unchanged, "norm must copy unchanged, not block-quantize");
        // A weight with the same role family stays quantizable.
        let w = Recipe::resolve(&r, &TensorTag { role: Role::FfnGate, place: Place { depth: 0.0, expert: None }, quantizable: true }, &lay).unwrap();
        assert!(!w.copy_unchanged, "a quantizable weight must not be forced to copy");
    }

    fn tag(role: Role) -> TensorTag {
        TensorTag { role, place: Place { depth: 0.5, expert: None }, quantizable: true }
    }

    #[test]
    fn block_float_bits_map_to_their_geometry() {
        assert_eq!(Bits::NVFP4.bpw(), Some(4.5));
        assert_eq!(Bits::MXFP4.bpw(), Some(4.25));
        assert_eq!(Bits::FP8_E4M3.bpw(), Some(8.0));
        assert_eq!(Bits::MXFP8.bpw(), Some(8.25));
        assert_eq!(Bits::Q4_K.bpw(), Some(4.5));
        // Round-trip through the ggml type id.
        for b in [Bits::NVFP4, Bits::MXFP4, Bits::MXFP8, Bits::FP8_E4M3, Bits::FP8_E5M2] {
            assert_eq!(Bits::from_ggml_type(b.to_ggml_type()), Some(b), "{}", b.name());
        }
    }

    #[test]
    fn iquant_bits_map_to_their_geometry_and_round_trip() {
        // bytes-per-256 from the ggml block_iq* structs; IQ4_NL is block 32.
        let cases = [
            (Bits::IQ1_S,   19, 1.5625),
            (Bits::IQ1_M,   29, 1.75),
            (Bits::IQ2_XXS, 16, 2.0625),
            (Bits::IQ2_XS,  17, 2.3125),
            (Bits::IQ2_S,   22, 2.5625),
            (Bits::IQ3_XXS, 18, 3.0625),
            (Bits::IQ3_S,   21, 3.4375),
            (Bits::IQ4_XS,  23, 4.25),
            (Bits::IQ4_NL,  20, 4.5),
        ];
        for (b, id, bpw) in cases {
            assert_eq!(b.to_ggml_type(), id, "{} -> id", b.name());
            assert_eq!(Bits::from_ggml_type(id), Some(b), "id {} -> {}", id, b.name());
            assert_eq!(Bits::from_name(b.name()).unwrap(), b, "{} name round-trip", b.name());
            let got = b.bpw().expect("bpw");
            assert!((got - bpw).abs() < 1e-9, "{} bpw {} != {}", b.name(), got, bpw);
        }
    }

    #[test]
    fn current_ggml_standard_ternary_and_internal_k_types_round_trip() {
        let cases = [
            (Bits::Q1_0, 41, 1.125),
            (Bits::TQ1_0, 34, 1.6875),
            (Bits::TQ2_0, 35, 2.0625),
            (Bits::Q2_0, 42, 2.25),
            (Bits::Q8_K, 15, 9.125),
            (Bits::NVFP4_GGUF, 40, 4.5),
        ];
        for (b, id, bpw) in cases {
            assert_eq!(b.to_ggml_type(), id);
            assert_eq!(Bits::from_ggml_type(id), Some(b));
            assert_eq!(Bits::from_name(b.name()), Some(b));
            assert!((b.bpw().unwrap() - bpw).abs() < 1e-9, "{}", b.name());
        }
    }

    #[test]
    fn dense_path_floor_rejects_sub_floor_assignments() {
        // The V4-Flash shape: experts NVFP4, everything always-on pinned at FP8.
        let toml = r#"
[defaults]
bits = "NVFP4"

[floors]
router = "FP8_E4M3"
dense_path = "FP8_E4M3"

[[rule]]
role = ["attn_q", "attn_k", "attn_v", "attn_o", "router", "shared_expert", "mtp", "embedding", "lm_head"]
bits = "FP8_E4M3"

[[rule]]
role = "routed_expert"
bits = "NVFP4"
"#;
        let r = Recipe::parse(toml).unwrap();
        let lay = layout_moe();
        // Experts get NVFP4 — allowed, they are the one role off the dense path.
        let e = r
            .resolve_named(&tag(Role::RoutedExpert { idx: 0, part: FfnPart::Down }), &lay, "blk.0.ffn_down_exps.weight")
            .unwrap();
        assert_eq!(e.ggml_type, requant_io::RQ_TYPE_NVFP4);
        // Attention / MTP / shared expert are pinned at FP8.
        for role in [Role::AttnQ, Role::Mtp, Role::SharedExpert(FfnPart::Up)] {
            let p = r.resolve_named(&tag(role), &lay, "t").unwrap();
            assert_eq!(p.ggml_type, requant_io::RQ_TYPE_FP8_E4M3, "role {}", role.label());
        }

        // Now break it: a rule that drops attention below the floor must be rejected, not
        // silently honoured. This is the whole point of the floor.
        let bad = format!("{toml}\n[[rule]]\nrole = \"attn_q\"\nbits = \"NVFP4\"\n");
        let r2 = Recipe::parse(&bad).unwrap();
        let err = r2
            .resolve_named(&tag(Role::AttnQ), &lay, "blk.0.attn_q.weight")
            .expect_err("attention below the dense-path floor must be an error");
        let msg = err.to_string();
        assert!(msg.contains("dense_path"), "unhelpful message: {msg}");
        assert!(msg.contains("blk.0.attn_q.weight"), "message should name the tensor: {msg}");
    }

    #[test]
    fn router_floor_is_configurable_but_still_enforced() {
        // Default floor is F16, so FP8 on the router is a violation...
        let strict = Recipe::parse("[defaults]\nbits = \"Q4_K\"\n[[rule]]\nrole = \"router\"\nbits = \"FP8_E4M3\"\n").unwrap();
        assert!(strict.resolve_named(&tag(Role::Router), &layout_moe(), "r").is_err());

        // ...unless the recipe lowers it deliberately, which is exactly what an all-FP8 dense
        // path wants. Deliberate is the operative word: it takes an explicit line in the recipe.
        let relaxed = Recipe::parse(
            "[defaults]\nbits = \"Q4_K\"\n[floors]\nrouter = \"FP8_E4M3\"\n[[rule]]\nrole = \"router\"\nbits = \"FP8_E4M3\"\n",
        )
        .unwrap();
        assert_eq!(
            relaxed.resolve_named(&tag(Role::Router), &layout_moe(), "r").unwrap().ggml_type,
            requant_io::RQ_TYPE_FP8_E4M3
        );

        // But NVFP4 is still below even the relaxed floor.
        let too_low = Recipe::parse(
            "[defaults]\nbits = \"Q4_K\"\n[floors]\nrouter = \"FP8_E4M3\"\n[[rule]]\nrole = \"router\"\nbits = \"NVFP4\"\n",
        )
        .unwrap();
        assert!(too_low.resolve_named(&tag(Role::Router), &layout_moe(), "r").is_err());
    }

    #[test]
    fn mtp_tensors_tag_onto_the_dense_path() {
        let lay = layout_moe();
        let t = TensorTag::tag("blk.3.nextn.shared_head_head.weight", &lay);
        assert_eq!(t.role, Role::Mtp);
        assert!(t.quantizable);
        assert!(t.role.is_dense_path());
        // `enorm` does not end in `_norm`, so it needs the explicit nextn handling.
        let n = TensorTag::tag("blk.3.nextn.enorm.weight", &lay);
        assert_eq!(n.role, Role::Mtp);
        assert!(!n.quantizable, "a 1-D MTP norm must not be block-quantized");
    }

    #[test]
    fn default_recipe_assigns_attention_policy_to_qkv_mla_and_ssm() {
        let r = Recipe::parse(DEFAULT_RECIPE_TOML).unwrap();
        let lay = layout_dense();
        // Fused QKV and MLA projections are attention -> Q5_K under the default recipe.
        for (name, role) in [
            ("blk.0.attn_qkv.weight", Role::AttnQkv),
            ("blk.0.attn_q_a.weight", Role::AttnMla(requant_io::MlaPart::QA)),
            ("blk.0.attn_kv_b.weight", Role::AttnMla(requant_io::MlaPart::KvB)),
        ] {
            let t = TensorTag::tag(name, &lay);
            assert_eq!(t.role, role, "{name}");
            let p = r.resolve_named(&t, &lay, name).unwrap();
            assert_eq!(p.ggml_type, 13, "{name} -> Q5_K (attention policy)");
        }
        // SSM weight matrices -> Q5_K.
        let t = TensorTag::tag("blk.0.ssm_in.weight", &lay);
        assert_eq!(t.role, Role::Ssm);
        assert_eq!(r.resolve_named(&t, &lay, "blk.0.ssm_in.weight").unwrap().ggml_type, 13);
        // SSM params are non-quantizable -> copied unchanged, whatever the default bits say.
        let t = TensorTag::tag("blk.0.ssm_dt.weight", &lay);
        assert_eq!(t.role, Role::SsmParam);
        assert!(!t.quantizable);
        assert!(r.resolve_named(&t, &lay, "blk.0.ssm_dt.weight").unwrap().copy_unchanged);
    }

    #[test]
    fn dense_path_floor_protects_fused_qkv_and_mla() {
        // An all-NVFP4 recipe with a dense-path floor of FP8 must reject a sub-floor assignment
        // to a fused-QKV / MLA tensor, exactly as it does for ordinary attention.
        let toml = r#"
[defaults]
bits = "NVFP4"
[floors]
router = "FP8_E4M3"
dense_path = "FP8_E4M3"
[[rule]]
role = "routed_expert"
bits = "NVFP4"
"#;
        let r = Recipe::parse(toml).unwrap();
        let lay = layout_moe();
        for name in ["blk.0.attn_qkv.weight", "blk.0.attn_q_a.weight", "blk.0.attn_kv_b.weight"] {
            let t = TensorTag::tag(name, &lay);
            assert!(t.role.is_dense_path(), "{name} must be dense-path for the floor to apply");
            let err = r.resolve_named(&t, &lay, name).expect_err("{name} below floor must error");
            let msg = err.to_string();
            assert!(msg.contains("dense_path"), "unhelpful message: {msg}");
            assert!(msg.contains(name), "message should name the tensor: {msg}");
        }
    }
}
