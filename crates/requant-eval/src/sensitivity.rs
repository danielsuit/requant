//! Per-tensor sensitivity: the `(bits → ΔKL)` curves the bit-allocation search consumes.
//!
//! DESIGN §3.6, last bullet: *quantize one layer/role at a time, leave the rest at fp16, measure
//! the KL it induces.* That ablation is the whole point. A round-trip error metric tells you how
//! badly a format mangles a tensor's *numbers*; it cannot tell you how much the model *cares*,
//! and those differ by orders of magnitude between a routed expert and a router. Only running the
//! model answers that.
//!
//! # Cost, and how the knobs control it
//!
//! One candidate model per (group × ladder level), each scored by a forward pass over the corpus.
//! Per-tensor grouping on a 300-tensor model with a 5-rung ladder is 1500 candidates — real money.
//! Two levers make it tractable, and they are the ones worth reaching for in order:
//!
//! 1. **[`Grouping::PerRole`]** collapses to ~10 groups (50 candidates). Sensitivity is far more a
//!    property of *role* than of individual tensor, which is the same observation the recipe
//!    language is built on, so this loses much less than the 30× cost cut suggests. Start here.
//! 2. **[`Grouping::PerRoleDepth`]** splits each role into depth buckets, which recovers the one
//!    within-role gradient that reliably matters (deep down-projections hurt more than shallow
//!    ones) at a few times the cost.
//!
//! Per-tensor is for small models and for confirming that a coarser grouping didn't hide anything.
//!
//! # What the ΔKL numbers mean
//!
//! Each entry is the KL a candidate induces *on its own*, against an otherwise-untouched model.
//! The allocator then treats total KL as the sum of the per-group ΔKLs. That additivity is an
//! approximation — quantization errors in different tensors interact — but it is the same
//! first-order independence assumption behind every diagonal-Hessian method in §1.1, it is
//! accurate in the small-perturbation regime a sane recipe stays in, and it is what makes the
//! problem a knapsack instead of a search over 5^300 joint configurations. When the final measured
//! KL of an allocated recipe diverges badly from the sum of its parts, that is the signal the
//! regime assumption broke — which is worth checking with `search --validate`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use requant_calib::Imatrix;
use requant_io::{
    ggml_type_name, is_float_type, packed_nbytes, GgufReader, ModelLayout, TensorPlan, TensorTag,
};
use requant_quant::{fallback_type, quantize_tensor, Bits};

use crate::evaluator::{Evaluator, Reference};

/// How tensors are batched into ablation candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grouping {
    /// One candidate per tensor. Most informative, most expensive.
    PerTensor,
    /// One candidate per role label (`attn_q`, `routed_expert.down`, …).
    PerRole,
    /// Role × depth bucket. `buckets = 4` gives quarter-of-the-stack resolution.
    PerRoleDepth { buckets: usize },
}

impl Grouping {
    fn key(&self, tag: &TensorTag, name: &str) -> String {
        match self {
            Grouping::PerTensor => name.to_string(),
            Grouping::PerRole => tag.role.label().to_string(),
            Grouping::PerRoleDepth { buckets } => {
                let b = (*buckets).max(1);
                // Depth is in [0, 1] inclusive, so the top of the range must clamp into the last
                // bucket rather than index one past it.
                let idx = ((tag.place.depth * b as f32) as usize).min(b - 1);
                format!("{}@d{}/{}", tag.role.label(), idx, b)
            }
        }
    }
}

/// What to ablate and how.
#[derive(Debug, Clone)]
pub struct SensitivityPlan {
    pub source: PathBuf,
    pub corpus: PathBuf,
    pub work_dir: PathBuf,
    /// Candidate precisions, cheapest first.
    pub ladder: Vec<Bits>,
    pub grouping: Grouping,
    pub imatrix: Option<Imatrix>,
    /// Keep candidate GGUFs on disk after scoring (needed for the external-dump workflow).
    pub keep_candidates: bool,
    /// Restrict to these role labels.
    pub roles: Option<Vec<String>>,
    /// Hard cap on candidates, as a guard against an accidental 1500-model run.
    pub max_candidates: Option<usize>,
}

impl SensitivityPlan {
    pub fn new(source: impl Into<PathBuf>, corpus: impl Into<PathBuf>, work_dir: impl Into<PathBuf>) -> Self {
        Self {
            source: source.into(),
            corpus: corpus.into(),
            work_dir: work_dir.into(),
            ladder: vec![Bits::Q2_K, Bits::Q3_K, Bits::Q4_K, Bits::Q5_K, Bits::Q6_K, Bits::Q8_0],
            grouping: Grouping::PerRole,
            imatrix: None,
            keep_candidates: false,
            roles: None,
            max_candidates: Some(256),
        }
    }
}

/// One `(bits → ΔKL)` sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityPoint {
    pub bits: String,
    /// The type actually used, after block-alignment fallback.
    pub ggml_type: u32,
    pub type_name: String,
    /// Bytes this group occupies at this precision.
    pub bytes: u64,
    /// Mean KL(fp16 ‖ candidate) induced by quantizing *only* this group to these bits.
    pub kl: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kl_p99: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top1_agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ppl: Option<f64>,
}

/// One group's curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityEntry {
    pub key: String,
    pub role: String,
    /// Mean fractional depth of the group's members.
    pub depth: f32,
    pub members: Vec<String>,
    /// Bytes the group occupies in the source model.
    pub source_bytes: u64,
    pub points: Vec<SensitivityPoint>,
}

/// The artifact the search consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityTable {
    pub schema: u32,
    pub source: String,
    pub corpus: String,
    pub evaluator: String,
    pub grouping: Grouping,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_ppl: Option<f64>,
    pub entries: Vec<SensitivityEntry>,
}

pub const SENSITIVITY_SCHEMA: u32 = 1;

impl SensitivityTable {
    pub fn write_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let s = serde_json::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), s)
            .with_context(|| format!("writing {}", path.as_ref().display()))?;
        Ok(())
    }

    pub fn read_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        let t: SensitivityTable = serde_json::from_str(&s)
            .with_context(|| format!("parsing sensitivity table {}", path.as_ref().display()))?;
        if t.schema != SENSITIVITY_SCHEMA {
            bail!(
                "sensitivity table schema {} (this build reads {SENSITIVITY_SCHEMA}) — re-run \
                 `requant sensitivity`",
                t.schema
            );
        }
        Ok(t)
    }

    /// Look up a tensor's curve. Falls back through the grouping hierarchy so a table captured
    /// per-role still answers per-tensor questions.
    pub fn lookup(&self, tensor_name: &str, role: &str) -> Option<&SensitivityEntry> {
        self.entries
            .iter()
            .find(|e| e.key == tensor_name)
            .or_else(|| self.entries.iter().find(|e| e.members.iter().any(|m| m == tensor_name)))
            .or_else(|| self.entries.iter().find(|e| e.key == role))
    }
}

/// A materialised ablation candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub group: String,
    pub bits: String,
    pub ggml_type: u32,
    pub path: PathBuf,
    pub group_bytes: u64,
}

/// The set of candidates for a plan, plus enough metadata to score them later (possibly on another
/// machine).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateManifest {
    pub source: String,
    pub corpus: String,
    pub grouping: Grouping,
    pub groups: Vec<GroupInfo>,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupInfo {
    pub key: String,
    pub role: String,
    pub depth: f32,
    pub members: Vec<String>,
    pub source_bytes: u64,
}

impl CandidateManifest {
    pub fn write_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        std::fs::write(path.as_ref(), serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.as_ref().display()))?;
        Ok(())
    }

    pub fn read_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        Ok(serde_json::from_str(&s)?)
    }
}

/// Filesystem-safe candidate id. Tensor names contain dots and slashes appear in depth buckets.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

struct Grouped {
    key: String,
    role: String,
    depth: f32,
    members: Vec<usize>,
    source_bytes: u64,
}

fn build_groups(
    reader: &GgufReader,
    layout: &ModelLayout,
    plan: &SensitivityPlan,
) -> Result<Vec<Grouped>> {
    let mut map: BTreeMap<String, Grouped> = BTreeMap::new();
    for (i, t) in reader.tensors.iter().enumerate() {
        let tag = TensorTag::tag(&t.name, layout);
        if !tag.quantizable {
            continue;
        }
        // Ablation only makes sense from a full-precision source: quantizing an already-quantized
        // tensor measures the *combined* damage, not this format's contribution.
        if !is_float_type(t.ggml_type) {
            continue;
        }
        let label = tag.role.label();
        if let Some(roles) = &plan.roles {
            if !roles.iter().any(|r| r == label) {
                continue;
            }
        }
        let key = plan.grouping.key(&tag, &t.name);
        let bytes = requant_io::tensor_nbytes(t).unwrap_or(0);
        let e = map.entry(key.clone()).or_insert_with(|| Grouped {
            key,
            role: label.to_string(),
            depth: 0.0,
            members: Vec::new(),
            source_bytes: 0,
        });
        e.members.push(i);
        e.source_bytes += bytes;
        // Running mean of depth.
        let n = e.members.len() as f32;
        e.depth += (tag.place.depth - e.depth) / n;
    }
    let groups: Vec<Grouped> = map.into_values().filter(|g| !g.members.is_empty()).collect();
    if groups.is_empty() {
        bail!(
            "no ablatable tensors in {} — sensitivity needs a full-precision (F16/BF16/F32) \
             source; an already-quantized model has nothing to ablate *from*",
            plan.source.display()
        );
    }
    Ok(groups)
}

/// Write every ablation candidate to disk and return the manifest.
///
/// Each candidate is byte-identical to the source except for one group of tensors. The write is
/// streaming — peak memory is one tensor, not one model.
pub fn emit_candidates(plan: &SensitivityPlan) -> Result<CandidateManifest> {
    let reader = GgufReader::open(&plan.source)
        .with_context(|| format!("opening {}", plan.source.display()))?;
    let layout = ModelLayout::from_kv(&reader.kv)?;
    let groups = build_groups(&reader, &layout, plan)?;

    let total = groups.len() * plan.ladder.len();
    if let Some(cap) = plan.max_candidates {
        if total > cap {
            bail!(
                "this plan needs {total} candidate models ({} groups × {} ladder levels), over the \
                 cap of {cap}. Each candidate is a full model write plus a forward pass over the \
                 corpus. Coarsen `grouping` (PerRole is ~10 groups), shorten the ladder, or raise \
                 the cap deliberately.",
                groups.len(),
                plan.ladder.len()
            );
        }
    }
    std::fs::create_dir_all(&plan.work_dir)
        .with_context(|| format!("creating {}", plan.work_dir.display()))?;

    let kv: Vec<_> = reader.kv.iter().filter(|(k, _)| k != "general.alignment").cloned().collect();
    let mut candidates = Vec::with_capacity(total);
    let mut done = 0usize;

    for g in &groups {
        for &bits in &plan.ladder {
            done += 1;
            let id = format!("{}@{}", sanitize(&g.key), bits.name());
            let path = plan.work_dir.join(format!("{id}.gguf"));

            // Resolve the per-tensor target once, so the header and the data agree.
            let mut targets: Vec<u32> = reader.tensors.iter().map(|t| t.ggml_type).collect();
            let mut group_bytes = 0u64;
            let mut actual_type = bits.to_ggml_type();
            for &i in &g.members {
                let (_, cols) = reader.tensors[i].rows_cols();
                let (actual, _) = fallback_type(bits.to_ggml_type(), cols);
                targets[i] = actual;
                actual_type = actual;
                group_bytes +=
                    packed_nbytes(actual, reader.tensors[i].n_elems(), &reader.tensors[i].name)?;
            }

            let plans: Vec<TensorPlan> = reader
                .tensors
                .iter()
                .zip(&targets)
                .map(|(t, &ty)| {
                    Ok(TensorPlan {
                        name: t.name.clone(),
                        dims: t.dims.clone(),
                        ggml_type: ty,
                        nbytes: packed_nbytes(ty, t.n_elems(), &t.name)?,
                    })
                })
                .collect::<Result<_>>()?;

            eprintln!(
                "sensitivity: [{done}/{total}] {} @ {} -> {}",
                g.key,
                bits.name(),
                path.display()
            );
            requant_io::write_gguf_streaming(
                &path,
                &kv,
                &plans,
                reader.alignment,
                reader.version,
                |i| {
                    let t = &reader.tensors[i];
                    if targets[i] == t.ggml_type {
                        return Ok(reader.tensor_bytes(i)?.to_vec());
                    }
                    let src = reader.tensor_to_f32(i)?;
                    let (rows, cols) = t.rows_cols();
                    let im = plan.imatrix.as_ref().and_then(|m| m.get(&t.name));
                    quantize_tensor(targets[i], &src, rows, cols, im).with_context(|| {
                        format!("quantizing `{}` to {}", t.name, ggml_type_name(targets[i]))
                    })
                },
            )?;

            candidates.push(Candidate {
                id,
                group: g.key.clone(),
                bits: bits.name().to_string(),
                ggml_type: actual_type,
                path,
                group_bytes,
            });
        }
    }

    Ok(CandidateManifest {
        source: plan.source.display().to_string(),
        corpus: plan.corpus.display().to_string(),
        grouping: plan.grouping,
        groups: groups
            .iter()
            .map(|g| GroupInfo {
                key: g.key.clone(),
                role: g.role.clone(),
                depth: g.depth,
                members: g.members.iter().map(|&i| reader.tensors[i].name.clone()).collect(),
                source_bytes: g.source_bytes,
            })
            .collect(),
        candidates,
    })
}

/// Score a previously emitted manifest.
pub fn score_candidates(
    manifest: &CandidateManifest,
    evaluator: &dyn Evaluator,
    reference: &Reference,
    corpus: &Path,
) -> Result<SensitivityTable> {
    let mut by_group: BTreeMap<&str, Vec<SensitivityPoint>> = BTreeMap::new();
    let n = manifest.candidates.len();
    for (i, c) in manifest.candidates.iter().enumerate() {
        eprintln!("sensitivity: scoring [{}/{n}] {}", i + 1, c.id);
        let score = evaluator
            .score(&c.path, reference, corpus)
            .with_context(|| format!("scoring candidate `{}`", c.id))?;
        by_group.entry(c.group.as_str()).or_default().push(SensitivityPoint {
            bits: c.bits.clone(),
            ggml_type: c.ggml_type,
            type_name: ggml_type_name(c.ggml_type),
            bytes: c.group_bytes,
            kl: score.kl.mean,
            kl_p99: Some(score.kl.p99),
            top1_agreement: Some(score.kl.top1_agreement),
            ppl: score.ppl,
        });
    }

    let mut entries = Vec::with_capacity(manifest.groups.len());
    for g in &manifest.groups {
        let mut points = by_group.remove(g.key.as_str()).unwrap_or_default();
        // Ascending in bytes is what the allocator's marginal analysis assumes.
        points.sort_by(|a, b| a.bytes.cmp(&b.bytes));
        entries.push(SensitivityEntry {
            key: g.key.clone(),
            role: g.role.clone(),
            depth: g.depth,
            members: g.members.clone(),
            source_bytes: g.source_bytes,
            points,
        });
    }

    Ok(SensitivityTable {
        schema: SENSITIVITY_SCHEMA,
        source: manifest.source.clone(),
        corpus: manifest.corpus.clone(),
        evaluator: evaluator.name().to_string(),
        grouping: manifest.grouping,
        reference_ppl: reference.reference_ppl(),
        entries,
    })
}

/// Emit candidates, score them, and (unless asked to keep them) clean up.
pub fn run_sensitivity(plan: &SensitivityPlan, evaluator: &dyn Evaluator) -> Result<SensitivityTable> {
    let manifest = emit_candidates(plan)?;
    let manifest_path = plan.work_dir.join("candidates.json");
    manifest.write_json(&manifest_path)?;

    if !evaluator.scores_local_candidates() {
        eprintln!(
            "sensitivity: evaluator `{}` cannot score local files. {} candidates are written to \
             {} with a manifest at {}. Run each through the serving stack, drop the logit dumps \
             next to the reference capture, then re-run scoring.",
            evaluator.name(),
            manifest.candidates.len(),
            plan.work_dir.display(),
            manifest_path.display()
        );
    }

    let reference = evaluator.prepare_reference(&plan.source, &plan.corpus)?;
    let table = score_candidates(&manifest, evaluator, &reference, &plan.corpus)?;

    if !plan.keep_candidates {
        for c in &manifest.candidates {
            let _ = std::fs::remove_file(&c.path);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use requant_io::{FfnPart, Place, Role};

    fn tag(role: Role, depth: f32) -> TensorTag {
        TensorTag { role, place: Place { depth, expert: None }, quantizable: true }
    }

    #[test]
    fn grouping_keys_are_what_they_claim() {
        let t = tag(Role::AttnQ, 0.8);
        assert_eq!(Grouping::PerTensor.key(&t, "blk.7.attn_q.weight"), "blk.7.attn_q.weight");
        assert_eq!(Grouping::PerRole.key(&t, "blk.7.attn_q.weight"), "attn_q");
        assert_eq!(
            Grouping::PerRoleDepth { buckets: 4 }.key(&t, "blk.7.attn_q.weight"),
            "attn_q@d3/4"
        );
        // Depth exactly 1.0 must not fall off the end of the bucket array.
        let deep = tag(Role::RoutedExpert { idx: 0, part: FfnPart::Down }, 1.0);
        assert_eq!(
            Grouping::PerRoleDepth { buckets: 4 }.key(&deep, "x"),
            "routed_expert.down@d3/4"
        );
    }

    #[test]
    fn sanitize_makes_tensor_names_into_filenames() {
        assert_eq!(sanitize("blk.7.attn_q.weight"), "blk.7.attn_q.weight");
        assert_eq!(sanitize("routed_expert.down@d3/4"), "routed_expert.down_d3_4");
    }

    #[test]
    fn table_round_trips_through_json_and_rejects_a_stale_schema() {
        let t = SensitivityTable {
            schema: SENSITIVITY_SCHEMA,
            source: "m.gguf".into(),
            corpus: "c.txt".into(),
            evaluator: "test".into(),
            grouping: Grouping::PerRole,
            reference_ppl: Some(16.25),
            entries: vec![SensitivityEntry {
                key: "attn_q".into(),
                role: "attn_q".into(),
                depth: 0.5,
                members: vec!["blk.0.attn_q.weight".into()],
                source_bytes: 1024,
                points: vec![SensitivityPoint {
                    bits: "Q4_K".into(),
                    ggml_type: 12,
                    type_name: "Q4_K".into(),
                    bytes: 288,
                    kl: 0.01,
                    kl_p99: Some(0.1),
                    top1_agreement: Some(0.99),
                    ppl: None,
                }],
            }],
        };
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sens.json");
        t.write_json(&p).unwrap();
        let back = SensitivityTable::read_json(&p).unwrap();
        assert_eq!(back.entries.len(), 1);
        assert_eq!(back.entries[0].points[0].kl, 0.01);
        // Lookup falls back from tensor name to role.
        assert!(back.lookup("blk.0.attn_q.weight", "attn_q").is_some());
        assert!(back.lookup("blk.9.attn_q.weight", "attn_q").is_some());
        assert!(back.lookup("blk.9.ffn_up.weight", "ffn_up").is_none());

        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        raw["schema"] = serde_json::json!(99);
        std::fs::write(&p, raw.to_string()).unwrap();
        assert!(SensitivityTable::read_json(&p).is_err(), "a stale schema must not load silently");
    }
}
