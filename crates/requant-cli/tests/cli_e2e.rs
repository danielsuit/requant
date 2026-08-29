//! End-to-end CLI tests: build tiny real-architecture GGUF fixtures, run the actual
//! `requant` pipeline functions (inspect tagging, quantize, search) against them, and
//! cross-check the output container against `llama-quantize --dry-run` when available.
//!
//! Fixtures use real llama.cpp tensor names + a real `general.architecture` so the role tagger
//! and recipe resolver exercise their real code paths. Tensor columns are block-aligned (256
//! for k-quants) so the quantizer actually accepts them.

use requant_io::gguf::{GgufReader, GgufType, GgufValue, GgufWriter, TensorSpec};
use requant_io::{ModelLayout, Role, TensorTag};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static UNIQ: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    let n = UNIQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("requant-e2e-{tag}-{}-{n}.gguf", std::process::id()))
}

/// Deterministic LCG vector in [-1.5, 1.5) — no Math.random.
fn lcg(n: usize) -> Vec<f32> {
    let mut s: u32 = 0x9e3779b9;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push((s as f32) / (u32::MAX as f32) * 3.0 - 1.5);
    }
    v
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
    }
    out
}

/// Add an F16 2-D weight tensor [rows, cols] with block-aligned cols.
fn add_f16_weight(w: &mut GgufWriter, name: &str, rows: usize, cols: usize) {
    assert!(cols % 256 == 0, "fixture cols must be k-quant-aligned (256), got {cols}");
    let data = f16_bytes(&lcg(rows * cols));
    w.add_tensor(TensorSpec {
        name: name.into(),
        dims: vec![rows as u64, cols as u64],
        ggml_type: 1, // F16
        data,
    });
}

/// Add an F16 1-D tensor (norm/bias) of length `n`.
fn add_f16_1d(w: &mut GgufWriter, name: &str, n: usize) {
    w.add_tensor(TensorSpec {
        name: name.into(),
        dims: vec![n as u64],
        ggml_type: 1,
        data: f16_bytes(&lcg(n)),
    });
}

fn llama_kv(w: &mut GgufWriter, n_layers: u32, hidden: u32) {
    w.add_kv("general.architecture", GgufValue::String("llama".to_string()));
    w.add_kv("general.name", GgufValue::String("requant-e2e-llama".to_string()));
    w.add_kv("general.file_type", GgufValue::U32(1)); // F16
    w.add_kv("llama.block_count", GgufValue::U32(n_layers));
    w.add_kv("llama.embedding_length", GgufValue::U32(hidden));
    w.add_kv("llama.context_length", GgufValue::U32(128));
    w.add_kv("llama.attention.head_count", GgufValue::U32(4));
    w.add_kv("llama.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5));
    w.add_kv("llama.rope.dimension_count", GgufValue::U32(64));
}

/// A 2-block dense llama with real tensor names. hidden=256 so cols align to 256.
fn build_dense_llama(path: &PathBuf) {
    let mut w = GgufWriter::new();
    llama_kv(&mut w, 2, 256);
    let h = 256usize;
    let inter = 256usize; // keep small; ffn_intermediate
    add_f16_1d(&mut w, "token_embd.weight", h);
    add_f16_1d(&mut w, "output_norm.weight", h);
    add_f16_weight(&mut w, "output.weight", h, h);
    for blk in 0..2u32 {
        let p = format!("blk.{blk}.");
        add_f16_1d(&mut w, &format!("{p}attn_norm.weight"), h);
        add_f16_weight(&mut w, &format!("{p}attn_q.weight"), h, h);
        add_f16_weight(&mut w, &format!("{p}attn_k.weight"), h, h);
        add_f16_weight(&mut w, &format!("{p}attn_v.weight"), h, h);
        add_f16_weight(&mut w, &format!("{p}attn_o.weight"), h, h);
        add_f16_1d(&mut w, &format!("{p}ffn_norm.weight"), h);
        add_f16_weight(&mut w, &format!("{p}ffn_gate.weight"), inter, h);
        add_f16_weight(&mut w, &format!("{p}ffn_up.weight"), inter, h);
        add_f16_weight(&mut w, &format!("{p}ffn_down.weight"), h, inter);
    }
    w.write_to(path).unwrap();
}

/// A 1-block MoE (qwen2moe arch) with a router + packed routed experts + a shared expert.
fn build_moe(path: &PathBuf) {
    let mut w = GgufWriter::new();
    w.add_kv("general.architecture", GgufValue::String("qwen2moe".to_string()));
    w.add_kv("general.name", GgufValue::String("requant-e2e-moe".to_string()));
    w.add_kv("general.file_type", GgufValue::U32(1));
    w.add_kv("qwen2moe.block_count", GgufValue::U32(1));
    w.add_kv("qwen2moe.embedding_length", GgufValue::U32(256));
    w.add_kv("qwen2moe.expert_count", GgufValue::U32(4));
    w.add_kv("qwen2moe.expert_used_count", GgufValue::U32(2));
    w.add_kv("qwen2moe.expert_shared_count", GgufValue::U32(1));
    let h = 256usize;
    let inter = 256usize;
    let n_exp = 4usize;
    add_f16_1d(&mut w, "token_embd.weight", h);
    add_f16_1d(&mut w, "output_norm.weight", h);
    add_f16_weight(&mut w, "output.weight", h, h);
    let p = "blk.0.";
    add_f16_1d(&mut w, &format!("{p}attn_norm.weight"), h);
    add_f16_weight(&mut w, &format!("{p}attn_q.weight"), h, h);
    add_f16_weight(&mut w, &format!("{p}attn_k.weight"), h, h);
    add_f16_weight(&mut w, &format!("{p}attn_v.weight"), h, h);
    add_f16_weight(&mut w, &format!("{p}attn_o.weight"), h, h);
    add_f16_1d(&mut w, &format!("{p}ffn_norm.weight"), h);
    // Router (gate logits) — must be protected.
    add_f16_weight(&mut w, &format!("{p}ffn_gate_inp.weight"), n_exp, h);
    // Packed routed experts: [n_exp*out, in].
    add_f16_weight(&mut w, &format!("{p}ffn_gate_exps.weight"), n_exp * inter, h);
    add_f16_weight(&mut w, &format!("{p}ffn_up_exps.weight"), n_exp * inter, h);
    add_f16_weight(&mut w, &format!("{p}ffn_down_exps.weight"), n_exp * h, inter);
    // Shared expert.
    add_f16_weight(&mut w, &format!("{p}ffn_gate_shexp.weight"), inter, h);
    add_f16_weight(&mut w, &format!("{p}ffn_up_shexp.weight"), inter, h);
    add_f16_weight(&mut w, &format!("{p}ffn_down_shexp.weight"), h, inter);
    w.write_to(path).unwrap();
}

fn tag_map(reader: &GgufReader, layout: &ModelLayout) -> std::collections::HashMap<String, Role> {
    reader
        .tensors
        .iter()
        .map(|t| (t.name.clone(), TensorTag::tag(&t.name, layout).role))
        .collect()
}

#[test]
fn dense_llama_roles_tagged_correctly() {
    let path = tmp_path("dense");
    build_dense_llama(&path);
    let reader = GgufReader::open(&path).unwrap();
    let layout = ModelLayout::from_kv(&reader.kv).unwrap();
    assert_eq!(layout.arch, "llama");
    assert_eq!(layout.n_layers, 2);
    assert!(!layout.is_moe);
    let roles = tag_map(&reader, &layout);
    assert_eq!(roles["token_embd.weight"], Role::Embedding);
    assert_eq!(roles["output.weight"], Role::LmHead);
    assert_eq!(roles["output_norm.weight"], Role::Norm);
    assert_eq!(roles["blk.0.attn_norm.weight"], Role::Norm);
    assert_eq!(roles["blk.0.attn_q.weight"], Role::AttnQ);
    assert_eq!(roles["blk.0.attn_o.weight"], Role::AttnO);
    assert_eq!(roles["blk.0.ffn_gate.weight"], Role::FfnGate);
    assert_eq!(roles["blk.1.ffn_down.weight"], Role::FfnDown);
    assert_eq!(roles["blk.1.attn_v.weight"], Role::AttnV);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_roles_tagged_and_router_identified() {
    let path = tmp_path("moe");
    build_moe(&path);
    let reader = GgufReader::open(&path).unwrap();
    let layout = ModelLayout::from_kv(&reader.kv).unwrap();
    assert!(layout.is_moe);
    assert_eq!(layout.expert_count, 4);
    assert_eq!(layout.expert_used, 2);
    let roles = tag_map(&reader, &layout);
    assert_eq!(roles["blk.0.ffn_gate_inp.weight"], Role::Router);
    assert!(roles["blk.0.ffn_gate_inp.weight"].is_router());
    assert!(matches!(roles["blk.0.ffn_gate_exps.weight"], Role::RoutedExpert { .. }));
    assert!(matches!(roles["blk.0.ffn_down_exps.weight"], Role::RoutedExpert { .. }));
    assert!(matches!(roles["blk.0.ffn_gate_shexp.weight"], Role::SharedExpert(_)));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn quantize_dense_llama_changes_types_and_shrinks() {
    let src = tmp_path("q-src");
    let out = tmp_path("q-out");
    build_dense_llama(&src);

    requant_cli::run_quantize(src.to_str().unwrap(), out.to_str().unwrap(), None, None).unwrap();

    let r = GgufReader::open(&out).unwrap();
    // The default MoE-aware recipe: attn/ffn_down/shared -> Q5_K, ffn gate/up -> Q4_K,
    // embedding -> Q8_0, lm_head/norm/router -> F16.
    let by_name: std::collections::HashMap<&str, (u32, &[u8])> = r
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), (t.ggml_type, &[][..])))
        .collect();
    let _ = by_name;
    let types: std::collections::HashMap<String, u32> =
        r.tensors.iter().map(|t| (t.name.clone(), t.ggml_type)).collect();
    // Attention should be Q5_K (13), FFN gate/up Q4_K (12), FFN down Q5_K (13).
    assert_eq!(types["blk.0.attn_q.weight"], 13, "attn should be Q5_K");
    assert_eq!(types["blk.0.ffn_gate.weight"], 12, "ffn_gate should be Q4_K");
    assert_eq!(types["blk.0.ffn_up.weight"], 12, "ffn_up should be Q4_K");
    assert_eq!(types["blk.0.ffn_down.weight"], 13, "ffn_down should be Q5_K");
    // Embedding quantizes to Q8_0 (8); protected tensors stay F16 (1).
    assert_eq!(types["token_embd.weight"], 8, "embedding quantizes to Q8_0");
    assert_eq!(types["output.weight"], 1, "lm_head stays F16");
    assert_eq!(types["blk.0.attn_norm.weight"], 1, "norm stays F16");

    // Output is smaller than source.
    let src_size = std::fs::metadata(&src).unwrap().len();
    let out_size = std::fs::metadata(&out).unwrap().len();
    assert!(out_size < src_size, "output ({out_size}) should be smaller than source ({src_size})");

    // Cross-check the output container parses in the reference quantizer.
    assert_llama_quantize_accepts(&out);

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn quantize_moe_protects_router() {
    let src = tmp_path("moe-src");
    let out = tmp_path("moe-out");
    build_moe(&src);

    requant_cli::run_quantize(src.to_str().unwrap(), out.to_str().unwrap(), None, None).unwrap();

    let r = GgufReader::open(&out).unwrap();
    let types: std::collections::HashMap<String, u32> =
        r.tensors.iter().map(|t| (t.name.clone(), t.ggml_type)).collect();
    // Router must remain F16 — the whole MoE-aware thesis.
    assert_eq!(
        types["blk.0.ffn_gate_inp.weight"],
        1,
        "router must stay F16, got {}",
        types["blk.0.ffn_gate_inp.weight"]
    );
    // Routed experts get the aggressive default (Q4_K = 12).
    assert_eq!(types["blk.0.ffn_gate_exps.weight"], 12, "routed experts should be Q4_K");
    assert_eq!(types["blk.0.ffn_down_exps.weight"], 12);
    // Shared expert is always-on — recipe treats it like attention (Q5_K = 13).
    assert_eq!(types["blk.0.ffn_gate_shexp.weight"], 13, "shared expert should be Q5_K");

    assert_llama_quantize_accepts(&out);
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn search_finds_allocation_under_budget() {
    let src = tmp_path("s-src");
    let recipe_out = tmp_path_text("s-recipe");
    build_dense_llama(&src);

    // A budget between Q4_K-only and Q6_K-for-attention size should force a mix.
    requant_cli::run_search(
        src.to_str().unwrap(),
        "14000", // bytes — tight enough to require tradeoffs on this tiny model
        None,
        None,
        Some(recipe_out.to_str().unwrap()),
        false, // --validate off (no perplexity run in unit tests)
        None,
    )
    .unwrap();

    let recipe = std::fs::read_to_string(&recipe_out).unwrap();
    assert!(recipe.contains("Auto-searched recipe"), "recipe header missing");
    // Every searchable tensor should appear in the per-tensor allocation comment.
    assert!(recipe.contains("blk.0.attn_q.weight"));
    assert!(recipe.contains("blk.0.ffn_gate.weight"));

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&recipe_out);
}

// ---------- helpers ----------

fn tmp_path_text(tag: &str) -> PathBuf {
    let n = UNIQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("requant-e2e-{tag}-{}-{n}.toml", std::process::id()))
}

/// Run `llama-quantize --dry-run` on a GGUF and assert the reference parser accepts the
/// container (prints "loaded meta data"). Skipped gracefully if the binary is absent.
fn assert_llama_quantize_accepts(path: &PathBuf) {
    let out_path = std::env::temp_dir().join(format!("requant-dryrun-{}.gguf", std::process::id()));
    let res = Command::new("llama-quantize")
        .args(["--dry-run", path.to_str().unwrap(), out_path.to_str().unwrap(), "Q8_0"])
        .output();
    let _ = std::fs::remove_file(&out_path);
    match res {
        Err(e) => eprintln!("llama-quantize unavailable, skipping cross-check: {e}"),
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            assert!(
                stderr.contains("loaded meta data"),
                "reference GGUF parser rejected our output container:\n{stderr}"
            );
        }
    }
}

// Suppress unused-import warning for GgufType when not all variants are used.
#[allow(dead_code)]
fn _gguf_type_marker(_t: GgufType) {}
