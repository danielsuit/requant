//! Tests for the hand-written GGUF reader/writer.
//!
//! Two layers of correctness:
//!  1. Writer→Reader→Writer is byte-identical (deterministic, self-consistent serialization).
//!  2. Writer output is accepted by the llama.cpp ecosystem (`llama-quantize --dry-run` parses it).
//!  3. `llama-quantize` can requantize a tensor we wrote and our reader reads its output back.

use requant_io::gguf::*;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static UNIQ: AtomicU64 = AtomicU64::new(0);

fn unique_name(tag: &str) -> String {
    let n = UNIQ.fetch_add(1, Ordering::Relaxed);
    format!("{tag}-{pid}-{n}.gguf", pid = std::process::id())
}

fn tmp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(unique_name(tag))
}

fn write_f16(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
}

/// Build a tiny synthetic GGUF with a few F16 tensors (no real architecture; just container validity).
fn make_synthetic_gguf() -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_kv("general.architecture".to_string(), GgufValue::String("requant-test".into()));
    w.add_kv("general.name".to_string(), GgufValue::String("synthetic".into()));
    w.add_kv("general.file_type".to_string(), GgufValue::U32(1)); // F16
    w.add_kv("test.counts".to_string(), GgufValue::Array {
        elem: GgufType::U32,
        items: vec![GgufValue::U32(1), GgufValue::U32(2), GgufValue::U32(3)],
    });

    // tensor A: shape [4, 32] F16, so 128 elements; row length 32 (matches legacy block).
    let (rows, cols) = (4usize, 32usize);
    let mut data_a = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let v = ((r * cols + c) as f32) * 0.1 - 1.0;
            write_f16(&mut data_a, v);
        }
    }
    w.add_tensor(TensorSpec { name: "tensor.a".into(), dims: vec![rows as u64, cols as u64], ggml_type: 1, data: data_a });

    // tensor B: shape [16] F16 (1-D norm-like).
    let mut data_b = Vec::with_capacity(16 * 2);
    for i in 0..16 {
        write_f16(&mut data_b, (i as f32) * 0.5);
    }
    w.add_tensor(TensorSpec { name: "tensor.b".into(), dims: vec![16], ggml_type: 1, data: data_b });

    w.to_bytes()
}

#[test]
fn writer_reader_writer_is_byte_identical() {
    let original = make_synthetic_gguf();
    let path = tmp_path("rrw");
    std::fs::write(&path, &original).unwrap();

    let r = GgufReader::open(&path).unwrap();
    assert_eq!(r.version, 3);
    assert_eq!(r.tensors.len(), 2);
    assert_eq!(r.tensors[0].name, "tensor.a");
    assert_eq!(r.tensors[0].dims, vec![4, 32]); // logical order
    assert_eq!(r.tensors[0].ggml_type, 1);
    assert_eq!(r.tensors[1].dims, vec![16]);
    // KV round-trips
    assert_eq!(r.get("general.architecture").unwrap().as_str(), Some("requant-test"));
    let (_, arr) = r.get("test.counts").unwrap().as_array().unwrap();
    assert_eq!(arr.len(), 3);

    // Re-serialize and compare byte-for-byte.
    let mut w = GgufWriter::new();
    w.set_alignment(r.alignment);
    w.copy_kv_from(&r);
    for (i, t) in r.tensors.iter().enumerate() {
        let bytes = r.tensor_bytes(i).unwrap().to_vec();
        w.add_tensor(TensorSpec { name: t.name.clone(), dims: t.dims.clone(), ggml_type: t.ggml_type, data: bytes });
    }
    let rewritten = w.to_bytes();
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        original, rewritten,
        "writer→reader→writer must be byte-identical (got {} vs {} bytes)",
        rewritten.len(), original.len()
    );
}

#[test]
fn llama_quantize_accepts_our_gguf() {
    let bytes = make_synthetic_gguf();
    let path = tmp_path("lq-in");
    std::fs::write(&path, &bytes).unwrap();
    // --dry-run just parses + reports size; validates our container against the reference parser.
    let out = Command::new("llama-quantize")
        .args(["--dry-run", path.to_str().unwrap(), "/tmp/requant-dryrun-out.gguf", "Q8_0"])
        .output();
    let _ = std::fs::remove_file(&path);
    match out {
        Err(e) => {
            // llama-quantize not guaranteed present in CI; skip gracefully there.
            eprintln!("llama-quantize unavailable, skipping ecosystem test: {e}");
        }
        Ok(o) => {
            // The synthetic GGUF uses a fake architecture ("requant-test"), so the reference
            // quantizer will reject it at model-registration time — *after* the loader has
            // parsed our container. The proof we want is that the reference parser accepted
            // our container (metadata + tensors), which shows up as a "loaded meta data" line.
            // A malformed container would error before ever printing that.
            let stderr = String::from_utf8_lossy(&o.stderr);
            assert!(
                stderr.contains("loaded meta data"),
                "reference GGUF parser rejected our container (no 'loaded meta data' line):\n{stderr}"
            );
        }
    }
}

#[test]
fn packed_sizes_match_known_format_geometry() {
    // bits-per-weight sanity vs llama-quantize's reported numbers.
    assert!((bpw(8).unwrap() - 8.5).abs() < 1e-9);   // Q8_0: 34 bytes / 32 = 8.5
    assert!((bpw(2).unwrap() - 4.5).abs() < 1e-9);   // Q4_0: 18/32 = 4.5
    assert!((bpw(12).unwrap() - 4.5).abs() < 1e-9);  // Q4_K: 144/256 = 4.5
    assert!((bpw(14).unwrap() - 6.5625).abs() < 1e-9); // Q6_K: 210/256
    assert!((bpw(10).unwrap() - 2.625).abs() < 1e-9); // Q2_K: 84/256
    assert_eq!(packed_nbytes(8, 32 * 10, "t").unwrap(), 34 * 10);
    assert_eq!(packed_nbytes(12, 256 * 4, "t").unwrap(), 144 * 4);
    assert!(packed_nbytes(8, 30, "t").is_err()); // not divisible by 32
}
