# requant

A fast, MoE-aware GGUF requantization tool with a recipe **search loop** that finds
size↔quality-optimal quant recipes without recomputing the calibration artifact.

The naive reading of "requantize quickly" is "make the rounding step fast." That's the
wrong target — round-to-nearest over a GGUF is I/O-bound. The two things that actually
make quant iteration slow are (1) recomputing the calibration artifact (imatrix / Hessian)
on every knob change, and (2) re-reading the fp16 source off disk. `requant` splits the
pipeline at the calibration boundary so every recipe experiment is the cheap projection
step over a memory-mapped source, and closes the loop with a per-layer sensitivity-driven
bit-allocation search.

See the [design doc](DESIGN.md) for the full thesis, math, and architecture.

## Status

**Implemented and verified:**
- **Qwen3.5-family dense-to-MoE warm-start conversion** (`moefy-qwen38`), including Qwen3.8-27B:
  streaming sharded safetensors rewrite into native Qwen3.5-MoE packed experts for both text-only
  and multimodal model classes. Full/linear/hybrid attention, GQA/MHA, vision, and MTP state pass
  through byte-for-byte; config/tensor mismatches and other model families fail closed. Includes
  deterministic router initialization, expected-output scaling, config migration, and a fail-loud
  `requires_training` manifest. Tests use tiny synthetic checkpoints only; see
  [the conversion guide](docs/QWEN38_MOE.md).
- **Qwen3.5-MoE inference packs** (`pack-qwen35`): quantize a Qwen3.5-MoE checkpoint in place in
  sharded safetensors — routed experts to Q4_K, attention/embeddings/LM head to Q6_K, the router
  and norms left BF16 — with a `minglang_pack.json` sidecar carrying each tensor's logical shape
  and block format (safetensors has no dtype that means "Q4_K"). Row lengths that are not
  block-aligned take the `llama-quantize` fallback rule rather than failing: `moe_intermediate_size
  = 544` puts `down_proj` on Q5_0. On Qwen3.8-27B this is 55.6 GB -> 19.6 GB, which is what moves
  the model into a 23 GB box's page cache. Streams row-chunk at a time, so the tool's own RSS never
  competes with the cache it is making room for.
- GGUF v3 read/write (mmap + streaming), sharded safetensors read, and role-tagged IR with MoE
  detection. Dense-to-MoE conversion writes sharded safetensors directly.
- Every quantized tensor type in the current GGML type table: Q1/Q2/Q4/Q5/Q8, all K-quants,
  all stored I-quants, TQ1/TQ2 ternary, MXFP4, and GGUF NVFP4. The established legacy/K/I
  kernels are **bit-exact with the pinned ggml oracle**; newer post-oracle types have synthetic
  layout/quantize/dequantize coverage. See the complete matrix below.
- imatrix-weighted scale search (the `make_qp_quants` super-block path) — bit-exact with
  ggml's imatrix kernels.
- Block-alignment fallback matching `llama-quantize` (Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0, …)
  so models with non-256 head dims (e.g. Qwen2.5-0.5B, hidden=896) quantize correctly.
- Per-role MoE-aware default recipe (router/lm_head/norm protected; routed experts
  aggressive; attention/shared-expert/deep-down-proj a notch higher).
- Recipe **search** under a byte budget: greedy-marginal over the *lower convex hull* of each
  tensor's cost curve (optimal for the LP relaxation, one-item integrality gap), recipe bits
  treated as a hard quality floor, emits a reproducible `[[rule]]` recipe. `--pareto` prints
  the whole size↔cost front in one run.
- **Sensitivity harness** (`requant sensitivity`): per-tensor `(bits → ΔKL)` curves by
  ablation — quantize one role/tensor at a time, leave the rest at fp16, measure the KL it
  induces. Feeds `search --sensitivity`, replacing the proxy with measurement.
- `eval` via `llama-perplexity`, with `--kl` for the far more sensitive KL comparison;
  `inspect` for tensor/role/MoE introspection.
- **Block-scaled floats**: FP4 (E2M1), FP8 (E4M3/E5M2), MXFP4, NVFP4, MXFP8. GGUF MXFP4 and
  NVFP4 are readable/writable; checkpoint layouts have safetensors readers plus grouped library
  output for vLLM/ModelOpt sidecars (see `crates/requant-quant/src/mxfp.rs` for conventions).
- Precision **floors** (`[floors]`): the router floor generalised to "no dense-path tensor
  below X", enforced against hand-written and search-emitted recipes alike.

**Not implemented:** GPTQ/AWQ algorithms, NF4, HQQ, EXL2, activation quantization, and
rotation/incoherence transforms. `QuantMethod::Gptq`/`Awq` are reserved recipe vocabulary, not
working kernels. The GGUF MXFP4 emit path has not yet been checked against ggml's
`quantize_row_mxfp4_ref` (the `ggml-oracle-mxfp4` feature exists for it); treat its packed-byte
identity as unverified until that optional oracle is run.

## Quantization and format support

The names below are the exact spellings accepted by recipe `bits = "..."`, explicit search
ladders, and the library `Bits::from_name` parser. “GGUF R/W” means requant can read, dequantize,
quantize, and write that tensor type in a GGUF. BPW includes block metadata.

| family | available names | BPW | GGUF R/W | calibration | verification/status |
|---|---|---:|:---:|---|---|
| float | `F32`, `F16`/`FP16`, `BF16` | 32, 16, 16 | yes | none | direct conversion/copy |
| standard Q | `Q1_0`, `Q2_0` | 1.125, 2.25 | yes | none | current GGML reference ports; synthetic tests |
| legacy Q | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` | 4.5, 5, 5.5, 6, 8.5, 9 | yes | none | packed-byte oracle coverage |
| K-quant | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` | 2.625, 3.4375, 4.5, 5.5, 6.5625 | yes | optional imatrix | packed-byte oracle coverage for RTN and weighted kernels |
| K intermediate | `Q8_K` | 9.125 | yes | none | tensor R/W + kernel support; GGML uses it internally, not as a whole-model preset |
| I-quant | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `IQ4_NL` | 1.5625–4.5 | yes | required for `IQ1_S`, `IQ2_XXS`, `IQ2_XS`; optional for the rest | packed-byte/dequant oracle coverage |
| ternary | `TQ1_0`, `TQ2_0` | 1.6875, 2.0625 | yes | none | current GGML reference ports; synthetic tests |
| GGUF block float | `MXFP4`, `NVFP4_GGUF`/`GGML_NVFP4` | 4.25, 4.5 | yes | none | synthetic layout/kernel tests; see MXFP4 oracle caveat above |
| checkpoint block float | `NVFP4`, `MXFP8` | 4.5, 8.25 | no | optional imatrix for NVFP4 scale refinement | library grouped output with sidecar scales; safetensors-oriented |
| checkpoint dense float | `FP8_E4M3`/`FP8`, `FP8_E5M2` | ~8 | no | none | library grouped output with per-channel scale sidecar |

`NVFP4_GGUF` and `NVFP4` deliberately have different names. Current GGML type 40 stores four
UE4M3 scales inside each 64-weight block. ModelOpt/vLLM checkpoint NVFP4 stores E2M1 weights,
an E4M3 `weight_scale`, and an additional FP32 `weight_scale_2` sidecar. Treating them as one
layout would silently produce incorrect weights.

The `full` search ladder contains every normal low-bit integer/codebook/ternary target from
`Q1_0` through `Q8_0`; `kquant`, `iquant`, and `blockfloat` select family-specific ladders. Any
accepted names can also be supplied as a comma-separated custom ladder. `Q8_K` is omitted from
automatic ladders because it is an intermediate block, while `Q8_0` is the normal high-quality
endpoint.

Container and checkpoint formats:

| container/layout | read | write | notes |
|---|:---:|:---:|---|
| GGUF v3 | yes | yes | mmap reader, regular and streaming writers, quant→quant source dequantization |
| single/sharded safetensors | yes | converter only | mmap reader and index resolution; `moefy-qwen38` writes native sharded output |
| safetensors scalar dtypes | yes | passthrough in converter | `BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`, `F16`, `BF16`, `F32`, `F64`, `F8_E4M3`, `F8_E5M2` |
| ModelOpt/vLLM grouped weights | yes via layout helpers | library output | NVFP4 `(weight, weight_scale, weight_scale_2)` and FP8 `(weight, weight_scale)` |

The compatibility baseline follows the current upstream
[`enum ggml_type`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h),
[`ggml-common.h` block layouts](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h),
and [`ggml-quants.c` reference kernels](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).

## Quickstart

```bash
# Validate a Qwen3.8-27B dense -> 32-expert/top-4 MoE conversion plan without writing weights:
requant moefy-qwen38 --input-dir Qwen3.8-27B --output-dir Qwen3.8-27B-MoE \
    --experts 32 --top-k 4 --dry-run

# What's in the model — MoE detection, per-tensor role/depth/quantizable:
requant inspect --input model.gguf

# Quantize with the built-in MoE-aware recipe (optionally imatrix-weighted):
requant quantize --input model-f16.gguf --output model-q4k.gguf [--imatrix imatrix.dat]

# Search for the best recipe under a byte budget, then quantize with it:
requant search --input model-f16.gguf --budget 400M --imatrix imatrix.dat --output recipe.toml
requant quantize --input model-f16.gguf --output model.opt.gguf --recipe recipe.toml --imatrix imatrix.dat

# The whole size<->quality front in one run (no budget needed):
requant search --input model-f16.gguf --imatrix imatrix.dat --pareto

# Import a llama.cpp imatrix into the content-addressed cache, then evaluate:
requant imatrix ...
requant eval --quant model.opt.gguf --reference model-f16.gguf --kl
```

### Closing the loop with measured sensitivity

The search runs on a free proxy (imatrix-weighted round-trip error) by default. To allocate
against measured ΔKL instead:

```bash
# Ablate one role at a time against the fp16 reference. ~10 candidate models at role
# granularity; use --grouping tensor (and raise --max-candidates) for finer curves.
requant sensitivity --input model-f16.gguf --corpus wiki.test.raw -o sens.json \
    --grouping role --imatrix imatrix.dat

requant search --input model-f16.gguf --budget 400M --sensitivity sens.json -o recipe.toml
```

**Before you commit to this path, check that something can run the architecture.** The
harness needs logits, and it gets them by running the model. If `llama.cpp` supports the
architecture, the above works as written. If it does not — a model with novel compressed
attention won't be supported on day one, and porting that attention just to eval it is a
model-porting project, not an eval project — use the external route instead: `requant
sensitivity --logits-dir dumps/` writes the candidates, you run each through the stack that
*does* implement it (vLLM), drop the logit dumps in `dumps/`, and the KL is computed here.
Same table, same allocator, no architecture work. See
`crates/requant-eval/src/evaluator.rs` for the full decision.

## Bit-exactness verification

The oracle tests link homebrew `libggml-base` (ggml 0.11.0) and compare our packed bytes
tensor-by-tensor against ggml's reference kernels:

```bash
cargo test --release -p requant-quant --features ggml-oracle
```

45 tests cover legacy, all five output K-quants, all nine I-quants, reference dequantization,
RTN/null-weight behavior, imatrix-weighted paths, multi-block, and multi-row cases. Without
`--features ggml-oracle` the oracle tests are skipped (the feature gates the FFI to libggml).

## Validation: Qwen2.5-0.5B

Same fp16 source, 1 MB wikitext corpus (445 chunks, `n_ctx=512`), measured with
`llama-perplexity` on Apple M3 Pro (Metal). fp16 reference PPL = **16.2491 ± 0.15**.

| recipe | size | PPL | vs fp16 |
|---|---|---|---|
| `llama-quantize q4_k` (reference, no imatrix) | 373.7 MiB | 16.7279 | +2.95% |
| `requant` default recipe + imatrix | 375.5 MiB | 16.7162 | +2.88% |
| `requant` search @ 400 MiB budget + imatrix | 381.6 MiB | 16.5269 | +1.71% |
| `requant` search @ 450 MiB budget + imatrix | 428.5 MiB | 16.3753 | +0.78% |

Two things to read from this honestly:

- **At the Q4_K point, `requant` matches `llama-quantize`.** 16.7162 vs 16.7279 is within the
  ±0.15 perplexity uncertainty — as it must be, since the k-quant kernels are byte-exact with
  ggml and Q4_K on a 0.5B dense model is near-lossless. `requant` defaults to imatrix-weighted
  scale search; `llama-quantize q4_k` here is unweighted (its default). The nominal edge is the
  imatrix contribution, small in this near-lossless regime.
- **The search walks a monotone Pareto front.** 16.72 → 16.53 → 16.38 as the budget relaxes,
  converging toward the fp16 floor. The greedy-marginal loop spends the next byte where it most
  reduces per-tensor weighted error — the capability `llama-quantize`'s fixed presets don't
  offer. The differentiator scales with model size and (especially) MoE routing structure,
  where per-role policies (protect the router, aggressive routed experts) are the real lever.

## Sub-4-bit MoE recipes (NVFP4 targets)

`recipes/v4-flash-nvfp4.toml` is the worked example: routed experts in NVFP4 with a sub-4-bit
cold tier through the middle of the stack, everything always-on (attention, router, shared
expert, MTP head, embeddings, LM head) pinned at FP8, and a `[floors]` block that makes
"nothing on the dense path below FP8" a constraint rather than a convention.

One caveat that no amount of bit allocation fixes: applying this to an NVFP4 checkpoint is
quant→quant *inside* the FP4 family, with no higher-precision master underneath. DESIGN §4
says to warn loudly about that, and `requant quantize` does. The tool will place the remaining
bits well; it cannot recover signal that was never stored. Whether the result holds task
quality is an empirical question — which is why the sensitivity harness is step one and not an
afterthought.

## Crates

`requant-io` · `requant-quant` · `requant-calib` · `requant-eval` · `requant-search` ·
`requant-cli`
