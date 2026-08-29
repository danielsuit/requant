# `requant`: a fast, MoE-aware, all-format requantization tool

*A design and implementation plan.*

---

## 0. Design thesis

The naive reading of "requantize quickly" is "make the rounding step fast." That's the wrong target — plain round-to-nearest (RTN) over a GGUF is I/O-bound, not compute-bound. The two things that actually make requant iteration slow are:

1. **Recomputing the calibration artifact** — the importance matrix (imatrix) for llama.cpp k/i-quants, or the layer-wise Hessian / activation statistics for GPTQ/AWQ. These require forward passes over calibration data and dominate wall-clock.
2. **Re-reading the fp16 source** off disk on every knob change.

So the valuable artifact is not a faster quantizer, it's a **requant *search loop* that never recomputes what didn't change.** The whole design falls out of four decisions:

- **Split the pipeline at the calibration boundary.** The calibration artifact is a function of `(model, calibration_set)` *only* — completely independent of target bit allocation. Compute once, cache content-addressed, then every recipe experiment is the cheap projection step over a memory-mapped source.
- **Be MoE-role-aware.** On a 744B MoE the routed experts dominate parameter count but each is sparsely activated and tolerates aggressive quantization; attention, the shared expert, embeddings, norms, and especially the **router/gate logits** do not. Per-role policies are the single biggest quality-per-bit lever and are absent from turnkey tools.
- **Parallelize on the natural grain.** Experts are independent (fan across devices); k-quant super-blocks are independent (rayon within a tensor). The secondhand-GPU zoo becomes a quant fabric.
- **Close the loop with a fast eval.** Per-layer KL divergence against fp16 reference logits turns "requantize quickly" into "find the Pareto-optimal recipe quickly," which is what you actually want.

---

## 1. Background: the quantization landscape

It helps to see the whole field as a small number of orthogonal ideas that formats mix and match. Everything below reduces to: *choose a grid, choose what error metric to minimize when snapping to it, and optionally transform the space first so the snapping hurts less.*

### 1.1 The error-minimization lineage (Hessian-based)

The foundational idea is second-order: the sensitivity of the loss to perturbing a weight is governed by the Hessian.

- **Optimal Brain Damage** — LeCun, Denker & Solla, 1990. Diagonal-Hessian saliency for pruning.
- **Optimal Brain Surgeon (OBS)** — Hassibi & Stork, 1993. Full-Hessian version with the closed form optimal *compensating update* to the remaining weights when one is removed. This update equation is the mathematical core of everything modern.
- **Optimal Brain Compression / OBQ** — Frantar & Alistarh, 2022. Reframes OBS from pruning to *quantization*: greedily quantize weights one at a time, each time applying the OBS optimal update to compensate. Exact but O(d³·d_row) — too slow for LLMs as-is.
- **GPTQ** — Frantar, Ashkboos, Hoefler & Alistarh, 2022/23. Makes OBQ tractable at LLM scale via three tricks (fixed column order shared across rows, Cholesky reformulation of the inverse-Hessian updates, and lazy batched updates). This is the workhorse behind most 4-/3-bit weight-only quants.

### 1.2 The activation-aware / equivalent-transform lineage

Instead of (or before) minimizing error, reshape the problem so outliers stop dominating.

- **LLM.int8()** — Dettmers, Lewis, Belkada & Zettlemoyer, 2022. Identifies a handful of emergent outlier *feature dimensions* and keeps them in fp16 while int8-ing the rest (mixed-precision decomposition).
- **SmoothQuant** — Xiao, Lin, Seznec, Wu, Demouth & Han, 2022/23. Migrates quantization difficulty from activations to weights with a per-input-channel scale, so both become quantizable.
- **AWQ** — Lin, Tang, Tang, Yang, Dang & Han, 2023. Protects the ~1% of "salient" weight channels identified by *activation* magnitude by scaling them up before quantizing (grid-searched per-channel scale), folding the inverse scale into the preceding op.
- **OmniQuant** — Shao et al., 2023. Makes the clipping thresholds (learnable weight clipping) and the equivalent transforms (learnable equivalent transformation) *learned* by block-wise gradient descent rather than grid-searched.

### 1.3 The rotation / incoherence lineage

Multiply weights and activations by paired orthogonal matrices `Q`, `Qᵀ` (identity in exact arithmetic) chosen to spread outliers and make the distribution incoherent / near-Gaussian, so a uniform grid fits well.

- **QuIP** — Chee, Cai, Kuleshov & De Sa, 2023. Incoherence processing + adaptive (LDLQ) rounding; first credible 2-bit.
- **QuIP#** — Tseng, Chee, Sun, Kuleshov & De Sa, 2024. Randomized Hadamard incoherence + a lattice codebook on the E8 lattice (vector quant) + fine-tuning.
- **QuaRot** — Ashkboos, Mohtashami, Croci, et al., 2024. Hadamard rotations that give end-to-end 4-bit including activations and KV cache, exploiting computational invariance.
- **SpinQuant** — Liu et al., 2024. Learns the rotation matrices instead of using random Hadamards.

### 1.4 The codebook / non-uniform lineage

Drop the uniform grid entirely.

- **SqueezeLLM** — Kim, Hooper, Gholami, et al., 2023. Sensitivity-weighted k-means for a non-uniform codebook, plus a dense-and-sparse split keeping outliers sparse.
- **SpQR** — Dettmers, Svirschevski, Egiazarian, et al., 2023. "Sparse-quantized representation": near-lossless by isolating outlier weights in a sparse high-precision side-channel.
- **AQLM** — Egiazarian, Panferov, Kuznedelev, et al., 2024. Additive (multi-codebook) vector quantization pushing toward 2 bits.
- **QTIP** — Tseng, Sun, Hou & De Sa, 2024. Trellis-coded quantization with incoherence — high-dimensional VQ at practical decode cost.
- **QLoRA / NF4** — Dettmers, Pagnoni, Holtzman & Zettlemoyer, 2023. The NormalFloat4 information-theoretically-motivated non-uniform 4-bit grid for normally-distributed weights, plus double quantization of the scales.
- **HQQ** — Badri & Shaji, 2023. Half-quadratic solver with a sparsity-promoting (Lp) loss on the quant error; robust to outliers with **no calibration data**, very fast.

### 1.5 The GGUF native formats (llama.cpp)

Not academic papers but the formats you must match bit-exactly. Primary author is Iwan Kawrakow (ikawrakow); documented in llama.cpp PRs/discussions rather than a paper.

- **Legacy quants**: `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` — simple per-block (32) scale (+min).
- **k-quants**: `Q2_K … Q6_K` — hierarchical super-blocks of 256 with sub-block scales that are themselves quantized. The quality workhorses.
- **i-quants**: `IQ1_S … IQ4_XS` — codebook-based (lattice-like) grids for very low bit-rates, and the reason a from-scratch tool is real work.
- **imatrix**: the importance matrix — per-input-channel mean of squared activations. As shown in §2.4, this is exactly the **diagonal of the GPTQ Hessian**, which unifies the GGUF world with the academic one.

### 1.6 Block-scaled float formats

- **FP8** (e4m3 / e5m2) — per-tensor / per-channel / blockwise scaling.
- **Microscaling (MX): MXFP4 / MXFP6 / MXFP8** — OCP Microscaling Formats spec, 2023. A shared power-of-two block scale (typically block=32) over low-bit floats. Increasingly the hardware-native path.

---

## 2. The math you actually implement

This section is the "how," in equations, for the paths worth building first.

### 2.1 Round-to-nearest (RTN), the baseline

For a block of weights `x` with symmetric range, scale `d = max|x| / q_max`, and `q_i = clamp(round(x_i / d), -q_max, q_max)`. Asymmetric adds a min/zero-point. This is the reference every other method is trying to beat, and the round-trip test target for format correctness.

### 2.2 The OBS / GPTQ core

Consider one linear layer, weight `W` (out × in), calibration inputs `X` (in × N). The layer-wise objective is

```
minimize  || W X − Ŵ X ||_F²
```

For a single output row `w` (length `in`), the error is a quadratic form `(w − ŵ) H (w − ŵ)ᵀ` with **Hessian**

```
H = 2 X Xᵀ            (in × in)
```

**OBS optimal update.** When you fix weight index `q` to its quantized value `quant(w_q)`, the error-minimizing adjustment to the *remaining* free weights is

```
δw  = − (w_q − quant(w_q)) / [H⁻¹]_qq  ·  H⁻¹[:, q]
```

and the resulting error increase is `(w_q − quant(w_q))² / [H⁻¹]_qq`. OBQ greedily picks the `q` minimizing that increase, applies `δw`, then removes row/col `q` from `H⁻¹`:

```
H⁻¹ ← H⁻¹ − (1 / [H⁻¹]_qq) · H⁻¹[:, q] · H⁻¹[q, :]
```

**GPTQ's three accelerations** over OBQ:

1. **Fixed order.** Quantize all rows in the *same* column order (left→right works as well as greedy for large layers). Now `H⁻¹` is shared across every row of the layer instead of being re-derived per weight.
2. **Cholesky reformulation.** The sequence of `H⁻¹` removals above is exactly a Cholesky factorization. Precompute `H⁻¹ = LLᵀ` once (with dampening below), and read the compensation directions off the Cholesky factor. This is faster and far more numerically stable than repeated rank-1 downdates.
3. **Lazy batch updates.** Apply the `δw` corrections in column blocks of ~128 so the work is a GEMM (compute-bound) rather than a stream of rank-1 updates (memory-bound).

**Dampening** for invertibility: `H ← H + λ · mean(diag(H)) · I`, typically `λ ≈ 0.01`.

Implementation sketch (per layer):

```
H = 2 X Xᵀ ; H += λ·mean(diag(H))·I
Hinv = cholesky_inverse(H)          # cache this — it's calibration-only
L = cholesky(Hinv)
for block of 128 columns:
    for col j in block:
        q_j = quantize_column(W[:, j])          # RTN into target grid
        err = (W[:, j] − q_j) / L[j, j]
        W[:, block after j] -= err ⊗ L[j, block after j]   # lazy, batched
    W[:, later blocks] -= Err_block ⊗ L[block, later blocks]
```

The point for us: **`Hinv` (and its Cholesky) is the cached calibration artifact.** Every recipe change re-runs only `quantize_column` into a different grid.

### 2.3 k-quant scale search (the GGUF path)

A k-quant block stores integer codes `q_i` plus a (quantized) scale `d` (and min for the asymmetric variants). The scale is chosen to minimize the **importance-weighted** reconstruction error:

```
minimize_d   Σ_i  g_i · (x_i − d · q_i)²      where q_i = round(x_i / d) clamped
```

with `g_i` the imatrix importance for that input channel (uniform if no imatrix). Because `q_i` is piecewise-constant in `d`, this isn't smooth — llama.cpp's `make_qx_quants` evaluates a **grid of candidate scales** around the RTN scale and keeps the weighted-RMSE minimizer. The hierarchical k-quant layout then quantizes the per-sub-block scales themselves against a super-block scale. Matching this exactly (candidate set, rounding, tie-breaks) is the bit-exactness challenge in §7.

### 2.4 Why imatrix *is* the diagonal Hessian

The imatrix entry for input channel `i` is `g_i = (1/N) Σ_n X[i,n]²`, i.e. the `i`-th diagonal of `X Xᵀ` up to scale — the diagonal of the GPTQ Hessian `H`. So **imatrix-weighted k-quant = diagonal-Hessian-weighted RTN**, and full GPTQ = the same objective with the off-diagonal (cross-channel) terms retained via the compensation updates. One `Stats` type in the cache serves both: store `X Xᵀ` (or its diagonal when memory-bound), and derive imatrix or full Hessian on demand.

### 2.5 AWQ per-channel scaling

Find a diagonal per-input-channel scale `s` so that `W' = W · diag(s)` and `X' = diag(1/s) · X` leave `W'X' = WX` unchanged, but `W'` quantizes better. AWQ parameterizes `s = (act_scale)^α · (weight_scale)^(−β)` and grid-searches `α, β ∈ [0,1]` to minimize the layer's quant error, then folds `diag(1/s)` into the preceding layernorm or linear. Cheap, calibration-light, composes cleanly with §2.2 (scale first, then GPTQ).

### 2.6 Rotation / incoherence (optional, phase 7)

Insert paired Hadamard matrices: `W X = (W H)(Hᵀ X)`. `H` is a normalized Walsh–Hadamard matrix applied in `O(n log n)` via the fast transform — no dense matmul. The rotation spreads outlier energy across channels so a uniform grid fits, and (QuaRot) lets you push activations/KV to 4-bit. For a *weight-only* requant tool this is a quality add-on, not core.

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────┐
                    │             Recipe (serde)               │
                    │  per-role / per-layer-range bit policy    │
                    └───────────────────┬─────────────────────┘
                                        │
  fp16 source ──► [Loader] ──► [IR: role-tagged tensor graph] ──► [Scheduler]
   (mmap)                                       │                     │
                                                ▼                     ▼
                                   [Calibration Cache]        fan tensors across
                                content-addressed, mmap        devices; rayon within
                                (XXᵀ / diag / Hessian)                │
                                                │                     ▼
                                                └──────────► [Quantizer trait impls]
                                                                      │
                                                                      ▼
                                                          [Writer] ──► GGUF / safetensors
                                                                      │
                                                                      ▼
                                                    [Eval loop: per-layer KL vs fp16]
                                                                      │
                                                                      ▼
                                                      [Bit-allocation search]
```

### 3.1 The calibration cache (first-class object)

- **Key**: `blake3(model_tensor_hashes ++ calib_set_hash ++ stat_type ++ stat_params)`. Content-addressed so a recipe change never invalidates it and a calib-set change invalidates exactly the affected entries.
- **Stat types**: `Diag` (imatrix; tiny, ~`in` floats per linear), `Gram` (full `XXᵀ`; `in²` — this is the memory sink, see §7), `ActScale` (AWQ/SmoothQuant per-channel max/mean).
- **Storage**: one memory-mapped file per (layer, stat_type), little-endian `f32`, with a small header (shape, dtype, params). `memmap2` for zero-copy reads on the projection pass.
- **Producer**: a single instrumented forward pass over the calib set that hooks every linear's input and accumulates the requested stat online (Welford for means, running `XXᵀ` accumulation in fp32/fp64). This is the only expensive step and it runs **once per (model, calib-set)**.

### 3.2 The IR: a role-tagged tensor graph

Parse GGUF or safetensors into a flat tensor list, then tag each tensor with a **role** by matching name patterns and inspecting the config:

```
enum Role {
    Embedding, LmHead, Norm,
    AttnQ, AttnK, AttnV, AttnO,
    FfnGate, FfnUp, FfnDown,          // dense FFN
    Router,                           // MoE gate logits — protect hard
    SharedExpert(FfnPart),            // always-on expert
    RoutedExpert { idx: u32, part: FfnPart },
}
```

MoE detection: presence of a router/gate tensor + stacked expert tensors (GGUF packs experts into a single 3-D tensor `[n_expert, out, in]`; safetensors usually splits them). The IR normalizes both into a per-expert view so policies address experts uniformly.

### 3.3 The `Quantizer` trait

```rust
trait Quantizer {
    /// Which calibration statistic this method needs.
    fn required_stat(&self, policy: &Policy) -> StatKind;
    /// Quantize one tensor into the target format's packed bytes.
    fn quantize(&self, w: TensorView<f32>, calib: Option<CalibView>, policy: &Policy)
        -> QuantTensor;
    /// Dequantize for eval / quant-from-quant source reads.
    fn dequantize(&self, q: &QuantTensor) -> Tensor<f32>;
}
```

One impl per format family (`KQuant`, `IQuant`, `Gptq`, `Awq`, `Nf4`, `Fp8`, `Mxfp`). RTN-vs-GPTQ is a *policy* flag inside a family, not a separate impl, so you can A/B them over the same cache.

### 3.4 The recipe / policy language

A serde config that a human edits and the search loop writes:

```toml
[defaults]
method = "kquant"          # kquant | iquant | gptq | awq | nf4 | fp8 | mxfp
bits   = "Q4_K"

[[rule]]  # protect the router everywhere
role  = "router"
bits  = "F16"

[[rule]]  # attention a notch higher than experts
role  = ["attn_q","attn_k","attn_v","attn_o"]
bits  = "Q6_K"

[[rule]]  # routed experts aggressive
role   = "routed_expert"
bits   = "Q4_K"

[[rule]]  # ...except deep-layer down-projections, which hurt most
role       = "routed_expert.down"
layer      = ">= 0.75"     # fractional depth
bits       = "Q5_K"

[[rule]]
role = ["embedding","lm_head","norm"]
bits = "F16"
```

Rules are last-match-wins over `(role, layer_range, expert_range)`. This is where the MoE-awareness lives.

### 3.5 The scheduler

- **Outer parallelism (across devices)**: a work queue of `(tensor, policy)` jobs. Experts and independent linears are dispatched to whichever device is free. VRAM-tier aware: big-Hessian GPTQ jobs → the M40 (24 GB); light RTN/k-quant jobs → CPU or the small cards. Work-stealing so the eBay-heterogeneous pool self-balances.
- **Inner parallelism (within a tensor)**: k-quant super-blocks and GPTQ column-blocks are independent → `rayon` `par_chunks`. A single Q6_K tensor saturates all cores without touching a GPU.
- CPU-only is a first-class mode; GPU only accelerates the Hessian/Cholesky-heavy GPTQ path.

### 3.6 The eval loop

- Run a few hundred calibration tokens through both the fp16 reference and the candidate quant, capturing logits.
- **Global metric**: KL(fp16 ‖ quant) on next-token distributions + perplexity. KL is more sensitive than PPL and cheaper to trust on small samples.
- **Per-layer sensitivity**: quantize one layer/role at a time (rest fp16), measure the KL it induces. This gives a per-tensor "cost of quantizing" curve that the search consumes.

### 3.7 Bit-allocation search

With per-tensor `(bits → ΔKL)` curves and per-tensor byte cost, choosing a recipe under a size budget is a **knapsack / marginal-analysis** problem:

- **Greedy marginal**: start everyone at the cheapest acceptable grid; repeatedly spend the next bit where `ΔKL_reduction / Δbytes` is largest until the budget is hit. Cheap, near-optimal in practice.
- **LP relaxation** for a principled Pareto front when you want it.

This is the step that turns the tool from "quantizer" into "optimizer," and it's only possible because §3.1 made recipe changes cheap.

---

## 4. Format support matrix

| Family | Formats | Calib stat | Notes / difficulty |
|---|---|---|---|
| GGUF legacy | Q4_0/1, Q5_0/1, Q8_0 | none | Trivial; do first for the round-trip harness. |
| GGUF k-quant | Q2_K…Q6_K | Diag (imatrix) | Core. Bit-exact match is the hard part (§2.3, §7). |
| GGUF i-quant | IQ1_S…IQ4_XS | Diag | Codebook/lattice grids; port ggml codebooks; hardest GGUF piece. |
| GPTQ | 2/3/4-bit, groupsize, act-order | Gram (Hessian) | §2.2. Safetensors + `quantize_config.json`. Hessian storage = risk. |
| AWQ | 4-bit, groupsize | ActScale | §2.5. Fold scales into prev op; emit AWQ packing. |
| bitsandbytes | NF4, FP4 + double-quant | none | NF4 grid is fixed (§1.4); double-quantize the block scales. |
| FP8 | e4m3, e5m2 | ActScale (opt) | Per-tensor/channel/block scale; hardware-native serving. |
| MX | MXFP4/6/8 | none | Shared pow2 block scale (block=32); align to OCP spec. |

**Quant→quant caveat (scope it honestly):** requantizing *from* an existing quant (e.g. Q8→Q4) is strictly worse than fp16→Q4 because information is already gone. Keep fp16 as source whenever available; support the quant→quant path only for the narrow "fp16 not on hand, degradation acceptable" case, and warn loudly.

**The severe case: no full-precision master at all.** A checkpoint that only ever shipped in NVFP4 has no fp16 anywhere in the chain, so a recipe over it is quant→quant *inside* the 4-bit regime. This is not a matter of degree — every other row in this table has a higher-precision source it could fall back to. Bit allocation still helps, because it decides where the remaining error lands; it cannot restore precision that was never stored, and no amount of search sophistication changes that. `requant quantize` prints a distinct, unmissable warning when it reads an FP4-family source, and the only thing that answers "does this still hold task quality" is measurement — which is why the eval harness (§3.6) is step one of this work and not an afterthought.

---

## 5. Rust crate ecosystem & repo layout

```
requant/
├── crates/
│   ├── requant-io       # GGUF + safetensors read/write, mmap loader, role tagging
│   ├── requant-quant    # Quantizer trait + one module per format family
│   │                    #   kquant/ (port of ggml-quants block kernels + scale search)
│   │                    #   iquant/  gptq/  awq/  nf4/  fp8/  mxfp/
│   ├── requant-calib    # instrumented forward pass, stat accumulation, cache
│   ├── requant-eval     # KL/PPL harness, per-layer sensitivity
│   ├── requant-search   # bit-allocation (greedy + LP)
│   ├── requant-sched    # device pool, work-stealing, VRAM tiers
│   └── requant-cli      # the `requant` binary
└── tests/
    └── bitexact/        # golden outputs vs llama.cpp (§7)
```

Dependencies: `gguf` (container), `safetensors`, `memmap2`, `rayon`, `half`, `serde`/`toml`, `ndarray` or raw slices, `blake3` (cache keys), `candle-core` (GGUF quant types, GPU tensors, and — if you want it — the forward pass for the calibration stats and eval). For the Hessian path, `candle` or `cudarc` for GPU Cholesky, or `faer` for a fast CPU dense linear-algebra fallback.

---

## 6. Implementation phases

- **Phase 0 — plumbing.** GGUF + safetensors read/write with a byte-exact round-trip test. Build the IR and role tagger; verify MoE detection on a real GLM-5.2 shard.
- **Phase 1 — RTN k-quants.** Implement Q4_K/Q5_K/Q6_K + legacy quants, RTN only. Gate on bit-exact match vs `llama-quantize` (§7). This proves the block kernels before any cleverness.
- **Phase 2 — imatrix cache + weighted k-quant.** Add the calibration forward pass, the `Diag` stat, the content-addressed cache, and imatrix-weighted scale search (§2.3). Match llama.cpp's imatrix output.
- **Phase 3 — eval harness.** KL/PPL + per-layer sensitivity (§3.6). Now you can *see* what a recipe costs.
- **Phase 4 — bit-allocation search.** Greedy marginal over the sensitivity curves (§3.7). This is the first moment the tool beats hand-tuning.
- **Phase 5 — multi-device scheduler.** Work-stealing across the GPU zoo + rayon within tensors (§3.5). Turns hours into minutes on big models.
- **Phase 6 — GPTQ + AWQ.** `Gram`/`ActScale` stats, Cholesky path, safetensors emit. The vLLM/SGLang serving target.
- **Phase 7 — i-quants, FP8, MX, rotations.** The long tail: codebook i-quants, block-float formats, optional Hadamard incoherence pre-processing.

---

## 7. The hard parts / risks

1. **Bit-exactness with llama.cpp** is the single biggest correctness risk. The k-quant scale search has specific candidate sets, rounding conventions, and sub-block scale quantization that must be replicated exactly or your GGUFs will silently underperform. Mitigation: golden-file tests in `tests/bitexact/` comparing your packed bytes against `llama-quantize` output tensor-by-tensor, added the moment each format lands.
2. **Hessian storage for full GPTQ.** `XXᵀ` is `in × in` per linear; for large hidden sizes that's hundreds of MB–GBs per layer, and you don't want it all resident. Mitigations: (a) compute and consume the Hessian per-layer streaming, never holding more than a few at once; (b) store only what a policy needs (`Diag` when the recipe is k-quant-only); (c) optional low-rank / blocked Hessian approximations.
3. **i-quant codebooks.** The lattice/codebook grids are the least documented part of GGUF. Budget real time to port the ggml codebooks and match them.
4. **Numerical stability.** Cholesky on near-singular Hessians needs the dampening term (§2.2); use fp32/fp64 accumulation for `XXᵀ`. Test on layers known to have outlier channels.
5. **Quant→quant degradation** (§4) — a correctness-of-expectations risk more than a code risk. Make it loud in the CLI.
6. **MoE role misdetection** silently wrecks quality if the router gets quantized. Add an assertion that any tensor tagged `Router` is never assigned below a configured floor.

   *Generalized (implemented as `[floors]`):* the router is the sharpest case of a broader rule — anything on the **dense path** is touched by every token, while a routed expert sees `expert_used / expert_count` of the traffic. So the floor extends to `dense_path`: attention, the shared expert, the MTP/nextn head, embeddings, and the LM head may not go below a configured precision (for an FP4-target recipe, FP8). Routed experts are the only role exempt, and that exemption is the entire reason a sub-4-bit MoE recipe can work at all. The check runs on every resolved policy, so it binds hand-written and search-emitted recipes identically — a search that could violate a floor is a search that will eventually violate a floor.

7. **The eval forward pass may not exist for your architecture.** §3.6's per-layer sensitivity needs logits, which means running the model. For anything in the GGUF ecosystem, `llama-perplexity` provides them and the loop is local. For a model with novel attention (V4-Flash-class CSA/HCA), neither llama.cpp nor candle is likely to implement it, and porting that attention *in order to eval it* is a model-porting project wearing an eval project's clothes.

   Mitigation, and the reason `requant-eval` is built around a portable logit-capture file rather than an in-process forward pass: the harness writes candidate models and scores them from **dumped logits**, wherever those were produced. Run each candidate through the stack that does implement the architecture (vLLM on the serving box), drop the dumps next to the reference capture, and the KL is computed here. Same table, same allocator, zero architecture work. Check architecture support *before* committing to the in-process path — it changes the shape of the loop (external scoring wants role-level grouping, not per-tensor).

---

## 8. Validation methodology

- **Bit-exactness**: golden tests vs `llama-quantize` per tensor (Phase 1+).
- **Fidelity**: KL(fp16 ‖ quant) and perplexity on a held-out wikitext-style slice; track the size↔KL Pareto front across recipes.
- **Downstream**: since this feeds Nightshift, the terminal metric is agentic — SWE-bench-style pass rate of the quantized student vs fp16 on a fixed task set. A recipe is "good" when it holds task pass-rate within tolerance at the smallest footprint.
- **Regression gate**: CI runs the round-trip + bit-exact tests on every format on every commit; fidelity/downstream run nightly on a fixed small model.

---

## Appendix: references

1. LeCun, Denker, Solla — *Optimal Brain Damage* (1990).
2. Hassibi, Stork — *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon* (1993).
3. Frantar, Alistarh — *Optimal Brain Compression* (OBQ) (2022).
4. Frantar, Ashkboos, Hoefler, Alistarh — *GPTQ* (2022/23).
5. Dettmers, Lewis, Belkada, Zettlemoyer — *LLM.int8()* (2022).
6. Xiao, Lin, Seznec, Wu, Demouth, Han — *SmoothQuant* (2022/23).
7. Lin, Tang, Tang, Yang, Dang, Han — *AWQ* (2023).
8. Shao et al. — *OmniQuant* (2023).
9. Chee, Cai, Kuleshov, De Sa — *QuIP* (2023).
10. Tseng, Chee, Sun, Kuleshov, De Sa — *QuIP#* (2024).
11. Ashkboos, Mohtashami, Croci, et al. — *QuaRot* (2024).
12. Liu et al. — *SpinQuant* (2024).
13. Kim, Hooper, Gholami, et al. — *SqueezeLLM* (2023).
14. Dettmers, Svirschevski, Egiazarian, et al. — *SpQR* (2023).
15. Egiazarian, Panferov, Kuznedelev, et al. — *AQLM* (2024).
16. Tseng, Sun, Hou, De Sa — *QTIP* (2024).
17. Dettmers, Pagnoni, Holtzman, Zettlemoyer — *QLoRA / NF4* (2023).
18. Badri, Shaji — *Half-Quadratic Quantization (HQQ)* (2023).
19. OCP — *Microscaling (MX) Formats Specification* (2023).
20. llama.cpp / ggml — k-quant & i-quant formats and imatrix (Kawrakow et al.; PRs/discussions, not a paper).
