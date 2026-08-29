# Qwen3.8-27B dense-to-MoE warm start

`requant moefy-qwen38` restructures the official BF16/F16/F32 Hugging Face checkpoint into the
native packed-expert layout used by Transformers' `Qwen3_5MoeForConditionalGeneration`. It does
not download weights, and it does not pretend that structural conversion alone produces a useful
MoE. The output is a training warm start.

The converter follows the official architecture definitions:

- [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json): 64 text
  layers, hidden size 5120, dense intermediate size 17408, hybrid linear/full attention, vision,
  and one MTP layer.
- [Transformers Qwen3.5-MoE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py): packed
  `experts.gate_up_proj`, packed `experts.down_proj`, a top-k router, and a shared expert.

## What the conversion does

With the recommended `32` experts and `top-k=4`:

- Each 17,408-wide dense SwiGLU MLP is split into 32 contiguous, 544-wide experts. No dense FFN
  neuron is dropped or duplicated, so total routed-FFN parameter count stays approximately flat.
- Gate and up rows are packed as `[32, 1088, 5120]`; down columns are packed as
  `[32, 5120, 544]`, matching the Transformers v5 Qwen3.5-MoE format.
- Expert down projections are multiplied by 32. Under balanced routing, this preserves the dense
  FFN output in expectation despite only a subset of the neuron partitions being selected.
- Every router is initialized with a deterministic small random matrix. This avoids the permanent
  top-k tie/collapse caused by an all-zero router.
- A width-1, zero-valued shared expert is added to satisfy the native architecture without changing
  the initial residual stream.
- The MTP MLP is converted too. Attention, linear-attention, vision, norms, embeddings, LM head,
  and other MTP tensors are copied byte-for-byte.
- `config.json` is changed from `qwen3_5` / `qwen3_5_text` to `qwen3_5_moe` /
  `qwen3_5_moe_text`, with router auxiliary loss output enabled.
- `requant_moe_conversion.json` records the exact conversion and explicitly marks the output
  `inference_ready: false` and `requires_training: true`.

At top-k 4, the routed FFN activates 2,176 intermediate units per token, 12.5% of the original
dense width. That is the intended sparsity target, not a claim of zero-shot quality preservation.

## Usage

Start with a plan-only validation. This opens headers and validates every expected tensor and
shape, but writes nothing:

```bash
cargo run --release -p requant-cli -- moefy-qwen38 \
  --input-dir /models/Qwen3.8-27B \
  --output-dir /models/Qwen3.8-27B-MoE-warmstart \
  --experts 32 \
  --top-k 4 \
  --dry-run
```

Then run the streaming conversion:

```bash
cargo run --release -p requant-cli -- moefy-qwen38 \
  --input-dir /models/Qwen3.8-27B \
  --output-dir /models/Qwen3.8-27B-MoE-warmstart \
  --experts 32 \
  --top-k 4 \
  --max-shard-size 5G
```

The output directory must not already exist. Existing files are never overwritten. Use the BF16
source checkpoint, not an FP8 or other quantized derivative; the converter rejects quantized MLP
tensors.

## Training handoff

The first training stage should distill from the original dense Qwen3.8-27B teacher while keeping
the copied attention/SSM/vision path frozen. Optimize the routers and expert MLPs with both a
teacher-matching objective and router load balancing. Monitor per-expert token counts, router
entropy, dropped/overflow tokens, hidden-state error at every converted MLP, and end-to-end KL.

Only after routing is balanced and validation quality has recovered should the dense path be
unfrozen or the checkpoint be quantized. The generated checkpoint is structurally loadable, but
running it as a finished inference model before that training step is expected to degrade quality.

## Weight-free verification

The repository tests construct tiny synthetic sharded safetensors checkpoints and verify:

- config migration and MTP conversion;
- packed gate/up and down layouts byte-for-byte;
- deterministic nonzero routers and zero shared experts;
- down-projection scaling;
- passthrough tensor preservation, sharding/index generation, dry-run behavior, and validation
  failures.

Run them with:

```bash
cargo test -p requant-cli moefy
```
