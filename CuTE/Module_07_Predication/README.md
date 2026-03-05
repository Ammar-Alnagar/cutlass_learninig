# Module 07: Predication — Handling Irregular Tensor Dimensions

## What This Module Teaches

Production kernels must handle tensor dimensions that are **not divisible by tile size**. This module teaches predication patterns that separate demo kernels from production-ready inference kernels:

1. **Predicated Copy**: `copy_if(tiled_copy, pred, src, dst)` — only copies where predicate is true
2. **Irregular Tile GEMM**: Two-path pattern (full tiles vs partial tiles)
3. **Variable Sequence Length**: Causal mask + OOB predication combined

## Why Predication Matters for LLM Inference

**Job mapping:**

- **Cerebras Performance Engineer** — "Production-quality kernel optimization" — Real models have arbitrary sequence lengths. A GEMM that only works on 128-divisible dimensions is not production.
- **NVIDIA DL Software Engineer** — "Efficient attention kernels for arbitrary sequence lengths" — FlashAttention-2 must handle `seqlen=97`, `seqlen=2047`, etc.

**The problem:** Tile size is 64, but sequence length is 97.
- Tile 0: elements 0-63 (full)
- Tile 1: elements 64-96 (partial, only 33 valid elements)
- Elements 97-127: out-of-bounds, must be masked

Without predication, you access illegal memory. With predication, you handle boundary tiles correctly.

## Prerequisites

- Module 01: Layouts (`make_layout`, `shape`, `stride`)
- Module 02: Tensors (`make_tensor`, `local_tile`)
- Module 03: TiledCopy (`make_tiled_copy`, `cp.async`)
- Module 04: TiledMMA (`make_tiled_mma`, `gemm`)

## The Predication Mental Model

```
For each element (i, j) in a tile:
  predicate(i, j) = (i < M && j < N)  // bounds check

copy_if(copy_op, pred, src, dst):
  for each thread:
    if pred[thread_id]:
      dst[thread_id] = src[thread_id]
    else:
      dst[thread_id] = 0  // or unchanged
```

**Key pattern:** Clear SMEM before predicated copy — zero-fill first, then `copy_if`, so unused lanes hold `0.0f` not garbage.

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_predicated_copy.cu | `copy_if` with bounds predicate | Variable-length tensor copy |
| ex02_irregular_tile_gemm.cu | Two-path GEMM (full vs partial tiles) | Padding vs predication tradeoff |
| ex03_fa2_variable_seqlen.cu | Causal mask + OOB predication | FlashAttention-2 with arbitrary seqlen |

## Exit Criteria

Before moving to Module 08, you must be able to:

1. Construct a predicate tensor using `make_identity_tensor` and bounds checking
2. Implement `copy_if` for a 1D tensor with irregular last tile
3. Run both padded and predicated GEMM versions and verify numerical equivalence
4. Combine causal mask + OOB predication in a single predicate tensor for FlashAttention-2

## Nsight Metrics

After each exercise, check:

- **nsight compute**: `nvtx` ranges to isolate predicated regions
- **Memory**: `l1tex__data_pipe_lookup_misc` — should show no illegal accesses
- **Stall**: `sm__inst_executed_pipe_tex_throttle` — predication may cause divergence

## Common Mistakes

1. **Forgetting to clear SMEM:** Before `copy_if`, zero-fill SMEM. Otherwise, unused lanes hold garbage from previous iterations.

2. **Wrong predicate shape:** Predicate must match the thread layout of the copy operation, not the tensor shape.

3. **Causal mask + OOB confusion:** In FlashAttention-2, causal mask (`i < j`) and OOB mask (`j < seqlen_kv`) are AND'd together. Both must be true for a valid attention score.

4. **Performance regression:** Predication adds branches. For boundary tiles, this is necessary. But don't use predication for full tiles — use the fast path.

---

Next: **ex01_predicated_copy.cu** — your first predicated copy.
