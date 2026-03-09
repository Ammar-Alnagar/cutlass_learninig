# Module 07 — Sparse GEMM

## Overview

Sparse GEMM (SpGEMM) exploits structured sparsity patterns to reduce computation. NVIDIA GPUs support 2:4 structured sparsity natively on Ampere+.

## 2:4 Structured Sparsity

Pattern: 2 non-zero values in every 4 consecutive elements.

```
Dense:  [x x x x x x x x]  → 8 values, 8 multiplies
2:4:    [x x 0 0 x x 0 0]  → 4 values, 4 multiplies (50% reduction)
```

## Sparsity Formats

CUTLASS 3.x supports:
- **2:4 Structured**: Hardware-accelerated on Ampere+
- **Block Sparse**: Software-managed
- **Unstructured**: Limited support (irregular access patterns)

## Performance Benefits

| Sparsity | Theoretical Speedup | Actual Speedup |
|----------|---------------------|----------------|
| 50% (2:4) | 2× | 1.5-1.8× |
| 75% | 4× | 2.5-3× |
| 87.5% | 8× | 4-5× |

Actual speedup is limited by:
- Sparse format overhead
- Irregular memory access
- Load imbalance

## Pruning for 2:4 Sparsity

To use 2:4 sparsity, weights must be pruned:
1. Train dense model
2. Apply 2:4 pruning (keep 2 largest in every 4)
3. Fine-tune pruned model

## Production Use Cases

| Use Case | Sparsity | Speedup |
|----------|----------|---------|
| Pruned LLM | 50% (2:4) | 1.5-1.8× |
| Compressed vision | 75% | 2.5-3× |
| Recommendation | 87.5% | 4-5× |

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_sparse_basic_FILL_IN.cu` | 2:4 sparse GEMM | Medium |
| `ex02_sparse_fp16_FILL_IN.cu` | Sparse FP16 | Medium |
