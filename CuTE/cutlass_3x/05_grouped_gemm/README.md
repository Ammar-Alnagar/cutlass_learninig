# Module 05 — Grouped GEMM

## Overview

Grouped GEMM executes multiple GEMMs of varying sizes in a single kernel launch. This is critical for:
- Mixture of Experts (MoE) routing
- Variable sequence length attention
- Batched operations with different shapes
- Multi-model inference

## The Problem

Naive approach for multiple GEMMs:
```cpp
for (int i = 0; i < num_gems; ++i) {
    launch_gemm(A[i], B[i], C[i], M[i], N[i], K[i]);  // Kernel launch overhead!
}
```

Problems:
- Kernel launch overhead per GEMM
- GPU underutilization (small GEMMs don't fill GPU)
- No work stealing between GEMMs

## Grouped GEMM Solution

Single kernel handles all GEMMs:
```cpp
launch_grouped_gemm(
    ptr_array_A, ptr_array_B, ptr_array_C,
    size_array_M, size_array_N, size_array_K,
    num_gems
);
```

Benefits:
- Single kernel launch
- Dynamic work distribution
- Better GPU utilization

## MoE Use Case

Mixture of Experts requires grouped GEMM:
```
Input tokens → Router → Expert assignments
Expert 0: 128 tokens
Expert 1: 45 tokens
Expert 2: 256 tokens
...
→ Grouped GEMM processes all experts in one launch
```

## CUTLASS 3.x Grouped GEMM API

```cpp
struct GroupedGemmArguments {
    // Problem sizes for each group
    std::vector<GemmCoord> problem_sizes;
    
    // Pointer arrays (device pointers)
    std::vector<ElementA*> ptr_A;
    std::vector<ElementB*> ptr_B;
    std::vector<ElementC*> ptr_C;
    std::vector<ElementD*> ptr_D;
    
    // Leading dimensions
    std::vector<int> lda, ldb, ldc, ldd;
};
```

## Production Use Cases

| Use Case | Groups | Speedup |
|----------|--------|---------|
| MoE (8 experts) | 8 | 3-5× |
| MoE (64 experts) | 64 | 10-20× |
| Variable seq attention | 32-128 | 5-10× |
| Multi-model batch | 4-16 | 2-4× |

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_grouped_basic_FILL_IN.cu` | Fixed-size groups | Medium |
| `ex02_grouped_moe_FILL_IN.cu` | Variable expert routing | Hard |
| `ex03_grouped_ragged_FILL_IN.cu` | Ragged batch | Hard |
