# Module 06 — Mixed Precision GEMM

## Overview

Mixed precision GEMM uses different precisions for inputs, accumulation, and outputs to optimize performance and accuracy. CUTLASS 3.x supports:
- TF32 (Ampere+)
- FP16/BF16 (all Tensor Core GPUs)
- FP8 E4M3/E5M2 (Hopper+)
- INT8 (all Tensor Core GPUs)

## Precision Comparison

| Precision | Bits | Dynamic Range | Use Case |
|-----------|------|---------------|----------|
| FP32 | 32 | High | Accumulation, sensitive models |
| TF32 | 19 (10 mantissa) | Medium | Ampere training |
| BF16 | 16 (7 mantissa) | Medium | LLM training/inference |
| FP16 | 16 (10 mantissa) | Low | Inference, vision |
| FP8 E4M3 | 8 (3 mantissa) | Low | Quantized inference |
| FP8 E5M2 | 8 (2 mantissa) | Medium | Quantized inference |
| INT8 | 8 | Low | Quantized inference |

## Scaling Tensors

Lower precision requires scaling to prevent overflow/underflow:

```cpp
// Per-tensor scaling
D = (A * scale_a) @ (B * scale_b) * scale_output

// Per-channel scaling (better accuracy)
D[i,j] = sum_k(A[i,k] * scale_a[k]) * (B[k,j] * scale_b[j]) * scale_output
```

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_tf32_gemm_FILL_IN.cu` | TF32 Tensor Core | Easy |
| `ex02_fp16_bf16_FILL_IN.cu` | FP16 vs BF16 | Easy |
| `ex03_fp8_e4m3_FILL_IN.cu` | FP8 quantized GEMM | Medium |
| `ex04_int8_gemm_FILL_IN.cu` | INT8 with per-channel scale | Medium |
