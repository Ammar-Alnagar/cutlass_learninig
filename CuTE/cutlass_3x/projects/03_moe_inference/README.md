# Project 03 — MoE Inference

## Overview

Implement Mixture of Experts (MoE) inference using grouped GEMM. Target 2× tokens/sec vs naive expert loop.

## Requirements

### Grouped Expert GEMM
- Variable tokens per expert
- Single kernel launch for all experts
- Work stealing for load balancing

### FP8 Quantized MoE
- FP8 E4M3 weights and activations
- Per-expert scaling
- EVT dequant epilogue

## Deliverables

| File | Description |
|------|-------------|
| `moe_grouped_gemm_FILL_IN.cu` | Expert routing + grouped GEMM |
| `moe_fp8_FILL_IN.cu` | Quantized MoE |
| `benchmark.cu` | Tokens/sec comparison |

## MoE Pattern

```
Input tokens → Router → Expert assignments
                ↓
Expert 0: tokens_0 × hidden → mlp → output_0
Expert 1: tokens_1 × hidden → mlp → output_1
...
                ↓
        Combine outputs
```

## Success Criteria

- 2× tokens/sec vs naive expert loop
- FP8 quantized: >1.5× over FP16
- Accuracy within 0.5% of dense model

## Interview Story

*"I implemented MoE inference using CUTLASS 3.x grouped GEMM. By processing all 8 experts in a single kernel launch with work stealing, I achieved 3.5× tokens/sec improvement over the naive expert loop. The FP8 quantized variant achieved an additional 1.8× speedup."*
