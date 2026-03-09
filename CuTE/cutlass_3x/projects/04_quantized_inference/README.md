# Project 04 — Quantized Inference

## Overview

Implement quantized linear layers using CUTLASS 3.x. Target FP8 >1.5× over FP16, INT8 with accuracy within 0.5% of FP16.

## Requirements

### INT8 Linear Layer
- Per-channel weight quantization
- Per-tensor activation quantization
- Fused dequant + GEMM + requant

### FP8 Linear Layer
- FP8 E4M3 weights and activations
- EVT dequant epilogue
- Scaling tensor management

## Deliverables

| File | Description |
|------|-------------|
| `int8_linear_FILL_IN.cu` | INT8 fused quant + GEMM + dequant |
| `fp8_linear_FILL_IN.cu` | FP8 with EVT dequant epilogue |
| `benchmark.cu` | vs FP16, accuracy table |

## Quantization Pattern

```
FP16 Input → Quantize → INT8/FP8 → GEMM → Dequantize → FP16 Output
              ↓                    ↓          ↓
           scale_in            quant_w    scale_out
```

## Success Criteria

- FP8: >1.5× over FP16 on H100
- INT8: >1.3× over FP16
- Accuracy: Within 0.5% of FP16 baseline

## Interview Story

*"I implemented quantized linear layers using CUTLASS 3.x EVT for fused dequantization. The FP8 variant achieved 1.7× speedup over FP16 on H100, with accuracy within 0.3% of the FP16 baseline on LLaMA-7B perplexity benchmark."*
