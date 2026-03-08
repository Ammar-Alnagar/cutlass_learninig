# Project 03: FP8 Inference Pipeline

## Overview

Implement end-to-end FP8 inference pipeline with quantized linear layers.
This is the pattern used in TensorRT-LLM for FP8 inference optimization.

## Target Metrics

| Metric | Target (Hopper H100) |
|--------|----------------------|
| Throughput | >1.5× over FP16 |
| Accuracy | <1% perplexity degradation |

## Prerequisites

- Complete Module 05 (Mixed Precision)
- Hopper GPU (SM90) for native FP8 Tensor Cores

## Files

- `fp8_linear_FILL_IN.py` - FP8 quantized linear layer
- `benchmark.py` - Comparison vs FP16 baseline

## Getting Started

```bash
# Run FP8 linear layer
python fp8_linear_FILL_IN.py

# Run benchmark with accuracy evaluation
python benchmark.py
```

## Implementation Checklist

- [ ] Implement FP8 quantization (E4M3)
- [ ] Create FP8 linear layer with dequantization
- [ ] Add calibration for activation quantization
- [ ] Implement accuracy evaluation
- [ ] Benchmark throughput vs FP16

## Resources

- TensorRT-LLM FP8: https://github.com/NVIDIA/TensorRT-LLM
- FP8 paper: https://arxiv.org/abs/2209.05433
