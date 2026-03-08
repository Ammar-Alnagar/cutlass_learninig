# Project 01: Fused Multi-Head Attention (Full Implementation)

## Overview

Implement production-ready Fused Multi-Head Attention (FMHA) kernels for three GPU architectures:
- **Ampere (SM80)**: Standard FlashAttention pattern
- **Hopper (SM90)**: Warp-specialized FlashAttention v2
- **Blackwell (SM100)**: PDL + persistent kernel optimization

This is the centerpiece project for NVIDIA DL Software Engineer applications.

## Target Metrics

| GPU | Target | Metric |
|-----|--------|--------|
| Ampere (A100) | Within 20% of FlashAttention-2 C++ | Tokens/sec |
| Hopper (H100) | Demonstrate warp-specialized pipeline | Occupancy |
| Blackwell (B200) | PDL + persistent scheduling | Latency |

## Prerequisites

- Complete Modules 01-07
- CUDA 12.x+ (13.x for Blackwell)
- FlashAttention reference: https://github.com/Dao-AILab/flash-attention

## Files

- `fmha_ampere_FILL_IN.py` - Ampere implementation
- `fmha_hopper_FILL_IN.py` - Hopper implementation  
- `fmha_blackwell_FILL_IN.py` - Blackwell implementation
- `benchmark.py` - Comparison vs FlashAttention reference

## Getting Started

```bash
# Verify FlashAttention reference
pip install flash-attn

# Run Ampere implementation
python fmha_ampere_FILL_IN.py

# Run benchmarks
python benchmark.py
```

## Implementation Checklist

### Ampere (SM80)
- [ ] Tiled attention over sequence dimension
- [ ] Online softmax (single pass)
- [ ] Shared memory tiling for Q, K, V
- [ ] FP16 compute with FP32 accumulation

### Hopper (SM90)
- [ ] Warp specialization (load/compute/store warps)
- [ ] TMA (Tensor Memory Accelerator) for loads
- [ ] Async pipeline with multiple stages
- [ ] FP8 support option

### Blackwell (SM100)
- [ ] Persistent kernel pattern
- [ ] PDL for dynamic dispatch
- [ ] tcgen05 MMA instructions
- [ ] FP4 quantization option

## Evaluation

Your implementation will be evaluated on:
1. **Correctness**: Output matches FlashAttention reference (rtol=1e-2)
2. **Performance**: Within target metrics for each GPU
3. **Code Quality**: Clean, well-documented, follows CUTLASS conventions

## Resources

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- CUTLASS Examples: https://github.com/NVIDIA/cutlass/tree/main/examples/python
