# Project 02: MoE Grouped GEMM

## Overview

Implement Mixture of Experts (MoE) with Grouped GEMM for efficient expert routing.
This is the pattern used in Mixtral, Grok, and GShard for sparse expert models.

## Target Metrics

| Metric | Target |
|--------|--------|
| Tokens/sec | 2× vs naive expert loop |
| Memory efficiency | <10% overhead vs dense GEMM |

## Prerequisites

- Complete Module 01 (Grouped GEMM exercise)
- Understanding of MoE routing

## Files

- `moe_gemm_FILL_IN.py` - MoE implementation with grouped GEMM
- `benchmark.py` - Comparison vs naive expert loop

## Getting Started

```bash
# Run MoE implementation
python moe_gemm_FILL_IN.py

# Run benchmark
python benchmark.py
```

## Implementation Checklist

- [ ] Implement expert routing (top-K selection)
- [ ] Group tokens by expert
- [ ] Use cutlass.op.GroupedGemm for expert processing
- [ ] Scatter outputs back to original positions
- [ ] Add load balancing loss

## Resources

- Mixtral paper: https://arxiv.org/abs/2401.04088
- GShard paper: https://arxiv.org/abs/2006.16668
