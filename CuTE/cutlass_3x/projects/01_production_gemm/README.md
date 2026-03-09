# Project 01 — Production GEMM

## Overview

Implement production-quality GEMM kernels for Ampere, Hopper, and Blackwell architectures. Target >90% of cuBLAS performance.

## Requirements

### Ampere (SM80)
- CollectiveBuilder with optimal tile configuration
- Pipeline staging for latency hiding
- Target: >85% cuBLAS performance

### Hopper (SM90)
- TMA-based loads
- Warp specialization (producer/consumer split)
- Target: >90% cuBLAS performance

### Blackwell (SM100)
- Persistent kernel scheduling
- FP8 Tensor Core
- Target: >95% cuBLAS performance

## Deliverables

| File | Description |
|------|-------------|
| `gemm_ampere.cu` | SM80 optimized GEMM |
| `gemm_hopper.cu` | SM90 TMA + warp-spec GEMM |
| `gemm_blackwell.cu` | SM100 persistent FP8 GEMM |
| `benchmark.cu` | vs cuBLAS comparison |

## Success Criteria

- >85% cuBLAS on Ampere (A100)
- >90% cuBLAS on Hopper (H100)
- >95% cuBLAS on Blackwell (B200)
- Roofline chart showing >80% theoretical peak

## Interview Story

*"I implemented production GEMM kernels using CUTLASS 3.x CollectiveBuilder. On H100, I achieved 92% of cuBLAS performance by configuring warp specialization with 4 producer warps and 28 consumer warps. The builder auto-selected 128x128x64 tiles with 6 pipeline stages."*
