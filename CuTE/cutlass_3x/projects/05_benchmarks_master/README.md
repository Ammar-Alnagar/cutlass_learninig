# Project 05 — Benchmarks Master

## Overview

Comprehensive benchmark suite comparing CUTLASS 3.x vs ThunderKittens vs cuBLAS. This is your primary interview artifact.

## Requirements

### Roofline Analysis
- Compute theoretical peak for each architecture
- Plot achieved TFLOPS vs arithmetic intensity
- Identify memory-bound vs compute-bound kernels

### Framework Comparison
- CUTLASS 3.x (this curriculum)
- ThunderKittens (your prior experience)
- cuBLAS (NVIDIA reference)

### Performance Tables
- TFLOPS for each kernel
- % of theoretical peak
- Speedup ratios

## Deliverables

| File | Description |
|------|-------------|
| `roofline.cu` | Roofline chart generation |
| `compare_tk.cu` | CUTLASS vs ThunderKittens |
| `compare_cublas.cu` | CUTLASS vs cuBLAS |
| `results/` | Benchmark results (Markdown tables) |

## Benchmark Kernels

| Kernel | Problem Size | Metric |
|--------|--------------|--------|
| Dense GEMM | 4096×4096×4096 | TFLOPS |
| FP8 GEMM | 4096×4096×4096 | TFLOPS, speedup |
| Warp-spec GEMM | 8192×8192×8192 | TFLOPS, speedup |
| Grouped GEMM | 8× 4096×4096×4096 | Tokens/sec |
| FA2/FA3 | B=8, H=64, S=4096, D=128 | Tokens/sec |

## Success Criteria

- Complete roofline chart for all kernels
- Side-by-side comparison tables
- GitHub-ready README with results

## Interview Artifact

This benchmark suite goes in your GitHub README. It demonstrates:
- Deep understanding of GPU performance
- Ability to benchmark rigorously
- Production-quality engineering

## Example Results Table

```
| Kernel | cuBLAS | ThunderKittens | CUTLASS 3.x | Speedup vs TK |
|--------|--------|----------------|-------------|---------------|
| Dense GEMM (FP16) | 312 TF | 285 TF | 298 TF | 1.05× |
| Dense GEMM (FP8) | 580 TF | N/A | 562 TF | - |
| Warp-spec GEMM | 620 TF | N/A | 595 TF | - |
| FA2 Attention | 125 TF | 118 TF | 121 TF | 1.03× |
| FA3 Attention | 180 TF | N/A | 172 TF | - |
```
