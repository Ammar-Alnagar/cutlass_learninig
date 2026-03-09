# Module 04 — StreamK Decomposition

## Overview

StreamK is a parallel decomposition strategy for GEMM that addresses the "wave quantization" problem in traditional tiled GEMM. It's particularly effective for:
- Non-square problem shapes
- Multi-GEMM workloads
- Irregular batch sizes

## The Wave Quantization Problem

Traditional tiled GEMM assigns fixed tiles to SMs. When the problem size doesn't divide evenly by tile size:

```
Problem: M=4100, Tile=128
Tiles needed: ceil(4100/128) = 33 tiles
Last tile: only 4 elements (124 unused)
```

This is **wave quantization** — the last "wave" of tiles is underutilized.

## StreamK Solution

StreamK decomposes the GEMM along the K dimension:

```
Traditional: Each SM processes fixed M×N tile
StreamK:     Each SM processes a "stripe" of K slices

     K
     ────────────────────────→
M    │ ████ │ ████ │ ████ │
│    │ ████ │ ████ │ ████ │
↓    │ ████ │ ████ │ ████ │
     │ ████ │ ████ │ ████ │
```

Each SM processes multiple K-slices, reducing to partial sums.

## StreamK vs Tiled GEMM

| Aspect | Tiled GEMM | StreamK |
|--------|------------|---------|
| Work distribution | Fixed tiles | Dynamic K-slices |
| Load balancing | Poor for irregular shapes | Excellent |
| Shared memory | Per-tile buffers | Per-stripe buffers |
| Reduction | None (direct write) | Partial sum reduction |
| Best for | Square, aligned problems | Irregular, batched problems |

## Wave Quantization Fix

StreamK eliminates wave quantization by:
1. Dividing K into fine-grained "work units"
2. Dynamically assigning work units to SMs
3. Reducing partial sums at the end

**Result:** Near-perfect utilization regardless of problem shape.

## Production Use Cases

| Use Case | StreamK Benefit |
|----------|-----------------|
| MoE expert GEMMs | Variable K per expert |
| Ragged batch attention | Irregular sequence lengths |
| Multi-model inference | Different layer sizes |
| Dynamic shape models | Runtime-varying dimensions |

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_streamk_gemm_FILL_IN.cu` | Basic StreamK GEMM | Medium |
| `ex02_streamk_vs_tiled_FILL_IN.cu` | StreamK vs tiled benchmark | Medium |
