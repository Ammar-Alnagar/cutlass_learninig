# ex03: Tiled GEMM — Timed (45 min)

Implement a tiled matrix multiply using shared memory in 45 minutes.

## What You Build

A CUDA kernel that multiplies two matrices using tiled shared memory. Each thread block loads tiles of A and B, computes partial dot products, and writes the result to C.

## Interview Rubric

- [ ] Shared memory declared (`__shared__`)
- [ ] Tile loading with boundary checks
- [ ] `__syncthreads()` after loading
- [ ] Dot product computation loop
- [ ] `__syncthreads()` before next tile
- [ ] Result write with boundary check

## Time Targets

- < 30 min: Expert
- 30-45 min: Strong
- 45-60 min: Acceptable
- > 60 min: Review shared memory

## Build Command

```bash
nvcc -std=c++17 -O2 -arch=sm_89 -o ex03 exercise.cu && ./ex03
```

## Background

This is the foundation of CUTLASS. Real CUTLASS adds tensor core intrinsics, double buffering, and software pipelining — but the tiled shared-memory pattern is identical.
