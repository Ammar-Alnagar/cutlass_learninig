# ex03: Memory Order Layers — Acquire/Release

Learn Layer 2 of atomics: memory ordering controls visibility timing.

## What You Build

A flag+data pattern using acquire/release memory ordering, plus a comparison with the broken relaxed-only pattern.

## What You Observe

The acquire/release pattern guarantees the reader sees `data=42`. The relaxed pattern may show `data=0` (stale value) because there's no ordering guarantee — only atomicity.

## CUTLASS/CUDA Mapping

CUDA memory fences map to C++ memory orders: `__threadfence()` ≈ `seq_cst`, `__threadfence_block()` ≈ `release`. CUTLASS uses `memory_order_relaxed` for counters (atomicity only) and `acquire/release` for data dependencies.

## Build Command

```bash
g++ -std=c++20 -O2 -o ex03 exercise.cpp -lpthread && ./ex03
```
