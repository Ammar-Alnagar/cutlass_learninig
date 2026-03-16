# ex02: Atomic Fixes Tearing

Add `std::atomic` to fix a tearing counter and learn atomic operations.

## What You Build

An atomic counter that replaces a non-atomic counter, plus demonstrations of `fetch_add`, `fetch_sub`, `exchange`, and `compare_exchange_weak`.

## What You Observe

The atomic counter produces correct results (400000) while the non-atomic counter loses updates. Compare-exchange enables lock-free patterns with retry loops.

## CUTLASS/CUDA Mapping

CUTLASS uses atomic operations for epilogue fusion and reduction. Device-side `atomicAdd`, `atomicCAS` provide the same guarantees. Lock-free queues on GPU use `atomicCAS` in retry loops.

## Build Command

```bash
g++ -std=c++20 -O2 -o ex02 exercise.cpp -lpthread && ./ex02
```
