# ex01: Deep Copy vs Shallow Copy

Debug a double-free crash caused by shallow copy of a pointer member.

## What You Build

A fix for a device buffer wrapper that crashes when copied because both objects try to free the same memory.

## What You Observe

The buggy version crashes with "double free or corruption". The fixed version completes successfully, with original and copy having independent memory. Modifying the copy does not affect the original.

## CUTLASS/CUDA Mapping

CUTLASS device memory wrappers face this exact issue. Copying a `cutlass::DeviceAllocation` without deep copy would cause double `cudaFree`. CUTLASS solves this by making device wrappers move-only (delete copy, implement move) — model weights are too large to deep copy anyway.

## Build Command

```bash
# Buggy version (will crash)
g++ -std=c++20 -O2 -fsanitize=address -o ex01_buggy exercise.cpp && ./ex01_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex01_fixed solution.cpp && ./ex01_fixed
```
