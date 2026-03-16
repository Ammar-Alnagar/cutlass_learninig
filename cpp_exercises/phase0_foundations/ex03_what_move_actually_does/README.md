# ex03: What Move Actually Does

Implement a move-only Buffer class from scratch to internalize the move pattern.

## What You Build

A move-only `Buffer` class with move constructor, move assignment, and deleted copy operations. Plus a function that returns a moved buffer to demonstrate RVO (return value optimization).

## What You Observe

Moved-from objects have `size=0` and `data=nullptr`. Destructors fire for all objects, but moved-from objects safely delete nullptr. No deep copies occur — only pointer theft.

## CUTLASS/CUDA Mapping

CUTLASS uses move-only types for device memory wrappers. When you construct a `cutlass::DeviceMemory` and move it into a kernel launcher, the pointer is transferred without copying gigabytes of model weights. This is essential for efficient LLM serving.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex03 exercise.cpp && ./ex03
```
