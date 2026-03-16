# ex02: The Named Rvalue Is Lvalue

Debug this broken move constructor that causes double-free crashes.

## What You Build

A fix for a move constructor that incorrectly copies instead of moving, leading to two objects owning the same memory.

## What You Observe

The buggy version prints "[WRONG] Copying data" and crashes on exit (double-free). The fixed version prints "[CORRECT] Moving data" and exits cleanly with the moved-from object in a safe state.

## CUTLASS/CUDA Mapping

Device-side move constructors follow the same rule. A kernel parameter `T&& param` is an lvalue inside the kernel. When you write custom CUDA allocators or device-side containers, move constructors must nullify the source pointer.

## Build Command

```bash
# Buggy version (will crash)
g++ -std=c++20 -O2 -o ex02_buggy exercise.cpp && ./ex02_buggy

# Fixed version (after applying solution)
g++ -std=c++20 -O2 -o ex02_fixed solution.cpp && ./ex02_fixed
```
