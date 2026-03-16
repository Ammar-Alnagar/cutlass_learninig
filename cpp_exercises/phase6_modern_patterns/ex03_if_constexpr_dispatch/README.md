# ex03: if constexpr Dispatch

Use `if constexpr` for compile-time type dispatch without SFINAE.

## What You Build

Generic functions that dispatch to type-specific code using `if constexpr` with type traits like `std::is_integral_v` and `std::is_floating_point_v`.

## What You Observe

A single `process_value<T>` template handles int, float, string, and unknown types. The false branches are discarded at compile time — no SFINAE needed.

## CUTLASS/CUDA Mapping

CUTLASS 3.x uses `if constexpr` for kernel dispatch instead of CUTLASS 2.x SFINAE. Cleaner code: `if constexpr (std::is_same_v<T, __half>)` selects Tensor Core kernels.

## Build Command

```bash
g++ -std=c++20 -O2 -o ex03 exercise.cpp && ./ex03
```
