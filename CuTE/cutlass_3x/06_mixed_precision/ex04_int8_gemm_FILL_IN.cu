/*
 * Module 06 — Mixed Precision
 * Exercise 04 — INT8 Quantized GEMM
 *
 * CUTLASS LAYER: INT8 Tensor Core with per-channel scaling
 *
 * WHAT YOU'RE BUILDING:
 *   INT8 quantized GEMM with per-channel scaling. Common
 *   in production quantized inference (RTN, PTQ).
 *
 * OBJECTIVE:
 *   - Configure INT8 Tensor Core GEMM
 *   - Apply per-channel scaling
 *   - Compare accuracy vs FP16
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"

using namespace cutlass;

using ArchTag = cutlass::arch::Sm80;
constexpr int M = 4096, N = 4096, K = 4096;

using ElementA = int8_t;
using ElementB = int8_t;
using ElementAccumulator = int32_t;
using ElementD = int8_t;

// TODO [MEDIUM]: Configure INT8 GEMM with per-channel scaling
// HINT: Use int32_t accumulator, apply scales in epilogue

int main() {
    std::cout << "=== Module 06, Exercise 04: INT8 GEMM ===" << std::endl;
    
    std::cout << "INT8 quantization characteristics:" << std::endl;
    std::cout << "  - 8-bit signed integer" << std::endl;
    std::cout << "  - Range: -128 to 127" << std::endl;
    std::cout << "  - Requires dequant/requant scales" << std::endl;
    std::cout << "  - Per-channel scaling for better accuracy" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement INT8 GEMM with per-channel scales" << std::endl;
    std::cout << "Expected: Similar accuracy to FP16 with 2× speedup" << std::endl;
    
    return 0;
}
