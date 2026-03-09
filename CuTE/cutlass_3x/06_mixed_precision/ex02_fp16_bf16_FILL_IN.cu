/*
 * Module 06 — Mixed Precision
 * Exercise 02 — FP16 vs BF16 GEMM
 *
 * CUTLASS LAYER: FP16 and BF16 Tensor Core configuration
 *
 * WHAT YOU'RE BUILDING:
 *   Comparison of FP16 and BF16 GEMMs. BF16 is preferred for
 *   LLM training due to better dynamic range.
 *
 * OBJECTIVE:
 *   - Configure FP16 and BF16 GEMMs
 *   - Compare performance and accuracy
 *   - Understand when to use each format
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"

using namespace cutlass;

using ArchTag = cutlass::arch::Sm80;
constexpr int M = 4096, N = 4096, K = 4096;

// TODO [EASY]: Define both FP16 and BF16 GEMM configurations
// HINT: Use cutlass::half_t for FP16, cutlass::bfloat16_t for BF16

int main() {
    std::cout << "=== Module 06, Exercise 02: FP16 vs BF16 GEMM ===" << std::endl;
    
    std::cout << "FP16 vs BF16 comparison:" << std::endl;
    std::cout << std::endl;
    std::cout << "FP16 (IEEE 754):" << std::endl;
    std::cout << "  - 1 sign, 5 exponent, 10 mantissa" << std::endl;
    std::cout << "  - Range: ±65504" << std::endl;
    std::cout << "  - Better for: Vision, inference" << std::endl;
    std::cout << std::endl;
    std::cout << "BF16 (Brain Float):" << std::endl;
    std::cout << "  - 1 sign, 8 exponent, 7 mantissa" << std::endl;
    std::cout << "  - Range: ±3.4e38 (same as FP32)" << std::endl;
    std::cout << "  - Better for: LLM training, gradient accumulation" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement and compare FP16 vs BF16 GEMMs" << std::endl;
    std::cout << "Expected: Similar performance, BF16 more stable for training" << std::endl;
    
    return 0;
}
