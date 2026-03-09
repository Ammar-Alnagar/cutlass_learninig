/*
 * Module 06 — Mixed Precision
 * Exercise 03 — FP8 E4M3 Quantized GEMM
 *
 * CUTLASS LAYER: FP8 Tensor Core with scaling
 *
 * WHAT YOU'RE BUILDING:
 *   FP8 E4M3 quantized GEMM for Hopper+. This is the format
 *   used in TRT-LLM FP8 inference.
 *
 * OBJECTIVE:
 *   - Configure FP8 E4M3 GEMM
 *   - Apply per-tensor scaling
 *   - Measure speedup vs FP16
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/float8.h"

#include "benchmark.cuh"
#include "reference.cuh"

using namespace cutlass;

using ArchTag = cutlass::arch::Sm90;  // FP8 requires Hopper+
constexpr int M = 4096, N = 4096, K = 4096;

// FP8 E4M3: 1 sign, 4 exponent, 3 mantissa
using ElementA = cutlass::float8_e4m3_t;
using ElementB = cutlass::float8_e4m3_t;
using ElementAccumulator = float;
using ElementD = cutlass::float8_e4m3_t;

// TODO [MEDIUM]: Configure FP8 GEMM with scaling tensors
// HINT: FP8 requires scale factors to prevent overflow

int main() {
    std::cout << "=== Module 06, Exercise 03: FP8 E4M3 GEMM ===" << std::endl;
    
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 9) {
        std::cout << "FP8 Tensor Core requires Hopper (SM90) or later." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "FP8 E4M3 characteristics:" << std::endl;
    std::cout << "  - 1 sign, 4 exponent, 3 mantissa" << std::endl;
    std::cout << "  - Range: ±448" << std::endl;
    std::cout << "  - 2× throughput vs FP16 on Hopper" << std::endl;
    std::cout << "  - Requires scaling tensors" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement FP8 GEMM with per-tensor scaling" << std::endl;
    std::cout << "Expected speedup over FP16: 1.8-2.2×" << std::endl;
    
    return 0;
}
