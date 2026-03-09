/*
 * Module 07 — Sparse GEMM
 * Exercise 01 — 2:4 Structured Sparsity
 *
 * CUTLASS LAYER: Sparse CollectiveMma with 2:4 pattern
 *
 * WHAT YOU'RE BUILDING:
 *   2:4 structured sparse GEMM using Ampere+ hardware support.
 *   This is used for pruned model inference.
 *
 * OBJECTIVE:
 *   - Configure 2:4 sparse GEMM
 *   - Understand sparse meta layout
 *   - Measure speedup vs dense
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"

using namespace cutlass;

using ArchTag = cutlass::arch::Sm80;  // 2:4 sparse requires Ampere+
constexpr int M = 4096, N = 4096, K = 4096;

using ElementA = cutlass::half_t;  // Dense A
using ElementB = cutlass::half_t;  // Sparse B (2:4 pattern)
using ElementAccumulator = float;
using ElementD = cutlass::half_t;

// TODO [MEDIUM]: Configure 2:4 sparse GEMM
// HINT: Use cutlass::gemm::SparseGemm or SparseCollectiveMma

int main() {
    std::cout << "=== Module 07, Exercise 01: 2:4 Sparse GEMM ===" << std::endl;
    
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 8) {
        std::cout << "2:4 sparsity requires Ampere (SM80) or later." << std::endl;
        return 0;
    }
    
    std::cout << "2:4 Structured Sparsity:" << std::endl;
    std::cout << "  - 2 non-zero values per 4 consecutive elements" << std::endl;
    std::cout << "  - Hardware-accelerated on Ampere+" << std::endl;
    std::cout << "  - 50% reduction in memory and compute" << std::endl;
    std::cout << "  - Requires pruning to 2:4 pattern" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement 2:4 sparse GEMM" << std::endl;
    std::cout << "Expected speedup vs dense: 1.5-1.8×" << std::endl;
    
    return 0;
}
