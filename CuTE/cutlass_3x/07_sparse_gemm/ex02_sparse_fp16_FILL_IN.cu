/*
 * Module 07 — Sparse GEMM
 * Exercise 02 — Sparse FP16 GEMM
 *
 * CUTLASS LAYER: Sparse FP16 Tensor Core
 *
 * WHAT YOU'RE BUILDING:
 *   Sparse FP16 GEMM combining sparsity with mixed precision.
 *   Common pattern for pruned LLM inference.
 *
 * OBJECTIVE:
 *   - Configure sparse FP16 GEMM
 *   - Combine sparsity with Tensor Core
 *   - Measure combined speedup
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"

using namespace cutlass;

using ArchTag = cutlass::arch::Sm80;
constexpr int M = 4096, N = 4096, K = 4096;

// TODO [MEDIUM]: Configure sparse FP16 GEMM
// HINT: Combine sparse layout with FP16 Tensor Core

int main() {
    std::cout << "=== Module 07, Exercise 02: Sparse FP16 GEMM ===" << std::endl;
    
    std::cout << "Sparse FP16 benefits:" << std::endl;
    std::cout << "  - 2:4 sparsity: 1.5-1.8× speedup" << std::endl;
    std::cout << "  - FP16 Tensor Core: High throughput" << std::endl;
    std::cout << "  - Combined: 2-3× vs dense FP32" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement sparse FP16 GEMM" << std::endl;
    
    return 0;
}
