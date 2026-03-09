/*
 * Module 08 — Kernel Fusion
 * Exercise 03 — GEMM + Online Softmax
 *
 * CUTLASS LAYER: EVT with online softmax
 *
 * WHAT YOU'RE BUILDING:
 *   GEMM with fused online softmax in epilogue.
 *   Numerically stable softmax in single pass.
 *
 * OBJECTIVE:
 *   - Implement online softmax EVT node
 *   - Fuse with GEMM epilogue
 *   - Understand numerical stability
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "benchmark.cuh"

using namespace cutlass;

// Online softmax (numerically stable):
// max_val = max(x)
// sum = sum(exp(x - max_val))
// softmax(x) = exp(x - max_val) / sum

// TODO [MEDIUM]: Implement online softmax EVT node
// HINT: Requires row-wise reduction (similar to LayerNorm challenge)

int main() {
    std::cout << "=== Module 08, Exercise 03: GEMM + Online Softmax ===" << std::endl;
    
    std::cout << "Online softmax:" << std::endl;
    std::cout << "  - Numerically stable (subtracts max)" << std::endl;
    std::cout << "  - Single pass (find max + compute sum together)" << std::endl;
    std::cout << "  - Used in Flash Attention" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement online softmax fusion" << std::endl;
    std::cout << "Expected speedup: 1.5-2× vs separate softmax kernel" << std::endl;
    
    return 0;
}
