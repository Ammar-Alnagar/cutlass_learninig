/*
 * Module 08 — Kernel Fusion
 * Exercise 01 — Fused LayerNorm + GEMM
 *
 * CUTLASS LAYER: Custom fused collective
 *
 * WHAT YOU'RE BUILDING:
 *   Fused LayerNorm + GEMM for Transformer pre-norm blocks.
 *   Eliminates intermediate memory traffic.
 *
 * OBJECTIVE:
 *   - Fuse LayerNorm into GEMM epilogue
 *   - Understand normalization in registers
 *   - Measure fusion speedup
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "benchmark.cuh"

using namespace cutlass;

// TODO [HARD]: Implement fused LayerNorm + GEMM
// HINT: LayerNorm requires row-wise reduction (challenging in EVT)

int main() {
    std::cout << "=== Module 08, Exercise 01: Fused LayerNorm + GEMM ===" << std::endl;
    
    std::cout << "LayerNorm fusion challenges:" << std::endl;
    std::cout << "  - Requires row-wise reduction (mean, variance)" << std::endl;
    std::cout << "  - Doesn't fit EVT elementwise model" << std::endl;
    std::cout << "  - Often implemented as separate kernel" << std::endl;
    std::cout << std::endl;
    std::cout << "Alternative approaches:" << std::endl;
    std::cout << "  - Fused attention pattern (QK + softmax)" << std::endl;
    std::cout << "  - Custom collective mainloop" << std::endl;
    std::cout << "  - Thread-block level reduction" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Explore LayerNorm fusion strategies" << std::endl;
    std::cout << "Expected speedup: 2-3× (eliminating intermediate writes)" << std::endl;
    
    return 0;
}
