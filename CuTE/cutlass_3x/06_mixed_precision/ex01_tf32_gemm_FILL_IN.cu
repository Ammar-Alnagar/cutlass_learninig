/*
 * Module 06 — Mixed Precision
 * Exercise 01 — TF32 GEMM
 *
 * CUTLASS LAYER: TF32 Tensor Core configuration
 *
 * WHAT YOU'RE BUILDING:
 *   TF32 GEMM for Ampere training workloads. TF32 provides
 *   FP32 dynamic range with reduced mantissa for 8× throughput.
 *
 * OBJECTIVE:
 *   - Configure TF32 Tensor Core GEMM
 *   - Compare TF32 vs FP32 performance
 *   - Understand TF32 accuracy tradeoffs
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

// TF32 uses FP32 inputs but computes in TF32 format
// CUTLASS handles the conversion automatically

using ArchTag = cutlass::arch::Sm80;  // TF32 requires Ampere+

constexpr int M = 4096, N = 4096, K = 4096;

// TF32 GEMM: FP32 inputs, TF32 compute, FP32 output
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cutlass::GemmlShape<128, 128, 64>;

// TODO [EASY]: Configure TF32 Tensor Core GEMM
// HINT: Use OpClassTensorOp with float element types

/*
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,  // Tensor Core (TF32)
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<128>,
   cutlass::gemm::collective::KernelScheduleAuto
   >::CollectiveOp;
*/

struct CollectiveMainloop {};

int main() {
    std::cout << "=== Module 06, Exercise 01: TF32 GEMM ===" << std::endl;
    
    // Check for Ampere+
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 8) {
        std::cout << "TF32 requires Ampere (SM80) or later." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
        return 0;
    }
    
    std::cout << "Device: " << prop.name << " (SM" << prop.major << prop.minor << ")" << std::endl;
    std::cout << "TF32 Tensor Core: Available" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TF32 characteristics:" << std::endl;
    std::cout << "  - 19 bits total (1 sign, 8 exponent, 10 mantissa)" << std::endl;
    std::cout << "  - Same dynamic range as FP32" << std::endl;
    std::cout << "  - 3 bits less precision than FP32" << std::endl;
    std::cout << "  - 8× throughput vs FP32 Tensor Core" << std::endl;
    std::cout << std::endl;
    
    std::cout << "TODO: Implement TF32 GEMM using CollectiveBuilder" << std::endl;
    std::cout << "Expected speedup over FP32: 5-8×" << std::endl;
    
    return 0;
}
