/*
 * Module 01 — CollectiveBuilder Anatomy
 * Exercise 03 — Hopper TMA-Based Collective
 *
 * CUTLASS LAYER: CollectiveMma with TMA (Tensor Memory Accelerator)
 *
 * WHAT YOU'RE BUILDING:
 *   SM90 Hopper GEMM using TMA for async memory operations.
 *   This is the foundation of FA3 (Flash Attention 3) on H100.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   TMA replaces your manual TiledCopy from CuTe Module 03.
 *   PipelineTmaAsync replaces your manual Pipeline from CuTe Module 06.
 *
 * OBJECTIVE:
 *   - Configure TMA load descriptors
 *   - Enable warp specialization for Hopper
 *   - Understand producer/consumer warp split
 */

// PREDICT BEFORE COMPILING
// Q1: How does TMA reduce instruction count vs manual LDG?
// Q2: What's the warp specialization split ratio for optimal H100 performance?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// SETUP — Hopper-specific configuration
// ============================================================================

// Architecture target — SM90 for Hopper
using ArchTag = cutlass::arch::Sm90;

// GEMM problem size
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

// Element types — BF16 is native on Hopper
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ============================================================================
// TMA CONFIGURATION — Tensor Memory Accelerator
// ============================================================================
// TMA provides:
//   - Async memory operations (overlap with compute)
//   - Hardware address calculation
//   - Automatic cache control
//   - Multi-cast support for cluster

// TODO [MEDIUM]: Configure CollectiveMainloop for Hopper with TMA
// Key differences from SM80:
//   1. Use PipelineTmaAsync instead of PipelineAsync
//   2. Enable warp specialization (producer/consumer split)
//   3. Configure cluster shape for multi-SM cooperation

/*
using TileShape = cutlass::GemmlShape<128, 128, 64>;  // Typical Hopper tile

// Cluster shape for multi-SM (SM90+ feature)
using ClusterShape = cutlass::GemmlShape<1, 1, 1>;  // Start with single SM

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   ClusterShape,
   cutlass::gemm::collective::StageCountAutoCarveout<230400>,  // H100 smem: 230KB
   cutlass::gemm::collective::KernelScheduleWarpSpecialized   // <-- Key for Hopper
   >::CollectiveOp;
*/

struct CollectiveMainloop {};

// ============================================================================
// WARP SPECIALIZATION — Producer/Consumer Split
// ============================================================================
// Hopper warp specialization divides warps into:
//   - Producer warps: Handle TMA loads (memory)
//   - Consumer warps: Handle MMA compute
// This enables true async memory/compute overlap.

// TODO [HARD]: Define warp specialization configuration
// HINT: cutlass::gemm::collective::WarpSpecialization<ProducerWarps, ConsumerWarps>
// REF: cutlass/examples/65_hopper_warp_specialized/

/*
// Typical split on H100 (108 SMs, 128 warps/SM):
//   - 16-24 producer warps per SM
//   - Remaining warps for consumer
// The builder auto-selects, but you can override:

using WarpSpecialization = cutlass::gemm::collective::WarpSpecializedPolicy
  <cutlass::gemm::collective::ProducerWarpCount<4>,   // 4 warps for TMA
   cutlass::gemm::collective::ConsumerWarpCount<28>   // 28 warps for MMA
   >;
*/

// ============================================================================
// COLLECTIVE EPILOGUE — Hopper-optimized
// ============================================================================

/*
using CollectiveEpilogue = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   ClusterShape,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;
*/

struct CollectiveEpilogue {};

// ============================================================================
// TMA DESCRIPTOR SETUP — Explicit TMA configuration
// ============================================================================
// For advanced use, you can manually configure TMA descriptors.
// This is needed when data layouts don't match standard patterns.

// TODO [HARD]: Create TMA load descriptor for matrix A
// HINT: cutlass::TmaLoadDescriptor requires:
//   - Global memory base pointer
//   - Stride dimensions
//   - Box dimensions (tile size)
//   - Element type

/*
struct TmaDescriptorA {
    using Element = ElementA;
    using Layout = LayoutA;
    
    // TMA requires specific memory alignment (128-byte for Hopper)
    static constexpr int kAlignment = 128;
    
    // Create descriptor from device pointer
    static auto make_descriptor(Element* ptr, int ld, int rows, int cols) {
        return cutlass::make_tma_load_descriptor(
            ptr,
            cutlass::make_coord(rows, cols),
            cutlass::make_coord(ld),
            cutlass::make_coord(TileShape::kM, TileShape::kK)
        );
    }
};
*/

// ============================================================================
// LAUNCH FUNCTION
// ============================================================================

template <typename GemmKernel>
cudaError_t launch_hopper_gemm(
    const typename GemmKernel::Arguments& args,
    cudaStream_t stream = 0
) {
    GemmKernel gemm_op;
    
    size_t workspace_bytes = 0;
    GemmKernel::get_workspace_size(args, &workspace_bytes);
    
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    // For Hopper TMA, workspace includes TMA descriptor setup
    cudaError_t status = gemm_op.run(args, workspace, stream);
    
    if (workspace) {
        cudaFree(workspace);
    }
    
    return status;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 01, Exercise 03: Hopper TMA Collective ===" << std::endl;
    
    // Check for Hopper GPU
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    if (prop.major < 9) {
        std::cout << "WARNING: This exercise requires Hopper (SM90) GPU." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
        std::cout << "Code will not launch, but you can still study the configuration." << std::endl;
        return 0;
    }
    
    std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Shared Memory per SM: " << (prop.sharedMemoryPerMultiprocessor / 1024) << " KB" << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory
    bfloat16_t *d_A, *d_B, *d_C, *d_D;
    size_t bytes_A = M * K * sizeof(bfloat16_t);
    size_t bytes_B = K * N * sizeof(bfloat16_t);
    size_t bytes_C = M * N * sizeof(bfloat16_t);
    size_t bytes_D = M * N * sizeof(bfloat16_t);
    
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_D, bytes_D);
    
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    cutlass_ref::init_matrix_zeros(d_C, M * N);
    cutlass_ref::init_matrix_zeros(d_D, M * N);
    
    // Reference: cuBLAS BF16
    std::cout << "Running cuBLAS BF16 reference..." << std::endl;
    cutlass_ref::gemm_ref_bf16(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // TODO [MEDIUM]: Build and launch Hopper TMA GEMM
    // HINT: Same pattern as SM80, but with warp specialization enabled
    
    /*
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<
        cutlass::gemm::GemmUniversal<
            cutlass::gemm::GemmShape<M, N, K, 1>,
            CollectiveMainloop,
            CollectiveEpilogue
        >
    >;
    
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_D, N},
        {1.0f, 0.0f}
    };
    
    std::cout << "Running Hopper TMA GEMM (warp-specialized)..." << std::endl;
    cutlass_bench::GpuTimer timer;
    timer.start();
    CUDA_CHECK(launch_hopper_gemm<GemmKernel>(args));
    timer.stop();
    
    float elapsed_ms = timer.elapsed_ms();
    double tflops = cutlass_bench::compute_gemm_tflops(M, N, K, elapsed_ms);
    std::cout << "Hopper TMA GEMM: " << elapsed_ms << " ms, " << tflops << " TFLOPS" << std::endl;
    
    // Compare to SM80 baseline
    std::cout << "\nExpected speedup over SM80: 1.5-2.0x (TMA + warp-spec)" << std::endl;
    */
    
    // Verify
    std::cout << "\nVerifying against cuBLAS..." << std::endl;
    // cutlass_ref::verify_gemm(d_D, d_C, M * N);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What does ncu show for l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum?" << std::endl;
    std::cout << "C2: How does TMA reduce load instruction count?" << std::endl;
    std::cout << "C3: Explain the producer/consumer warp split in your own words." << std::endl;
    std::cout << "C4: How does this map to FA3 architecture?" << std::endl;
    
    return 0;
}
