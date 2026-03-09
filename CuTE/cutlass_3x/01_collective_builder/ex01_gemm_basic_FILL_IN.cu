/*
 * Module 01 — CollectiveBuilder Anatomy
 * Exercise 01 — Basic GEMM with CollectiveBuilder
 *
 * CUTLASS LAYER: CollectiveBuilder → CollectiveMma → GemmUniversalAdapter
 *
 * WHAT YOU'RE BUILDING:
 *   Your first production GEMM using CUTLASS 3.x CollectiveBuilder.
 *   This is the exact pattern used in TRT-LLM for dense linear layers.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   CollectiveMma wraps TiledMMA — you wired this manually in CuTe Module 04.
 *   CollectiveBuilder auto-selects the optimal TiledMMA configuration per arch.
 *
 * OBJECTIVE:
 *   - Understand CollectiveBuilder template parameter anatomy
 *   - Launch a GEMM using GemmUniversalAdapter
 *   - Verify correctness against cuBLAS reference
 */

// PREDICT BEFORE COMPILING
// Q1: What SM architecture will the builder target if you specify SM80?
// Q2: Will this GEMM be memory-bound or compute-bound for M=N=K=4096?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUTLASS 3.x includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

// Utilities
#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// SETUP — Problem definition and type aliases
// ============================================================================

// Architecture target
using ArchTag = cutlass::arch::Sm80;  // TODO: Change to Sm90 for Hopper

// GEMM problem size
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

// Element types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

// Layouts (row-major for all)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // B is column-major for C = A*B
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Tile shape — CollectiveBuilder will auto-select optimal if you use placeholders
// For learning, we specify explicitly:
using TileShape = cutlass::GemmlShape<int, int, int>;  // M, N, K tile sizes

// ============================================================================
// COLLECTIVE BUILDER — Mainloop configuration
// ============================================================================
// This is the heart of CUTLASS 3.x. The builder auto-selects:
//   - TiledMMA atom (16x8x16, 32x8x16, etc.)
//   - Shared memory layout
//   - Pipeline stages
//   - Warp specialization strategy (for Hopper+)

// TODO [EASY]: Complete the CollectiveBuilder typedef for the mainloop
// HINT: Follow the pattern from cutlass/examples/60_collective_builder/
// REF: cutlass/include/cutlass/gemm/collective/collective_builder.hpp

/*
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,    // Tensor Core MMA
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,  // SM90+ only
   cutlass::gemm::collective::StageCountAutoCarveout<128>,  // Auto stages
   cutlass::gemm::collective::KernelScheduleAuto  // Auto warp-spec
   >::CollectiveOp;
*/

// Placeholder for now — solution provided in solutions/ex01_gemm_basic.cu
struct CollectiveMainloop {};

// ============================================================================
// COLLECTIVE EPILOGUE — Output configuration
// ============================================================================

// TODO [EASY]: Complete the CollectiveBuilder typedef for the epilogue
// HINT: Epilogue handles C + D output with optional fusion

/*
using CollectiveEpilogue = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;
*/

struct CollectiveEpilogue {};

// ============================================================================
// GEMM KERNEL — Universal adapter
// ============================================================================

// TODO [MEDIUM]: Define the GemmUniversal kernel type
// HINT: GemmUniversal<Shape, CollectiveMainloop, CollectiveEpilogue>

/*
using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<
  cutlass::gemm::GemmUniversal<
    cutlass::gemm::GemmShape<M, N, K, 1>,  // Batch size = 1
    CollectiveMainloop,
    CollectiveEpilogue
  >
>;
*/

struct GemmKernel {};

// ============================================================================
// LAUNCH WRAPPER
// ============================================================================

template <typename GemmKernel_>
cudaError_t launch_gemm(
    const typename GemmKernel_::Arguments& args,
    cudaStream_t stream = 0
) {
    GemmKernel_ gemm_op;
    
    // Query workspace size
    size_t workspace_bytes = 0;
    GemmKernel_::get_workspace_size(args, &workspace_bytes);
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    // Launch
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
    std::cout << "=== Module 01, Exercise 01: Basic CollectiveBuilder GEMM ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Allocate device memory
    half *d_A, *d_B, *d_C, *d_D;
    size_t bytes_A = M * K * sizeof(half);
    size_t bytes_B = K * N * sizeof(half);
    size_t bytes_C = M * N * sizeof(half);
    size_t bytes_D = M * N * sizeof(half);
    
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_D, bytes_D);
    
    // Initialize matrices
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    cutlass_ref::init_matrix_zeros(d_C, M * N);
    cutlass_ref::init_matrix_zeros(d_D, M * N);
    
    // Reference: cuBLAS
    std::cout << "Running cuBLAS reference..." << std::endl;
    cutlass_ref::gemm_ref_fp16(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // TODO [EASY]: Build GemmKernel arguments and launch
    // HINT: typename GemmKernel::Arguments{...}
    // REF: cutlass/examples/60_collective_builder/60_collective_builder_gemm.cu
    
    /*
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},  // A pointer + leading dimension
        {d_B, N},  // B pointer + leading dimension
        {d_C, N},  // C pointer + leading dimension
        {d_D, N},  // D pointer + leading dimension
        {1.0f, 0.0f}  // alpha, beta
    };
    
    std::cout << "Running CUTLASS CollectiveBuilder GEMM..." << std::endl;
    cutlass_bench::GpuTimer timer;
    timer.start();
    CUDA_CHECK(launch_gemm<GemmKernel>(args));
    timer.stop();
    
    float elapsed_ms = timer.elapsed_ms();
    double tflops = cutlass_bench::compute_gemm_tflops(M, N, K, elapsed_ms);
    std::cout << "CUTLASS GEMM: " << elapsed_ms << " ms, " << tflops << " TFLOPS" << std::endl;
    */
    
    // Verify correctness
    std::cout << "\nVerifying against cuBLAS..." << std::endl;
    // cutlass_ref::verify_gemm(d_D, d_C, M * N);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: Did your predictions match the actual behavior?" << std::endl;
    std::cout << "C2: What does ncu show for sm__inst_executed_pipe_tensor.sum?" << std::endl;
    std::cout << "C3: How does this map to TRT-LLM linear layer implementation?" << std::endl;
    
    return 0;
}
