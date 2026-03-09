/*
 * Module 05 — Grouped GEMM
 * Exercise 01 — Basic Grouped GEMM (Fixed-Size Groups)
 *
 * CUTLASS LAYER: Grouped GEMM universal adapter
 *
 * WHAT YOU'RE BUILDING:
 *   Grouped GEMM for fixed-size expert batches. This is the
 *   foundation of MoE inference in TRT-LLM.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual batched GEMM from CuTe, now extended to
 *   handle variable problem sizes via pointer arrays.
 *
 * OBJECTIVE:
 *   - Configure grouped GEMM with pointer arrays
 *   - Execute multiple GEMMs in single launch
 *   - Measure speedup vs sequential launches
 */

// PREDICT BEFORE COMPILING
// Q1: What's the kernel launch overhead for 8 sequential GEMMs?
// Q2: Does grouped GEMM help if all GEMMs are the same size?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"

using namespace cutlass;

// ============================================================================
// SETUP — Group configuration
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

// All groups have the same size in this exercise
constexpr int GROUP_SIZE = 8;  // Number of GEMMs in the group
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ============================================================================
// BASELINE — Sequential GEMM launches
// ============================================================================

using TileShape = cutlass::GemmlShape<128, 128, 64>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<128>,
   cutlass::gemm::collective::KernelScheduleAuto
   >::CollectiveOp;

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

using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpilogue
    >
>;

// ============================================================================
// GROUPED GEMM CONFIGURATION
// ============================================================================
// Grouped GEMM uses a different kernel type that accepts:
//   - Array of problem sizes
//   - Array of pointers
//   - Number of groups

// TODO [MEDIUM]: Define grouped GEMM kernel type
// HINT: Use cutlass::gemm::device::GemmGrouped

/*
using GroupedGemmKernel = cutlass::gemm::device::GemmGrouped<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<_, _, _, _>,  // Dynamic sizes
        CollectiveMainloop,
        CollectiveEpilogue
    >
>;
*/

struct GroupedGemmKernel {};

// ============================================================================
// ARGUMENT STRUCTURES
// ============================================================================

struct GroupedGemmArgs {
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementA*> ptr_A;
    std::vector<ElementB*> ptr_B;
    std::vector<ElementC*> ptr_C;
    std::vector<ElementD*> ptr_D;
    std::vector<int> lda, ldb, ldc, ldd;
};

// ============================================================================
// SEQUENTIAL LAUNCH (BASELINE)
// ============================================================================

float launch_sequential_gemm(
    const std::vector<ElementA*>& d_A,
    const std::vector<ElementB*>& d_B,
    const std::vector<ElementD*>& d_D,
    int num_groups,
    int warmup = 5, int iters = 50
) {
    // Warmup
    for (int iter = 0; iter < warmup; ++iter) {
        for (int g = 0; g < num_groups; ++g) {
            typename GemmKernel::Arguments args{
                cutlass::gemm::GemmCoord{M, N, K},
                {d_A[g], K},
                {d_B[g], N},
                {d_D[g], N},
                {d_D[g], N},
                {1.0f, 0.0f}
            };
            
            GemmKernel gemm_op;
            size_t ws_size = 0;
            GemmKernel::get_workspace_size(args, &ws_size);
            void* ws = nullptr;
            if (ws_size > 0) cudaMalloc(&ws, ws_size);
            gemm_op.run(args, ws, 0);
            if (ws) cudaFree(ws);
        }
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cutlass_bench::GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < iters; ++iter) {
        for (int g = 0; g < num_groups; ++g) {
            typename GemmKernel::Arguments args{
                cutlass::gemm::GemmCoord{M, N, K},
                {d_A[g], K},
                {d_B[g], N},
                {d_D[g], N},
                {d_D[g], N},
                {1.0f, 0.0f}
            };
            
            GemmKernel gemm_op;
            size_t ws_size = 0;
            GemmKernel::get_workspace_size(args, &ws_size);
            void* ws = nullptr;
            if (ws_size > 0) cudaMalloc(&ws, ws_size);
            gemm_op.run(args, ws, 0);
            if (ws) cudaFree(ws);
        }
    }
    timer.stop();
    
    return timer.elapsed_ms() / iters;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 05, Exercise 01: Basic Grouped GEMM ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    std::cout << "Group configuration:" << std::endl;
    std::cout << "  Number of groups: " << GROUP_SIZE << std::endl;
    std::cout << "  Each GEMM: " << M << "x" << N << "x" << K << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory for all groups
    std::vector<ElementA*> d_A(GROUP_SIZE);
    std::vector<ElementB*> d_B(GROUP_SIZE);
    std::vector<ElementD*> d_D_sequential(GROUP_SIZE);
    std::vector<ElementD*> d_D_grouped(GROUP_SIZE);
    
    size_t bytes_per_gemm = M * K * sizeof(ElementA);
    size_t bytes_out = M * N * sizeof(ElementD);
    
    for (int g = 0; g < GROUP_SIZE; ++g) {
        cudaMalloc(&d_A[g], bytes_per_gemm);
        cudaMalloc(&d_B[g], K * N * sizeof(ElementB));
        cudaMalloc(&d_D_sequential[g], bytes_out);
        cudaMalloc(&d_D_grouped[g], bytes_out);
        
        cutlass_ref::init_matrix_random(d_A[g], M * K);
        cutlass_ref::init_matrix_random(d_B[g], K * N);
    }
    
    // ========================================================================
    // BASELINE: SEQUENTIAL LAUNCHES
    // ========================================================================
    
    std::cout << "=== BASELINE: Sequential GEMM Launches ===" << std::endl;
    std::cout << "Running " << GROUP_SIZE << " sequential GEMM launches..." << std::endl;
    
    float time_sequential = launch_sequential_gemm(
        d_A, d_B, d_D_sequential, GROUP_SIZE);
    
    std::cout << "  Total time: " << time_sequential << " ms" << std::endl;
    std::cout << "  Per-GEMM time: " << (time_sequential / GROUP_SIZE) << " ms" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // GROUPED GEMM
    // ========================================================================
    
    std::cout << "=== GROUPED GEMM ===" << std::endl;
    std::cout << "Running grouped GEMM (single launch)..." << std::endl;
    
    // TODO [MEDIUM]: Build grouped GEMM arguments and launch
    // HINT: Create GroupedGemmKernel::Arguments with pointer arrays
    
    /*
    // Build grouped GEMM arguments
    GroupedGemmKernel::Arguments grouped_args;
    
    for (int g = 0; g < GROUP_SIZE; ++g) {
        grouped_args.problem_sizes.push_back({M, N, K});
        grouped_args.ptr_A.push_back(d_A[g]);
        grouped_args.ptr_B.push_back(d_B[g]);
        grouped_args.ptr_C.push_back(nullptr);  // No C operand
        grouped_args.ptr_D.push_back(d_D_grouped[g]);
        grouped_args.lda.push_back(K);
        grouped_args.ldb.push_back(N);
        grouped_args.ldc.push_back(N);
        grouped_args.ldd.push_back(N);
    }
    
    // Warmup
    GroupedGemmKernel grouped_gemm_op;
    for (int i = 0; i < 5; ++i) {
        size_t ws_size = 0;
        GroupedGemmKernel::get_workspace_size(grouped_args, &ws_size);
        void* ws = nullptr;
        if (ws_size > 0) cudaMalloc(&ws, ws_size);
        grouped_gemm_op.run(grouped_args, ws, 0);
        if (ws) cudaFree(ws);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cutlass_bench::GpuTimer timer;
    timer.start();
    for (int i = 0; i < 50; ++i) {
        size_t ws_size = 0;
        GroupedGemmKernel::get_workspace_size(grouped_args, &ws_size);
        void* ws = nullptr;
        if (ws_size > 0) cudaMalloc(&ws, ws_size);
        grouped_gemm_op.run(grouped_args, ws, 0);
        if (ws) cudaFree(ws);
    }
    timer.stop();
    
    float time_grouped = timer.elapsed_ms() / 50;
    float speedup = time_sequential / time_grouped;
    
    std::cout << "  Total time: " << time_grouped << " ms" << std::endl;
    std::cout << "  Per-GEMM time: " << (time_grouped / GROUP_SIZE) << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    */
    
    // ========================================================================
    // ANALYSIS
    // ========================================================================
    
    std::cout << "\n=== ANALYSIS ===" << std::endl;
    std::cout << "Speedup sources:" << std::endl;
    std::cout << "  1. Kernel launch overhead eliminated" << std::endl;
    std::cout << "  2. Better GPU utilization (work stealing)" << std::endl;
    std::cout << "  3. Reduced host-side scheduling" << std::endl;
    std::cout << std::endl;
    std::cout << "For MoE with 8 experts:" << std::endl;
    std::cout << "  Expected speedup: 2-4×" << std::endl;
    std::cout << std::endl;
    std::cout << "For MoE with 64 experts:" << std::endl;
    std::cout << "  Expected speedup: 10-20×" << std::endl;
    
    // Cleanup
    for (int g = 0; g < GROUP_SIZE; ++g) {
        cudaFree(d_A[g]);
        cudaFree(d_B[g]);
        cudaFree(d_D_sequential[g]);
        cudaFree(d_D_grouped[g]);
    }
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What speedup did grouped GEMM provide?" << std::endl;
    std::cout << "C2: How much was kernel launch overhead?" << std::endl;
    std::cout << "C3: When would grouped GEMM help most?" << std::endl;
    
    return 0;
}
