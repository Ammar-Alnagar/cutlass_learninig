/*
 * Module 03 — Warp-Specialized GEMM
 * Exercise 01 — Basic Hopper Warp-Specialized GEMM
 *
 * CUTLASS LAYER: CollectiveMma with warp specialization
 *
 * WHAT YOU'RE BUILDING:
 *   SM90 Hopper GEMM with explicit producer/consumer warp split.
 *   This is the foundation of FA3 and all Hopper-optimized kernels.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual warp-level pipeline from CuTe Module 06, now
 *   automated by CollectiveBuilder with warp specialization.
 *
 * OBJECTIVE:
 *   - Enable warp specialization in CollectiveBuilder
 *   - Configure producer/consumer warp split
 *   - Measure speedup vs non-specialized baseline
 */

// PREDICT BEFORE COMPILING
// Q1: What's the optimal producer/consumer warp split for H100?
// Q2: How does warp specialization improve Tensor Core utilization?

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

using ArchTag = cutlass::arch::Sm90;

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cutlass::GemmlShape<128, 128, 64>;

// ============================================================================
// WARP SPECIALIZATION CONFIGURATION
// ============================================================================
// Key parameter: producer/consumer warp split
// 
// H100 has 128 warps per SM. Typical split:
//   - Producer warps: 4-8 (handle TMA loads)
//   - Consumer warps: 28-32 (handle MMA compute)
//   - Remaining: coordination, epilogue
//
// The split affects:
//   - Memory bandwidth utilization (more producers = better)
//   - Compute utilization (more consumers = better)
//   - Shared memory usage (more stages = better hiding)

// TODO [MEDIUM]: Configure warp-specialized CollectiveMainloop
// HINT: Use KernelScheduleWarpSpecialized instead of KernelScheduleAuto

/*
using CollectiveMainloopWS = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<230400>,  // H100: 230KB
   cutlass::gemm::collective::KernelScheduleWarpSpecialized     // <-- Key!
   >::CollectiveOp;
*/

// Non-specialized baseline for comparison
using CollectiveMainloopStandard = typename cutlass::gemm::collective::CollectiveBuilder
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

struct CollectiveMainloopWS {};

// ============================================================================
// COLLECTIVE EPILOGUE
// ============================================================================

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

// ============================================================================
// GEMM KERNEL TYPES
// ============================================================================

using GemmKernelWS = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloopWS,
        CollectiveEpilogue
    >
>;

using GemmKernelStandard = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloopStandard,
        CollectiveEpilogue
    >
>;

// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================

template <typename GemmKernel>
float benchmark_gemm(
    bfloat16_t* d_A, bfloat16_t* d_B, bfloat16_t* d_D,
    int M, int N, int K,
    int warmup = 10, int iters = 100
) {
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {d_D, N},
        {d_D, N},
        {1.0f, 0.0f}
    };
    
    // Warmup
    GemmKernel gemm_op;
    for (int i = 0; i < warmup; ++i) {
        size_t ws_size = 0;
        GemmKernel::get_workspace_size(args, &ws_size);
        void* ws = nullptr;
        if (ws_size > 0) cudaMalloc(&ws, ws_size);
        gemm_op.run(args, ws, 0);
        if (ws) cudaFree(ws);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cutlass_bench::GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        size_t ws_size = 0;
        GemmKernel::get_workspace_size(args, &ws_size);
        void* ws = nullptr;
        if (ws_size > 0) cudaMalloc(&ws, ws_size);
        gemm_op.run(args, ws, 0);
        if (ws) cudaFree(ws);
    }
    timer.stop();
    
    return timer.elapsed_ms() / iters;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 03, Exercise 01: Hopper Warp-Specialized GEMM ===" << std::endl;
    
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
        std::cout << "\nKey concepts:" << std::endl;
        std::cout << "  - Producer warps handle TMA loads" << std::endl;
        std::cout << "  - Consumer warps handle MMA compute" << std::endl;
        std::cout << "  - Expected speedup: 1.5-2.0x on H100" << std::endl;
        return 0;
    }
    
    std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Warps per SM: " << (prop.maxThreadsPerMultiProcessor / 32) << std::endl;
    std::cout << "  Shared Memory per SM: " << (prop.sharedMemoryPerMultiprocessor / 1024) << " KB" << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory
    bfloat16_t *d_A, *d_B, *d_D_ws, *d_D_standard;
    
    size_t bytes = M * K * sizeof(bfloat16_t);
    size_t bytes_out = M * N * sizeof(bfloat16_t);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_D_ws, bytes_out);
    cudaMalloc(&d_D_standard, bytes_out);
    
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    std::cout << "Problem: GEMM " << M << "x" << N << "x" << K << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // BASELINE: Standard (non-specialized) GEMM
    // ========================================================================
    
    std::cout << "=== BASELINE: Standard GEMM ===" << std::endl;
    std::cout << "Running standard (non-warp-specialized) GEMM..." << std::endl;
    
    float time_standard = benchmark_gemm<GemmKernelStandard>(
        d_A, d_B, d_D_standard, M, N, K);
    
    double tflops_standard = cutlass_bench::compute_gemm_tflops(M, N, K, time_standard);
    std::cout << "  Time: " << time_standard << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_standard << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // WARP-SPECIALIZED GEMM
    // ========================================================================
    
    std::cout << "=== WARP-SPECIALIZED GEMM ===" << std::endl;
    std::cout << "Running warp-specialized GEMM..." << std::endl;
    
    // TODO [MEDIUM]: Benchmark warp-specialized GEMM
    /*
    float time_ws = benchmark_gemm<GemmKernelWS>(
        d_A, d_B, d_D_ws, M, N, K);
    
    double tflops_ws = cutlass_bench::compute_gemm_tflops(M, N, K, time_ws);
    float speedup = time_standard / time_ws;
    
    std::cout << "  Time: " << time_ws << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_ws << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    */
    
    // ========================================================================
    // WARP SPLIT ANALYSIS
    // ========================================================================
    
    std::cout << "\n=== WARP SPLIT ANALYSIS ===" << std::endl;
    std::cout << "H100 warp configuration:" << std::endl;
    std::cout << "  Total warps per SM: 128" << std::endl;
    std::cout << "  Typical producer warps: 4-8" << std::endl;
    std::cout << "  Typical consumer warps: 28-32" << std::endl;
    std::cout << "  Remaining: coordination, epilogue" << std::endl;
    std::cout << std::endl;
    std::cout << "The CollectiveBuilder auto-selects optimal split based on:" << std::endl;
    std::cout << "  - Tile size (larger tiles need more producers)" << std::endl;
    std::cout << "  - Pipeline stages (more stages need more coordination)" << std::endl;
    std::cout << "  - Problem shape (M/N/K ratio affects load/compute balance)" << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Verify warp specialization with ncu:" << std::endl;
    std::cout << "  ncu --metrics smsp__thread_inst_executed_per_pipe_tensor.ratio,\\" << std::endl;
    std::cout << "              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\" << std::endl;
    std::cout << "              smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct \\" << std::endl;
    std::cout << "       ./hopper_ws_gemm" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected observations:" << std::endl;
    std::cout << "  - Higher tensor instruction ratio vs standard GEMM" << std::endl;
    std::cout << "  - Higher load bandwidth (producer warps feeding consumers)" << std::endl;
    std::cout << "  - Lower stall percentage (better latency hiding)" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_ws);
    cudaFree(d_D_standard);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What speedup did warp specialization provide?" << std::endl;
    std::cout << "C2: How does the producer/consumer split affect performance?" << std::endl;
    std::cout << "C3: Why is warp specialization critical for FA3?" << std::endl;
    
    return 0;
}
