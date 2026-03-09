/*
 * Module 03 — Warp-Specialized GEMM
 * Exercise 02 — Ping-Pong Pipeline
 *
 * CUTLASS LAYER: Warp-specialized mainloop with ping-pong buffering
 *
 * WHAT YOU'RE BUILDING:
 *   Advanced warp-specialized GEMM with ping-pong pipeline for
 *   even better latency hiding. Used in production FA3 implementations.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual double-buffering from CuTe Module 06, now
 *   extended to multi-buffer ping-pong with warp specialization.
 *
 * OBJECTIVE:
 *   - Configure multi-stage ping-pong pipeline
 *   - Understand buffer switching overhead
 *   - Measure latency hiding improvement
 */

// PREDICT BEFORE COMPILING
// Q1: How does ping-pong differ from standard multi-stage pipeline?
// Q2: What's the tradeoff between more stages and shared memory usage?

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
// SETUP — Hopper configuration
// ============================================================================

using ArchTag = cutlass::arch::Sm90;

constexpr int M = 8192;  // Larger problem to show latency hiding
constexpr int N = 8192;
constexpr int K = 8192;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ============================================================================
// PIPELINE STAGE CONFIGURATION
// ============================================================================
// Ping-pong pipeline uses multiple buffer "slots" in shared memory.
// While consumer processes slot N, producer loads slot N+1.
//
// Stage count tradeoffs:
//   - More stages = better latency hiding
//   - More stages = more shared memory usage
//   - Optimal: enough stages to cover memory latency
//
// H100 memory latency: ~500 cycles
// At 1.5 GHz: ~330 ns
// GDDR6X bandwidth: 3.35 TB/s
// Tile size (128x128 BF16): ~32 KB
// Load time: ~10 ns per tile
// Stages needed: ~5-8 to cover latency

// TODO [HARD]: Configure ping-pong pipeline with different stage counts
// HINT: StageCountAutoCarveout<smem_capacity> controls stages

/*
// Conservative: 4 stages (less smem, more stalls)
using TileShape4Stage = cutlass::GemmlShape<128, 128, 64>;
using CollectiveMainloop4Stage = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape4Stage,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<4>,  // 4 stages
   cutlass::gemm::collective::KernelScheduleWarpSpecialized
   >::CollectiveOp;

// Aggressive: 8 stages (more smem, better hiding)
using TileShape8Stage = cutlass::GemmlShape<128, 128, 64>;
using CollectiveMainloop8Stage = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape8Stage,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<8>,  // 8 stages
   cutlass::gemm::collective::KernelScheduleWarpSpecialized
   >::CollectiveOp;
*/

struct CollectiveMainloop4Stage {};
struct CollectiveMainloop8Stage {};

// ============================================================================
// COLLECTIVE EPILOGUE
// ============================================================================

using CollectiveEpilogue = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   cutlass::GemmlShape<128, 128, 64>,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;

// ============================================================================
// GEMM KERNEL TYPES
// ============================================================================

using GemmKernel4Stage = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop4Stage,
        CollectiveEpilogue
    >
>;

using GemmKernel8Stage = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop8Stage,
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
    int warmup = 10, int iters = 50
) {
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {d_D, N},
        {d_D, N},
        {1.0f, 0.0f}
    };
    
    GemmKernel gemm_op;
    
    // Warmup
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
// SHARED MEMORY ANALYSIS
// ============================================================================

void analyze_smem_usage(int stages) {
    // Per-stage shared memory usage (approximate)
    // For 128x128x64 tile with BF16:
    //   - A tile: 128 * 64 * 2 bytes = 16 KB
    //   - B tile: 64 * 128 * 2 bytes = 16 KB
    //   - Total per stage: ~32 KB
    
    int smem_per_stage = 32 * 1024;  // 32 KB
    int total_smem = stages * smem_per_stage;
    
    std::cout << "  Stages: " << stages << std::endl;
    std::cout << "  SMem per stage: ~" << (smem_per_stage / 1024) << " KB" << std::endl;
    std::cout << "  Total SMem: ~" << (total_smem / 1024) << " KB" << std::endl;
    std::cout << "  H100 SMem capacity: 230 KB" << std::endl;
    std::cout << "  SMem utilization: " << (100 * total_smem / (230 * 1024)) << "%" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 03, Exercise 02: Ping-Pong Pipeline ===" << std::endl;
    
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
    bfloat16_t *d_A, *d_B, *d_D_4stage, *d_D_8stage;
    
    size_t bytes = M * K * sizeof(bfloat16_t);
    size_t bytes_out = M * N * sizeof(bfloat16_t);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_D_4stage, bytes_out);
    cudaMalloc(&d_D_8stage, bytes_out);
    
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    std::cout << "Problem: GEMM " << M << "x" << N << "x" << K << std::endl;
    std::cout << "(Large problem to demonstrate latency hiding)" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // SHARED MEMORY ANALYSIS
    // ========================================================================
    
    std::cout << "=== SHARED MEMORY ANALYSIS ===" << std::endl;
    std::cout << "4-stage pipeline:" << std::endl;
    analyze_smem_usage(4);
    std::cout << std::endl;
    std::cout << "8-stage pipeline:" << std::endl;
    analyze_smem_usage(8);
    std::cout << std::endl;
    
    // ========================================================================
    // BENCHMARK DIFFERENT STAGE COUNTS
    // ========================================================================
    
    std::cout << "=== BENCHMARK RESULTS ===" << std::endl;
    
    // 4-stage pipeline
    std::cout << "Running 4-stage ping-pong pipeline..." << std::endl;
    /*
    float time_4stage = benchmark_gemm<GemmKernel4Stage>(
        d_A, d_B, d_D_4stage, M, N, K);
    double tflops_4stage = cutlass_bench::compute_gemm_tflops(M, N, K, time_4stage);
    std::cout << "  Time: " << time_4stage << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_4stage << std::endl;
    */
    
    // 8-stage pipeline
    std::cout << "Running 8-stage ping-pong pipeline..." << std::endl;
    /*
    float time_8stage = benchmark_gemm<GemmKernel8Stage>(
        d_A, d_B, d_D_8stage, M, N, K);
    double tflops_8stage = cutlass_bench::compute_gemm_tflops(M, N, K, time_8stage);
    float speedup = time_4stage / time_8stage;
    std::cout << "  Time: " << time_8stage << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_8stage << std::endl;
    std::cout << "  Speedup (8 vs 4 stage): " << speedup << "x" << std::endl;
    */
    
    // ========================================================================
    // LATENCY HIDING ANALYSIS
    // ========================================================================
    
    std::cout << "\n=== LATENCY HIDING ANALYSIS ===" << std::endl;
    std::cout << "H100 memory characteristics:" << std::endl;
    std::cout << "  Memory latency: ~500 cycles" << std::endl;
    std::cout << "  At 1.5 GHz: ~330 ns" << std::endl;
    std::cout << "  GDDR6X bandwidth: 3.35 TB/s" << std::endl;
    std::cout << std::endl;
    std::cout << "Tile load time (128x128x64 BF16, ~32 KB):" << std::endl;
    std::cout << "  ~10 ns per tile" << std::endl;
    std::cout << std::endl;
    std::cout << "Stages needed to cover latency:" << std::endl;
    std::cout << "  500 cycles / (load time in cycles) ≈ 5-8 stages" << std::endl;
    std::cout << std::endl;
    std::cout << "Optimal configuration:" << std::endl;
    std::cout << "  - Small problems (K < 2048): 4 stages sufficient" << std::endl;
    std::cout << "  - Medium problems (K 2048-8192): 6 stages recommended" << std::endl;
    std::cout << "  - Large problems (K > 8192): 8 stages for max throughput" << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Verify pipeline efficiency with ncu:" << std::endl;
    std::cout << "  ncu --metrics smsp__warp_issue_stalled_mem_barrier_per_warp_active.pct,\\" << std::endl;
    std::cout << "              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \\" << std::endl;
    std::cout << "       ./pingpong_pipeline" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected observations:" << std::endl;
    std::cout << "  - 8-stage has lower stall percentage than 4-stage" << std::endl;
    std::cout << "  - 8-stage has higher load bandwidth (better pipelining)" << std::endl;
    std::cout << "  - Diminishing returns beyond 8 stages (smem pressure)" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_4stage);
    cudaFree(d_D_8stage);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What speedup did 8-stage provide over 4-stage?" << std::endl;
    std::cout << "C2: When would you choose fewer stages?" << std::endl;
    std::cout << "C3: How does ping-pong relate to FA3 implementation?" << std::endl;
    
    return 0;
}
