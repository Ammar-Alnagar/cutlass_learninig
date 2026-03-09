/*
 * Module 01 — CollectiveBuilder Anatomy
 * Exercise 02 — Tile Sizes, Stages, and Data Types
 *
 * CUTLASS LAYER: CollectiveBuilder configuration tuning
 *
 * WHAT YOU'RE BUILDING:
 *   Production GEMM with explicit tile size and stage tuning.
 *   This is what you configure when targeting specific GPU SKUs
 *   in TRT-LLM deployment.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   TileShape maps to the M/N/K tile dimensions you set in TiledMMA.
 *   StageCount maps to your manual pipeline stage count from CuTe Module 06.
 *
 * OBJECTIVE:
 *   - Tune tile sizes for different problem shapes
 *   - Configure pipeline stages for latency hiding
 *   - Compare FP16 vs BF16 performance
 */

// PREDICT BEFORE COMPILING
// Q1: How does increasing K-tile size affect shared memory usage?
// Q2: What's the tradeoff between more stages vs register pressure?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <tuple>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// CONFIGURATION VARIANTS — Compare different tile/stage configs
// ============================================================================

struct GemmConfig {
    const char* name;
    int tile_m;
    int tile_n;
    int tile_k;
    int stages;
};

// TODO [MEDIUM]: Define 3 configurations to compare
// 1. Small tiles (64x64x32) — good for small problems
// 2. Medium tiles (128x128x64) — balanced
// 3. Large tiles (256x256x128) — good for large problems
// HINT: Use StageCountAutoCarveout<smem_capacity>

constexpr GemmConfig configs[] = {
    // TODO: Fill in configurations
    {"small_tiles", 64, 64, 32, 3},
    {"medium_tiles", 128, 128, 64, 4},
    {"large_tiles", 256, 256, 128, 5},
};

// ============================================================================
// PROBLEM SIZES — Test different GEMM shapes
// ============================================================================

struct ProblemSize {
    const char* name;
    int M, N, K;
};

constexpr ProblemSize problems[] = {
    {"square_1024", 1024, 1024, 1024},
    {"square_4096", 4096, 4096, 4096},
    {"tall_8192x1024", 8192, 1024, 4096},    // Q projection in attention
    {"wide_1024x8192", 1024, 8192, 4096},    // K/V projection in attention
    {"mlp_inner", 4096, 16384, 4096},        // MLP up projection
};

// ============================================================================
// COLLECTIVE BUILDER TEMPLATE — Parameterized by config
// ============================================================================

template <int TileM, int TileN, int TileK, int Stages, typename ElementAccumulator>
using MakeCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
  <cutlass::arch::Sm80,
   cutlass::arch::OpClassTensorOp,
   cutlass::half_t, cutlass::layout::RowMajor,      // A
   cutlass::half_t, cutlass::layout::ColumnMajor,   // B
   ElementAccumulator,
   cutlass::GemmlShape<TileM, TileN, TileK>,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<Stages>,
   cutlass::gemm::collective::KernelScheduleAuto
   >::CollectiveOp;

template <int TileM, int TileN, int TileK, int Stages, typename ElementAccumulator>
using MakeCollectiveEpilogue = typename cutlass::gemm::collective::CollectiveBuilder
  <cutlass::arch::Sm80,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   cutlass::GemmlShape<TileM, TileN, TileK>,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::half_t, cutlass::layout::RowMajor,  // C
   cutlass::half_t, cutlass::layout::RowMajor,  // D
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;

// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================

template <typename CollectiveMainloop, typename CollectiveEpilogue>
float benchmark_gemm_config(
    int M, int N, int K,
    cutlass::half_t* d_A, cutlass::half_t* d_B,
    cutlass::half_t* d_C, cutlass::half_t* d_D,
    int warmup = 10, int iters = 100
) {
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
    std::cout << "=== Module 01, Exercise 02: Tile Sizes and Stages ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Allocate buffers (use max size)
    constexpr int MAX_M = 8192;
    constexpr int MAX_N = 16384;
    constexpr int MAX_K = 16384;
    
    cutlass::half_t *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, MAX_M * MAX_K * sizeof(cutlass::half_t));
    cudaMalloc(&d_B, MAX_K * MAX_N * sizeof(cutlass::half_t));
    cudaMalloc(&d_C, MAX_M * MAX_N * sizeof(cutlass::half_t));
    cudaMalloc(&d_D, MAX_M * MAX_N * sizeof(cutlass::half_t));
    
    cutlass_ref::init_matrix_random(d_A, MAX_M * MAX_K);
    cutlass_ref::init_matrix_random(d_B, MAX_K * MAX_N);
    
    std::vector<cutlass_bench::BenchmarkResult> results;
    
    // TODO [MEDIUM]: Run benchmarks for each config × problem combination
    // HINT: Nested loop over configs[] and problems[]
    // Record results in the results vector
    
    for (const auto& cfg : configs) {
        std::cout << "\n--- Config: " << cfg.name 
                  << " (tile=" << cfg.tile_m << "x" << cfg.tile_n << "x" << cfg.tile_k
                  << ", stages=" << cfg.stages << ") ---" << std::endl;
        
        for (const auto& prob : problems) {
            std::cout << "  Problem: " << prob.name 
                      << " (" << prob.M << "x" << prob.N << "x" << prob.K << ")... ";
            
            // TODO: Instantiate CollectiveMainloop and CollectiveEpilogue with config
            // TODO: Call benchmark_gemm_config and record result
            
            float elapsed_ms = 0.0f;  // Placeholder
            double tflops = cutlass_bench::compute_gemm_tflops(prob.M, prob.N, prob.K, elapsed_ms);
            
            results.push_back({
                std::string(cfg.name) + "_" + prob.name,
                prob.M, prob.N, prob.K,
                elapsed_ms, tflops, 0.0
            });
            
            std::cout << elapsed_ms << " ms, " << tflops << " TFLOPS" << std::endl;
        }
    }
    
    // Print summary table
    cutlass_bench::print_benchmark_table(results);
    
    // TODO [HARD]: Analyze results
    // - Which config wins for each problem shape?
    // - Why do large tiles hurt small problems?
    // - What's the optimal stage count for your GPU?
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: Which tile configuration won for square_4096?" << std::endl;
    std::cout << "C2: How does stage count affect occupancy?" << std::endl;
    std::cout << "C3: What would you change for Hopper (SM90) with TMA?" << std::endl;
    
    return 0;
}
