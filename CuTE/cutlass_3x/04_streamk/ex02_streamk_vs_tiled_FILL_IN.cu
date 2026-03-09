/*
 * Module 04 — StreamK Decomposition
 * Exercise 02 — StreamK vs Tiled Benchmark Comparison
 *
 * CUTLASS LAYER: Tile scheduler comparison
 *
 * WHAT YOU'RE BUILDING:
 *   Comprehensive benchmark comparing StreamK vs traditional tiled
 *   GEMM across multiple problem shapes. This tells you when to
 *   use each scheduling strategy.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your understanding of grid configuration from CuTe, now
 *   comparing automated scheduling strategies.
 *
 * OBJECTIVE:
 *   - Benchmark both schedulers across problem shapes
 *   - Identify crossover points
 *   - Build intuition for scheduler selection
 */

// PREDICT BEFORE COMPILING
// Q1: For which problem shapes will StreamK win?
// Q2: What's the overhead of StreamK for perfectly-aligned problems?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// PROBLEM SHAPE DEFINITIONS
// ============================================================================

struct ProblemShape {
    std::string name;
    int M, N, K;
    std::string description;
};

// Test various problem shapes
constexpr ProblemShape problems[] = {
    // Square, aligned (traditional should win or tie)
    {"square_aligned_1024", 1024, 1024, 1024, "Square, 1024-aligned"},
    {"square_aligned_4096", 4096, 4096, 4096, "Square, 4096-aligned"},
    
    // Square, misaligned (StreamK should win)
    {"square_misaligned_1025", 1025, 1025, 1025, "Square, 1025 (not aligned)"},
    {"square_misaligned_4100", 4100, 4100, 4100, "Square, 4100 (not aligned)"},
    
    // Tall/skinny (StreamK should win)
    {"tall_8192x512", 8192, 512, 4096, "Tall (Q projection)"},
    {"tall_16384x256", 16384, 256, 2048, "Very tall/skinny"},
    
    // Short/wide (StreamK should win)
    {"wide_512x8192", 512, 8192, 4096, "Wide (K/V projection)"},
    {"wide_256x16384", 256, 16384, 2048, "Very short/wide"},
    
    // MLP shapes (mixed)
    {"mlp_up_4096x16384", 4096, 16384, 4096, "MLP up-projection"},
    {"mlp_down_16384x4096", 16384, 4096, 16384, "MLP down-projection"},
    
    // Attention shapes
    {"attn_qk_32768x32768", 32768, 32768, 128, "Attention QK^T (S=4096, H=8)"},
};

// ============================================================================
// KERNEL CONFIGURATIONS
// ============================================================================

using ArchTag = cutlass::arch::Sm80;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cutlass::GemmlShape<128, 128, 64>;

// Traditional tiled GEMM
using CollectiveMainloopTraditional = typename cutlass::gemm::collective::CollectiveBuilder
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

using CollectiveEpilogueTraditional = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;

// StreamK GEMM
// TODO [MEDIUM]: Define StreamK collective mainloop
// HINT: Use KernelScheduleStreamK

/*
using CollectiveMainloopStreamK = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<128>,
   cutlass::gemm::collective::KernelScheduleStreamK
   >::CollectiveOp;
*/

struct CollectiveMainloopStreamK {};

using CollectiveEpilogueStreamK = typename cutlass::gemm::collective::CollectiveBuilder
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
// BENCHMARK FUNCTION (template on M, N, K)
// ============================================================================

template <int M, int N, int K, typename GemmKernel>
float benchmark_gemm(
    half_t* d_A, half_t* d_B, half_t* d_D,
    int warmup = 5, int iters = 50
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
// RESULTS TABLE
// ============================================================================

struct BenchmarkResult {
    std::string problem;
    int M, N, K;
    float time_traditional;
    float time_streamk;
    float speedup;  // traditional / streamk (>1 means StreamK wins)
    std::string winner;
};

void print_results_table(const std::vector<BenchmarkResult>& results) {
    std::cout << std::left << std::setw(25) << "Problem"
              << std::right << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(12) << "Trad(ms)"
              << std::setw(12) << "StreamK(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Winner" << std::endl;
    
    std::cout << std::string(103, '-') << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(25) << r.problem
                  << std::right << std::setw(8) << r.M
                  << std::setw(8) << r.N
                  << std::setw(8) << r.K
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.time_traditional
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.time_streamk
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.speedup
                  << std::setw(10) << r.winner << std::endl;
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 04, Exercise 02: StreamK vs Tiled Comparison ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Find max dimensions for buffer allocation
    int max_M = 0, max_N = 0, max_K = 0;
    for (const auto& p : problems) {
        max_M = max(max_M, p.M);
        max_N = max(max_N, p.N);
        max_K = max(max_K, p.K);
    }
    
    // Allocate device memory (max size)
    half_t *d_A, *d_B, *d_D_traditional, *d_D_streamk;
    
    size_t bytes_AB = max_M * max_K * sizeof(half_t);
    size_t bytes_out = max_M * max_N * sizeof(half_t);
    
    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_D_traditional, bytes_out);
    cudaMalloc(&d_D_streamk, bytes_out);
    
    // Initialize (use max size)
    cutlass_ref::init_matrix_random(d_A, max_M * max_K);
    cutlass_ref::init_matrix_random(d_B, max_K * max_N);
    
    std::vector<BenchmarkResult> results;
    
    // ========================================================================
    // BENCHMARK ALL PROBLEMS
    // ========================================================================
    
    std::cout << "=== RUNNING BENCHMARKS ===" << std::endl;
    
    for (const auto& problem : problems) {
        std::cout << "\nProblem: " << problem.name << " (" << problem.description << ")" << std::endl;
        std::cout << "  Shape: " << problem.M << "x" << problem.N << "x" << problem.K << std::endl;
        
        // TODO [MEDIUM]: Benchmark both traditional and StreamK for each problem
        // HINT: Use benchmark_gemm template function
        
        float time_traditional = 0.0f;  // Placeholder
        float time_streamk = 0.0f;       // Placeholder
        
        /*
        // Traditional tiled GEMM
        using GemmTraditional = cutlass::gemm::device::GemmUniversalAdapter<
            cutlass::gemm::GemmUniversal<
                cutlass::gemm::GemmShape<problem.M, problem.N, problem.K, 1>,
                CollectiveMainloopTraditional,
                CollectiveEpilogueTraditional
            >
        >;
        
        time_traditional = benchmark_gemm<problem.M, problem.N, problem.K, GemmTraditional>(
            d_A, d_B, d_D_traditional);
        
        // StreamK GEMM
        using GemmStreamK = cutlass::gemm::device::GemmUniversalAdapter<
            cutlass::gemm::GemmUniversal<
                cutlass::gemm::GemmShape<problem.M, problem.N, problem.K, 1>,
                CollectiveMainloopStreamK,
                CollectiveEpilogueStreamK
            >
        >;
        
        time_streamk = benchmark_gemm<problem.M, problem.N, problem.K, GemmStreamK>(
            d_A, d_B, d_D_streamk);
        */
        
        float speedup = time_traditional / time_streamk;
        std::string winner = (speedup > 1.05f) ? "StreamK" : 
                             (speedup < 0.95f) ? "Traditional" : "Tie";
        
        results.push_back({
            problem.name,
            problem.M, problem.N, problem.K,
            time_traditional, time_streamk, speedup, winner
        });
        
        std::cout << "  Traditional: " << time_traditional << " ms" << std::endl;
        std::cout << "  StreamK:     " << time_streamk << " ms" << std::endl;
        std::cout << "  Speedup:     " << speedup << "x (" << winner << ")" << std::endl;
    }
    
    // ========================================================================
    // PRINT RESULTS TABLE
    // ========================================================================
    
    std::cout << "\n=== RESULTS SUMMARY ===" << std::endl;
    print_results_table(results);
    
    // ========================================================================
    // ANALYSIS
    // ========================================================================
    
    std::cout << "\n=== ANALYSIS ===" << std::endl;
    
    // Count wins
    int streamk_wins = 0, traditional_wins = 0, ties = 0;
    for (const auto& r : results) {
        if (r.winner == "StreamK") streamk_wins++;
        else if (r.winner == "Traditional") traditional_wins++;
        else ties++;
    }
    
    std::cout << "Summary:" << std::endl;
    std::cout << "  StreamK wins: " << streamk_wins << "/" << results.size() << std::endl;
    std::cout << "  Traditional wins: " << traditional_wins << "/" << results.size() << std::endl;
    std::cout << "  Ties: " << ties << "/" << results.size() << std::endl;
    std::cout << std::endl;
    
    std::cout << "When to use StreamK:" << std::endl;
    std::cout << "  ✓ Irregular problem shapes (not divisible by tile size)" << std::endl;
    std::cout << "  ✓ Tall/skinny or short/wide matrices" << std::endl;
    std::cout << "  ✓ Batched GEMM with variable sizes" << std::endl;
    std::cout << "  ✓ MoE expert GEMMs" << std::endl;
    std::cout << std::endl;
    std::cout << "When to use Traditional:" << std::endl;
    std::cout << "  ✓ Square, aligned problems" << std::endl;
    std::cout << "  ✓ Production deployment with fixed shapes" << std::endl;
    std::cout << "  ✓ When minimal overhead is critical" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_traditional);
    cudaFree(d_D_streamk);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: Which problem shapes benefited most from StreamK?" << std::endl;
    std::cout << "C2: What was the overhead for aligned problems?" << std::endl;
    std::cout << "C3: When would you choose StreamK for production?" << std::endl;
    
    return 0;
}
