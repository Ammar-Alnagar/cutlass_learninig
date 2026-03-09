/*
 * Module 04 — StreamK Decomposition
 * Exercise 01 — Basic StreamK GEMM
 *
 * CUTLASS LAYER: StreamK tile scheduler
 *
 * WHAT YOU'RE BUILDING:
 *   StreamK GEMM for irregular problem shapes. This fixes the
 *   wave quantization problem that hurts traditional tiled GEMM.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual grid-stride loops from CuTe, now automated by
 *   StreamK's dynamic work unit scheduler.
 *
 * OBJECTIVE:
 *   - Enable StreamK scheduling in CollectiveBuilder
 *   - Understand work unit decomposition
 *   - Measure improvement for irregular shapes
 */

// PREDICT BEFORE COMPILING
// Q1: For M=4100 with tile=128, how much utilization is lost to wave quantization?
// Q2: Does StreamK add overhead for perfectly-aligned problems?

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
// SETUP — Problem definition (irregular shape)
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

// Irregular problem shape (not divisible by typical tile sizes)
constexpr int M = 4100;  // 4100 % 128 = 4 (wasteful!)
constexpr int N = 3072;
constexpr int K = 8192;

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
// WAVE QUANTIZATION ANALYSIS
// ============================================================================

void analyze_wave_quantization(int M, int N, int tile_m, int tile_n) {
    int tiles_m = (M + tile_m - 1) / tile_m;
    int tiles_n = (N + tile_n - 1) / tile_n;
    int total_tiles = tiles_m * tiles_n;
    
    int full_tile_m = M / tile_m;
    int full_tile_n = N / tile_n;
    int full_tiles = full_tile_m * full_tile_n;
    
    int partial_tiles = total_tiles - full_tiles;
    
    // Compute wasted elements
    int partial_m_elements = M % tile_m;
    int partial_n_elements = N % tile_n;
    if (partial_m_elements == 0) partial_m_elements = tile_m;
    if (partial_n_elements == 0) partial_n_elements = tile_n;
    
    float utilization = 100.0f * M * N / (tiles_m * tile_m * tiles_n * tile_n);
    
    std::cout << "  Tile size: " << tile_m << "x" << tile_n << std::endl;
    std::cout << "  Total tiles: " << total_tiles << " (" << tiles_m << " × " << tiles_n << ")" << std::endl;
    std::cout << "  Full tiles: " << full_tiles << std::endl;
    std::cout << "  Partial tiles: " << partial_tiles << std::endl;
    std::cout << "  Utilization: " << utilization << "%" << std::endl;
    std::cout << "  Wasted: " << (100 - utilization) << "%" << std::endl;
}

// ============================================================================
// TRADITIONAL TILED GEMM
// ============================================================================

using TileShapeTraditional = cutlass::GemmlShape<128, 128, 64>;

using CollectiveMainloopTraditional = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShapeTraditional,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<128>,
   cutlass::gemm::collective::KernelScheduleAuto  // Traditional scheduling
   >::CollectiveOp;

using CollectiveEpilogueTraditional = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShapeTraditional,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;

using GemmKernelTraditional = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloopTraditional,
        CollectiveEpilogueTraditional
    >
>;

// ============================================================================
// STREAMK GEMM
// ============================================================================
// StreamK uses a different tile scheduler:
//   - Divides K dimension into fine-grained work units
//   - Dynamically assigns work units to SMs
//   - Reduces partial sums across work units

// TODO [MEDIUM]: Configure StreamK scheduling
// HINT: Use KernelScheduleStreamK instead of KernelScheduleAuto

/*
using TileShapeStreamK = cutlass::GemmlShape<128, 128, 64>;

using CollectiveMainloopStreamK = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShapeStreamK,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<128>,
   cutlass::gemm::collective::KernelScheduleStreamK  // <-- StreamK!
   >::CollectiveOp;
*/

struct CollectiveMainloopStreamK {};

using CollectiveEpilogueStreamK = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShapeStreamK,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;

using GemmKernelStreamK = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloopStreamK,
        CollectiveEpilogueStreamK
    >
>;

// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================

template <typename GemmKernel>
float benchmark_gemm(
    half_t* d_A, half_t* d_B, half_t* d_D,
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
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 04, Exercise 01: StreamK GEMM ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Analyze wave quantization
    std::cout << "=== WAVE QUANTIZATION ANALYSIS ===" << std::endl;
    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << std::endl;
    std::cout << "Traditional tiled GEMM:" << std::endl;
    analyze_wave_quantization(M, N, 128, 128);
    std::cout << std::endl;
    
    // Allocate device memory
    half_t *d_A, *d_B, *d_D_traditional, *d_D_streamk;
    
    size_t bytes_AB = M * K * sizeof(half_t);
    size_t bytes_out = M * N * sizeof(half_t);
    
    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_D_traditional, bytes_out);
    cudaMalloc(&d_D_streamk, bytes_out);
    
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    // ========================================================================
    // BENCHMARK TRADITIONAL TILED GEMM
    // ========================================================================
    
    std::cout << "=== BENCHMARK RESULTS ===" << std::endl;
    std::cout << "Running traditional tiled GEMM..." << std::endl;
    
    float time_traditional = benchmark_gemm<GemmKernelTraditional>(
        d_A, d_B, d_D_traditional, M, N, K);
    
    double tflops_traditional = cutlass_bench::compute_gemm_tflops(M, N, K, time_traditional);
    std::cout << "  Time: " << time_traditional << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_traditional << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // BENCHMARK STREAMK GEMM
    // ========================================================================
    
    std::cout << "Running StreamK GEMM..." << std::endl;
    
    // TODO [MEDIUM]: Benchmark StreamK GEMM
    /*
    float time_streamk = benchmark_gemm<GemmKernelStreamK>(
        d_A, d_B, d_D_streamk, M, N, K);
    
    double tflops_streamk = cutlass_bench::compute_gemm_tflops(M, N, K, time_streamk);
    float speedup = time_traditional / time_streamk;
    
    std::cout << "  Time: " << time_streamk << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_streamk << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    */
    
    // ========================================================================
    // STREAMK ANALYSIS
    // ========================================================================
    
    std::cout << "\n=== STREAMK ANALYSIS ===" << std::endl;
    std::cout << "StreamK work unit decomposition:" << std::endl;
    std::cout << "  K dimension (" << K << ") divided into work units" << std::endl;
    std::cout << "  Each work unit: fixed K-slice (e.g., K/64)" << std::endl;
    std::cout << "  SMs dynamically pick up work units" << std::endl;
    std::cout << "  Partial sums reduced at the end" << std::endl;
    std::cout << std::endl;
    std::cout << "Benefits for irregular shapes:" << std::endl;
    std::cout << "  - No wave quantization waste" << std::endl;
    std::cout << "  - Better load balancing across SMs" << std::endl;
    std::cout << "  - Works well for variable K (MoE, ragged batch)" << std::endl;
    std::cout << std::endl;
    std::cout << "Tradeoffs:" << std::endl;
    std::cout << "  - Partial sum reduction overhead" << std::endl;
    std::cout << "  - May be slower for perfectly-aligned problems" << std::endl;
    std::cout << "  - Best for: irregular shapes, batched GEMM, MoE" << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Compare StreamK vs traditional with ncu:" << std::endl;
    std::cout << "  ncu --metrics smsp__inst_executed.sum,\\" << std::endl;
    std::cout << "              l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \\" << std::endl;
    std::cout << "       ./streamk_gemm" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected observations:" << std::endl;
    std::cout << "  - StreamK has more instructions (reduction overhead)" << std::endl;
    std::cout << "  - StreamK has similar or fewer global stores" << std::endl;
    std::cout << "  - StreamK shows better SM utilization for irregular shapes" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_traditional);
    cudaFree(d_D_streamk);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What speedup did StreamK provide for this irregular shape?" << std::endl;
    std::cout << "C2: When would traditional tiled GEMM be better?" << std::endl;
    std::cout << "C3: How does StreamK help with MoE expert GEMMs?" << std::endl;
    
    return 0;
}
