/*
 * Module 02 — Epilogue Visitor Tree (EVT)
 * Exercise 02 — ReLU and GELU Activation Fusion
 *
 * CUTLASS LAYER: EVT with activation operations
 *
 * WHAT YOU'RE BUILDING:
 *   Fused GEMM + activation — the core of Transformer MLP layers.
 *   This is exactly what TRT-LLM uses for dense MLP inference.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual activation epilogue from CuTe Module 04, now
 *   replaced with composable EVT activation nodes.
 *
 * OBJECTIVE:
 *   - Fuse ReLU activation into GEMM epilogue
 *   - Fuse GELU activation (LLM standard)
 *   - Compare activation fusion speedups
 */

// PREDICT BEFORE COMPILING
// Q1: ReLU is simpler than GELU — will ReLU fusion be faster?
// Q2: What's the theoretical speedup limit for activation fusion?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// SETUP — Problem definition
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 16384;  // MLP inner dimension (4x hidden)

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

// ============================================================================
// ACTIVATION FUNCTIONS — Reference implementations
// ============================================================================

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float gelu(float x) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x * x * x)));
}

__device__ __forceinline__ float fast_gelu(float x) {
    // Faster GELU approximation used in practice
    constexpr float kAlpha = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + 0.044715f * x * x * x)));
}

// ============================================================================
// UNFUSED REFERENCE — Separate activation kernel
// ============================================================================

template <typename ActivationFn>
__global__ void activation_kernel(
    half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = float(output[idx]);
        val = ActivationFn{}(val);
        output[idx] = __float2half(val);
    }
}

struct ReLU { __device__ float operator()(float x) { return relu(x); } };
struct GELU { __device__ float operator()(float x) { return gelu(x); } };
struct FastGELU { __device__ float operator()(float x) { return fast_gelu(x); } };

template <typename ActivationFn>
void launch_unfused_gemm_activation(
    half* d_A, half* d_B, half* d_D,
    int M, int N, int K
) {
    // Step 1: GEMM
    cutlass_ref::gemm_ref_fp16(M, N, K, d_A, d_B, d_D, 1.0f, 0.0f);
    
    // Step 2: Separate activation kernel
    int size = M * N;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    activation_kernel<ActivationFn><<<blocks, threads>>>(d_D, size);
}

// ============================================================================
// EVT CONFIGURATION — Activation Fusion
// ============================================================================
// EVT provides built-in activation visitors:
//   - EpilogueVisitorReLu
//   - EpilogueVisitorGelu
//   - EpilogueVisitorFastGelu
//   - EpilogueVisitorSigmoid
//   - EpilogueVisitorTanh

// TODO [MEDIUM]: Define EVT operations for GEMM + ReLU
// HINT: cutlass::epilogue::collective::EpilogueVisitorReLu

/*
using EvtOperationsReLU = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,
    cutlass::epilogue::collective::EpilogueVisitorReLu<ElementAccumulator>
>;
*/

// TODO [MEDIUM]: Define EVT operations for GEMM + GELU
// HINT: cutlass::epilogue::collective::EpilogueVisitorGelu

/*
using EvtOperationsGELU = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,
    cutlass::epilogue::collective::EpilogueVisitorGelu<ElementAccumulator>
>;
*/

// TODO [MEDIUM]: Define EVT operations for GEMM + FastGELU
// HINT: cutlass::epilogue::collective::EpilogueVisitorFastGelu

/*
using EvtOperationsFastGELU = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,
    cutlass::epilogue::collective::EpilogueVisitorFastGelu<ElementAccumulator>
>;
*/

struct EvtOperationsReLU {};
struct EvtOperationsGELU {};
struct EvtOperationsFastGELU {};

// ============================================================================
// COLLECTIVE MAINLOOP — Standard GEMM
// ============================================================================

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

// ============================================================================
// COLLECTIVE EPILOGUE — With Activation Fusion
// ============================================================================

// TODO [MEDIUM]: Configure CollectiveEpilogue with ReLU EVT
/*
using CollectiveEpilogueReLU = typename cutlass::epilogue::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto,
   EvtOperationsReLU
   >::CollectiveOp;
*/

// TODO [MEDIUM]: Configure CollectiveEpilogue with GELU EVT
/*
using CollectiveEpilogueGELU = typename cutlass::epilogue::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto,
   EvtOperationsGELU
   >::CollectiveOp;
*/

struct CollectiveEpilogueReLU {};
struct CollectiveEpilogueGELU {};

// ============================================================================
// GEMM KERNEL TYPES
// ============================================================================

using GemmKernelReLU = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpilogueReLU
    >
>;

using GemmKernelGELU = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpilogueGELU
    >
>;

// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================

template <typename GemmKernel>
float benchmark_fused_gemm(
    half* d_A, half* d_B, half* d_D,
    int M, int N, int K,
    int warmup = 10, int iters = 100
) {
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {d_D, N},  // C = D for in-place
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
    std::cout << "=== Module 02, Exercise 02: EVT Activation Fusion ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Allocate device memory
    half *d_A, *d_B, *d_D_relu, *d_D_gelu, *d_D_unfused;
    
    size_t bytes_AB = M * K * sizeof(half);
    size_t bytes_out = M * N * sizeof(half);
    
    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_D_relu, bytes_out);
    cudaMalloc(&d_D_gelu, bytes_out);
    cudaMalloc(&d_D_unfused, bytes_out);
    
    // Initialize
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    std::cout << "Problem: GEMM " << M << "x" << N << "x" << K << " (MLP up-projection)" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // UNFUSED BASELINES
    // ========================================================================
    
    std::cout << "=== UNFUSED BASELINES ===" << std::endl;
    
    // ReLU unfused
    std::cout << "Running UNFUSED GEMM + ReLU..." << std::endl;
    cutlass_bench::GpuTimer timer_relu_unfused;
    timer_relu_unfused.start();
    launch_unfused_gemm_activation<ReLU>(d_A, d_B, d_D_unfused, M, N, K);
    timer_relu_unfused.stop();
    float time_relu_unfused = timer_relu_unfused.elapsed_ms();
    std::cout << "  Time: " << time_relu_unfused << " ms" << std::endl;
    
    // GELU unfused
    std::cout << "Running UNFUSED GEMM + GELU..." << std::endl;
    cutlass_bench::GpuTimer timer_gelu_unfused;
    timer_gelu_unfused.start();
    launch_unfused_gemm_activation<GELU>(d_A, d_B, d_D_unfused, M, N, K);
    timer_gelu_unfused.stop();
    float time_gelu_unfused = timer_gelu_unfused.elapsed_ms();
    std::cout << "  Time: " << time_gelu_unfused << " ms" << std::endl;
    
    // ========================================================================
    // EVT-FUSED GEMMS
    // ========================================================================
    
    std::cout << "\n=== EVT-FUSED GEMMS ===" << std::endl;
    
    // TODO [MEDIUM]: Benchmark fused ReLU GEMM
    /*
    std::cout << "Running EVT-FUSED GEMM + ReLU..." << std::endl;
    float time_relu_fused = benchmark_fused_gemm<GemmKernelReLU>(
        d_A, d_B, d_D_relu, M, N, K);
    double tflops_relu = cutlass_bench::compute_gemm_tflops(M, N, K, time_relu_fused);
    std::cout << "  Time: " << time_relu_fused << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_relu << std::endl;
    std::cout << "  Speedup: " << (time_relu_unfused / time_relu_fused) << "x" << std::endl;
    */
    
    // TODO [MEDIUM]: Benchmark fused GELU GEMM
    /*
    std::cout << "Running EVT-FUSED GEMM + GELU..." << std::endl;
    float time_gelu_fused = benchmark_fused_gemm<GemmKernelGELU>(
        d_A, d_B, d_D_gelu, M, N, K);
    double tflops_gelu = cutlass_bench::compute_gemm_tflops(M, N, K, time_gelu_fused);
    std::cout << "  Time: " << time_gelu_fused << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops_gelu << std::endl;
    std::cout << "  Speedup: " << (time_gelu_unfused / time_gelu_fused) << "x" << std::endl;
    */
    
    // ========================================================================
    // RESULTS SUMMARY
    // ========================================================================
    
    std::cout << "\n=== RESULTS SUMMARY ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Configuration"
              << std::right << std::setw(12) << "Time(ms)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(47, '-') << std::endl;
    
    // Print results (fill in after implementing)
    std::cout << std::left << std::setw(20) << "GEMM+ReLU (unfused)"
              << std::right << std::setw(12) << time_relu_unfused
              << std::setw(15) << "1.00x" << std::endl;
    // std::cout << std::left << std::setw(20) << "GEMM+ReLU (fused)"
    //           << std::right << std::setw(12) << time_relu_fused
    //           << std::setw(15) << (time_relu_unfused / time_relu_fused) << "x" << std::endl;
    
    std::cout << std::left << std::setw(20) << "GEMM+GELU (unfused)"
              << std::right << std::setw(12) << time_gelu_unfused
              << std::setw(15) << "1.00x" << std::endl;
    // std::cout << std::left << std::setw(20) << "GEMM+GELU (fused)"
    //           << std::right << std::setw(12) << time_gelu_fused
    //           << std::setw(15) << (time_gelu_unfused / time_gelu_fused) << "x" << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Run ncu to verify fusion:" << std::endl;
    std::cout << "  ncu --metrics smsp__inst_executed.sum,\\" << std::endl;
    std::cout << "              l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \\" << std::endl;
    std::cout << "       ./evt_activation" << std::endl;
    std::cout << "\nExpected observations:" << std::endl;
    std::cout << "  - Fused version has fewer global stores" << std::endl;
    std::cout << "  - GELU has more instructions than ReLU (expected)" << << std::endl;
    std::cout << "  - Speedup should be ~1.3-1.5x (activation is cheap)" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_relu);
    cudaFree(d_D_gelu);
    cudaFree(d_D_unfused);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: Why is GELU fusion speedup lower than ReLU?" << std::endl;
    std::cout << "C2: What's the theoretical maximum speedup for activation fusion?" << std::endl;
    std::cout << "C3: How does this map to Transformer MLP implementation?" << std::endl;
    
    return 0;
}
