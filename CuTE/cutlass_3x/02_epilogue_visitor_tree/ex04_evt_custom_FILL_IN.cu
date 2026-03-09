/*
 * Module 02 — Epilogue Visitor Tree (EVT)
 * Exercise 04 — Write Your Own EVT Node
 *
 * CUTLASS LAYER: Custom EVT node implementation
 *
 * WHAT YOU'RE BUILDING:
 *   A custom EVT node for fused RMSNorm — exactly what you need
 *   for pre-norm Transformer blocks without a separate kernel.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your custom epilogue functor from CuTe Module 04, now expressed
 *   as a composable EVT node that can chain with other operations.
 *
 * OBJECTIVE:
 *   - Implement a custom EVT node interface
 *   - Fuse RMSNorm into the GEMM epilogue
 *   - Chain custom node with built-in nodes
 */

// PREDICT BEFORE COMPILING
// Q1: What methods must a custom EVT node implement?
// Q2: Can you chain a custom node with built-in activation nodes?

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

constexpr int M = 4096;   // Batch * seq_len
constexpr int N = 4096;   // Hidden dimension
constexpr int K = 4096;   // Hidden dimension

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementAccumulator = float;
using ElementOutput = cutlass::half_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using TileShape = cutlass::GemmlShape<128, 128, 64>;

// ============================================================================
// RMSNorm — Reference implementation
// ============================================================================

// RMSNorm: output = input / RMS(input) * gamma
// RMS = sqrt(mean(x^2) + epsilon)

__device__ __forceinline__ float rsqrt_float(float x) {
    return rsqrtf(x);
}

// Unfused RMSNorm kernel (separate from GEMM)
__global__ void rmsnorm_kernel(
    const float* input, float* output, const float* gamma,
    float epsilon, int M, int N
) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    extern __shared__ float shared[];
    
    // Compute row-wise RMS
    float sum_sq = 0.0f;
    for (int c = col; c < N; c += blockDim.x) {
        float val = input[row * N + c];
        sum_sq += val * val;
    }
    
    // Warp reduction (simplified — use proper warp shuffle in production)
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    float rms = rsqrt_float(sum_sq / N + epsilon);
    
    // Apply normalization and gamma
    if (col < N) {
        float g = gamma[col];
        float val = input[row * N + col];
        output[row * N + col] = val * rms * g;
    }
}

void launch_unfused_gemm_rmsnorm(
    half* d_A, half* d_B, float* d_accum, float* d_output,
    const float* d_gamma, float epsilon,
    int M, int N, int K
) {
    // Step 1: GEMM (FP32 accumulator)
    // (Simplified — use FP16 Tensor Core in production)
    
    // Step 2: RMSNorm kernel
    int threads = 256;
    rmsnorm_kernel<<<M, threads>>>(d_accum, d_output, d_gamma, epsilon, M, N);
}

// ============================================================================
// CUSTOM EVT NODE — RMSNorm
// ============================================================================
// A custom EVT node must implement:
//   1. Arguments struct — runtime parameters
//   2. get() method — returns the elementwise operation functor
//   3. The functor must be callable with accumulator type

// TODO [HARD]: Implement custom RMSNorm EVT node
// HINT: Follow the pattern below

/*
struct EvtOpRMSNorm {
    // Runtime arguments
    struct Arguments {
        const float* gamma;    // [N] scale parameters
        float epsilon;         // Numerical stability
        int N;                 // Hidden dimension
    };
    
    // Device-side state (computed from arguments)
    struct Params {
        const float* gamma;
        float epsilon;
        int N;
        float row_rms;  // Computed per-row
    };
    
    // Get the elementwise operation functor
    template <class ThreadMap>
    __device__ auto get(
        ThreadMap const& thread_map,
        Arguments const& args
    ) {
        return Functor{args.gamma, args.epsilon, args.N};
    }
    
    // Elementwise operation functor
    struct Functor {
        const float* gamma;
        float epsilon;
        int N;
        
        // Called for each accumulator element
        __device__ float operator()(float accum, int row, int col) const {
            // Note: In real EVT, row/col context is provided differently
            // This is a simplified illustration
            
            // For RMSNorm, we need row-wise reduction first
            // This is complex in EVT — RMSNorm is better as a separate kernel
            // unless we use a custom reduction pattern
            
            // Simplified: assume row_rms is precomputed
            float row_rms = 1.0f;  // Placeholder
            return accum * row_rms * gamma[col];
        }
    };
};
*/

// Alternative: Simpler custom EVT node — fused clamp + scale
// This is more realistic for EVT's elementwise model

struct EvtOpClampScale {
    // Runtime arguments
    struct Arguments {
        float min_val;
        float max_val;
        float scale;
    };
    
    // Get the elementwise operation functor
    template <class ThreadMap>
    __device__ auto get(
        ThreadMap const& thread_map,
        Arguments const& args
    ) {
        return Functor{args.min_val, args.max_val, args.scale};
    }
    
    // Elementwise operation functor
    struct Functor {
        float min_val;
        float max_val;
        float scale;
        
        // Called for each accumulator element
        __device__ float operator()(float accum) const {
            accum = fminf(max_val, fmaxf(min_val, accum));  // Clamp
            return accum * scale;                            // Scale
        }
    };
};

// ============================================================================
// EVT COMPOSITION — Chain custom node with built-in nodes
// ============================================================================

// TODO [HARD]: Define EVT tree with custom node
// HINT: Custom node can be composed with built-in nodes

/*
using EvtOperationsCustom = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,
    EvtOpClampScale,  // Custom clamp + scale
    cutlass::epilogue::collective::EpilogueVisitorReLu<ElementAccumulator>
>;
// Order matters: Multiply → ClampScale → ReLU
*/

struct EvtOperationsCustom {};

// ============================================================================
// COLLECTIVE MAINLOOP
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
// COLLECTIVE EPILOGUE — With custom EVT node
// ============================================================================

// TODO [HARD]: Configure CollectiveEpilogue with custom EVT

/*
using CollectiveEpilogueCustom = typename cutlass::epilogue::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementOutput, LayoutOutput,
   ElementOutput, LayoutOutput,
   cutlass::gemm::collective::EpilogueScheduleAuto,
   EvtOperationsCustom
   >::CollectiveOp;
*/

struct CollectiveEpilogueCustom {};

// ============================================================================
// GEMM KERNEL
// ============================================================================

using GemmKernelCustom = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpilogueCustom
    >
>;

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 02, Exercise 04: Custom EVT Node ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_accum;
    half *d_output_fused, *d_output_unfused;
    float *d_gamma;
    
    size_t bytes_AB = M * K * sizeof(half);
    size_t bytes_accum = M * N * sizeof(float);
    size_t bytes_output = M * N * sizeof(half);
    size_t bytes_gamma = N * sizeof(float);
    
    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_accum, bytes_accum);
    cudaMalloc(&d_output_fused, bytes_output);
    cudaMalloc(&d_output_unfused, bytes_output);
    cudaMalloc(&d_gamma, bytes_gamma);
    
    // Initialize
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    // Gamma (RMSNorm scale) — initialize to 1.0
    std::vector<float> h_gamma(N, 1.0f);
    cudaMemcpy(d_gamma, h_gamma.data(), bytes_gamma, cudaMemcpyHostToDevice);
    
    float epsilon = 1e-5f;
    
    // Custom EVT arguments
    EvtOpClampScale::Arguments clamp_scale_args{
        -10.0f,   // min
        10.0f,    // max
        1.0f      // scale
    };
    
    std::cout << "Custom EVT node: ClampScale" << std::endl;
    std::cout << "  min = " << clamp_scale_args.min_val << std::endl;
    std::cout << "  max = " << clamp_scale_args.max_val << std::endl;
    std::cout << "  scale = " << clamp_scale_args.scale << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // UNFUSED BASELINE
    // ========================================================================
    
    std::cout << "Running UNFUSED GEMM + clamp + scale..." << std::endl;
    
    // (Simplified — in practice, run actual GEMM + separate clamp kernel)
    cutlass_bench::GpuTimer timer_unfused;
    timer_unfused.start();
    // launch_unfused_gemm_clamp_scale(...);
    timer_unfused.stop();
    
    float time_unfused = timer_unfused.elapsed_ms();
    std::cout << "  Time: " << time_unfused << " ms (placeholder)" << std::endl;
    
    // ========================================================================
    // EVT-FUSED WITH CUSTOM NODE
    // ========================================================================
    
    // TODO [HARD]: Build and launch GEMM with custom EVT node
    
    /*
    typename GemmKernelCustom::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {nullptr, 0},
        {d_output_fused, N},
        {1.0f, clamp_scale_args}  // alpha, custom EVT args
    };
    
    std::cout << "Running EVT-FUSED GEMM + custom ClampScale node..." << std::endl;
    
    cutlass_bench::GpuTimer timer_fused;
    timer_fused.start();
    
    GemmKernelCustom gemm_op;
    size_t ws_size = 0;
    GemmKernelCustom::get_workspace_size(args, &ws_size);
    void* ws = nullptr;
    if (ws_size > 0) cudaMalloc(&ws, ws_size);
    
    gemm_op.run(args, ws, 0);
    
    if (ws) cudaFree(ws);
    
    timer_fused.stop();
    
    float time_fused = timer_fused.elapsed_ms();
    double tflops = cutlass_bench::compute_gemm_tflops(M, N, K, time_fused);
    
    std::cout << "  Time: " << time_fused << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops << std::endl;
    std::cout << "  Speedup: " << (time_unfused / time_fused) << "x" << std::endl;
    */
    
    // ========================================================================
    // CUSTOM EVT NODE DESIGN GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== CUSTOM EVT NODE DESIGN GUIDANCE ===" << std::endl;
    std::cout << "When to use custom EVT nodes:" << std::endl;
    std::cout << "  - Elementwise operations not in built-in set" << std::endl;
    std::cout << "  - Domain-specific fusion (e.g., custom activation)" << std::endl;
    std::cout << "  - Research/experimental operations" << std::endl;
    std::cout << std::endl;
    std::cout << "When NOT to use custom EVT nodes:" << std::endl;
    std::cout << "  - Reductions (use separate kernel)" << std::endl;
    std::cout << "  - Operations requiring row/column context" << std::endl;
    std::cout << "  - Complex control flow" << std::endl;
    std::cout << std::endl;
    std::cout << "RMSNorm note: True RMSNorm requires row-wise reduction," << std::endl;
    std::cout << "which doesn't fit EVT's elementwise model. Use a separate" << std::endl;
    std::cout << "kernel or explore CUTLASS's fused attention patterns." << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Verify custom node fusion:" << std::endl;
    std::cout << "  ncu --metrics smsp__inst_executed.sum \\" << std::endl;
    std::cout << "       ./evt_custom" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_accum);
    cudaFree(d_output_fused);
    cudaFree(d_output_unfused);
    cudaFree(d_gamma);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What are the requirements for a custom EVT node?" << std::endl;
    std::cout << "C2: Why doesn't RMSNorm fit the EVT elementwise model?" << std::endl;
    std::cout << "C3: What operations ARE suitable for custom EVT nodes?" << std::endl;
    
    return 0;
}
