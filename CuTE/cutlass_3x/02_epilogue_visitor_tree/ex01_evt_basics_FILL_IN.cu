/*
 * Module 02 — Epilogue Visitor Tree (EVT)
 * Exercise 01 — Scale + Bias Fusion
 *
 * CUTLASS LAYER: EVT (Epilogue Visitor Tree)
 *
 * WHAT YOU'RE BUILDING:
 *   Fused GEMM + bias add — exactly what TRT-LLM uses for
 *   linear layers with bias terms.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   CollectiveEpilogue wraps your manual epilogue from CuTe Module 04.
 *   EVT replaces your custom epilogue functor with composable nodes.
 *
 * OBJECTIVE:
 *   - Configure EVT with Multiply + Add operations
 *   - Fuse bias vector addition into epilogue
 *   - Verify fusion via Nsight Compute metrics
 */

// PREDICT BEFORE COMPILING
// Q1: How many global memory writes does unfused GEMM+bias require?
// Q2: What metric proves the bias was fused (not a separate kernel)?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/tensor_visitor_plan.hpp"

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
constexpr int K = 4096;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;  // Bias input
using ElementD = cutlass::half_t;  // Output
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cutlass::GemmlShape<128, 128, 64>;

// ============================================================================
// EVT CONFIGURATION — Scale + Bias
// ============================================================================
// EVT composes operations as a tree:
//   1. Multiply: D = alpha * accum
//   2. Add:      D = D + beta * C (where C is the bias vector)
//
// The key insight: both operations happen in registers before
// the first global memory write.

// TODO [EASY]: Define EVT operations for scale + bias
// HINT: Use cutlass::epilogue::collective::EpilogueVisitorTree
// REF: cutlass/include/cutlass/epilogue/collective/

/*
// Define the EVT operation sequence
using EvtOperations = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,  // alpha * accum
    cutlass::epilogue::collective::EpilogueVisitorAdd<ElementAccumulator>        // + beta * C
>;

// EVT arguments (runtime parameters)
struct EvtArguments {
    ElementAccumulator alpha = 1.0f;
    ElementAccumulator beta = 1.0f;
};
*/

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
// COLLECTIVE EPILOGUE — With EVT Fusion
// ============================================================================

// TODO [MEDIUM]: Configure CollectiveEpilogue with EVT
// HINT: Pass EvtOperations to the builder

/*
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto,
   EvtOperations  // <-- Fuse these operations
   >::CollectiveOp;
*/

struct CollectiveEpilogue {};

// ============================================================================
// GEMM KERNEL
// ============================================================================

using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpilogue
    >
>;

// ============================================================================
// REFERENCE: UNFUSED GEMM + BIAS
// ============================================================================

// Unfused bias addition (separate kernel)
__global__ void add_bias_kernel(
    half* output, const half* bias,
    float scale, int M, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        float val = float(output[idx]) * scale + float(bias[col]);
        output[idx] = __float2half(val);
    }
}

void launch_unfused_gemm_bias(
    half* d_A, half* d_B, half* d_D, half* d_bias,
    int M, int N, int K, float alpha = 1.0f
) {
    // Step 1: GEMM (accumulates into d_D)
    cutlass_ref::gemm_ref_fp16(M, N, K, d_A, d_B, d_D, alpha, 0.0f);
    
    // Step 2: Separate bias kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    add_bias_kernel<<<grid, block>>>(d_D, d_bias, alpha, M, N);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 02, Exercise 01: EVT Scale + Bias Fusion ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Allocate device memory
    half *d_A, *d_B, *d_C, *d_D, *d_D_unfused, *d_bias;
    
    size_t bytes_AB = M * K * sizeof(half);
    size_t bytes_out = M * N * sizeof(half);
    size_t bytes_bias = N * sizeof(half);  // Bias is per-output-column
    
    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_C, bytes_out);
    cudaMalloc(&d_D, bytes_out);
    cudaMalloc(&d_D_unfused, bytes_out);
    cudaMalloc(&d_bias, bytes_bias);
    
    // Initialize
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    cutlass_ref::init_matrix_zeros(d_C, M * N);
    cutlass_ref::init_matrix_zeros(d_D, M * N);
    cutlass_ref::init_matrix_zeros(d_D_unfused, M * N);
    
    // Bias vector (random per-column)
    std::vector<half> h_bias(N);
    for (int i = 0; i < N; ++i) {
        h_bias[i] = __float2half((float(rand()) / RAND_MAX - 0.5f) * 2.0f);
    }
    cudaMemcpy(d_bias, h_bias.data(), bytes_bias, cudaMemcpyHostToDevice);
    
    float alpha = 1.0f;
    float beta = 1.0f;
    
    // ========================================================================
    // UNFUSED BASELINE
    // ========================================================================
    
    std::cout << "Running UNFUSED GEMM + bias (separate kernel)..." << std::endl;
    
    cutlass_bench::GpuTimer timer_unfused;
    timer_unfused.start();
    launch_unfused_gemm_bias(d_A, d_B, d_D_unfused, d_bias, M, N, K, alpha);
    timer_unfused.stop();
    
    float time_unfused = timer_unfused.elapsed_ms();
    std::cout << "  Unfused time: " << time_unfused << " ms" << std::endl;
    
    // ========================================================================
    // EVT-FUSED GEMM
    // ========================================================================
    
    // TODO [EASY]: Build GemmKernel arguments with EVT
    // HINT: Pass alpha and beta as EVT arguments
    
    /*
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {d_bias, N},  // Bias vector as C operand
        {d_D, N},
        {alpha, beta}  // EVT arguments: alpha * accum + beta * bias
    };
    
    std::cout << "Running EVT-FUSED GEMM + bias..." << std::endl;
    
    cutlass_bench::GpuTimer timer_fused;
    timer_fused.start();
    
    GemmKernel gemm_op;
    size_t workspace_bytes = 0;
    GemmKernel::get_workspace_size(args, &workspace_bytes);
    
    void* workspace = nullptr;
    if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);
    
    gemm_op.run(args, workspace, 0);
    
    if (workspace) cudaFree(workspace);
    
    timer_fused.stop();
    
    float time_fused = timer_fused.elapsed_ms();
    double tflops = cutlass_bench::compute_gemm_tflops(M, N, K, time_fused);
    
    std::cout << "  Fused time: " << time_fused << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops << std::endl;
    std::cout << "  Speedup: " << (time_unfused / time_fused) << "x" << std::endl;
    */
    
    // ========================================================================
    // VERIFICATION
    // ========================================================================
    
    std::cout << "\nVerifying fused vs unfused..." << std::endl;
    // cutlass_ref::verify_gemm(d_D, d_D_unfused, M * N, 1e-2f);
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Run ncu to verify fusion:" << std::endl;
    std::cout << "  ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \\" << std::endl;
    std::cout << "       ./evt_bias_fused" << std::endl;
    std::cout << "\nExpected: Fused version has ~50% fewer global stores" << std::endl;
    std::cout << "  (GEMM write eliminated by fusion)" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_D_unfused);
    cudaFree(d_bias);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: Did your predictions match the actual speedup?" << std::endl;
    std::cout << "C2: What does ncu show for l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum?" << std::endl;
    std::cout << "C3: How much speedup did you observe? Is it close to 2x?" << std::endl;
    
    return 0;
}
