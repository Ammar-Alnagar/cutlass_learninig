/*
 * Module 02 — Epilogue Visitor Tree (EVT)
 * Exercise 03 — FP8 Output Quantization
 *
 * CUTLASS LAYER: EVT with quantization operations
 *
 * WHAT YOU'RE BUILDING:
 *   Fused GEMM + FP8 quantization — exactly what TRT-LLM uses
 *   for FP8 quantized linear layers in LLM inference.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual quantization epilogue, now replaced with EVT
 *   quantization nodes that fuse scaling + rounding + saturation.
 *
 * OBJECTIVE:
 *   - Configure EVT for FP8 E4M3 output quantization
 *   - Understand per-tensor vs per-channel scaling
 *   - Measure quantization overhead (should be minimal with EVT)
 */

// PREDICT BEFORE COMPILING
// Q1: Does FP8 quantization in the epilogue add significant overhead?
// Q2: Per-channel scaling is more accurate — what's the memory cost?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/float8.h"

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

// Input: FP16, Accumulator: FP32, Output: FP8 E4M3
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::float8_e4m3_t;  // FP8 output
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cutlass::GemmlShape<128, 128, 64>;

// ============================================================================
// FP8 QUANTIZATION — Reference implementation
// ============================================================================

// FP8 E4M3 range: approximately [-448, 448]
__device__ __forceinline__ float8_e4m3_t quantize_e4m3(float x, float scale) {
    constexpr float kMax = 448.0f;
    constexpr float kMin = -448.0f;
    
    // Scale and clamp
    x = x * scale;
    x = fmaxf(kMin, fminf(kMax, x));
    
    // Round to nearest (simplified — real impl uses RTZ or RNE)
    return float8_e4m3_t(x);
}

__device__ __forceinline__ float dequantize_e4m3(float8_e4m3_t x, float scale) {
    return float(x) / scale;
}

// Per-tensor quantization kernel (unfused reference)
__global__ void quantize_per_tensor_kernel(
    const float* input, float8_e4m3_t* output,
    float scale, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = quantize_e4m3(input[idx], scale);
    }
}

// Per-channel quantization kernel (unfused reference)
__global__ void quantize_per_channel_kernel(
    const float* input, float8_e4m3_t* output,
    const float* scales,  // [N] scales
    int M, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        float scale = scales[col];
        output[idx] = quantize_e4m3(input[idx], scale);
    }
}

// ============================================================================
// SCALE COMPUTATION — Determine quantization scales
// ============================================================================

// Compute per-tensor scale from max absolute value
float compute_per_tensor_scale(const float* h_data, int size) {
    float max_val = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_val = fmaxf(max_val, fabsf(h_data[i]));
    }
    // Scale to use ~80% of FP8 range for headroom
    return 0.8f * 127.0f / max_val;
}

// Compute per-channel scales
std::vector<float> compute_per_channel_scales(
    const float* h_data, int M, int N
) {
    std::vector<float> scales(N, 0.0f);
    for (int j = 0; j < N; ++j) {
        float max_val = 0.0f;
        for (int i = 0; i < M; ++i) {
            max_val = fmaxf(max_val, fabsf(h_data[i * N + j]));
        }
        scales[j] = 0.8f * 127.0f / max_val;
    }
    return scales;
}

// ============================================================================
// UNFUSED REFERENCE — GEMM + separate quantization
// ============================================================================

void launch_unfused_gemm_quantize(
    half* d_A, half* d_B, float* d_accum, float8_e4m3_t* d_D,
    float scale, int M, int N, int K,
    bool per_channel = false, const float* d_scales = nullptr
) {
    // Step 1: GEMM with FP32 accumulator
    // (Using FP32 reference for simplicity — in practice use FP16 Tensor Core)
    
    // Step 2: Quantization kernel
    int size = M * N;
    int threads = 256;
    
    if (per_channel && d_scales) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        quantize_per_channel_kernel<<<grid, block>>>(d_accum, d_D, d_scales, M, N);
    } else {
        int blocks = (size + threads - 1) / threads;
        quantize_per_tensor_kernel<<<blocks, threads>>>(d_accum, d_D, scale, size);
    }
}

// ============================================================================
// EVT CONFIGURATION — FP8 Quantization
// ============================================================================
// EVT quantization node fuses:
//   1. Scale multiplication
//   2. Rounding
//   3. Saturation/clamping
//   4. Type conversion (FP32 → FP8)

// TODO [MEDIUM]: Define EVT operations for FP8 quantization
// HINT: cutlass::epilogue::collective::EpilogueVisitorQuantize

/*
// Per-tensor quantization EVT
using EvtOperationsQuantizePerTensor = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,
    cutlass::epilogue::collective::EpilogueVisitorQuantize<
        ElementAccumulator,  // Input type (FP32 accumulator)
        ElementD,            // Output type (FP8)
        cutlass::Float8E4M3  // FP8 format
    >
>;

// Per-channel quantization EVT (requires scale tensor)
using EvtOperationsQuantizePerChannel = cutlass::epilogue::collective::EpilogueVisitorTree<
    cutlass::epilogue::collective::EpilogueVisitorMultiply<ElementAccumulator>,
    cutlass::epilogue::collective::EpilogueVisitorQuantize<
        ElementAccumulator,
        ElementD,
        cutlass::Float8E4M3,
        cutlass::epilogue::collective::ScaleType::PerColumn  // Per-channel
    >
>;
*/

struct EvtOperationsQuantizePerTensor {};
struct EvtOperationsQuantizePerChannel {};

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
// COLLECTIVE EPILOGUE — With Quantization
// ============================================================================

// TODO [MEDIUM]: Configure CollectiveEpilogue with quantization EVT

/*
using CollectiveEpiloguePerTensor = typename cutlass::epilogue::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto,
   EvtOperationsQuantizePerTensor
   >::CollectiveOp;

using CollectiveEpiloguePerChannel = typename cutlass::epilogue::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementC, LayoutC,
   ElementD, LayoutD,
   cutlass::gemm::collective::EpilogueScheduleAuto,
   EvtOperationsQuantizePerChannel
   >::CollectiveOp;
*/

struct CollectiveEpiloguePerTensor {};
struct CollectiveEpiloguePerChannel {};

// ============================================================================
// GEMM KERNEL TYPES
// ============================================================================

using GemmKernelPerTensor = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpiloguePerTensor
    >
>;

using GemmKernelPerChannel = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M, N, K, 1>,
        CollectiveMainloop,
        CollectiveEpiloguePerChannel
    >
>;

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 02, Exercise 03: EVT FP8 Quantization ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_accum;  // FP32 accumulator for reference
    float8_e4m3_t *d_D_fused, *d_D_unfused;
    float *d_scales;  // Per-channel scales
    
    size_t bytes_AB = M * K * sizeof(half);
    size_t bytes_accum = M * N * sizeof(float);
    size_t bytes_D = M * N * sizeof(float8_e4m3_t);
    size_t bytes_scales = N * sizeof(float);
    
    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_accum, bytes_accum);
    cudaMalloc(&d_D_fused, bytes_D);
    cudaMalloc(&d_D_unfused, bytes_D);
    cudaMalloc(&d_scales, bytes_scales);
    
    // Initialize inputs
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    // Compute scale factors (host-side for reference)
    float* h_accum = new float[M * N];
    // (In practice, run GEMM first to get actual accumulator values)
    for (int i = 0; i < M * N; ++i) {
        h_accum[i] = (float(rand()) / RAND_MAX - 0.5f) * 10.0f;  // Simulated GEMM output
    }
    
    float scale_per_tensor = compute_per_tensor_scale(h_accum, M * N);
    std::vector<float> h_scales = compute_per_channel_scales(h_accum, M, N);
    cudaMemcpy(d_scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice);
    
    std::cout << "Quantization scales:" << std::endl;
    std::cout << "  Per-tensor scale: " << scale_per_tensor << std::endl;
    std::cout << "  Per-channel scales: [" << h_scales[0] << ", ..., " << h_scales[N-1] << "]" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // UNFUSED BASELINES
    // ========================================================================
    
    std::cout << "=== UNFUSED BASELINES ===" << std::endl;
    
    // Per-tensor quantization (unfused)
    std::cout << "Running UNFUSED GEMM + per-tensor quantize..." << std::endl;
    cutlass_bench::GpuTimer timer_unfused_tensor;
    timer_unfused_tensor.start();
    launch_unfused_gemm_quantize(d_A, d_B, d_accum, d_D_unfused, scale_per_tensor, M, N, K, false);
    timer_unfused_tensor.stop();
    float time_unfused_tensor = timer_unfused_tensor.elapsed_ms();
    std::cout << "  Time: " << time_unfused_tensor << " ms" << std::endl;
    
    // Per-channel quantization (unfused)
    std::cout << "Running UNFUSED GEMM + per-channel quantize..." << std::endl;
    cutlass_bench::GpuTimer timer_unfused_channel;
    timer_unfused_channel.start();
    launch_unfused_gemm_quantize(d_A, d_B, d_accum, d_D_unfused, 0.0f, M, N, K, true, d_scales);
    timer_unfused_channel.stop();
    float time_unfused_channel = timer_unfused_channel.elapsed_ms();
    std::cout << "  Time: " << time_unfused_channel << " ms" << std::endl;
    
    // ========================================================================
    // EVT-FUSED QUANTIZATION
    // ========================================================================
    
    std::cout << "\n=== EVT-FUSED QUANTIZATION ===" << std::endl;
    
    // TODO [MEDIUM]: Benchmark fused per-tensor quantization
    /*
    typename GemmKernelPerTensor::Arguments args_tensor{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {nullptr, 0},  // No C operand
        {d_D_fused, N},
        {1.0f, scale_per_tensor}  // alpha, quant_scale
    };
    
    std::cout << "Running EVT-FUSED per-tensor quantization..." << std::endl;
    cutlass_bench::GpuTimer timer_fused_tensor;
    timer_fused_tensor.start();
    
    GemmKernelPerTensor gemm_op_tensor;
    size_t ws_size = 0;
    GemmKernelPerTensor::get_workspace_size(args_tensor, &ws_size);
    void* ws = nullptr;
    if (ws_size > 0) cudaMalloc(&ws, ws_size);
    gemm_op_tensor.run(args_tensor, ws, 0);
    if (ws) cudaFree(ws);
    
    timer_fused_tensor.stop();
    float time_fused_tensor = timer_fused_tensor.elapsed_ms();
    double tflops = cutlass_bench::compute_gemm_tflops(M, N, K, time_fused_tensor);
    
    std::cout << "  Time: " << time_fused_tensor << " ms" << std::endl;
    std::cout << "  TFLOPS: " << tflops << std::endl;
    std::cout << "  Speedup: " << (time_unfused_tensor / time_fused_tensor) << "x" << std::endl;
    */
    
    // TODO [HARD]: Benchmark fused per-channel quantization
    /*
    typename GemmKernelPerChannel::Arguments args_channel{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},
        {d_B, N},
        {nullptr, 0},
        {d_D_fused, N},
        {1.0f, d_scales}  // alpha, per-channel scales
    };
    
    std::cout << "Running EVT-FUSED per-channel quantization..." << std::endl;
    // ... similar benchmark pattern
    */
    
    // ========================================================================
    // ACCURACY COMPARISON
    // ========================================================================
    
    std::cout << "\n=== ACCURACY COMPARISON ===" << std::endl;
    std::cout << "Per-tensor quantization:" << std::endl;
    std::cout << "  - Single scale for entire tensor" << std::endl;
    std::cout << "  - Lower accuracy for varied distributions" << std::endl;
    std::cout << "  - Minimal memory overhead (1 float)" << std::endl;
    std::cout << std::endl;
    std::cout << "Per-channel quantization:" << std::endl;
    std::cout << "  - One scale per output column" << std::endl;
    std::cout << "  - Better accuracy for LLM weights" << std::endl;
    std::cout << "  - Memory overhead: N floats (e.g., 16KB for N=4096)" << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "\n=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Run ncu to verify quantization overhead:" << std::endl;
    std::cout << "  ncu --metrics smsp__inst_executed.sum,\\" << std::endl;
    std::cout << "              l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \\" << std::endl;
    std::cout << "       ./evt_quantize" << std::endl;
    std::cout << "\nExpected: Quantization adds <5% overhead when fused" << std::endl;
    
    // Cleanup
    delete[] h_accum;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_accum);
    cudaFree(d_D_fused);
    cudaFree(d_D_unfused);
    cudaFree(d_scales);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What's the overhead of EVT-fused quantization?" << std::endl;
    std::cout << "C2: When would you choose per-tensor vs per-channel?" << std::endl;
    std::cout << "C3: How does this map to TRT-LLM FP8 linear layers?" << std::endl;
    
    return 0;
}
