/*
 * Module 01 — CollectiveBuilder Anatomy
 * Exercise 04 — Blackwell Persistent Kernel
 *
 * CUTLASS LAYER: CollectiveBuilder for SM100 (Blackwell)
 *
 * WHAT YOU'RE BUILDING:
 *   SM100 Blackwell GEMM with persistent kernel strategy.
 *   Blackwell introduces native FP8 Tensor Core and persistent kernels.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Same TiledMMA abstraction, but Blackwell has new MMA atoms.
 *   Persistent kernel = your manual grid-stride loop from CuTe, but auto-generated.
 *
 * OBJECTIVE:
 *   - Configure Blackwell-specific CollectiveBuilder
 *   - Enable FP8 Tensor Core (E4M3/E5M2)
 *   - Understand persistent kernel scheduling
 */

// PREDICT BEFORE COMPILING
// Q1: How does persistent kernel scheduling differ from standard grid launch?
// Q2: What's the FP8 Tensor Core throughput advantage on Blackwell?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/float8.h"

#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// SETUP — Blackwell-specific configuration
// ============================================================================

// Architecture target — SM100 for Blackwell
using ArchTag = cutlass::arch::Sm100;

// GEMM problem size
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

// Element types — FP8 E4M3 is native on Blackwell
// E4M3: 1 sign, 4 exponent, 3 mantissa (max ~448)
// E5M2: 1 sign, 5 exponent, 2 mantissa (max ~57344)
using ElementA = cutlass::float8_e4m3_t;
using ElementB = cutlass::float8_e4m3_t;
using ElementC = cutlass::float8_e4m3_t;
using ElementD = cutlass::float8_e4m3_t;
using ElementAccumulator = float;  // Accumulate in FP32 for accuracy

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ============================================================================
// BLACKWELL COLLECTIVE BUILDER
// ============================================================================
// Blackwell improvements:
//   1. Native FP8 Tensor Core (2× throughput vs Hopper)
//   2. Persistent kernel support (no grid launch overhead)
//   3. Improved shared memory bandwidth
//   4. TMA multicast enhancements

// TODO [MEDIUM]: Configure CollectiveMainloop for Blackwell SM100
// Key differences from Hopper:
//   1. Use KernelSchedulePersistent for persistent kernel
//   2. FP8 Tensor Core atom (different from Hopper)
//   3. Larger shared memory per SM (512 KB on B200)

/*
using TileShape = cutlass::GemmlShape<128, 128, 128>;  // FP8 benefits from larger K tile

// Persistent kernel configuration
using PersistentConfig = cutlass::gemm::collective::PersistentKernelConfig
    <cutlass::gemm::collective::PersistentScheduleAuto,
     cutlass::gemm::collective::TileSchedulerAuto>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementA, LayoutA,
   ElementB, LayoutB,
   ElementAccumulator,
   TileShape,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<512000>,  // B200 smem: 512KB
   PersistentConfig
   >::CollectiveOp;
*/

struct CollectiveMainloop {};

// ============================================================================
// FP8 SCALING TENSORS
// ============================================================================
// FP8 GEMM requires scaling factors to prevent overflow/underflow.
// CUTLASS 3.x supports:
//   - Per-tensor scale (single scalar)
//   - Per-channel scale (vector)
//   - Block-wise scale (2D grid)

// TODO [HARD]: Define scaling tensor configuration for FP8 GEMM
// HINT: cutlass::gemm::ScaledElementDescriptor

/*
struct Fp8ScalingConfig {
    // Per-tensor scaling (simplest)
    using ScaleType = cutlass::gemm::ScaleType::PerTensor;
    
    // Scale factors (typically computed during quantization)
    float scale_a = 1.0f;  // A = FP8_A * scale_a
    float scale_b = 1.0f;  // B = FP8_B * scale_b
    float scale_d = 1.0f;  // D = accum * scale_d (for FP8 output)
    
    // For per-channel scaling:
    // using ScaleType = cutlass::gemm::ScaleType::PerColumn;
    // float* d_scale_a;  // [K] scales for A
    // float* d_scale_b;  // [N] scales for B
};
*/

// ============================================================================
// COLLECTIVE EPILOGUE — FP8 quantization
// ============================================================================

/*
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
*/

struct CollectiveEpilogue {};

// ============================================================================
// FP8 QUANTIZATION HELPERS
// ============================================================================

// Convert FP32 to FP8 E4M3 with saturation
__device__ __forceinline__ float8_e4m3_t float_to_e4m3(float x) {
    // E4M3 range: approximately [-448, 448]
    constexpr float kMax = 448.0f;
    constexpr float kMin = -448.0f;
    
    x = fmaxf(kMin, fminf(kMax, x));
    return cutlass::float8_e4m3_t(x);
}

// Convert FP8 E4M3 to FP32
__device__ __forceinline__ float e4m3_to_float(float8_e4m3_t x) {
    return float(x);
}

// Quantize FP32 matrix to FP8
__global__ void quantize_fp8_kernel(
    const float* input, float8_e4m3_t* output,
    float scale, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * scale;
        output[idx] = float_to_e4m3(val);
    }
}

// Dequantize FP8 matrix to FP32
__global__ void dequantize_fp8_kernel(
    const float8_e4m3_t* input, float* output,
    float scale, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = e4m3_to_float(input[idx]);
        output[idx] = val / scale;
    }
}

// ============================================================================
// LAUNCH FUNCTION
// ============================================================================

template <typename GemmKernel>
cudaError_t launch_blackwell_gemm(
    const typename GemmKernel::Arguments& args,
    cudaStream_t stream = 0
) {
    GemmKernel gemm_op;
    
    size_t workspace_bytes = 0;
    GemmKernel::get_workspace_size(args, &workspace_bytes);
    
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    cudaError_t status = gemm_op.run(args, workspace, stream);
    
    if (workspace) {
        cudaFree(workspace);
    }
    
    return status;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 01, Exercise 04: Blackwell Persistent Kernel ===" << std::endl;
    
    // Check for Blackwell GPU
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    if (prop.major < 10) {
        std::cout << "WARNING: This exercise requires Blackwell (SM100) GPU." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
        std::cout << "Code will not launch, but you can still study the configuration." << std::endl;
        std::cout << "\nKey Blackwell features for FP8 GEMM:" << std::endl;
        std::cout << "  - 2× FP8 Tensor Core throughput vs Hopper" << std::endl;
        std::cout << "  - Persistent kernel scheduling" << std::endl;
        std::cout << "  - 512 KB shared memory per SM (B200)" << std::endl;
        return 0;
    }
    
    std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Shared Memory per SM: " << (prop.sharedMemoryPerMultiprocessor / 1024) << " KB" << std::endl;
    std::cout << std::endl;
    
    // For FP8 GEMM, we need to quantize inputs first
    float *h_A_fp32, *h_B_fp32;
    float8_e4m3_t *d_A, *d_B, *d_C, *d_D;
    
    size_t bytes_fp32 = M * K * sizeof(float);
    size_t bytes_fp8 = M * K * sizeof(float8_e4m3_t);
    
    h_A_fp32 = new float[M * K];
    h_B_fp32 = new float[K * N];
    
    // Initialize FP32 data
    for (int i = 0; i < M * K; ++i) {
        h_A_fp32[i] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_fp32[i] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // Compute scale factors (simple max-based scaling)
    float max_a = 0.0f, max_b = 0.0f;
    for (int i = 0; i < M * K; ++i) max_a = fmaxf(max_a, fabsf(h_A_fp32[i]));
    for (int i = 0; i < K * N; ++i) max_b = fmaxf(max_b, fabsf(h_B_fp32[i]));
    
    float scale_a = 255.0f / max_a;  // Scale to use full E4M3 range
    float scale_b = 255.0f / max_b;
    
    std::cout << "FP8 scaling factors:" << std::endl;
    std::cout << "  scale_a = " << scale_a << " (max_A = " << max_a << ")" << std::endl;
    std::cout << "  scale_b = " << scale_b << " (max_B = " << max_b << ")" << std::endl;
    
    // Allocate device memory
    cudaMalloc(&d_A, bytes_fp8);
    cudaMalloc(&d_B, bytes_fp8);
    cudaMalloc(&d_C, M * N * sizeof(float8_e4m3_t));
    cudaMalloc(&d_D, M * N * sizeof(float8_e4m3_t));
    
    // TODO [MEDIUM]: Launch quantization kernels
    // HINT: Use quantize_fp8_kernel defined above
    
    /*
    float *d_A_fp32, *d_B_fp32;
    cudaMalloc(&d_A_fp32, bytes_fp32);
    cudaMalloc(&d_B_fp32, bytes_fp32);
    cudaMemcpy(d_A_fp32, h_A_fp32, bytes_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B_fp32, bytes_fp32, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks_a = (M * K + threads - 1) / threads;
    int blocks_b = (K * N + threads - 1) / threads;
    
    quantize_fp8_kernel<<<blocks_a, threads>>>(d_A_fp32, d_A, scale_a, M * K);
    quantize_fp8_kernel<<<blocks_b, threads>>>(d_B_fp32, d_B, scale_b, K * N);
    cudaDeviceSynchronize();
    */
    
    // Reference: cuBLAS FP8 (if available) or FP32 reference
    std::cout << "\nRunning reference GEMM (FP32 for comparison)..." << std::endl;
    // Note: cuBLAS FP8 support varies by version
    
    // TODO [HARD]: Build and launch Blackwell FP8 GEMM
    // HINT: Same pattern as previous exercises
    
    /*
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
    
    std::cout << "Running Blackwell FP8 GEMM (persistent kernel)..." << std::endl;
    cutlass_bench::GpuTimer timer;
    timer.start();
    CUDA_CHECK(launch_blackwell_gemm<GemmKernel>(args));
    timer.stop();
    
    float elapsed_ms = timer.elapsed_ms();
    double tflops = cutlass_bench::compute_gemm_tflops(M, N, K, elapsed_ms);
    std::cout << "Blackwell FP8 GEMM: " << elapsed_ms << " ms, " << tflops << " TFLOPS" << std::endl;
    
    // Expected: 2× FP16 TFLOPS due to 2× throughput
    std::cout << "\nExpected FP8 speedup over FP16: ~2.0x (native FP8 Tensor Core)" << std::endl;
    */
    
    // Cleanup
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What is the key advantage of persistent kernels?" << std::endl;
    std::cout << "C2: Why does FP8 require scaling tensors?" << std::endl;
    std::cout << "C3: How does Blackwell FP8 compare to Hopper FP8?" << std::endl;
    std::cout << "C4: What quantization strategy would you use for LLM weights?" << std::endl;
    
    return 0;
}
