/*
 * Project 01 — Production GEMM
 * Hopper (SM90) Implementation
 *
 * Target: >90% cuBLAS performance
 * Features: TMA + Warp Specialization
 */

#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../../utils/benchmark.cuh"
#include "../../utils/reference.cuh"
#include "../../utils/roofline.cuh"

using namespace cutlass;

// ============================================================================
// CONFIGURATION — Hopper SM90
// ============================================================================

using ArchTag = cutlass::arch::Sm90;

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

using ElementA = cutlass::bfloat16_t;  // BF16 native on Hopper
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// TODO: Configure optimal tile size for Hopper
// Typical: 128x128x64 with TMA

// TODO: Configure CollectiveMainloop for Hopper
// - Use KernelScheduleWarpSpecialized
// - Configure producer/consumer warp split
// - StageCountAutoCarveout<230400> for H100 (230KB smem)

// TODO: Configure CollectiveEpilogue

// TODO: Define GemmKernel type

// ============================================================================
// BENCHMARK
// ============================================================================

int main() {
    std::cout << "=== Project 01: Production GEMM (Hopper SM90) ===" << std::endl;
    std::cout << "Features: TMA + Warp Specialization" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Verify Hopper GPU
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 9) {
        std::cout << "This benchmark requires Hopper (SM90) or later." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
        std::cout << std::endl;
        std::cout << "Key Hopper features:" << std::endl;
        std::cout << "  - TMA (Tensor Memory Accelerator)" << std::endl;
        std::cout << "  - Warp specialization (producer/consumer split)" << std::endl;
        std::cout << "  - 230 KB shared memory per SM" << std::endl;
        return 0;
    }
    
    std::cout << "Hopper features available:" << std::endl;
    std::cout << "  - TMA: Yes" << std::endl;
    std::cout << "  - Warp specialization: Yes" << std::endl;
    std::cout << "  - Shared memory per SM: " << (prop.sharedMemoryPerMultiprocessor / 1024) << " KB" << std::endl;
    std::cout << std::endl;
    
    // Allocate memory
    ElementA *d_A;
    ElementB *d_B;
    ElementD *d_D_cutlass, *d_D_cublas;
    
    cudaMalloc(&d_A, M * K * sizeof(ElementA));
    cudaMalloc(&d_B, K * N * sizeof(ElementB));
    cudaMalloc(&d_D_cutlass, M * N * sizeof(ElementD));
    cudaMalloc(&d_D_cublas, M * N * sizeof(ElementD));
    
    cutlass_ref::init_matrix_random(d_A, M * K);
    cutlass_ref::init_matrix_random(d_B, K * N);
    
    // cuBLAS reference
    std::cout << "Running cuBLAS reference..." << std::endl;
    cutlass_bench::GpuTimer timer_cublas;
    timer_cublas.start();
    cutlass_ref::gemm_ref_bf16(M, N, K, d_A, d_B, d_D_cublas);
    timer_cublas.stop();
    float time_cublas = timer_cublas.elapsed_ms();
    
    double tflops_cublas = cutlass_bench::compute_gemm_tflops(M, N, K, time_cublas);
    std::cout << "  cuBLAS: " << time_cublas << " ms, " << tflops_cublas << " TFLOPS" << std::endl;
    
    // TODO: Run CUTLASS GEMM with TMA + warp specialization
    // TODO: Compare with cuBLAS
    // TODO: Verify correctness
    
    std::cout << "\nTarget: >90% cuBLAS performance" << std::endl;
    std::cout << "Expected speedup over Ampere: 1.5-2.0×" << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_cutlass);
    cudaFree(d_D_cublas);
    
    return 0;
}
