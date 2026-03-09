/*
 * Project 01 — Production GEMM
 * Ampere (SM80) Implementation
 *
 * Target: >85% cuBLAS performance
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
// CONFIGURATION — Ampere SM80
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

// Problem sizes for benchmark
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// TODO: Configure optimal tile size for Ampere
// Typical: 128x128x64 or 256x128x64
using TileShape = cutlass::GemmlShape<128, 128, 64>;

// TODO: Configure CollectiveMainloop for Ampere
// - Use OpClassTensorOp for Tensor Core
// - StageCountAutoCarveout for pipeline staging
// - KernelScheduleAuto for standard scheduling

// TODO: Configure CollectiveEpilogue

// TODO: Define GemmKernel type

// ============================================================================
// BENCHMARK
// ============================================================================

int main() {
    std::cout << "=== Project 01: Production GEMM (Ampere SM80) ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Verify Ampere GPU
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major != 8 || prop.minor < 0) {
        std::cout << "This benchmark targets Ampere (SM80)." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
    }
    
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
    cutlass_ref::gemm_ref_fp16(M, N, K, d_A, d_B, d_D_cublas);
    timer_cublas.stop();
    float time_cublas = timer_cublas.elapsed_ms();
    
    double tflops_cublas = cutlass_bench::compute_gemm_tflops(M, N, K, time_cublas);
    std::cout << "  cuBLAS: " << time_cublas << " ms, " << tflops_cublas << " TFLOPS" << std::endl;
    
    // TODO: Run CUTLASS GEMM
    // TODO: Compare with cuBLAS
    // TODO: Verify correctness
    
    std::cout << "\nTarget: >85% cuBLAS performance" << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D_cutlass);
    cudaFree(d_D_cublas);
    
    return 0;
}
