/*
 * Project 02 — Fused Attention
 * FA2 in CUTLASS 3.x
 *
 * Target: Within 10% of FA2 reference performance
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../../utils/benchmark.cuh"
#include "../../utils/reference.cuh"

using namespace cutlass;

// ============================================================================
// ATTENTION CONFIGURATION
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

// Attention dimensions
constexpr int B = 8;       // Batch
constexpr int H = 64;      // Heads
constexpr int S = 4096;    // Sequence length
constexpr int D = 128;     // Head dimension

// GEMM equivalent for QK^T
constexpr int M_QK = B * H * S;  // Flattened batch * heads * seq
constexpr int N_QK = S;
constexpr int K_QK = D;

using ElementQKV = cutlass::half_t;
using ElementAccumulator = float;
using ElementOutput = cutlass::half_t;

// ============================================================================
// SOFTMAX — Reference
// ============================================================================

__global__ void softmax_kernel(float* data, int M, int N, float scale) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    
    // Find max for numerical stability
    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum = shared + blockDim.x * gridDim.x;
    
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        float val = data[row * N + c];
        local_max = fmaxf(local_max, val);
    }
    
    // Block reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (threadIdx.x == 0) max_val[blockIdx.x] = local_max;
    __syncthreads();
    
    local_max = max_val[blockIdx.x];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    if (col < N) {
        float val = expf((data[row * N + col] * scale) - local_max);
        data[row * N + col] = val;
        local_sum = val;
    }
    
    // Block reduction for sum
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (threadIdx.x == 0) sum[blockIdx.x] = local_sum;
    __syncthreads();
    
    local_sum = sum[blockIdx.x];
    
    // Normalize
    if (col < N) {
        data[row * N + col] /= local_sum;
    }
}

// ============================================================================
// FA2 CUTLASS IMPLEMENTATION
// ============================================================================

// TODO: Implement FA2 using CUTLASS collectives
// Steps:
// 1. QK^T GEMM using CollectiveBuilder
// 2. Softmax (separate kernel or fused)
// 3. PV GEMM using CollectiveBuilder

struct FA2Config {
    int B, H, S, D;
    float scale;
};

void fa2_cutlass(
    const ElementQKV* d_Q,
    const ElementQKV* d_K,
    const ElementQKV* d_V,
    ElementOutput* d_O,
    float* d_scores,
    FA2Config config
) {
    int M = config.B * config.H * config.S;
    int N = config.S;
    int K = config.D;
    
    // Step 1: QK^T GEMM
    // TODO: Use CollectiveBuilder for QK^T
    
    // Step 2: Softmax
    // TODO: Launch softmax kernel
    
    // Step 3: PV GEMM
    // TODO: Use CollectiveBuilder for PV
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Project 02: FA2 in CUTLASS 3.x ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    std::cout << "Attention configuration:" << std::endl;
    std::cout << "  Batch: " << B << ", Heads: " << H << std::endl;
    std::cout << "  Sequence: " << S << ", Head dim: " << D << std::endl;
    std::cout << "  QK^T GEMM: " << M_QK << "x" << N_QK << "x" << K_QK << std::endl;
    std::cout << std::endl;
    
    // Allocate memory
    ElementQKV *d_Q, *d_K, *d_V;
    ElementOutput *d_O;
    float *d_scores;
    
    size_t bytes_qkv = B * H * S * D * sizeof(ElementQKV);
    size_t bytes_out = B * H * S * D * sizeof(ElementOutput);
    size_t bytes_scores = B * H * S * S * sizeof(float);
    
    cudaMalloc(&d_Q, bytes_qkv);
    cudaMalloc(&d_K, bytes_qkv);
    cudaMalloc(&d_V, bytes_qkv);
    cudaMalloc(&d_O, bytes_out);
    cudaMalloc(&d_scores, bytes_scores);
    
    cutlass_ref::init_matrix_random(d_Q, B * H * S * D);
    cutlass_ref::init_matrix_random(d_K, B * H * S * D);
    cutlass_ref::init_matrix_random(d_V, B * H * S * D);
    
    FA2Config config{B, H, S, D, 1.0f / sqrtf(float(D))};
    
    // TODO: Run FA2 CUTLASS
    // TODO: Compare with FA2 reference
    
    std::cout << "\nTarget: Within 10% of FA2 reference" << std::endl;
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_scores);
    
    return 0;
}
