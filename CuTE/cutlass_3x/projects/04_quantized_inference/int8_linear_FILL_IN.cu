/*
 * Project 04 — Quantized Inference
 * INT8 Linear Layer
 *
 * Target: >1.3× over FP16, accuracy within 0.5%
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
// INT8 LINEAR CONFIGURATION
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

constexpr int M = 4096;  // Batch * seq
constexpr int N = 16384; // Output dim (4x hidden)
constexpr int K = 4096;  // Input dim (hidden)

using ElementA = int8_t;  // Quantized activation
using ElementB = int8_t;  // Quantized weight
using ElementAccumulator = int32_t;
using ElementD = int8_t;  // Quantized output

// ============================================================================
// QUANTIZATION HELPERS
// ============================================================================

// Per-tensor quantization
__device__ __forceinline__ int8_t quantize_per_tensor(float x, float scale) {
    int val = __float2int_rn(x * scale);
    return static_cast<int8_t>(max(-128, min(127, val)));
}

__device__ __forceinline__ float dequantize_per_tensor(int8_t x, float scale) {
    return static_cast<float>(x) / scale;
}

// Per-channel quantization (for weights)
__device__ __forceinline__ int8_t quantize_per_channel(float x, float scale) {
    int val = __float2int_rn(x * scale);
    return static_cast<int8_t>(max(-128, min(127, val)));
}

// ============================================================================
// INT8 LINEAR LAYER
// ============================================================================

// TODO: Implement INT8 linear layer with fused quantization
// Pattern:
// 1. Quantize input (per-tensor)
// 2. INT8 GEMM (per-channel weight scales)
// 3. Dequantize output (with combined scale)

struct Int8LinearConfig {
    float input_scale;       // Per-tensor input scale
    std::vector<float> weight_scales;  // Per-channel weight scales
    float output_scale;      // Output scale
};

void int8_linear(
    const float* d_input_fp16,
    const float* d_weight_fp16,
    float* d_output_fp16,
    Int8LinearConfig config,
    int M, int N, int K
) {
    // TODO: 
    // 1. Quantize input to int8
    // 2. Quantize weights to int8 (per-channel)
    // 3. INT8 GEMM with accumulation in int32
    // 4. Dequantize output with combined scale
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Project 04: INT8 Quantized Linear Layer ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    std::cout << "Linear layer configuration:" << std::endl;
    std::cout << "  Input:  " << M << "x" << K << " (FP16)" << std::endl;
    std::cout << "  Weight: " << K << "x" << N << " (FP16 → INT8)" << std::endl;
    std::cout << "  Output: " << M << "x" << N << " (INT8 → FP16)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "INT8 quantization:" << std::endl;
    std::cout << "  - Input: Per-tensor scale" << std::endl;
    std::cout << "  - Weights: Per-channel scales" << std::endl;
    std::cout << "  - Accumulation: int32" << std::endl;
    std::cout << "  - Output: Per-tensor scale" << std::endl;
    std::cout << std::endl;
    
    std::cout << "\nTarget: >1.3× over FP16, accuracy within 0.5%" << std::endl;
    
    return 0;
}
