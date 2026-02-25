/**
 * Exercise 02: Tensor Core Operation Simulation
 * 
 * Objective: Simulate Tensor Core operations to understand
 *            how they accelerate matrix multiplication
 * 
 * Tasks:
 * 1. Understand Tensor Core capabilities
 * 2. Simulate a small Tensor Core operation
 * 3. Compare with scalar multiplication
 * 4. Calculate throughput improvement
 * 
 * Key Concepts:
 * - Tensor Core: Hardware unit for matrix math
 * - Throughput: Operations per cycle
 * - Mixed Precision: FP16 inputs, FP32 accumulation
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 02: Tensor Core Operation Simulation ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Tensor Core specifications
    std::cout << "Task 1 - Tensor Core Specifications (sm_80):" << std::endl;
    std::cout << "A100 Tensor Core capabilities:" << std::endl;
    std::cout << "  FP64: 64 TFLOPS (8 warps per SM)" << std::endl;
    std::cout << "  FP32: 19.5 TFLOPS" << std::endl;
    std::cout << "  FP16: 312 TFLOPS (with Tensor Cores)" << std::endl;
    std::cout << "  BF16: 312 TFLOPS" << std::endl;
    std::cout << "  INT8: 624 TOPS" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simulate 16x16x16 MMA operation
    std::cout << "Task 2 - Simulate 16x16x16 MMA:" << std::endl;
    
    // Create small matrices for simulation
    float A[256], B[256], C[256], D[256];
    
    for (int i = 0; i < 256; ++i) {
        A[i] = static_cast<float>(i % 16) / 10.0f;
        B[i] = static_cast<float>(i % 13) / 10.0f;
        C[i] = 0.0f;
        D[i] = 0.0f;
    }

    auto layout_A = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto layout_B = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto layout_C = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    auto tensor_D = make_tensor(make_gmem_ptr(D), layout_C);

    std::cout << "Matrix dimensions: 16x16 × 16x16 = 16x16" << std::endl;
    std::cout << "Total multiply-accumulates: 16 × 16 × 16 = 4096" << std::endl;
    std::cout << std::endl;

    // Simulate Tensor Core operation (simplified)
    std::cout << "Simulating Tensor Core MMA..." << std::endl;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < 16; ++k) {
                sum += tensor_A(m, k) * tensor_B(k, n);
            }
            tensor_D(m, n) = sum + tensor_C(m, n);
        }
    }
    std::cout << "MMA complete!" << std::endl;
    std::cout << std::endl;

    // Show sample results
    std::cout << "Sample results (top-left 4x4):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%6.3f ", tensor_D(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Compare scalar vs Tensor Core
    std::cout << "Task 3 - Scalar vs Tensor Core:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Scalar FP32 multiplication:" << std::endl;
    std::cout << "  Operations per cycle: ~8 (per SM)" << std::endl;
    std::cout << "  Cycles for 16x16x16 MMA: 4096 / 8 = 512 cycles" << std::endl;
    std::cout << std::endl;

    std::cout << "Tensor Core FP16 multiplication:" << std::endl;
    std::cout << "  Operations per cycle: ~256 (per SM)" << std::endl;
    std::cout << "  Cycles for 16x16x16 MMA: 4096 / 256 = 16 cycles" << std::endl;
    std::cout << "  Speedup: 512 / 16 = 32x faster!" << std::endl;
    std::cout << std::endl;

    // TASK 4: Throughput calculation
    std::cout << "Task 4 - Throughput Calculation:" << std::endl;
    std::cout << "For 16x16x16 MMA at 1.4 GHz (A100):" << std::endl;
    std::cout << "  Tensor Core ops/clock: 256 FMA = 512 FLOPs" << std::endl;
    std::cout << "  Warps per SM: 8" << std::endl;
    std::cout << "  SMs in A100: 108" << std::endl;
    std::cout << "  Peak FP16 TFLOPS: 512 × 8 × 108 × 1.4e9 / 1e12 = 312 TFLOPS" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate for different sizes
    std::cout << "=== Challenge: Different MMA Sizes ===" << std::endl;
    
    struct MMASize { int M, N, K; };
    MMASize sizes[] = {{16, 16, 16}, {16, 8, 32}, {8, 8, 4}};
    
    for (auto& s : sizes) {
        int ops = s.M * s.N * s.K;
        std::cout << s.M << "x" << s.N << "x" << s.K << " MMA:" << std::endl;
        std::cout << "  Total operations: " << ops << std::endl;
        std::cout << "  Scalar cycles (est.): " << ops / 8 << std::endl;
        std::cout << "  Tensor Core cycles (est.): " << ops / 256 << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Tensor Cores accelerate matrix math 32x" << std::endl;
    std::cout << "2. FP16 Tensor Cores: 312 TFLOPS on A100" << std::endl;
    std::cout << "3. MMA atoms map to Tensor Core instructions" << std::endl;
    std::cout << "4. Throughput scales with matrix size" << std::endl;

    return 0;
}
