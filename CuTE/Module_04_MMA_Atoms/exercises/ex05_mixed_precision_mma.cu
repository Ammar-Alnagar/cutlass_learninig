/**
 * Exercise 05: Mixed Precision MMA
 * 
 * Objective: Learn to use mixed precision in MMA operations
 *            for optimal performance and accuracy
 * 
 * Tasks:
 * 1. Understand mixed precision benefits
 * 2. Work with FP16 inputs and FP32 accumulation
 * 3. Compare precision configurations
 * 4. Handle type conversions
 * 
 * Key Concepts:
 * - Mixed Precision: Different precisions for different operations
 * - FP16: Fast computation, reduced memory
 * - FP32: Accurate accumulation
 * - BF16: Brain floating point (ML optimized)
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 05: Mixed Precision MMA ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Precision comparison
    std::cout << "Task 1 - Precision Comparison:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "| Format | Bits | Range       | Precision  |" << std::endl;
    std::cout << "|--------|------|-------------|------------|" << std::endl;
    std::cout << "| FP32   | 32   | ±3.4e38     | 7 digits   |" << std::endl;
    std::cout << "| FP16   | 16   | ±65504      | 3 digits   |" << std::endl;
    std::cout << "| BF16   | 16   | ±3.4e38     | 3 digits   |" << std::endl;
    std::cout << "| INT8   | 8    | -128 to 127 | Integer    |" << std::endl;
    std::cout << std::endl;

    // TASK 2: Mixed precision MMA configuration
    std::cout << "Task 2 - Mixed Precision MMA:" << std::endl;
    std::cout << "Common Tensor Core configurations:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "1. F16 × F16 -> F32 (most common):" << std::endl;
    std::cout << "   Inputs: FP16, Accumulation: FP32" << std::endl;
    std::cout << "   Use case: Deep learning training" << std::endl;
    std::cout << std::endl;

    std::cout << "2. BF16 × BF16 -> F32:" << std::endl;
    std::cout << "   Inputs: BF16, Accumulation: FP32" << std::endl;
    std::cout << "   Use case: ML with FP32 range needed" << std::endl;
    std::cout << std::endl;

    std::cout << "3. INT8 × INT8 -> INT32:" << std::endl;
    std::cout << "   Inputs: INT8, Accumulation: INT32" << std::endl;
    std::cout << "   Use case: Inference quantization" << std::endl;
    std::cout << std::endl;

    std::cout << "4. F64 × F64 -> F64:" << std::endl;
    std::cout << "   Inputs: FP64, Accumulation: FP64" << std::endl;
    std::cout << "   Use case: Scientific computing" << std::endl;
    std::cout << std::endl;

    // TASK 3: Simulate mixed precision computation
    std::cout << "Task 3 - Mixed Precision Simulation:" << std::endl;
    
    // FP16 inputs (simulated with float)
    float A_fp16[64];
    float B_fp16[64];
    for (int i = 0; i < 64; ++i) {
        A_fp16[i] = static_cast<float>(i % 8) / 10.0f;
        B_fp16[i] = static_cast<float>(i % 7) / 10.0f;
    }

    // FP32 accumulator
    float C_fp32[64];
    for (int i = 0; i < 64; ++i) {
        C_fp32[i] = 0.0f;
    }

    auto layout_A = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto layout_B = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto layout_C = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});

    auto A_tensor = make_tensor(make_gmem_ptr(A_fp16), layout_A);
    auto B_tensor = make_tensor(make_gmem_ptr(B_fp16), layout_B);
    auto C_tensor = make_tensor(make_gmem_ptr(C_fp32), layout_C);

    std::cout << "FP16 inputs, FP32 accumulation:" << std::endl;
    std::cout << "Computing 8x8 × 8x8 = 8x8 matrix multiply" << std::endl;
    std::cout << std::endl;

    // Simulate mixed precision MMA
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 8; ++k) {
                // FP16 multiply, FP32 accumulate
                sum += A_tensor(i, k) * B_tensor(k, j);
            }
            C_tensor(i, j) = sum;  // FP32 result
        }
    }

    std::cout << "Result (FP32 accumulation):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%6.4f ", C_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Performance comparison
    std::cout << "Task 4 - Performance Comparison:" << std::endl;
    std::cout << "Relative throughput on A100:" << std::endl;
    std::cout << "  FP64: 1x (baseline)" << std::endl;
    std::cout << "  FP32: 2x" << std::endl;
    std::cout << "  FP16 (Tensor Core): 16x" << std::endl;
    std::cout << "  BF16 (Tensor Core): 16x" << std::endl;
    std::cout << "  INT8 (Tensor Core): 32x" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Choose precision for application
    std::cout << "=== Challenge: Choose Precision ===" << std::endl;
    std::cout << "Scenario 1: Neural network training" << std::endl;
    std::cout << "  Answer: FP16 inputs + FP32 accumulation" << std::endl;
    std::cout << "  Reason: Good accuracy, 16x speedup" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 2: Neural network inference" << std::endl;
    std::cout << "  Answer: INT8" << std::endl;
    std::cout << "  Reason: Maximum throughput, acceptable accuracy" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 3: Scientific simulation" << std::endl;
    std::cout << "  Answer: FP64" << std::endl;
    std::cout << "  Reason: Maximum accuracy required" << std::endl;
    std::cout << std::endl;

    // MMA INSTRUCTION EXAMPLE
    std::cout << "=== MMA Instruction Example ===" << std::endl;
    std::cout << R"(
// FP16 MMA instruction (sm_80+)
mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32
    {d0, d1, ...},  // FP32 destination (accumulator)
    {a0, a1, ...},  // FP16 operand A
    {b0, b1, ...},  // FP16 operand B
    {c0, c1, ...};  // FP32 accumulator (input)

// INT8 MMA instruction
mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32
    {d0, ...},  // INT32 destination
    {a0, ...},  // INT8 operand A
    {b0, ...},  // INT8 operand B
    {c0, ...};  // INT32 accumulator
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Mixed precision balances speed and accuracy" << std::endl;
    std::cout << "2. FP16/BF16 for inputs, FP32 for accumulation" << std::endl;
    std::cout << "3. Tensor Cores accelerate mixed precision" << std::endl;
    std::cout << "4. Choose precision based on application needs" << std::endl;

    return 0;
}
