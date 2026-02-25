/**
 * Exercise 01: MMA Atom Basics
 * 
 * Objective: Understand the fundamentals of MMA (Matrix Multiply-Accumulate)
 *            atoms in CuTe and Tensor Core operations
 * 
 * Tasks:
 * 1. Understand what an MMA atom is
 * 2. Learn the MMA operation: D = A * B + C
 * 3. See how threads cooperate in MMA
 * 4. Practice with small matrix multiplication
 * 
 * Key Concepts:
 * - MMA Atom: Fundamental Tensor Core operation unit
 * - Matrix Multiply-Accumulate: D = A * B + C
 * - Tensor Cores: Hardware units for fast matrix math
 * - Warp-Level: 32 threads cooperate for MMA
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: MMA Atom Basics ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Understand MMA operation
    std::cout << "Task 1 - MMA Operation (D = A * B + C):" << std::endl;
    std::cout << "A: Operand A matrix (M x K)" << std::endl;
    std::cout << "B: Operand B matrix (K x N)" << std::endl;
    std::cout << "C: Accumulator matrix (M x N)" << std::endl;
    std::cout << "D: Result matrix (M x N)" << std::endl;
    std::cout << std::endl;

    // Small example: 2x4 × 4x2 = 2x2
    float A_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // 2x4
    float B_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // 4x2
    float C_data[4] = {0, 0, 0, 0};              // 2x2 accumulator
    float D_data[4];                              // 2x2 result

    auto A_layout = make_layout(make_shape(Int<2>{}, Int<4>{}), GenRowMajor{});
    auto B_layout = make_layout(make_shape(Int<4>{}, Int<2>{}), GenRowMajor{});
    auto C_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{});
    
    auto A_tensor = make_tensor(make_gmem_ptr(A_data), A_layout);
    auto B_tensor = make_tensor(make_gmem_ptr(B_data), B_layout);
    auto C_tensor = make_tensor(make_gmem_ptr(C_data), C_layout);
    auto D_tensor = make_tensor(make_gmem_ptr(D_data), C_layout);

    std::cout << "Matrix A (2x4):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(A_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Matrix B (4x2):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%3d ", static_cast<int>(B_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 2: Perform matrix multiplication manually
    std::cout << "Task 2 - Manual Matrix Multiplication:" << std::endl;
    std::cout << "D[i,j] = sum_k(A[i,k] * B[k,j])" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float sum = 0.0f;
            std::cout << "D[" << i << "," << j << "] = ";
            for (int k = 0; k < 4; ++k) {
                sum += A_tensor(i, k) * B_tensor(k, j);
                std::cout << "(" << A_tensor(i, k) << "*" << B_tensor(k, j) << ")";
                if (k < 3) std::cout << " + ";
            }
            std::cout << " = " << sum << std::endl;
            D_tensor(i, j) = sum;
        }
    }
    std::cout << std::endl;

    std::cout << "Result D (2x2):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%6.1f ", D_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Understand Tensor Core MMA
    std::cout << "Task 3 - Tensor Core MMA:" << std::endl;
    std::cout << "Tensor Cores perform D = A * B + C in hardware" << std::endl;
    std::cout << std::endl;
    std::cout << "Common Tensor Core configurations (sm_80+):" << std::endl;
    std::cout << "  16x8x16 FP16: 16x8 output, 16-wide reduction" << std::endl;
    std::cout << "  16x8x32 INT8: 16x8 output, 32-wide reduction" << std::endl;
    std::cout << "  8x8x4 FP64: 8x8 output, 4-wide reduction" << std::endl;
    std::cout << std::endl;

    // TASK 4: Thread cooperation in MMA
    std::cout << "Task 4 - Thread Cooperation:" << std::endl;
    std::cout << "A warp (32 threads) cooperates for MMA:" << std::endl;
    std::cout << "  - Each thread loads operands to registers" << std::endl;
    std::cout << "  - Threads feed data to Tensor Core" << std::endl;
    std::cout << "  - Results distributed across threads" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate MMA operations
    std::cout << "=== Challenge: MMA Calculation ===" << std::endl;
    std::cout << "For a 16x16x16 MMA operation:" << std::endl;
    std::cout << "  A: 16x16 matrix" << std::endl;
    std::cout << "  B: 16x16 matrix" << std::endl;
    std::cout << "  C: 16x16 accumulator" << std::endl;
    std::cout << "  Total multiply-accumulates: 16 × 16 × 16 = 4096" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. MMA performs D = A * B + C" << std::endl;
    std::cout << "2. Tensor Cores accelerate matrix math" << std::endl;
    std::cout << "3. Threads cooperate at warp level" << std::endl;
    std::cout << "4. MMA atoms are the building blocks" << std::endl;

    return 0;
}
