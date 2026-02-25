/**
 * Exercise 03: Thread to Tensor Core Mapping
 * 
 * Objective: Understand how threads are organized to feed data
 *            to Tensor Cores efficiently
 * 
 * Tasks:
 * 1. Learn warp-level organization for MMA
 * 2. Understand thread roles in MMA
 * 3. Map threads to Tensor Core operands
 * 4. Practice with different configurations
 * 
 * Key Concepts:
 * - Warp: 32 threads that execute together
 * - MMA Warp: Threads organized for matrix multiply
 * - Operand Loading: Each thread loads specific elements
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 03: Thread to Tensor Core Mapping ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Warp organization
    std::cout << "Task 1 - Warp Organization:" << std::endl;
    std::cout << "A warp has 32 threads (threadIdx 0-31)" << std::endl;
    std::cout << std::endl;
    std::cout << "For 16x16x16 MMA with 32 threads:" << std::endl;
    std::cout << "  Output elements: 16 × 16 = 256" << std::endl;
    std::cout << "  Elements per thread: 256 / 32 = 8" << std::endl;
    std::cout << "  Each thread computes 8 output elements" << std::endl;
    std::cout << std::endl;

    // TASK 2: Thread arrangement for MMA
    std::cout << "Task 2 - Thread Arrangement:" << std::endl;
    std::cout << "Common arrangement: 8 rows × 4 columns of threads" << std::endl;
    std::cout << "  8 × 4 = 32 threads (one warp)" << std::endl;
    std::cout << std::endl;

    std::cout << "Thread layout (8x4):" << std::endl;
    for (int ti = 0; ti < 8; ++ti) {
        for (int tj = 0; tj < 4; ++tj) {
            int thread_id = ti * 4 + tj;
            printf("T%2d ", thread_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Output element assignment
    std::cout << "Task 3 - Output Element Assignment:" << std::endl;
    std::cout << "Each thread handles 2x2 output elements:" << std::endl;
    std::cout << std::endl;

    std::cout << "16x16 output matrix divided among 32 threads:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            // Calculate which thread handles this element
            int thread_row = i / 2;  // 8 thread rows
            int thread_col = j / 4;  // 4 thread columns
            int thread_id = thread_row * 4 + thread_col;
            
            // Local position within thread's 2x2 block
            int local_i = i % 2;
            int local_j = j % 4;
            
            printf("T%2d ", thread_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Operand loading pattern
    std::cout << "Task 4 - Operand Loading:" << std::endl;
    std::cout << "For 16x16x16 MMA:" << std::endl;
    std::cout << "  A operand (16x16): Each thread loads specific rows" << std::endl;
    std::cout << "  B operand (16x16): Each thread loads specific columns" << std::endl;
    std::cout << std::endl;

    std::cout << "Thread 0 loads:" << std::endl;
    std::cout << "  From A: rows 0-1, all 16 columns" << std::endl;
    std::cout << "  From B: all 16 rows, columns 0-3" << std::endl;
    std::cout << std::endl;

    // TASK 5: MMA instruction flow
    std::cout << "Task 5 - MMA Instruction Flow:" << std::endl;
    std::cout << "1. Load A operands to registers (cooperative)" << std::endl;
    std::cout << "2. Load B operands to registers (cooperative)" << std::endl;
    std::cout << "3. Execute mma.sync instruction" << std::endl;
    std::cout << "4. Accumulate results in registers" << std::endl;
    std::cout << "5. Repeat for K dimension" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate thread responsibilities
    std::cout << "=== Challenge: Thread Responsibilities ===" << std::endl;
    std::cout << "For thread 15 in a 16x16x16 MMA:" << std::endl;
    
    int thread_id = 15;
    int thread_row = thread_id / 4;
    int thread_col = thread_id % 4;
    
    std::cout << "Thread ID: " << thread_id << std::endl;
    std::cout << "Thread position: (" << thread_row << ", " << thread_col << ")" << std::endl;
    std::cout << "Output rows: " << (thread_row * 2) << "-" << (thread_row * 2 + 1) << std::endl;
    std::cout << "Output cols: " << (thread_col * 4) << "-" << (thread_col * 4 + 3) << std::endl;
    std::cout << std::endl;

    // CUDA WARP MMA PATTERN
    std::cout << "=== CUDA Warp MMA Pattern ===" << std::endl;
    std::cout << R"(
// Conceptual warp-level MMA
__global__ void warp_mma_kernel(float* A, float* B, float* C, int K) {
    // Each warp performs 16x16x16 MMA
    
    // Thread's register fragments
    float accum[8];  // 8 elements per thread
    float frag_a[...];
    float frag_b[...];
    
    // Load operands cooperatively
    load_operands(frag_a, frag_b, A, B, warp_id);
    
    // Perform MMA
    for (int k = 0; k < K / 16; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32"
            "{%0, %1, ...}, {%2, ...}, {%3, ...}, {%4, ...};"
            : /* outputs */ : /* inputs */
        );
    }
    
    // Store results
    store_results(accum, C, warp_id);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. 32 threads cooperate for warp-level MMA" << std::endl;
    std::cout << "2. Each thread handles specific output elements" << std::endl;
    std::cout << "3. Threads load operands cooperatively" << std::endl;
    std::cout << "4. mma.sync instruction executes Tensor Core op" << std::endl;

    return 0;
}
