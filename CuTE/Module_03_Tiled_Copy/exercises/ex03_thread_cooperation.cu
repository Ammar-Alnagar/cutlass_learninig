/**
 * Exercise 03: Thread Cooperation
 * 
 * Objective: Understand how threads cooperate in tiled copy operations
 * 
 * Tasks:
 * 1. See how threads divide copy work
 * 2. Understand thread indexing
 * 3. Calculate work per thread
 * 4. Practice with different thread configurations
 * 
 * Key Concepts:
 * - Thread Cooperation: Threads work together on a task
 * - Work Division: Each thread handles part of the data
 * - Efficiency: Parallel copy is faster than serial
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 03: Thread Cooperation ===" << std::endl;
    std::cout << std::endl;

    // Simulate data for a 16x16 matrix
    float data[256];
    for (int i = 0; i < 256; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto tensor = make_tensor(make_gmem_ptr(data), layout);

    // TASK 1: Thread block configuration
    std::cout << "Task 1 - Thread Block Configuration:" << std::endl;
    std::cout << "Using a 4x4 thread block (16 threads total)" << std::endl;
    std::cout << "Matrix size: 16x16 = 256 elements" << std::endl;
    std::cout << "Work per thread: " << 256 / 16 << " elements" << std::endl;
    std::cout << std::endl;

    // TASK 2: Thread assignment visualization
    std::cout << "Task 2 - Thread Assignment:" << std::endl;
    std::cout << "Each thread handles 1 element in a 4x4 pattern:" << std::endl;
    std::cout << std::endl;

    // Show which thread handles which element
    std::cout << "Thread ID assignment (threadIdx.y, threadIdx.x):" << std::endl;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            // Thread ID based on position in 4x4 repeating pattern
            int thread_id = (i % 4) * 4 + (j % 4);
            printf("T%2d ", thread_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Calculate global coordinates from thread ID
    std::cout << "Task 3 - Coordinate Calculation:" << std::endl;
    std::cout << "For thread (ti, tj) in a 4x4 block:" << std::endl;
    std::cout << "  Global row = blockIdx.y * 4 + ti" << std::endl;
    std::cout << "  Global col = blockIdx.x * 4 + tj" << std::endl;
    std::cout << std::endl;

    // Example for block (0, 0)
    std::cout << "Block (0, 0) - Thread to Global Mapping:" << std::endl;
    for (int ti = 0; ti < 4; ++ti) {
        for (int tj = 0; tj < 4; ++tj) {
            int global_row = 0 * 4 + ti;
            int global_col = 0 * 4 + tj;
            int thread_id = ti * 4 + tj;
            std::cout << "  Thread " << thread_id << " -> (" << global_row << ", " << global_col << ")" << std::endl;
        }
    }
    std::cout << std::endl;

    // TASK 4: Multiple blocks cooperation
    std::cout << "Task 4 - Multiple Thread Blocks:" << std::endl;
    std::cout << "Using 4x4 blocks to cover 16x16 matrix:" << std::endl;
    std::cout << "  Blocks needed: (16/4) × (16/4) = 4 × 4 = 16 blocks" << std::endl;
    std::cout << std::endl;

    std::cout << "Block assignment:" << std::endl;
    for (int bi = 0; bi < 4; ++bi) {
        for (int bj = 0; bj < 4; ++bj) {
            printf("B(%d,%d) ", bi, bj);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Work distribution analysis
    std::cout << "Task 5 - Work Distribution Analysis:" << std::endl;
    std::cout << "Configuration: 16 blocks × 16 threads = 256 threads" << std::endl;
    std::cout << "Matrix: 16 × 16 = 256 elements" << std::endl;
    std::cout << "Work per thread: 1 element" << std::endl;
    std::cout << std::endl;

    std::cout << "Alternative: 4 blocks × 16 threads = 64 threads" << std::endl;
    std::cout << "Work per thread: 256 / 64 = 4 elements" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate for different configurations
    std::cout << "=== Challenge: Configuration Analysis ===" << std::endl;
    
    struct Config { int blocks; int threads_per_block; int matrix_size; };
    Config configs[] = {
        {1, 256, 256},
        {4, 64, 256},
        {16, 16, 256},
        {64, 4, 256}
    };

    for (auto& cfg : configs) {
        int total_threads = cfg.blocks * cfg.threads_per_block;
        float work_per_thread = (float)cfg.matrix_size / total_threads;
        std::cout << cfg.blocks << " blocks × " << cfg.threads_per_block 
                  << " threads = " << total_threads << " threads" << std::endl;
        std::cout << "  Work per thread: " << work_per_thread << " elements" << std::endl;
    }
    std::cout << std::endl;

    // COOPERATIVE COPY PATTERN
    std::cout << "=== Cooperative Copy Pattern ===" << std::endl;
    std::cout << R"(
__global__ void cooperative_copy(float* src, float* dst, int M, int N) {
    // Calculate global coordinates from thread/block indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread copies one element
    if (row < M && col < N) {
        int idx = row * N + col;
        dst[idx] = src[idx];
    }
}

// Launch configuration
dim3 block(4, 4);  // 16 threads per block
dim3 grid(4, 4);   // 16 blocks for 16x16 matrix
cooperative_copy<<<grid, block>>>(src, dst, 16, 16);
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Threads cooperate by dividing work" << std::endl;
    std::cout << "2. Thread indices determine work assignment" << std::endl;
    std::cout << "3. Block configuration affects parallelism" << std::endl;
    std::cout << "4. More threads = less work per thread" << std::endl;

    return 0;
}
