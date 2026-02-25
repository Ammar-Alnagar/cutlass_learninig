/**
 * Exercise 06: Hierarchical Layouts
 * 
 * Objective: Master hierarchical layouts for organizing thread blocks, warps,
 *            and threads in CUDA kernels
 * 
 * Tasks:
 * 1. Create a layout representing thread block hierarchy
 * 2. Map warps to threads within a block
 * 3. Create a 3-level hierarchy: block -> warp -> thread
 * 4. Understand how hierarchy enables scalable kernels
 * 
 * Key Concepts:
 * - Hierarchy: Multiple levels of organization
 * - Thread Blocks: Groups of threads that can cooperate
 * - Warps: Groups of 32 threads executed together
 * - Scalability: Same code works for different problem sizes
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 06: Hierarchical Layouts ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a thread block layout (128 threads)
    // Organize as 4 warps of 32 threads each
    auto block_layout = make_layout(make_shape(Int<4>{}, Int<32>{}), GenRowMajor{});
    
    std::cout << "Task 1 - Thread Block Layout (4 warps x 32 threads):" << std::endl;
    std::cout << "Block layout: " << block_layout << std::endl;
    std::cout << "Shape: " << block_layout.shape() << std::endl;
    std::cout << "Total threads: " << 4 * 32 << std::endl;
    std::cout << std::endl;

    // TASK 2: Create a warp layout within a thread block
    // Each warp handles a portion of the computation
    auto warp_layout = make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{});
    
    std::cout << "Task 2 - Warp Layout (4 warps):" << std::endl;
    std::cout << "Warp layout: " << warp_layout << std::endl;
    std::cout << std::endl;

    // TASK 3: Create a thread-within-warp layout
    // 32 threads per warp, organized for efficient memory access
    auto thread_in_warp_layout = make_layout(make_shape(Int<32>{}, Int<1>{}), GenRowMajor{});
    
    std::cout << "Task 3 - Thread-within-Warp Layout (32 threads):" << std::endl;
    std::cout << "Thread layout: " << thread_in_warp_layout << std::endl;
    std::cout << std::endl;

    // TASK 4: Create a 2D thread block layout (8x16 threads)
    // Common configuration for matrix operations
    auto block_2d_layout = make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{});
    
    std::cout << "Task 4 - 2D Thread Block Layout (8x16):" << std::endl;
    std::cout << "2D Block layout: " << block_2d_layout << std::endl;
    std::cout << "Total threads: " << 8 * 16 << std::endl;
    std::cout << std::endl;

    // Visualize warp assignment in 2D block
    std::cout << "=== Warp Assignment in 2D Block (8x16) ===" << std::endl;
    std::cout << "Assuming 4 warps (warp 0-3), each with 32 threads:" << std::endl;
    std::cout << std::endl;
    
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            int thread_id = i * 16 + j;
            int warp_id = thread_id / 32;
            printf("W%d ", warp_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Create a hierarchical layout for matrix multiplication
    // Thread block processes a tile, threads within block cooperate
    std::cout << "=== Matrix Multiplication Hierarchy ===" << std::endl;
    std::cout << "Problem: Multiply two 64x64 matrices" << std::endl;
    std::cout << "Thread block size: 8x16 threads (128 threads)" << std::endl;
    std::cout << "Each thread computes: 2x2 elements of output" << std::endl;
    std::cout << std::endl;

    // Layout for output tile per block (8x16 elements)
    auto output_tile_layout = make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{});
    
    std::cout << "Output tile per block (8x16):" << std::endl;
    print(output_tile_layout);
    std::cout << std::endl;

    // Layout for thread's portion (2x2 elements)
    auto thread_work_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{});
    
    std::cout << "Each thread's work (2x2):" << std::endl;
    print(thread_work_layout);
    std::cout << std::endl;

    // CHALLENGE: Calculate thread-to-output mapping
    std::cout << "=== Challenge: Thread to Output Mapping ===" << std::endl;
    std::cout << "For a thread at position (ti, tj) in an 8x16 block:" << std::endl;
    std::cout << "  Output row = ti * 2 to ti * 2 + 1" << std::endl;
    std::cout << "  Output col = tj * 2 to tj * 2 + 1" << std::endl;
    std::cout << std::endl;

    std::cout << "Example mappings:" << std::endl;
    int thread_positions[][2] = {{0, 0}, {0, 1}, {1, 0}, {7, 15}};
    for (auto& pos : thread_positions) {
        int ti = pos[0];
        int tj = pos[1];
        std::cout << "  Thread (" << ti << "," << tj << ") computes output elements:" << std::endl;
        std::cout << "    (" << (ti*2) << "," << (tj*2) << "), (" << (ti*2) << "," << (tj*2+1) << ")" << std::endl;
        std::cout << "    (" << (ti*2+1) << "," << (tj*2) << "), (" << (ti*2+1) << "," << (tj*2+1) << ")" << std::endl;
    }
    std::cout << std::endl;

    // Visualize the full hierarchy
    std::cout << "=== Full Hierarchy Visualization ===" << std::endl;
    std::cout << "Level 1: Grid of thread blocks" << std::endl;
    std::cout << "Level 2: Thread block (8x16 threads)" << std::endl;
    std::cout << "Level 3: Each thread (2x2 elements)" << std::endl;
    std::cout << std::endl;

    std::cout << "64x64 Matrix divided among blocks (8x8 blocks):" << std::endl;
    for (int bi = 0; bi < 8; ++bi) {
        for (int bj = 0; bj < 8; ++bj) {
            printf("B%2d ", bi * 8 + bj);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Hierarchical layouts organize threads at multiple levels" << std::endl;
    std::cout << "2. Thread blocks enable cooperation and synchronization" << std::endl;
    std::cout << "3. Warps are the hardware execution unit (32 threads)" << std::endl;
    std::cout << "4. Hierarchies enable scalable kernel design" << std::endl;

    return 0;
}
