/**
 * Exercise 01: Tiled Copy Basics
 * 
 * Objective: Understand the fundamentals of tiled copy operations
 *            where threads cooperate to copy data tiles
 * 
 * Tasks:
 * 1. Understand what tiled copy means
 * 2. See how threads divide copy work
 * 3. Practice with simple tile copy patterns
 * 4. Calculate copy efficiency
 * 
 * Key Concepts:
 * - Tiled Copy: Copying data in tiles rather than element-by-element
 * - Thread Cooperation: Multiple threads work together
 * - Efficiency: Better than naive element-wise copy
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: Tiled Copy Basics ===" << std::endl;
    std::cout << std::endl;

    // Simulate source and destination data
    float src_data[64];
    float dst_data[64];
    
    for (int i = 0; i < 64; ++i) {
        src_data[i] = static_cast<float>(i);
        dst_data[i] = 0.0f;
    }

    auto layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto src_tensor = make_tensor(make_gmem_ptr(src_data), layout);
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), layout);

    std::cout << "Source 8x8 Matrix:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(src_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 1: Understand tile division
    std::cout << "Task 1 - Tile Division:" << std::endl;
    std::cout << "Dividing 8x8 matrix into 2x2 tiles (each tile is 4x4):" << std::endl;
    std::cout << std::endl;
    
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int tile_row = i / 4;
            int tile_col = j / 4;
            printf("T%d%d ", tile_row, tile_col);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 2: Simulate tiled copy (one tile at a time)
    std::cout << "Task 2 - Simulated Tiled Copy:" << std::endl;
    
    // Copy tile 0,0 (top-left 4x4)
    std::cout << "Copying tile (0,0)..." << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            dst_tensor(i, j) = src_tensor(i, j);
        }
    }
    
    // Copy tile 0,1 (top-right 4x4)
    std::cout << "Copying tile (0,1)..." << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 4; j < 8; ++j) {
            dst_tensor(i, j) = src_tensor(i, j);
        }
    }
    
    // Copy tile 1,0 (bottom-left 4x4)
    std::cout << "Copying tile (1,0)..." << std::endl;
    for (int i = 4; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            dst_tensor(i, j) = src_tensor(i, j);
        }
    }
    
    // Copy tile 1,1 (bottom-right 4x4)
    std::cout << "Copying tile (1,1)..." << std::endl;
    for (int i = 4; i < 8; ++i) {
        for (int j = 4; j < 8; ++j) {
            dst_tensor(i, j) = src_tensor(i, j);
        }
    }
    
    std::cout << "Copy complete!" << std::endl;
    std::cout << std::endl;

    // Verify copy
    std::cout << "Destination 8x8 Matrix (after tiled copy):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(dst_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Compare with element-wise copy
    std::cout << "Task 3 - Copy Method Comparison:" << std::endl;
    std::cout << "Element-wise copy: 64 separate memory operations" << std::endl;
    std::cout << "Tiled copy: 4 tiles, each can be copied efficiently" << std::endl;
    std::cout << "Benefits of tiled copy:" << std::endl;
    std::cout << "  - Better memory coalescing" << std::endl;
    std::cout << "  - Enables vectorized loads/stores" << std::endl;
    std::cout << "  - Thread cooperation" << std::endl;
    std::cout << "  - Overlap with computation" << std::endl;
    std::cout << std::endl;

    // TASK 4: Calculate work per thread
    std::cout << "Task 4 - Work Distribution:" << std::endl;
    std::cout << "With 4 threads, each handling one 4x4 tile:" << std::endl;
    std::cout << "  Thread 0: Tile (0,0) - 16 elements" << std::endl;
    std::cout << "  Thread 1: Tile (0,1) - 16 elements" << std::endl;
    std::cout << "  Thread 2: Tile (1,0) - 16 elements" << std::endl;
    std::cout << "  Thread 3: Tile (1,1) - 16 elements" << std::endl;
    std::cout << "  Total: 64 elements" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Different tile sizes
    std::cout << "=== Challenge: Different Tile Sizes ===" << std::endl;
    std::cout << "What if we use 2x2 tiles instead of 4x4?" << std::endl;
    std::cout << "Number of tiles: " << (8/2) << " × " << (8/2) << " = " << (8/2)*(8/2) << " tiles" << std::endl;
    std::cout << "Elements per tile: " << 2*2 << std::endl;
    std::cout << std::endl;

    std::cout << "What if we use 8x1 tiles (row tiles)?" << std::endl;
    std::cout << "Number of tiles: " << 8 << std::endl;
    std::cout << "Elements per tile: " << 8 << std::endl;
    std::cout << std::endl;

    // TILED COPY IN CUDA
    std::cout << "=== Tiled Copy in CUDA Kernels ===" << std::endl;
    std::cout << R"(
// Conceptual CUDA kernel for tiled copy
__global__ void tiled_copy_kernel(float* src, float* dst, int M, int N) {
    // Each thread block handles one tile
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    
    // Each thread within block handles elements
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Calculate global coordinates
    int row = tile_row * TILE_M + tid / TILE_N;
    int col = tile_col * TILE_N + tid % TILE_N;
    
    // Copy element
    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Tiled copy divides work among threads" << std::endl;
    std::cout << "2. Each thread handles a tile of data" << std::endl;
    std::cout << "3. Tiled copy enables better memory patterns" << std::endl;
    std::cout << "4. Tile size affects parallelism and efficiency" << std::endl;

    return 0;
}
