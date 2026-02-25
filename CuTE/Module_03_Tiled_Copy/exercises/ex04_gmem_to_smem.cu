/**
 * Exercise 04: Global to Shared Memory Copy
 * 
 * Objective: Learn to copy data from global memory to shared memory
 *            efficiently using tiled copy patterns
 * 
 * Tasks:
 * 1. Understand the gmem -> smem copy pattern
 * 2. Practice with tile-based loading
 * 3. Handle shared memory allocation
 * 4. Optimize for coalesced access
 * 
 * Key Concepts:
 * - Global Memory: Large, slow, off-chip
 * - Shared Memory: Small, fast, on-chip
 * - Tiled Loading: Load data in tiles for efficiency
 * - Coalescing: Consecutive threads access consecutive addresses
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: Global to Shared Memory Copy ===" << std::endl;
    std::cout << std::endl;

    // Simulate global memory data (large matrix)
    float gmem_data[256];
    for (int i = 0; i < 256; ++i) {
        gmem_data[i] = static_cast<float>(i);
    }

    // Simulate shared memory buffer (smaller tile buffer)
    float smem_buffer[64];  // Holds one 8x8 tile

    auto gmem_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto smem_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});

    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_data), gmem_layout);
    auto smem_tensor = make_tensor(make_smem_ptr(smem_buffer), smem_layout);

    // TASK 1: Understand the copy pattern
    std::cout << "Task 1 - Copy Pattern:" << std::endl;
    std::cout << "Global memory: 16x16 matrix (256 elements)" << std::endl;
    std::cout << "Shared memory: 8x8 buffer (64 elements)" << std::endl;
    std::cout << "Strategy: Load one 8x8 tile at a time" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simulate loading tile (0,0)
    std::cout << "Task 2 - Loading Tile (0,0):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            smem_tensor(i, j) = gmem_tensor(i, j);
        }
    }

    std::cout << "Shared memory after loading tile (0,0):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%5.1f ", smem_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Simulate loading tile (0,1)
    std::cout << "Task 3 - Loading Tile (0,1):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            smem_tensor(i, j) = gmem_tensor(i, j + 8);
        }
    }

    std::cout << "Shared memory after loading tile (0,1):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%5.1f ", smem_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Analyze access pattern
    std::cout << "Task 4 - Access Pattern Analysis:" << std::endl;
    std::cout << "When loading tile (0,0) with 8x8 threads:" << std::endl;
    std::cout << "  Thread (ti, tj) loads from gmem(ti, tj)" << std::endl;
    std::cout << "  Thread (ti, tj) stores to smem(ti, tj)" << std::endl;
    std::cout << std::endl;

    std::cout << "Coalescing analysis:" << std::endl;
    std::cout << "  Consecutive threads in x dimension:" << std::endl;
    std::cout << "    Thread (0,0) accesses gmem(0,0) = offset 0" << std::endl;
    std::cout << "    Thread (0,1) accesses gmem(0,1) = offset 1" << std::endl;
    std::cout << "    Thread (0,2) accesses gmem(0,2) = offset 2" << std::endl;
    std::cout << "  -> COALESCED! Consecutive threads access consecutive addresses" << std::endl;
    std::cout << std::endl;

    // TASK 5: Calculate memory traffic
    std::cout << "Task 5 - Memory Traffic:" << std::endl;
    std::cout << "Total data to process: 16x16 = 256 elements" << std::endl;
    std::cout << "Tile size: 8x8 = 64 elements" << std::endl;
    std::cout << "Number of tiles: " << (16*16) / (8*8) << " tiles" << std::endl;
    std::cout << "Global memory reads: 256 elements" << std::endl;
    std::cout << "Shared memory writes: 256 elements (64 per tile × 4 tiles)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Optimize with vectorized loads
    std::cout << "=== Challenge: Vectorized Loading ===" << std::endl;
    std::cout << "How can we improve the copy using vectorized loads?" << std::endl;
    std::cout << "Answer: Load 4 elements per thread using 128-bit loads" << std::endl;
    std::cout << "  Original: 64 threads × 1 element = 64 loads" << std::endl;
    std::cout << "  Vectorized: 16 threads × 4 elements = 16 loads" << std::endl;
    std::cout << "  Improvement: 4x fewer load instructions" << std::endl;
    std::cout << std::endl;

    // CUDA KERNEL PATTERN
    std::cout << "=== CUDA Kernel Pattern ===" << std::endl;
    std::cout << R"(
__global__ void gmem_to_smem_copy(float* global_data, float* shared_data, 
                                   int M, int N, int tile_M, int tile_N) {
    // Shared memory declaration
    extern __shared__ float smem[];
    
    // Create tensors
    auto gmem_tensor = make_tensor(make_gmem_ptr(global_data), ...);
    auto smem_tensor = make_tensor(make_smem_ptr(smem), ...);
    
    // Thread coordinates
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    
    // Global coordinates for this tile
    int global_i = blockIdx.y * tile_M + ti;
    int global_j = blockIdx.x * tile_N + tj;
    
    // Load from global to shared (coalesced)
    if (global_i < M && global_j < N) {
        smem_tensor(ti, tj) = gmem_tensor(global_i, global_j);
    }
    
    __syncthreads();  // Ensure all loads complete
    
    // Now smem is ready for computation...
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Load data from gmem to smem in tiles" << std::endl;
    std::cout << "2. Use coalesced access patterns for efficiency" << std::endl;
    std::cout << "3. Synchronize threads after loading" << std::endl;
    std::cout << "4. Shared memory enables data reuse" << std::endl;

    return 0;
}
