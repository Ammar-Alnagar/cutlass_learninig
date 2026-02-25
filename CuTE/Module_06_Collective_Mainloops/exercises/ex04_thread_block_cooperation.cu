/**
 * Exercise 04: Thread Block Cooperation
 * 
 * Objective: Understand how thread blocks cooperate in large-scale
 *            matrix operations
 * 
 * Tasks:
 * 1. Learn thread block organization
 * 2. Understand block-level work division
 * 3. Practice with block cooperation patterns
 * 4. Handle synchronization
 * 
 * Key Concepts:
 * - Thread Block: Group of threads that can cooperate
 * - Grid: Collection of thread blocks
 * - Block Cooperation: Multiple blocks work together
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: Thread Block Cooperation ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Thread block organization
    std::cout << "Task 1 - Thread Block Organization:" << std::endl;
    std::cout << "CUDA hierarchy:" << std::endl;
    std::cout << "  Grid: Collection of thread blocks" << std::endl;
    std::cout << "  Block: Group of threads (up to 1024)" << std::endl;
    std::cout << "  Warp: 32 threads that execute together" << std::endl;
    std::cout << "  Thread: Individual execution unit" << std::endl;
    std::cout << std::endl;

    // TASK 2: Block work division for GEMM
    std::cout << "Task 2 - Block Work Division for GEMM:" << std::endl;
    std::cout << "For 128x128 GEMM with 64x64 thread blocks:" << std::endl;
    std::cout << "  Output tiles: (128/64) × (128/64) = 2 × 2 = 4 blocks" << std::endl;
    std::cout << std::endl;

    std::cout << "Block assignment:" << std::endl;
    for (int by = 0; by < 2; ++by) {
        for (int bx = 0; bx < 2; ++bx) {
            std::cout << "  Block (" << bx << ", " << by << "): ";
            std::cout << "Output rows " << (by * 64) << "-" << (by * 64 + 63);
            std::cout << ", cols " << (bx * 64) << "-" << (bx * 64 + 63) << std::endl;
        }
    }
    std::cout << std::endl;

    // TASK 3: Visualize block grid
    std::cout << "Task 3 - Block Grid Visualization:" << std::endl;
    std::cout << "128x128 output with 64x64 blocks:" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            int block_y = i / 64;
            int block_x = j / 64;
            int block_id = block_y * 2 + block_x;
            printf("B%d ", block_id);
        }
        std::cout << std::endl;
        if (i == 63) std::cout << std::endl;  // Visual break
    }
    std::cout << std::endl;

    // TASK 4: Block cooperation patterns
    std::cout << "Task 4 - Block Cooperation Patterns:" << std::endl;
    std::cout << std::endl;

    std::cout << "Pattern 1: Independent Blocks" << std::endl;
    std::cout << "  Each block computes its output tile independently" << std::endl;
    std::cout << "  No inter-block communication needed" << std::endl;
    std::cout << "  Most common for GEMM" << std::endl;
    std::cout << std::endl;

    std::cout << "Pattern 2: Cooperative Blocks" << std::endl;
    std::cout << "  Multiple blocks cooperate on single output tile" << std::endl;
    std::cout << "  Requires synchronization" << std::endl;
    std::cout << "  Used for very large tiles" << std::endl;
    std::cout << std::endl;

    std::cout << "Pattern 3: Pipelined Blocks" << std::endl;
    std::cout << "  Blocks form a pipeline" << std::endl;
    std::cout << "  Each block handles different stage" << std::endl;
    std::cout << "  Complex but high throughput" << std::endl;
    std::cout << std::endl;

    // TASK 5: Calculate block requirements
    std::cout << "Task 5 - Block Requirements:" << std::endl;
    
    struct GEMMConfig { int M, N, K, tile_M, tile_N; };
    GEMMConfig configs[] = {
        {64, 64, 64, 32, 32},
        {128, 128, 128, 64, 64},
        {256, 256, 256, 128, 128},
        {512, 512, 512, 128, 128},
    };

    std::cout << "| M   | N   | K   | Tile    | Blocks    |" << std::endl;
    std::cout << "|-----|-----|-----|---------|-----------|" << std::endl;
    
    for (auto& cfg : configs) {
        int blocks_m = (cfg.M + cfg.tile_M - 1) / cfg.tile_M;
        int blocks_n = (cfg.N + cfg.tile_N - 1) / cfg.tile_N;
        int total_blocks = blocks_m * blocks_n;
        
        printf("| %-3d | %-3d | %-3d | %dx%-3d  | %9d |\n",
               cfg.M, cfg.N, cfg.K, cfg.tile_M, cfg.tile_N, total_blocks);
    }
    std::cout << std::endl;

    // CHALLENGE: Design block configuration
    std::cout << "=== Challenge: Block Configuration Design ===" << std::endl;
    std::cout << "For 1024x1024 GEMM:" << std::endl;
    std::cout << "Option 1: 128x128 tiles" << std::endl;
    std::cout << "  Blocks: (1024/128)² = 64 blocks" << std::endl;
    std::cout << "  Work per block: 128×128 = 16,384 elements" << std::endl;
    std::cout << std::endl;

    std::cout << "Option 2: 64x64 tiles" << std::endl;
    std::cout << "  Blocks: (1024/64)² = 256 blocks" << std::endl;
    std::cout << "  Work per block: 64×64 = 4,096 elements" << std::endl;
    std::cout << std::endl;

    std::cout << "Trade-off: More blocks = more parallelism but less work each" << std::endl;
    std::cout << std::endl;

    // BLOCK COOPERATION PATTERN
    std::cout << "=== Block Cooperation Pattern ===" << std::endl;
    std::cout << R"(
__global__ void block_coop_gemm(float* A, float* B, float* C, 
                                 int M, int N, int K) {
    // Block coordinates
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    
    // Output tile coordinates
    int out_m = block_m * TILE_M;
    int out_n = block_n * TILE_N;
    
    // Each block computes its tile independently
    // No inter-block synchronization needed!
    
    // Shared memory for this block
    extern __shared__ float smem[];
    
    // Compute tile
    for (int k = 0; k < K / TILE_K; ++k) {
        // Load, compute, store
        ...
    }
    
    // Write results
    store_tile(C, accum, out_m, out_n);
}

// Launch configuration
dim3 block(128, 128);  // Threads per block
dim3 grid(M / 128, N / 128);  // Blocks
gemm_kernel<<<grid, block>>>(...);
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Thread blocks divide work in GEMM" << std::endl;
    std::cout << "2. Each block computes output tile independently" << std::endl;
    std::cout << "3. Block count depends on tile size" << std::endl;
    std::cout << "4. No inter-block sync needed for standard GEMM" << std::endl;

    return 0;
}
