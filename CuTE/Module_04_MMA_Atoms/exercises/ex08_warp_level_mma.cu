/**
 * Exercise 08: Warp-Level Matrix Multiply
 * 
 * Objective: Understand warp-level matrix multiply operations
 *            and how to orchestrate warps for GEMM
 * 
 * Tasks:
 * 1. Learn warp-level primitives
 * 2. Understand warp cooperation
 * 3. Practice with warp assignment
 * 4. Implement warp-level GEMM
 * 
 * Key Concepts:
 * - Warp: 32 threads executing together
 * - Warp-Level MMA: Entire warp performs one MMA
 * - Warp Assignment: Warps handle different tiles
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 08: Warp-Level Matrix Multiply ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Warp basics
    std::cout << "Task 1 - Warp Basics:" << std::endl;
    std::cout << "A warp consists of 32 threads" << std::endl;
    std::cout << "Threads in a warp execute in lockstep (SIMT)" << std::endl;
    std::cout << "Warps are scheduled independently" << std::endl;
    std::cout << std::endl;

    std::cout << "Thread block to warp mapping:" << std::endl;
    std::cout << "  128-thread block: 128 / 32 = 4 warps" << std::endl;
    std::cout << "  256-thread block: 256 / 32 = 8 warps" << std::endl;
    std::cout << "  512-thread block: 512 / 32 = 16 warps" << std::endl;
    std::cout << std::endl;

    // TASK 2: Warp-level MMA operation
    std::cout << "Task 2 - Warp-Level MMA:" << std::endl;
    std::cout << "For 16x16x16 MMA with 32 threads:" << std::endl;
    std::cout << "  Each warp produces 16x16 = 256 output elements" << std::endl;
    std::cout << "  Each thread computes 256 / 32 = 8 elements" << std::endl;
    std::cout << std::endl;

    std::cout << "Warp MMA instruction:" << std::endl;
    std::cout << "  mma.sync.aligned.m16n16k16..." << std::endl;
    std::cout << "  - All 32 threads participate" << std::endl;
    std::cout << "  - Each thread provides operands" << std::endl;
    std::cout << "  - Results distributed to threads" << std::endl;
    std::cout << std::endl;

    // TASK 3: Multi-warp GEMM
    std::cout << "Task 3 - Multi-Warp GEMM:" << std::endl;
    std::cout << "For a 64x64x64 GEMM with 16x16x16 MMA:" << std::endl;
    std::cout << "  Output tiles: (64/16) × (64/16) = 4 × 4 = 16 tiles" << std::endl;
    std::cout << "  With 4 warps per block:" << std::endl;
    std::cout << "    Warps needed: 16 tiles / 4 warps = 4 blocks" << std::endl;
    std::cout << "    Each warp handles 4 tiles" << std::endl;
    std::cout << std::endl;

    // Visualize warp assignment
    std::cout << "Warp assignment for 4x4 output tiles with 4 warps:" << std::endl;
    std::cout << std::endl;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int warp_id = (i / 2) * 2 + (j / 2);
            printf("W%d ", warp_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Warp synchronization
    std::cout << "Task 4 - Warp Synchronization:" << std::endl;
    std::cout << "Within a warp: implicit synchronization (lockstep)" << std::endl;
    std::cout << "Between warps: __syncthreads() required" << std::endl;
    std::cout << std::endl;
    std::cout << "Warp-level primitives:" << std::endl;
    std::cout << "  __shfl_sync(value, src_lane): Share value within warp" << std::endl;
    std::cout << "  __shfl_down_sync(value, delta): Share with offset" << std::endl;
    std::cout << "  __shfl_up_sync(value, delta): Share backwards" << std::endl;
    std::cout << std::endl;

    // TASK 5: Warp-level GEMM structure
    std::cout << "Task 5 - Warp-Level GEMM Structure:" << std::endl;
    std::cout << R"(
For a thread block with 4 warps:

Warp 0: Handles tiles (0,0), (0,1), (1,0), (1,1)
Warp 1: Handles tiles (0,2), (0,3), (1,2), (1,3)
Warp 2: Handles tiles (2,0), (2,1), (3,0), (3,1)
Warp 3: Handles tiles (2,2), (2,3), (3,2), (3,3)

Each warp:
1. Loads its assigned tiles from global memory
2. Performs MMA operations for K dimension
3. Stores results to output matrix
)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate warp requirements
    std::cout << "=== Challenge: Warp Requirements ===" << std::endl;
    std::cout << "For a 128x128x128 GEMM with 16x16x16 MMA:" << std::endl;
    
    int M = 128, N = 128, K = 128;
    int tile_M = 16, tile_N = 16, tile_K = 16;
    int tiles_M = M / tile_M;
    int tiles_N = N / tile_N;
    int total_tiles = tiles_M * tiles_N;
    int warps_per_block = 4;
    
    std::cout << "  Output tiles: " << tiles_M << " × " << tiles_N << " = " << total_tiles << std::endl;
    std::cout << "  Warps per block: " << warps_per_block << std::endl;
    std::cout << "  Tiles per warp: " << total_tiles / warps_per_block << std::endl;
    std::cout << "  Thread blocks needed: " << (total_tiles + warps_per_block - 1) / warps_per_block << std::endl;
    std::cout << std::endl;

    // WARP-LEVEL GEMM PATTERN
    std::cout << "=== Warp-Level GEMM Pattern ===" << std::endl;
    std::cout << R"(
__global__ void warp_level_gemm(float* A, float* B, float* C, 
                                 int M, int N, int K) {
    // Warp ID within block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Determine which tiles this warp handles
    int tile_m = (warp_id / warps_n) * TILE_M;
    int tile_n = (warp_id % warps_n) * TILE_N;
    
    // Accumulator for this warp
    float accum[8];  // 8 elements per thread
    
    // Loop over K dimension
    for (int k = 0; k < K; k += TILE_K) {
        // Load operands cooperatively within warp
        float frag_a[...], frag_b[...];
        load_operands(A, B, frag_a, frag_b, tile_m, tile_n, k);
        
        // Perform warp-level MMA
        mma_sync(accum, frag_a, frag_b);
    }
    
    // Store results
    store_results(C, accum, tile_m, tile_n);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Warps are the unit of MMA execution" << std::endl;
    std::cout << "2. 32 threads cooperate for warp-level MMA" << std::endl;
    std::cout << "3. Multiple warps handle different tiles" << std::endl;
    std::cout << "4. Warp primitives enable intra-warp communication" << std::endl;

    return 0;
}
