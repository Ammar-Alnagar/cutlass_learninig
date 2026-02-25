/**
 * Exercise 05: Mainloop Scheduling
 * 
 * Objective: Learn to schedule the mainloop of GEMM kernels
 *            for optimal performance
 * 
 * Tasks:
 * 1. Understand mainloop structure
 * 2. Learn scheduling strategies
 * 3. Handle K-dimension iteration
 * 4. Optimize for occupancy
 * 
 * Key Concepts:
 * - Mainloop: The K-dimension iteration loop
 * - Scheduling: Order of operations
 * - Occupancy: Active warps per SM
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 05: Mainloop Scheduling ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Mainloop structure
    std::cout << "Task 1 - Mainloop Structure:" << std::endl;
    std::cout << "GEMM mainloop iterates over K dimension:" << std::endl;
    std::cout << std::endl;

    std::cout << "Basic structure:" << std::endl;
    std::cout << "  for (k_tile = 0; k_tile < K / TILE_K; ++k_tile) {" << std::endl;
    std::cout << "    1. Load A tile from global to shared" << std::endl;
    std::cout << "    2. Load B tile from global to shared" << std::endl;
    std::cout << "    3. Synchronize threads" << std::endl;
    std::cout << "    4. Perform MMA operations" << std::endl;
    std::cout << "    5. Accumulate results" << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << std::endl;

    // TASK 2: Scheduling strategies
    std::cout << "Task 2 - Scheduling Strategies:" << std::endl;
    std::cout << std::endl;

    std::cout << "Strategy 1: Sequential" << std::endl;
    std::cout << "  Load A, Load B, Sync, Compute, Repeat" << std::endl;
    std::cout << "  Simple but low occupancy" << std::endl;
    std::cout << std::endl;

    std::cout << "Strategy 2: Pipelined" << std::endl;
    std::cout << "  Load next while computing current" << std::endl;
    std::cout << "  Better occupancy, more complex" << std::endl;
    std::cout << std::endl;

    std::cout << "Strategy 3: Multi-buffered" << std::endl;
    std::cout << "  Multiple shared memory buffers" << std::endl;
    std::cout << "  Maximum overlap, highest complexity" << std::endl;
    std::cout << std::endl;

    // TASK 3: K-dimension iteration
    std::cout << "Task 3 - K-Dimension Iteration:" << std::endl;
    std::cout << "For K=1024 with TILE_K=64:" << std::endl;
    std::cout << "  Number of iterations: 1024 / 64 = 16" << std::endl;
    std::cout << std::endl;

    std::cout << "Iteration breakdown:" << std::endl;
    for (int k = 0; k < 16; k += 4) {
        std::cout << "  Iterations " << k << "-" << (k + 3) << ": K = " 
                  << (k * 64) << "-" << ((k + 4) * 64 - 1) << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Occupancy considerations
    std::cout << "Task 4 - Occupancy Considerations:" << std::endl;
    std::cout << "Factors affecting occupancy:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Register usage:" << std::endl;
    std::cout << "   More registers = fewer active warps" << std::endl;
    std::cout << "   Target: < 64 registers per thread" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Shared memory:" << std::endl;
    std::cout << "   More smem = fewer active blocks" << std::endl;
    std::cout << "   A100: 192 KB per SM" << std::endl;
    std::cout << std::endl;

    std::cout << "3. Thread count:" << std::endl;
    std::cout << "   More threads = better latency hiding" << std::endl;
    std::cout << "   Target: > 20 warps per SM" << std::endl;
    std::cout << std::endl;

    // TASK 5: Optimal scheduling
    std::cout << "Task 5 - Optimal Scheduling:" << std::endl;
    std::cout << "For best performance:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Use async copy (cp.async)" << std::endl;
    std::cout << "   Overlap load and compute" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Pipeline multiple tiles" << std::endl;
    std::cout << "   2-4 stage pipeline common" << std::endl;
    std::cout << std::endl;

    std::cout << "3. Unroll K loop" << std::endl;
    std::cout << "   Reduce loop overhead" << std::endl;
    std::cout << "   Enable better scheduling" << std::endl;
    std::cout << std::endl;

    std::cout << "4. Balance register usage" << std::endl;
    std::cout << "   Enough for performance" << std::endl;
    std::cout << "   Not so many that occupancy suffers" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate optimal configuration
    std::cout << "=== Challenge: Optimal Configuration ===" << std::endl;
    std::cout << "For A100 (192 KB smem, 255K registers per SM):" << std::endl;
    std::cout << std::endl;

    std::cout << "Configuration: 128 threads/block, 64 registers/thread" << std::endl;
    int threads_per_block = 128;
    int regs_per_thread = 64;
    int smem_per_block = 64 * 1024;  // 64 KB
    
    int max_blocks_by_smem = (192 * 1024) / smem_per_block;
    int max_blocks_by_regs = (255 * 1024) / (threads_per_block * regs_per_thread);
    int max_blocks_by_warps = 64 / ((threads_per_block + 31) / 32);
    
    int active_blocks = std::min({max_blocks_by_smem, max_blocks_by_regs, max_blocks_by_warps});
    int occupancy = active_blocks * threads_per_block / 1024;  // Warps per SM
    
    std::cout << "  Max blocks by smem: " << max_blocks_by_smem << std::endl;
    std::cout << "  Max blocks by regs: " << max_blocks_by_regs << std::endl;
    std::cout << "  Max blocks by warps: " << max_blocks_by_warps << std::endl;
    std::cout << "  Active blocks: " << active_blocks << std::endl;
    std::cout << "  Occupancy: " << occupancy << " warps/SM" << std::endl;
    std::cout << std::endl;

    // MAINLOOP PATTERN
    std::cout << "=== Mainloop Pattern ===" << std::endl;
    std::cout << R"(
__global__ void scheduled_gemm(float* A, float* B, float* C, 
                                int M, int N, int K) {
    extern __shared__ float smem[];
    float accum[...];
    
    // Initialize accumulator
    zero_fill(accum);
    
    // Mainloop over K dimension
    #pragma unroll 4
    for (int k = 0; k < K / TILE_K; ++k) {
        // Async load
        cp_async(smem, A, k);
        cp_async(smem, B, k);
        cp_async_fence();
        
        // Wait for load
        cp_async_wait();
        
        // MMA compute
        mma_sync(accum, smem);
    }
    
    // Store results
    store(C, accum);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Mainloop iterates over K dimension" << std::endl;
    std::cout << "2. Pipelining improves throughput" << std::endl;
    std::cout << "3. Balance registers and occupancy" << std::endl;
    std::cout << "4. Async copy enables overlap" << std::endl;

    return 0;
}
