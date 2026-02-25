/**
 * Exercise 06: Double Buffering
 * 
 * Objective: Learn double buffering technique for hiding
 *            memory latency
 * 
 * Tasks:
 * 1. Understand double buffering concept
 * 2. Implement double buffered pipeline
 * 3. Analyze latency hiding
 * 4. Compare with single buffering
 * 
 * Key Concepts:
 * - Double Buffering: Two buffers for ping-pong operation
 * - Latency Hiding: Overlap load with compute
 * - Ping-Pong: Alternate between buffers
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 06: Double Buffering ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Double buffering concept
    std::cout << "Task 1 - Double Buffering Concept:" << std::endl;
    std::cout << "Use two buffers to overlap operations:" << std::endl;
    std::cout << "  Buffer 0: Being computed" << std::endl;
    std::cout << "  Buffer 1: Being loaded" << std::endl;
    std::cout << std::endl;

    std::cout << "Timeline:" << std::endl;
    std::cout << "  Time 0: Load to Buffer 0" << std::endl;
    std::cout << "  Time 1: Compute Buffer 0, Load to Buffer 1" << std::endl;
    std::cout << "  Time 2: Compute Buffer 1, Load to Buffer 0" << std::endl;
    std::cout << "  Time 3: Compute Buffer 0, Load to Buffer 1" << std::endl;
    std::cout << std::endl;

    // TASK 2: Compare single vs double buffering
    std::cout << "Task 2 - Single vs Double Buffering:" << std::endl;
    std::cout << std::endl;

    std::cout << "Single Buffering:" << std::endl;
    std::cout << "  [Load 0] [Compute 0] [Load 1] [Compute 1] [Load 2] [Compute 2]" << std::endl;
    std::cout << "  Total time: 6 units for 3 tiles" << std::endl;
    std::cout << std::endl;

    std::cout << "Double Buffering:" << std::endl;
    std::cout << "  [Load 0] [Compute 0]" << std::endl;
    std::cout << "           [Load 1] [Compute 1]" << std::endl;
    std::cout << "                    [Load 2] [Compute 2]" << std::endl;
    std::cout << "  Total time: 4 units for 3 tiles" << std::endl;
    std::cout << "  Speedup: 6/4 = 1.5x" << std::endl;
    std::cout << std::endl;

    // TASK 3: Simulate double buffering
    std::cout << "Task 3 - Double Buffering Simulation:" << std::endl;
    
    float buffer[2][64];  // Two buffers
    float result[64];
    
    int num_tiles = 4;
    int load_time = 10;
    int compute_time = 10;

    std::cout << "Simulating " << num_tiles << " tiles with double buffering:" << std::endl;
    std::cout << std::endl;

    std::cout << "Time | Buffer 0          | Buffer 1" << std::endl;
    std::cout << "-----|-------------------|-------------------" << std::endl;

    for (int t = 0; t < num_tiles + 1; ++t) {
        printf("  %2d  | ", t);
        
        // Buffer 0
        if (t == 0) {
            printf("Load Tile 0      ");
        } else if (t < num_tiles) {
            printf("Compute Tile %d  ", t - 1);
        } else {
            printf("Compute Tile %d  ", num_tiles - 1);
        }
        
        printf("| ");
        
        // Buffer 1
        if (t > 0 && t < num_tiles) {
            printf("Load Tile %d     ", t);
        } else {
            printf("-                ");
        }
        
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Memory requirements
    std::cout << "Task 4 - Memory Requirements:" << std::endl;
    std::cout << "Double buffering requires 2x shared memory:" << std::endl;
    std::cout << std::endl;

    int tile_size = 64 * 64;  // elements
    int element_size = 2;  // bytes (FP16)
    int single_buffer = tile_size * element_size;
    int double_buffer = 2 * single_buffer;

    std::cout << "Single buffer: " << single_buffer / 1024 << " KB" << std::endl;
    std::cout << "Double buffer: " << double_buffer / 1024 << " KB" << std::endl;
    std::cout << "Overhead: 100% (but worth it for performance)" << std::endl;
    std::cout << std::endl;

    // TASK 5: Multi-buffering extension
    std::cout << "Task 5 - Multi-Buffering Extension:" << std::endl;
    std::cout << "Can extend to 3 or 4 buffers:" << std::endl;
    std::cout << std::endl;

    std::cout << "Triple Buffering:" << std::endl;
    std::cout << "  Even more overlap potential" << std::endl;
    std::cout << "  3x memory requirement" << std::endl;
    std::cout << "  Useful when load time > compute time" << std::endl;
    std::cout << std::endl;

    std::cout << "Quad Buffering:" << std::endl;
    std::cout << "  Maximum overlap" << std::endl;
    std::cout << "  4x memory requirement" << std::endl;
    std::cout << "  For memory-bound operations" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate speedup
    std::cout << "=== Challenge: Speedup Calculation ===" << std::endl;
    std::cout << "For 8 tiles, load=10, compute=10:" << std::endl;
    int n = 8;
    int l = 10, c = 10;
    
    int single_time = n * (l + c);
    int double_time = l + n * c;  // First load + all computes
    
    std::cout << "  Single buffering: " << single_time << " time units" << std::endl;
    std::cout << "  Double buffering: " << double_time << " time units" << std::endl;
    std::cout << "  Speedup: " << (float)single_time / double_time << "x" << std::endl;
    std::cout << std::endl;

    // DOUBLE BUFFERING PATTERN
    std::cout << "=== Double Buffering Pattern ===" << std::endl;
    std::cout << R"(
__global__ void double_buffer_gemm(float* A, float* B, float* C, int K) {
    // Two shared memory buffers
    __shared__ float smem[2][TILE_M][TILE_K];
    
    int buf_idx = 0;
    
    // Prologue: Load first tile
    load_tile(A, B, smem[0], 0);
    
    for (int k = 1; k < K / TILE_K; ++k) {
        // Toggle buffer index
        buf_idx = 1 - buf_idx;
        int prev_idx = 1 - buf_idx;
        
        // Async load next tile
        load_tile_async(A, B, smem[buf_idx], k);
        
        // Compute previous tile (already loaded)
        mma_sync(accum, smem[prev_idx]);
    }
    
    // Epilogue: Compute last tile
    mma_sync(accum, smem[1 - buf_idx]);
    
    store_results(C, accum);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Double buffering overlaps load and compute" << std::endl;
    std::cout << "2. Requires 2x shared memory" << std::endl;
    std::cout << "3. ~1.5-2x speedup typical" << std::endl;
    std::cout << "4. Can extend to multi-buffering" << std::endl;

    return 0;
}
