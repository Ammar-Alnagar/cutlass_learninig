/**
 * Exercise 01: Producer-Consumer Pipeline
 * 
 * Objective: Understand the producer-consumer pipeline pattern
 *            for overlapping memory operations with computation
 * 
 * Tasks:
 * 1. Learn producer-consumer concepts
 * 2. Understand pipeline stages
 * 3. Simulate pipeline execution
 * 4. Calculate throughput improvement
 * 
 * Key Concepts:
 * - Producer: Loads data from global to shared memory
 * - Consumer: Performs computation on loaded data
 * - Pipeline: Overlap load and compute phases
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: Producer-Consumer Pipeline ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Pipeline concept
    std::cout << "Task 1 - Pipeline Concept:" << std::endl;
    std::cout << "Producer-Consumer pipeline separates concerns:" << std::endl;
    std::cout << "  Producer: Memory operations (load/store)" << std::endl;
    std::cout << "  Consumer: Compute operations (MMA)" << std::endl;
    std::cout << "  Pipeline: Overlap producer and consumer" << std::endl;
    std::cout << std::endl;

    std::cout << "Without pipeline:" << std::endl;
    std::cout << "  [Load Tile 0] [Compute Tile 0] [Load Tile 1] [Compute Tile 1] ..." << std::endl;
    std::cout << "  Sequential execution" << std::endl;
    std::cout << std::endl;

    std::cout << "With pipeline:" << std::endl;
    std::cout << "  [Load 0] [Compute 0]" << std::endl;
    std::cout << "         [Load 1] [Compute 1]" << std::endl;
    std::cout << "                 [Load 2] [Compute 2]" << std::endl;
    std::cout << "  Overlapped execution!" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simulate pipeline execution
    std::cout << "Task 2 - Pipeline Simulation:" << std::endl;
    std::cout << "Simulating 4-tile GEMM with pipeline:" << std::endl;
    std::cout << std::endl;

    int num_tiles = 4;
    int load_time = 10;  // Arbitrary time units
    int compute_time = 10;

    std::cout << "Time | Producer     | Consumer" << std::endl;
    std::cout << "-----|--------------|--------------" << std::endl;

    for (int t = 0; t < (num_tiles + 1) * compute_time; t += compute_time) {
        std::cout << "  " << t << "  | ";
        
        // Producer
        int load_tile = t / compute_time;
        if (load_tile < num_tiles) {
            std::cout << "Load Tile " << load_tile;
        } else {
            std::cout << "-";
        }
        
        std::cout << " | ";
        
        // Consumer
        int compute_tile = (t / compute_time) - 1;
        if (compute_tile >= 0 && compute_tile < num_tiles) {
            std::cout << "Compute Tile " << compute_tile;
        } else {
            std::cout << "-";
        }
        
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Calculate throughput
    std::cout << "Task 3 - Throughput Calculation:" << std::endl;
    
    int sequential_time = num_tiles * (load_time + compute_time);
    int pipelined_time = load_time + num_tiles * compute_time;  // First load + all computes
    float speedup = (float)sequential_time / pipelined_time;

    std::cout << "Sequential execution: " << sequential_time << " time units" << std::endl;
    std::cout << "Pipelined execution: " << pipelined_time << " time units" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    std::cout << std::endl;

    // TASK 4: Pipeline stages
    std::cout << "Task 4 - Pipeline Stages:" << std::endl;
    std::cout << "Complete GEMM pipeline:" << std::endl;
    std::cout << "  Stage 1: Global -> Shared (Producer)" << std::endl;
    std::cout << "  Stage 2: Shared -> Registers (Consumer Load)" << std::endl;
    std::cout << "  Stage 3: MMA Compute (Consumer Compute)" << std::endl;
    std::cout << "  Stage 4: Registers -> Global (Store)" << std::endl;
    std::cout << std::endl;

    std::cout << "Full pipeline timeline:" << std::endl;
    std::cout << "  Time 0:  Load Tile 0 (G->S)" << std::endl;
    std::cout << "  Time 1:  Load Tile 1 (G->S), Load Tile 0 (S->R)" << std::endl;
    std::cout << "  Time 2:  Load Tile 2 (G->S), Load Tile 1 (S->R), Compute Tile 0" << std::endl;
    std::cout << "  Time 3:  Load Tile 3 (G->S), Load Tile 2 (S->R), Compute Tile 1" << std::endl;
    std::cout << "  Time 4:  Load Tile 3 (S->R), Compute Tile 2" << std::endl;
    std::cout << "  Time 5:  Compute Tile 3" << std::endl;
    std::cout << "  Time 6:  Store Results (R->G)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate for different configurations
    std::cout << "=== Challenge: Pipeline Configurations ===" << std::endl;
    std::cout << "For 8 tiles with load=10, compute=10:" << std::endl;
    int n = 8;
    int seq = n * 20;
    int pipe = 10 + n * 10;
    std::cout << "  Sequential: " << seq << " time units" << std::endl;
    std::cout << "  Pipelined: " << pipe << " time units" << std::endl;
    std::cout << "  Speedup: " << (float)seq / pipe << "x" << std::endl;
    std::cout << std::endl;

    // PIPELINE PATTERN
    std::cout << "=== Pipeline Pattern ===" << std::endl;
    std::cout << R"(
__global__ void pipeline_gemm(float* A, float* B, float* C, int K) {
    extern __shared__ float smem[];
    
    // Pipeline registers
    float accum[...];
    
    // Prologue: Load first tile
    load_tile(A, B, smem, 0);
    cp_async_fence();
    
    for (int k = 1; k < K / TILE_K; ++k) {
        // Wait for previous load
        cp_async_wait();
        
        // Consumer: Compute previous tile
        mma_sync(accum, smem);
        
        // Producer: Load next tile
        load_tile(A, B, smem, k);
        cp_async_fence();
    }
    
    // Epilogue: Final compute and store
    cp_async_wait();
    mma_sync(accum, smem);
    store_results(C, accum);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Producer-consumer separates load and compute" << std::endl;
    std::cout << "2. Pipelining overlaps operations" << std::endl;
    std::cout << "3. Throughput improves ~2x for 2-stage pipeline" << std::endl;
    std::cout << "4. More stages = more overlap potential" << std::endl;

    return 0;
}
