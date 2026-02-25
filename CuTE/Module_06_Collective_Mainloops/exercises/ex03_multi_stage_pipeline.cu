/**
 * Exercise 03: Multi-Stage Pipeline
 * 
 * Objective: Learn to design and implement multi-stage pipelines
 *            for maximum throughput
 * 
 * Tasks:
 * 1. Understand multi-stage pipeline concepts
 * 2. Design pipeline stages
 * 3. Calculate optimal stage count
 * 4. Handle pipeline hazards
 * 
 * Key Concepts:
 * - Pipeline Stages: Multiple overlapping operations
 * - Throughput: Operations completed per unit time
 * - Latency: Time for single operation
 * - Hazards: Dependencies that stall pipeline
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 03: Multi-Stage Pipeline ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Multi-stage concept
    std::cout << "Task 1 - Multi-Stage Pipeline Concept:" << std::endl;
    std::cout << "Breaking operations into smaller stages:" << std::endl;
    std::cout << "  Stage 1: Load from global to shared" << std::endl;
    std::cout << "  Stage 2: Load from shared to registers" << std::endl;
    std::cout << "  Stage 3: Compute (MMA)" << std::endl;
    std::cout << "  Stage 4: Store results" << std::endl;
    std::cout << std::endl;

    std::cout << "Benefit: More overlap = higher throughput" << std::endl;
    std::cout << std::endl;

    // TASK 2: 2-stage vs 3-stage vs 4-stage
    std::cout << "Task 2 - Pipeline Stage Comparison:" << std::endl;
    std::cout << std::endl;

    auto show_pipeline = [](const char* name, int stages, int tiles, int stage_time) {
        int total_time = stage_time + (tiles - 1) * stage_time + stage_time;
        float throughput = (float)tiles / total_time;
        
        std::cout << name << " (" << stages << " stages):" << std::endl;
        std::cout << "  Time for " << tiles << " tiles: " << total_time << " units" << std::endl;
        std::cout << "  Throughput: " << throughput << " tiles/unit time" << std::endl;
        std::cout << std::endl;
    };

    int tiles = 8;
    show_pipeline("2-stage pipeline", 2, tiles, 10);
    show_pipeline("3-stage pipeline", 3, tiles, 10);
    show_pipeline("4-stage pipeline", 4, tiles, 10);

    // TASK 3: Pipeline timing diagram
    std::cout << "Task 3 - 4-Stage Pipeline Timing:" << std::endl;
    std::cout << std::endl;

    std::cout << "Time | Stage 1    | Stage 2    | Stage 3    | Stage 4" << std::endl;
    std::cout << "-----|------------|------------|------------|------------" << std::endl;

    for (int t = 0; t < tiles + 3; ++t) {
        printf("  %2d  | ", t);
        
        for (int s = 0; s < 4; ++s) {
            int tile = t - s;
            if (tile >= 0 && tile < tiles) {
                printf("Tile %d  ", tile);
            } else {
                printf("-         ");
            }
            printf("| ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Pipeline hazards
    std::cout << "Task 4 - Pipeline Hazards:" << std::endl;
    std::cout << "Hazards cause pipeline stalls:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Structural Hazard:" << std::endl;
    std::cout << "   Resource conflict (e.g., memory port busy)" << std::endl;
    std::cout << "   Solution: Duplicate resources or schedule carefully" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Data Hazard:" << std::endl;
    std::cout << "   Dependency on previous result" << std::endl;
    std::cout << "   Solution: Forwarding or stall" << std::endl;
    std::cout << std::endl;

    std::cout << "3. Control Hazard:" << std::endl;
    std::cout << "   Branch changes pipeline flow" << std::endl;
    std::cout << "   Solution: Branch prediction or delay" << std::endl;
    std::cout << std::endl;

    // TASK 5: Optimal stage count
    std::cout << "Task 5 - Optimal Stage Count:" << std::endl;
    std::cout << "More stages = more overlap BUT:" << std::endl;
    std::cout << "  - More complexity" << std::endl;
    std::cout << "  - More register pressure" << std::endl;
    std::cout << "  - Diminishing returns" << std::endl;
    std::cout << std::endl;

    std::cout << "Common configurations:" << std::endl;
    std::cout << "  2-stage: Simple, good for small problems" << std::endl;
    std::cout << "  3-stage: Good balance" << std::endl;
    std::cout << "  4-stage: Maximum throughput for large problems" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate pipeline efficiency
    std::cout << "=== Challenge: Pipeline Efficiency ===" << std::endl;
    std::cout << "For 16 tiles with 4-stage pipeline:" << std::endl;
    int n = 16;
    int stages = 4;
    int stage_time = 10;
    int total = stage_time * (stages + n - 1);
    int sequential = n * stages * stage_time;
    
    std::cout << "  Sequential time: " << sequential << std::endl;
    std::cout << "  Pipelined time: " << total << std::endl;
    std::cout << "  Speedup: " << (float)sequential / total << "x" << std::endl;
    std::cout << "  Efficiency: " << (float)n / (stages + n - 1) * 100 << "%" << std::endl;
    std::cout << std::endl;

    // MULTI-STAGE PIPELINE PATTERN
    std::cout << "=== Multi-Stage Pipeline Pattern ===" << std::endl;
    std::cout << R"(
__global__ void multi_stage_gemm(float* A, float* B, float* C, int K) {
    extern __shared__ float smem[];
    
    // Pipeline state
    float accum[...];
    int produce_idx = 0;
    int consume_idx = 0;
    
    // Prologue: Fill pipeline
    #pragma unroll
    for (int i = 0; i < NUM_STAGES - 1; ++i) {
        load_tile(A, B, smem, i);
        cp_async_fence();
    }
    
    // Main loop
    for (int k = NUM_STAGES - 1; k < K / TILE_K; ++k) {
        cp_async_wait(NUM_STAGES - 2);
        
        // Compute
        mma_sync(accum, smem[consume_idx]);
        
        // Load next
        load_tile(A, B, smem, k);
        cp_async_fence();
        
        consume_idx = (consume_idx + 1) % NUM_STAGES;
    }
    
    // Epilogue: Drain pipeline
    cp_async_wait(0);
    mma_sync(accum, smem[consume_idx]);
    store_results(C, accum);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Multi-stage pipelines increase overlap" << std::endl;
    std::cout << "2. More stages = higher throughput (to a point)" << std::endl;
    std::cout << "3. Handle hazards carefully" << std::endl;
    std::cout << "4. 3-4 stages is common for GEMM" << std::endl;

    return 0;
}
