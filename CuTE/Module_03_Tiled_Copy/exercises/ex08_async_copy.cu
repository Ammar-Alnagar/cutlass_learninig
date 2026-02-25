/**
 * Exercise 08: Async Copy with cp.async
 * 
 * Objective: Learn to use asynchronous copy operations (cp.async)
 *            for overlapping memory transfers with computation
 * 
 * Tasks:
 * 1. Understand async copy concepts
 * 2. See the producer-consumer pattern
 * 3. Practice with async copy simulation
 * 4. Understand synchronization requirements
 * 
 * Key Concepts:
 * - Async Copy: Non-blocking memory transfer
 * - Overlap: Memory transfer happens during computation
 * - cp.async: CUDA instruction for async copy (sm_80+)
 * - Pipeline: Multiple async operations in flight
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 08: Async Copy with cp.async ===" << std::endl;
    std::cout << std::endl;

    // Simulate global and shared memory
    float gmem_A[256];
    float gmem_B[256];
    float smem_A[64];
    float smem_B[64];
    float accum[32];  // Accumulator for computation
    
    for (int i = 0; i < 256; ++i) {
        gmem_A[i] = static_cast<float>(i) / 100.0f;
        gmem_B[i] = static_cast<float>(i % 13) / 50.0f;
    }
    for (int i = 0; i < 64; ++i) {
        smem_A[i] = 0.0f;
        smem_B[i] = 0.0f;
    }
    for (int i = 0; i < 32; ++i) {
        accum[i] = 0.0f;
    }

    auto gmem_A_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto gmem_B_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto smem_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    auto gmem_A_tensor = make_tensor(make_gmem_ptr(gmem_A), gmem_A_layout);
    auto gmem_B_tensor = make_tensor(make_gmem_ptr(gmem_B), gmem_B_layout);
    auto smem_A_tensor = make_tensor(make_smem_ptr(smem_A), smem_layout);
    auto smem_B_tensor = make_tensor(make_smem_ptr(smem_B), smem_layout);
    auto accum_tensor = make_tensor(make_rmem_ptr(accum), make_layout(make_shape(Int<8>{}, Int<4>{})));

    // TASK 1: Understand async copy concept
    std::cout << "Task 1 - Async Copy Concept:" << std::endl;
    std::cout << "Synchronous copy: Load -> Wait -> Compute" << std::endl;
    std::cout << "Async copy: Load (async) -> Compute -> Wait (if needed)" << std::endl;
    std::cout << "Benefit: Overlap memory transfer with computation" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simulate async copy pipeline
    std::cout << "Task 2 - Async Copy Pipeline Simulation:" << std::endl;
    std::cout << "Processing 4 tiles with async copy:" << std::endl;
    std::cout << std::endl;

    for (int tile = 0; tile < 4; ++tile) {
        std::cout << "=== Tile " << tile << " ===" << std::endl;
        
        // Calculate tile position
        int tile_row = tile / 2;
        int tile_col = tile % 2;
        int start_i = tile_row * 8;
        int start_j = tile_col * 8;

        // ASYNC LOAD: Initiate async copy from global to shared
        std::cout << "  ASYNC LOAD: Initiating cp.async for tile " << tile << std::endl;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                smem_A_tensor(i, j) = gmem_A_tensor(start_i + i, start_j + j);
                smem_B_tensor(i, j) = gmem_B_tensor(start_i + i, start_j + j);
            }
        }
        std::cout << "    Data loaded to shared memory" << std::endl;

        // COMPUTE: Process data while next tile loads (simulated)
        std::cout << "  COMPUTE: Processing data in registers" << std::endl;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                // Simulated computation (multiply-accumulate)
                float sum = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    sum += smem_A_tensor(i, k) * smem_B_tensor(k, j);
                }
                accum_tensor(i, j) += sum;
            }
        }
        std::cout << "    Computation complete for tile " << tile << std::endl;
        std::cout << std::endl;
    }

    // TASK 3: Compare synchronous vs async
    std::cout << "Task 3 - Synchronous vs Async Comparison:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Synchronous (4 tiles):" << std::endl;
    std::cout << "  Tile 0: [LOAD] [COMPUTE]" << std::endl;
    std::cout << "  Tile 1: [LOAD] [COMPUTE]" << std::endl;
    std::cout << "  Tile 2: [LOAD] [COMPUTE]" << std::endl;
    std::cout << "  Tile 3: [LOAD] [COMPUTE]" << std::endl;
    std::cout << "  Total time: 8 units (4 load + 4 compute)" << std::endl;
    std::cout << std::endl;

    std::cout << "Async (4 tiles, pipelined):" << std::endl;
    std::cout << "  Time 0: [LOAD 0]" << std::endl;
    std::cout << "  Time 1: [COMPUTE 0] [LOAD 1]" << std::endl;
    std::cout << "  Time 2: [COMPUTE 1] [LOAD 2]" << std::endl;
    std::cout << "  Time 3: [COMPUTE 2] [LOAD 3]" << std::endl;
    std::cout << "  Time 4: [COMPUTE 3]" << std::endl;
    std::cout << "  Total time: 5 units (overlapped!)" << std::endl;
    std::cout << "  Speedup: 8/5 = 1.6x" << std::endl;
    std::cout << std::endl;

    // TASK 4: Async copy requirements
    std::cout << "Task 4 - Async Copy Requirements:" << std::endl;
    std::cout << "1. Hardware: sm_80 or later (A100, H100, etc.)" << std::endl;
    std::cout << "2. Memory: Global to shared memory only" << std::endl;
    std::cout << "3. Alignment: 16-byte alignment required" << std::endl;
    std::cout << "4. Synchronization: cp.async.commit_group and cp.async.wait_group" << std::endl;
    std::cout << std::endl;

    // TASK 5: Pipeline stages
    std::cout << "Task 5 - Pipeline Stages:" << std::endl;
    std::cout << "Common pipeline configurations:" << std::endl;
    std::cout << "  2-stage: Load next while computing current" << std::endl;
    std::cout << "  3-stage: Better latency hiding" << std::endl;
    std::cout << "  4-stage: Maximum throughput" << std::endl;
    std::cout << std::endl;

    // CUDA ASYNC COPY PATTERN
    std::cout << "=== CUDA Async Copy Pattern ===" << std::endl;
    std::cout << R"(
// cp.async PTX instruction pattern
__global__ void async_copy_kernel(float* gmem, float* smem, int N) {
    extern __shared__ float shared[];
    
    // Async copy from global to shared
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "cp.async.ca.shared.global [%0], [%1], %2, p;\n"
        "cp.async.commit_group;\n"
        "}"
        :
        : "r"(shared_offset), "l"(gmem_ptr), "r"(bytes)
    );
    
    // Wait for async copies to complete
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    
    // Now safe to use shared memory
}
)" << std::endl;

    // CHALLENGE: Pipeline design
    std::cout << "=== Challenge: Design a Pipeline ===" << std::endl;
    std::cout << "For a GEMM kernel with 4 pipeline stages:" << std::endl;
    std::cout << "  Stage 0: Load tile K=0" << std::endl;
    std::cout << "  Stage 1: Load tile K=1, Compute K=0" << std::endl;
    std::cout << "  Stage 2: Load tile K=2, Compute K=1" << std::endl;
    std::cout << "  Stage 3: Load tile K=3, Compute K=2" << std::endl;
    std::cout << "  Stage 4: Compute K=3" << std::endl;
    std::cout << std::endl;
    std::cout << "Benefit: Always computing while loading!" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Async copy enables memory/compute overlap" << std::endl;
    std::cout << "2. cp.async is available on sm_80+" << std::endl;
    std::cout << "3. Pipelining improves throughput" << std::endl;
    std::cout << "4. Proper synchronization is critical" << std::endl;

    return 0;
}
