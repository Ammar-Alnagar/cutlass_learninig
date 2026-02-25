/**
 * Exercise 06: Shared Memory Layouts for GEMM
 * 
 * Objective: Design optimal shared memory layouts for GEMM kernels
 * 
 * Tasks:
 * 1. Understand GEMM shared memory requirements
 * 2. Design layouts for operands A and B
 * 3. Avoid bank conflicts in both load and compute phases
 * 4. Practice with different tile sizes
 * 
 * Key Concepts:
 * - GEMM Tiles: A and B matrices loaded to shared memory
 * - Access Patterns: Row access for one, column for other
 * - Conflict-Free: Both phases must be optimized
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 06: Shared Memory Layouts for GEMM ===" << std::endl;
    std::cout << std::endl;

    // GEMM setup
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 64;

    // TASK 1: GEMM shared memory requirements
    std::cout << "Task 1 - GEMM Shared Memory Requirements:" << std::endl;
    std::cout << "For GEMM C = A × B:" << std::endl;
    std::cout << "  Tile A: " << TILE_M << "x" << TILE_K << " elements" << std::endl;
    std::cout << "  Tile B: " << TILE_K << "x" << TILE_N << " elements" << std::endl;
    std::cout << std::endl;

    std::cout << "Memory requirements (FP16, 2 bytes):" << std::endl;
    std::cout << "  Tile A: " << TILE_M * TILE_K * 2 << " bytes" << std::endl;
    std::cout << "  Tile B: " << TILE_K * TILE_N * 2 << " bytes" << std::endl;
    std::cout << "  Total: " << (TILE_M * TILE_K + TILE_K * TILE_N) * 2 << " bytes" << std::endl;
    std::cout << std::endl;

    // TASK 2: Access pattern analysis
    std::cout << "Task 2 - Access Pattern Analysis:" << std::endl;
    std::cout << std::endl;

    std::cout << "During load phase (global -> shared):" << std::endl;
    std::cout << "  Tile A: Row-major access (coalesced)" << std::endl;
    std::cout << "  Tile B: Row-major access (coalesced)" << std::endl;
    std::cout << std::endl;

    std::cout << "During compute phase (MMA):" << std::endl;
    std::cout << "  Tile A: Each warp accesses rows" << std::endl;
    std::cout << "  Tile B: Each warp accesses columns" << std::endl;
    std::cout << "  -> B needs conflict-free column access!" << std::endl;
    std::cout << std::endl;

    // TASK 3: Layout design for Tile A
    std::cout << "Task 3 - Tile A Layout:" << std::endl;
    std::cout << "Row-major is fine (row access in both phases)" << std::endl;
    
    auto smem_A_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), GenRowMajor{});
    std::cout << "Layout A: " << smem_A_layout << std::endl;
    std::cout << std::endl;

    // TASK 4: Layout design for Tile B (with padding)
    std::cout << "Task 4 - Tile B Layout (Padded):" << std::endl;
    std::cout << "Column access needs padding or swizzling" << std::endl;
    
    // Padded layout for B
    auto smem_B_layout_padded = make_layout(
        make_shape(Int<TILE_K>{}, Int<TILE_N>{}),
        make_stride(Int<TILE_N + 1>{}, Int<1>{})
    );
    std::cout << "Layout B (padded): " << smem_B_layout_padded << std::endl;
    std::cout << "Stride: " << smem_B_layout_padded.stride() << std::endl;
    std::cout << std::endl;

    // Analyze column access for B
    std::cout << "Column access analysis for B (first 8 threads):" << std::endl;
    for (int t = 0; t < 8; ++t) {
        int addr = t * (TILE_N + 1);  // Padded stride
        int bank = addr % 32;
        std::cout << "  Thread " << t << " -> Bank " << bank << std::endl;
    }
    std::cout << "  Result: Different banks (no conflict!)" << std::endl;
    std::cout << std::endl;

    // TASK 5: Alternative - swizzled layout for B
    std::cout << "Task 5 - Tile B Layout (Swizzled):" << std::endl;
    std::cout << "Instead of padding, use XOR swizzling" << std::endl;
    std::cout << "No memory overhead, but more complex addressing" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate optimal layout
    std::cout << "=== Challenge: Optimal Layout Design ===" << std::endl;
    std::cout << "For 128x128x128 GEMM tiles:" << std::endl;
    std::cout << std::endl;
    
    int m = 128, n = 128, k = 128;
    
    std::cout << "Option 1 - Padding:" << std::endl;
    std::cout << "  Tile A: " << m << "x" << k << " (no padding needed)" << std::endl;
    std::cout << "  Tile B: " << k << "x" << (n + 1) << " (padded)" << std::endl;
    std::cout << "  Overhead: " << k << " elements" << std::endl;
    std::cout << std::endl;

    std::cout << "Option 2 - Swizzling:" << std::endl;
    std::cout << "  Tile A: " << m << "x" << k << " (row-major)" << std::endl;
    std::cout << "  Tile B: " << k << "x" << n << " (swizzled)" << std::endl;
    std::cout << "  Overhead: 0 elements" << std::endl;
    std::cout << std::endl;

    // GEMM SHARED MEMORY PATTERN
    std::cout << "=== GEMM Shared Memory Pattern ===" << std::endl;
    std::cout << R"(
__global__ void gemm_shared(float* A, float* B, float* C, 
                            int M, int N, int K) {
    // Shared memory with padding for B
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N + 1];  // Padded!
    
    // Load tiles from global to shared
    for (int k_tile = 0; k_tile < K / TILE_K; ++k_tile) {
        // Coalesced load for A (row access)
        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_K; j += blockDim.x) {
                As[i][j] = A[...];
            }
        }
        
        // Coalesced load for B (row access)
        for (int i = threadIdx.y; i < TILE_K; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                Bs[i][j] = B[...];
            }
        }
        
        __syncthreads();
        
        // Compute with MMA (B accessed column-wise)
        // Padded layout prevents bank conflicts
        mma_kernel(As, Bs, accum);
        
        __syncthreads();
    }
    
    // Store results
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. GEMM needs two shared memory tiles (A and B)" << std::endl;
    std::cout << "2. Tile B needs padding/swizzling for column access" << std::endl;
    std::cout << "3. Both load and compute phases must be optimized" << std::endl;
    std::cout << "4. Padding adds memory, swizzling adds complexity" << std::endl;

    return 0;
}
