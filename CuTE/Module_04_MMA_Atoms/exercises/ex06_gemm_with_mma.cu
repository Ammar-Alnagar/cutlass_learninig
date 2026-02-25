/**
 * Exercise 06: GEMM with MMA Atoms
 * 
 * Objective: Build a complete GEMM (General Matrix Multiply)
 *            using MMA atoms
 * 
 * Tasks:
 * 1. Understand GEMM structure
 * 2. Tile GEMM for MMA operations
 * 3. Implement multi-level tiling
 * 4. Handle edge cases
 * 
 * Key Concepts:
 * - GEMM: C = A × B + C (General Matrix Multiply)
 * - Tiling: Divide computation into tiles
 * - MMA Atom: Building block for GEMM
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 06: GEMM with MMA Atoms ===" << std::endl;
    std::cout << std::endl;

    // GEMM parameters
    const int M = 64;  // Rows of A and C
    const int N = 64;  // Cols of B and C
    const int K = 64;  // Cols of A, Rows of B

    // Create matrices
    float A_data[M * K];
    float B_data[K * N];
    float C_data[M * N];

    for (int i = 0; i < M * K; ++i) {
        A_data[i] = static_cast<float>(i % 16) / 10.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        B_data[i] = static_cast<float>(i % 13) / 10.0f;
    }
    for (int i = 0; i < M * N; ++i) {
        C_data[i] = 0.0f;
    }

    auto layout_A = make_layout(make_shape(Int<M>{}, Int<K>{}), GenRowMajor{});
    auto layout_B = make_layout(make_shape(Int<K>{}, Int<N>{}), GenRowMajor{});
    auto layout_C = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});

    auto A_tensor = make_tensor(make_gmem_ptr(A_data), layout_A);
    auto B_tensor = make_tensor(make_gmem_ptr(B_data), layout_B);
    auto C_tensor = make_tensor(make_gmem_ptr(C_data), layout_C);

    // TASK 1: GEMM structure
    std::cout << "Task 1 - GEMM Structure:" << std::endl;
    std::cout << "C = A × B + C" << std::endl;
    std::cout << "A: " << M << "x" << K << std::endl;
    std::cout << "B: " << K << "x" << N << std::endl;
    std::cout << "C: " << M << "x" << N << std::endl;
    std::cout << std::endl;

    // TASK 2: Tiling for MMA
    std::cout << "Task 2 - Tiling for MMA:" << std::endl;
    std::cout << "Using 16x16x16 MMA atoms:" << std::endl;
    std::cout << "  M tiles: " << M << " / 16 = " << M / 16 << std::endl;
    std::cout << "  N tiles: " << N << " / 16 = " << N / 16 << std::endl;
    std::cout << "  K tiles: " << K << " / 16 = " << K / 16 << std::endl;
    std::cout << "  Total MMA ops: " << (M/16) << " × " << (N/16) << " × " << (K/16) 
              << " = " << (M/16)*(N/16)*(K/16) << std::endl;
    std::cout << std::endl;

    // TASK 3: Visualize tiling
    std::cout << "Task 3 - GEMM Tiling Visualization:" << std::endl;
    std::cout << "Output matrix C (64x64) divided into 16x16 tiles:" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            int tile_m = i / 16;
            int tile_n = j / 16;
            int tile_id = tile_m * 4 + tile_n;
            printf("T%2d ", tile_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Simulate tiled GEMM
    std::cout << "Task 4 - Simulated Tiled GEMM:" << std::endl;
    std::cout "Processing " << (M/16) << " × " << (N/16) << " = " << (M/16)*(N/16) 
              << " output tiles" << std::endl;
    std::cout << "Each tile requires " << (K/16) << " MMA operations" << std::endl;
    std::cout << std::endl;

    // Simulate GEMM computation (simplified)
    std::cout << "Computing GEMM..." << std::endl;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A_tensor(m, k) * B_tensor(k, n);
            }
            C_tensor(m, n) = sum;
        }
    }
    std::cout << "GEMM complete!" << std::endl;
    std::cout << std::endl;

    // Show sample results
    std::cout << "Result C (top-left 8x8):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%6.3f ", C_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: GEMM hierarchy
    std::cout << "Task 5 - GEMM Hierarchy:" << std::endl;
    std::cout << "Level 1: Grid of thread blocks" << std::endl;
    std::cout << "  Each block computes one or more C tiles" << std::endl;
    std::cout << std::endl;
    std::cout << "Level 2: Thread block" << std::endl;
    std::cout << "  Threads cooperate to compute tile" << std::endl;
    std::cout << std::endl;
    std::cout << "Level 3: Warp" << std::endl;
    std::cout << "  Each warp performs MMA operations" << std::endl;
    std::cout << std::endl;
    std::cout << "Level 4: Thread" << std::endl;
    std::cout << "  Each thread handles fragment of operands" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate GEMM complexity
    std::cout << "=== Challenge: GEMM Complexity ===" << std::endl;
    std::cout << "For " << M << "x" << N << "x" << K << " GEMM:" << std::endl;
    std::cout << "  Total multiply-accumulates: " << M << " × " << N << " × " << K 
              << " = " << M*N*K << std::endl;
    std::cout << "  With 16x16x16 MMA: " << (M*N*K) / (16*16*16) << " MMA operations" << std::endl;
    std::cout << "  Each MMA does " << 16*16*16 << " multiply-accumulates" << std::endl;
    std::cout << std::endl;

    // GEMM KERNEL STRUCTURE
    std::cout << "=== GEMM Kernel Structure ===" << std::endl;
    std::cout << R"(
__global__ void gemm_kernel(float* A, float* B, float* C, 
                            int M, int N, int K) {
    // Thread block tile coordinates
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    
    // Shared memory for tiles
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = &smem[TILE_M * TILE_K];
    
    // Accumulator for this thread
    float accum[8];
    
    // Loop over K dimension
    for (int k_tile = 0; k_tile < K / TILE_K; ++k_tile) {
        // Load A and B tiles to shared memory
        load_tiles(A, B, As, Bs, tile_m, tile_n, k_tile);
        
        __syncthreads();
        
        // Perform MMA operations
        for (int m = 0; m < TILE_M / 16; ++m) {
            for (int n = 0; n < TILE_N / 16; ++n) {
                mma_sync(accum, As, Bs);
            }
        }
        
        __syncthreads();
    }
    
    // Store results to C
    store_results(C, accum, tile_m, tile_n);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. GEMM is tiled for MMA operations" << std::endl;
    std::cout << "2. Multi-level hierarchy: grid -> block -> warp -> thread" << std::endl;
    std::cout << "3. K dimension is reduced through multiple MMA steps" << std::endl;
    std::cout << "4. Shared memory enables data reuse" << std::endl;

    return 0;
}
