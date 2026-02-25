/**
 * Exercise 07: Matrix Transpose Copy
 * 
 * Objective: Learn to implement matrix transpose using tiled copy
 *            with optimized memory access patterns
 * 
 * Tasks:
 * 1. Understand transpose as a copy operation
 * 2. Implement tiled transpose
 * 3. Optimize for coalesced access
 * 4. Handle edge cases
 * 
 * Key Concepts:
 * - Transpose: Swap rows and columns (A^T[i,j] = A[j,i])
 * - Tiled Transpose: Process matrix in tiles
 * - Memory Efficiency: Coalesced read and write
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 07: Matrix Transpose Copy ===" << std::endl;
    std::cout << std::endl;

    // Create an 8x8 matrix
    float src_data[64];
    float dst_data[64];
    
    for (int i = 0; i < 64; ++i) {
        src_data[i] = static_cast<float>(i);
        dst_data[i] = 0.0f;
    }

    auto src_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto dst_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    auto src_tensor = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), dst_layout);

    // TASK 1: Visualize source matrix
    std::cout << "Task 1 - Source Matrix (8x8):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(src_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 2: Naive transpose (element by element)
    std::cout << "Task 2 - Naive Transpose:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            dst_tensor(j, i) = src_tensor(i, j);
        }
    }

    std::cout << "Transposed Matrix:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(dst_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Analyze access pattern
    std::cout << "Task 3 - Access Pattern Analysis:" << std::endl;
    std::cout << "Naive transpose: dst[j][i] = src[i][j]" << std::endl;
    std::cout << std::endl;
    std::cout << "Read pattern (src):" << std::endl;
    std::cout << "  Row-wise access: COALESCED (stride 1)" << std::endl;
    std::cout << std::endl;
    std::cout << "Write pattern (dst):" << std::endl;
    std::cout << "  Column-wise access: UNCOALESCED (stride 8)" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem: Uncoalesced writes reduce bandwidth!" << std::endl;
    std::cout << std::endl;

    // Reset destination
    for (int i = 0; i < 64; ++i) {
        dst_data[i] = 0.0f;
    }

    // TASK 4: Tiled transpose (4x4 tiles)
    std::cout << "Task 4 - Tiled Transpose (4x4 tiles):" << std::endl;
    std::cout << "Strategy: Load tile to shared memory, transpose, store" << std::endl;
    std::cout << std::endl;

    // Process each 4x4 tile
    for (int tile_i = 0; tile_i < 2; ++tile_i) {
        for (int tile_j = 0; tile_j < 2; ++tile_j) {
            std::cout << "Processing tile (" << tile_i << ", " << tile_j << ")" << std::endl;
            
            // Load tile from source (coalesced row access)
            float tile[4][4];
            for (int ti = 0; ti < 4; ++ti) {
                for (int tj = 0; tj < 4; ++tj) {
                    int src_i = tile_i * 4 + ti;
                    int src_j = tile_j * 4 + tj;
                    tile[ti][tj] = src_tensor(src_i, src_j);
                }
            }
            
            // Transpose tile and store to destination (coalesced row access)
            for (int ti = 0; ti < 4; ++ti) {
                for (int tj = 0; tj < 4; ++tj) {
                    int dst_i = tile_j * 4 + ti;  // Note: swapped
                    int dst_j = tile_i * 4 + tj;  // Note: swapped
                    dst_tensor(dst_i, dst_j) = tile[tj][ti];  // Transpose
                }
            }
        }
    }

    std::cout << "Transposed Matrix (tiled):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(dst_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Verify transpose correctness
    std::cout << "Task 5 - Verification:" << std::endl;
    bool correct = true;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (dst_tensor(i, j) != src_tensor(j, i)) {
                correct = false;
            }
        }
    }
    std::cout << "Transpose correct: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Shared memory optimization
    std::cout << "=== Challenge: Shared Memory Optimization ===" << std::endl;
    std::cout << "In a real CUDA kernel, use shared memory:" << std::endl;
    std::cout << R"(
// Each block loads one tile to shared memory
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Load with coalesced access
int ti = threadIdx.y, tj = threadIdx.x;
int src_i = blockIdx.y * TILE_SIZE + ti;
int src_j = blockIdx.x * TILE_SIZE + tj;
tile[ti][tj] = src[src_i * N + src_j];

__syncthreads();

// Read transposed with coalesced access
int dst_i = blockIdx.x * TILE_SIZE + ti;
int dst_j = blockIdx.y * TILE_SIZE + tj;
dst[dst_i * N + dst_j] = tile[tj][ti];
)" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Transpose swaps rows and columns" << std::endl;
    std::cout << "2. Naive transpose has uncoalesced writes" << std::endl;
    std::cout << "3. Tiled transpose enables coalesced access" << std::endl;
    std::cout << "4. Shared memory helps reorganize data" << std::endl;

    return 0;
}
