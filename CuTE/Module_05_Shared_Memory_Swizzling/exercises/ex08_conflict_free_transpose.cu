/**
 * Exercise 08: Bank Conflict-Free Matrix Transpose
 * 
 * Objective: Implement a bank conflict-free matrix transpose
 *            using padding and swizzling techniques
 * 
 * Tasks:
 * 1. Understand transpose access patterns
 * 2. Implement padded transpose
 * 3. Implement swizzled transpose
 * 4. Compare performance characteristics
 * 
 * Key Concepts:
 * - Transpose: Swap rows and columns
 * - Read Pattern: Row-major (usually fine)
 * - Write Pattern: Column-major (needs optimization)
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 08: Bank Conflict-Free Matrix Transpose ===" << std::endl;
    std::cout << std::endl;

    const int SIZE = 32;

    // Create source matrix
    float src_data[SIZE * SIZE];
    float dst_padded[SIZE * SIZE];
    float dst_swizzled[SIZE * SIZE];
    
    for (int i = 0; i < SIZE * SIZE; ++i) {
        src_data[i] = static_cast<float>(i);
        dst_padded[i] = 0.0f;
        dst_swizzled[i] = 0.0f;
    }

    auto src_layout = make_layout(make_shape(Int<SIZE>{}, Int<SIZE>{}), GenRowMajor{});
    auto src_tensor = make_tensor(make_gmem_ptr(src_data), src_layout);

    std::cout << "Source Matrix (first 8x8):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(src_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 1: Naive transpose (with conflicts)
    std::cout << "Task 1 - Naive Transpose Analysis:" << std::endl;
    std::cout << "Read: Row-major (coalesced, no conflicts)" << std::endl;
    std::cout << "Write: Column-major (uncoalesced, 32-way conflicts!)" << std::endl;
    std::cout << std::endl;

    std::cout << "Write pattern analysis (first 8 writes):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int addr = i * SIZE;  // Column access
        int bank = addr % 32;
        std::cout << "  Write to (" << i << ", 0) -> Addr " << addr 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "  All writes to Bank 0! (32-way conflict)" << std::endl;
    std::cout << std::endl;

    // TASK 2: Padded transpose
    std::cout << "Task 2 - Padded Transpose:" << std::endl;
    const int PADDED_SIZE = SIZE + 1;
    
    auto padded_layout = make_layout(
        make_shape(Int<SIZE>{}, Int<PADDED_SIZE>{}),
        make_stride(Int<PADDED_SIZE>{}, Int<1>{})
    );
    auto padded_tensor = make_tensor(make_gmem_ptr(dst_padded), padded_layout);

    std::cout << "Using padded layout with stride " << PADDED_SIZE << std::endl;
    
    // Simulate padded transpose
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            padded_tensor(j, i) = src_tensor(i, j);
        }
    }

    std::cout << "Write pattern analysis (first 8 writes):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int addr = i * PADDED_SIZE;  // Padded column access
        int bank = addr % 32;
        std::cout << "  Write to (" << i << ", 0) -> Addr " << addr 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "  Different banks! (no conflict)" << std::endl;
    std::cout << std::endl;

    std::cout << "Transposed result (first 8x8):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(padded_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Swizzled transpose
    std::cout << "Task 3 - Swizzled Transpose:" << std::endl;
    
    auto xor_swizzle = [](int addr) {
        return addr ^ (addr >> 5);
    };

    std::cout << "Using XOR swizzle: addr XOR (addr >> 5)" << std::endl;
    
    // Simulate swizzled transpose
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            int write_addr = j * SIZE + i;  // Column-major write
            int swizzled_addr = xor_swizzle(write_addr);
            dst_swizzled[swizzled_addr] = src_tensor(i, j);
        }
    }

    std::cout << "Swizzled write pattern (first 8 writes):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int addr = i * SIZE;
        int swizzled = xor_swizzle(addr);
        int bank = swizzled % 32;
        std::cout << "  Addr " << addr << " -> Swizzled " << swizzled 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "  Different banks! (no conflict)" << std::endl;
    std::cout << std::endl;

    // TASK 4: Compare approaches
    std::cout << "Task 4 - Approach Comparison:" << std::endl;
    
    std::cout << std::endl;
    std::cout << "| Aspect        | Naive    | Padded   | Swizzled |" << std::endl;
    std::cout << "|---------------|----------|----------|----------|" << std::endl;
    std::cout << "| Conflicts     | 32-way   | None     | None     |" << std::endl;
    std::cout << "| Memory        | 100%     | 103%     | 100%     |" << std::endl;
    std::cout << "| Complexity    | Low      | Low      | Medium   |" << std::endl;
    std::cout << "| Performance   | Poor     | Excellent| Excellent|" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Optimize for different sizes
    std::cout << "=== Challenge: Different Matrix Sizes ===" << std::endl;
    std::cout << "For 64x64 matrix:" << std::endl;
    std::cout << "  Padding: +1 element (stride = 65)" << std::endl;
    std::cout << "  Overhead: 64 / 4096 = 1.6%" << std::endl;
    std::cout << std::endl;

    std::cout << "For 128x128 matrix:" << std::endl;
    std::cout << "  Padding: +1 element (stride = 129)" << std::endl;
    std::cout << "  Overhead: 128 / 16384 = 0.8%" << std::endl;
    std::cout << std::endl;

    // CUDA TRANSPOSE KERNEL
    std::cout << "=== CUDA Transpose Kernel ===" << std::endl;
    std::cout << R"(
// Padded transpose kernel
__global__ void transpose_padded(float* src, float* dst, int width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // Padded!
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Coalesced read (row access)
    tile[threadIdx.y][threadIdx.x] = src[y * width + x];
    
    __syncthreads();
    
    // Compute transposed coordinates
    int tx = blockIdx.y * TILE_SIZE + threadIdx.x;
    int ty = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Conflict-free write (padded column access)
    dst[tx * width + ty] = tile[threadIdx.x][threadIdx.y];
}

// Swizzled transpose kernel
__global__ void transpose_swizzled(float* src, float* dst, int width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    // Read with coalesced access
    // Write with swizzled addressing
    int addr = threadIdx.y * TILE_SIZE + threadIdx.x;
    int swizzled_addr = addr ^ (addr >> 5);
    
    tile[swizzled_addr / TILE_SIZE][swizzled_addr % TILE_SIZE] = ...;
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Transpose has conflicting access patterns" << std::endl;
    std::cout << "2. Padding eliminates write conflicts" << std::endl;
    std::cout << "3. Swizzling provides conflict-free access" << std::endl;
    std::cout << "4. Both approaches achieve excellent performance" << std::endl;

    return 0;
}
