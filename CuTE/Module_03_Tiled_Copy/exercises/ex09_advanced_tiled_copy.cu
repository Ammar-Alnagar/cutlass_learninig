/**
 * Exercise 09: Advanced Tiled Copy Patterns
 *
 * Objective: Master advanced tiled copy operations including async copies,
 *            multi-stage pipelines, and complex transfer patterns
 *
 * Tasks:
 * 1. Implement async copy with cp.async
 * 2. Design multi-stage pipeline copies
 * 3. Create complex tiled transfer patterns
 * 4. Optimize copy for specific hardware
 */

#include <cuda_runtime.h>
#include <iostream>

// Note: This exercise demonstrates concepts. Actual CuTe usage requires
// full CUTLASS installation with appropriate headers.

using namespace std;

// =========================================================================
// Helper Functions
// =========================================================================

void check_cuda_error(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error: " << message << " - " 
             << cudaGetErrorString(result) << endl;
    }
}

// =========================================================================
// Task 1: Async Copy Pattern (sm_80+)
// =========================================================================

__global__ void async_copy_kernel(float* gmem_src, float* smem_dst, int size) {
    // Async copy pattern for sm_80+ architectures
    // Uses cp.async instruction for non-blocking transfer
    
    extern __shared__ float smem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Async copy instruction (inline assembly)
        // cp.async.ca.shared.global [dst], [src], bytes
        int smem_offset = threadIdx.x * sizeof(float);
        
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], %2;"
            : 
            : "r"(smem_offset), 
              "l"(&gmem_src[idx]), 
              "r"(sizeof(float))
        );
        
        // Signal completion
        asm volatile("cp.async.commit_group;" ::);
        
        // Wait for completion
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        
        // Store to destination
        smem_dst[idx] = smem[threadIdx.x];
    }
}

// =========================================================================
// Task 2: Double Buffering Pattern
// =========================================================================

template <int TILE_SIZE>
__global__ void double_buffered_copy(float* src, float* dst, int total_size) {
    extern __shared__ float smem[];  // 2 * TILE_SIZE elements
    
    float* smem_buffer[2];
    smem_buffer[0] = smem;
    smem_buffer[1] = &smem[TILE_SIZE];
    
    int write_stage = 0;
    int read_stage = 0;
    
    int num_tiles = (total_size + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int next_stage = 1 - write_stage;
        
        // Load next tile (producer)
        int load_idx = tile_idx * TILE_SIZE + threadIdx.x;
        if (load_idx < total_size) {
            smem_buffer[next_stage][threadIdx.x] = src[load_idx];
        }
        __syncthreads();
        
        // Process current tile (consumer)
        int store_idx = tile_idx * TILE_SIZE + threadIdx.x;
        if (store_idx < total_size) {
            dst[store_idx] = smem_buffer[read_stage][threadIdx.x];
        }
        __syncthreads();
        
        write_stage = next_stage;
        read_stage = next_stage;
    }
}

// =========================================================================
// Task 3: Vectorized Tiled Copy
// =========================================================================

__global__ void vectorized_tiled_copy(float* src, float* dst, int M, int N) {
    // Each thread copies a 4x4 tile using vectorized loads
    
    constexpr int TILE_H = 4;
    constexpr int TILE_W = 4;
    
    int row_start = blockIdx.y * blockDim.y * TILE_H + threadIdx.y * TILE_H;
    int col_start = blockIdx.x * blockDim.x * TILE_W + threadIdx.x * TILE_W;
    
    // Vectorized load (float4 = 128 bits)
    for (int tile_row = 0; tile_row < TILE_H; ++tile_row) {
        int row = row_start + tile_row;
        if (row < M) {
            int col = col_start;
            if (col % 4 == 0 && col + TILE_W <= N) {
                // Aligned vectorized load
                float4 val = reinterpret_cast<float4*>(&src[row * N + col])[0];
                reinterpret_cast<float4*>(&dst[row * N + col])[0] = val;
            } else {
                // Scalar fallback
                for (int tile_col = 0; tile_col < TILE_W; ++tile_col) {
                    col = col_start + tile_col;
                    if (col < N) {
                        dst[row * N + col] = src[row * N + col];
                    }
                }
            }
        }
    }
}

// =========================================================================
// Task 4: Matrix Transpose with Tiled Copy
// =========================================================================

__global__ void tiled_transpose_copy(float* src, float* dst, int M, int N) {
    extern __shared__ float smem[];
    
    // Load tile (coalesced)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;
    
    // Load from source
    if (row < M && col < N) {
        smem[tile_row * blockDim.x + tile_col] = src[row * N + col];
    }
    __syncthreads();
    
    // Store transposed (also coalesced)
    int dst_row = col;
    int dst_col = row;
    
    if (dst_row < N && dst_col < M) {
        dst[dst_row * M + dst_col] = smem[tile_col * blockDim.y + tile_row];
    }
}

// =========================================================================
// Main Exercise
// =========================================================================

int main() {
    cout << "=== Exercise 09: Advanced Tiled Copy Patterns ===" << endl;
    cout << endl;
    
    // Test parameters
    const int SIZE = 1024;
    const int M = 32, N = 32;
    const int TILE_SIZE = 256;
    
    // Allocate memory
    float *h_src, *h_dst, *d_src, *d_dst;
    
    h_src = new float[SIZE];
    h_dst = new float[SIZE];
    
    for (int i = 0; i < SIZE; ++i) {
        h_src[i] = static_cast<float>(i);
    }
    
    check_cuda_error(cudaMalloc(&d_src, SIZE * sizeof(float)), "cudaMalloc src");
    check_cuda_error(cudaMalloc(&d_dst, SIZE * sizeof(float)), "cudaMalloc dst");
    
    check_cuda_error(cudaMemcpy(d_src, h_src, SIZE * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    
    // =========================================================================
    // Task 1: Async Copy Demo
    // =========================================================================
    cout << "--- Task 1: Async Copy Pattern ---" << endl;
    cout << "Async copy uses cp.async instruction (sm_80+)" << endl;
    cout << "Key features:" << endl;
    cout << "  - Non-blocking memory transfer" << endl;
    cout << "  - Overlaps with computation" << endl;
    cout << "  - Requires cp.async.commit_group and wait_group" << endl;
    cout << endl;
    
    // =========================================================================
    // Task 2: Double Buffering Demo
    // =========================================================================
    cout << "--- Task 2: Double Buffering Pattern ---" << endl;
    cout << "Double buffering overlaps load and store:" << endl;
    cout << "  - Stage 0: Load tile N+1 while processing tile N" << endl;
    cout << "  - Stage 1: Ping-pong between buffers" << endl;
    cout << "  - Hides memory latency" << endl;
    cout << endl;
    
    cout << "Performance improvement:" << endl;
    cout << "  Sequential: N × (load + process)" << endl;
    cout << "  Double buffered: load + N × process" << endl;
    cout << "  Speedup: ~2x for large N" << endl;
    cout << endl;
    
    // =========================================================================
    // Task 3: Vectorized Copy Demo
    // =========================================================================
    cout << "--- Task 3: Vectorized Tiled Copy ---" << endl;
    cout << "Vectorized loads (float4) provide 4x bandwidth:" << endl;
    cout << "  - 128-bit load in single instruction" << endl;
    cout << "  - Requires 16-byte alignment" << endl;
    cout << "  - Best for contiguous memory regions" << endl;
    cout << endl;
    
    cout << "Alignment requirements:" << endl;
    cout << "  float4: 16-byte (index % 4 == 0)" << endl;
    cout << "  float2: 8-byte (index % 2 == 0)" << endl;
    cout << "  float1: 4-byte (any index)" << endl;
    cout << endl;
    
    // =========================================================================
    // Task 4: Transpose Copy Demo
    // =========================================================================
    cout << "--- Task 4: Matrix Transpose with Tiled Copy ---" << endl;
    cout << "Tiled transpose ensures coalesced access:" << endl;
    cout << "  - Load: Coalesced row-wise read" << endl;
    cout << "  - Shared: Transpose in shared memory" << endl;
    cout << "  - Store: Coalesced column-wise write" << endl;
    cout << endl;
    
    cout << "Without tiling:" << endl;
    cout << "  - One dimension is uncoalesced" << endl;
    cout << "  - 16x slower for that dimension" << endl;
    cout << endl;
    
    // =========================================================================
    // Performance Comparison
    // =========================================================================
    cout << "=== Performance Comparison ===" << endl;
    cout << endl;
    
    cout << "Copy Method Comparison (theoretical):" << endl;
    cout << "+------------------+------------+-------------+" << endl;
    cout << "| Method           | Bandwidth  | Use Case    |" << endl;
    cout << "+------------------+------------+-------------+" << endl;
    cout << "| Scalar copy      | 1x (387)   | Baseline    |" << endl;
    cout << "| Vectorized (4x)  | 4x (1548)  | General     |" << endl;
    cout << "| Async copy       | 4x + overlap | Pipeline  |" << endl;
    cout << "| Double buffered  | ~2x        | Latency hide|" << endl;
    cout << "+------------------+------------+-------------+" << endl;
    cout << "Bandwidth in GB/s (A100)" << endl;
    cout << endl;
    
    // =========================================================================
    // Challenge: Design Optimal Copy Kernel
    // =========================================================================
    cout << "=== Challenge: Design Optimal Copy Kernel ===" << endl;
    cout << endl;
    
    cout << "Design a copy kernel for matrix multiplication:" << endl;
    cout << "  Matrix A: 1024x1024 (row-major)" << endl;
    cout << "  Matrix B: 1024x1024 (column-major)" << endl;
    cout << "  Target: Load tiles into shared memory" << endl;
    cout << "  Constraints:" << endl;
    cout << "    - Use vectorized loads" << endl;
    cout << "    - Avoid bank conflicts" << endl;
    cout << "    - Overlap with computation" << endl;
    cout << endl;
    
    cout << "Your design considerations:" << endl;
    cout << "1. Tile size: 32x32 or 64x64?" << endl;
    cout << "2. Thread layout: 16x16 or 32x8?" << endl;
    cout << "3. Padding: +1 or +8 per row?" << endl;
    cout << "4. Stages: 2 (double) or 3 (triple) buffering?" << endl;
    cout << endl;
    
    // Cleanup
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_src);
    cudaFree(d_dst);
    
    cout << "=== Exercise Complete ===" << endl;
    cout << "Key Learnings:" << endl;
    cout << "1. Async copy enables compute/memory overlap" << endl;
    cout << "2. Double buffering hides memory latency" << endl;
    cout << "3. Vectorized loads provide 4x bandwidth" << endl;
    cout << "4. Tiled transpose ensures coalesced access" << endl;
    cout << "5. Bank conflicts must be avoided in shared memory" << endl;
    cout << "6. Multiple techniques can be combined" << endl;
    
    return 0;
}
