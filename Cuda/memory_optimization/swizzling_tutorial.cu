/*
 * CUDA Shared Memory Swizzling Tutorial
 * 
 * This tutorial demonstrates shared memory swizzling techniques to avoid bank conflicts.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Define swizzling function
__device__ __forceinline__ unsigned int swizzle_row(unsigned int addr, unsigned int width) {
    // Simple swizzling: XOR with right-shifted version
    return addr ^ (addr >> 5);  // 5 = log2(32 banks)
}

// Kernel 1: Basic swizzling example
__global__ void basicSwizzling(float* input, float* output, int n) {
    __shared__ float raw_sdata[1024];  // Raw shared memory array
    int tid = threadIdx.x;
    
    if (tid < n) {
        // Calculate swizzled address
        int swizzled_addr = swizzle_row(tid, 32);
        
        // Store with swizzling
        raw_sdata[swizzled_addr] = input[tid];
        __syncthreads();
        
        // Retrieve with swizzling
        output[tid] = raw_sdata[swizzled_addr];
    }
}

// Swizzle function for 2D access
__device__ __forceinline__ int swizzle_2d(int row, int col, int width) {
    return row * width + (col ^ (row & 31));  // XOR with row's lower 5 bits
}

// Kernel 2: Matrix tiling with swizzling
#define TILE_SIZE 32
__global__ void swizzledTiling(float* input, float* output, int n) {
    // Use swizzled shared memory layout
    __shared__ float sdata[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    if (x < n && y < n) {
        // Calculate swizzled address for 2D access
        int linear_idx = threadIdx.y * TILE_SIZE + threadIdx.x;
        int swizzled_idx = swizzle_2d(threadIdx.y, threadIdx.x, TILE_SIZE);
        
        // Ensure swizzled_idx is within bounds
        if (swizzled_idx < TILE_SIZE * TILE_SIZE) {
            sdata[0][swizzled_idx] = input[y * n + x];
        }
    }
    __syncthreads();
    
    // Process and write back
    if (x < n && y < n) {
        int linear_idx = threadIdx.y * TILE_SIZE + threadIdx.x;
        int swizzled_idx = swizzle_2d(threadIdx.y, threadIdx.x, TILE_SIZE);
        
        if (swizzled_idx < TILE_SIZE * TILE_SIZE) {
            output[y * n + x] = sdata[0][swizzled_idx] * 2.0f;
        }
    }
}

// Kernel 3: More sophisticated swizzling for GEMM-like access
__device__ __forceinline__ int advanced_swizzle(int row, int col, int width) {
    // More complex swizzling pattern
    int bank_offset = (row / 2) % 32;  // Distribute based on row
    int swizzle_factor = (col ^ bank_offset) & 31;  // XOR with bank offset
    return row * width + (col ^ swizzle_factor);
}

__global__ void advancedSwizzling(float* input, float* output, int n) {
    __shared__ float sdata[1024];  // Flattened shared memory
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        int row = tid / 32;
        int col = tid % 32;
        
        if (row < 32 && col < 32) {  // Keep within reasonable bounds
            int swizzled_addr = advanced_swizzle(row, col, 32);
            
            // Bounds check for swizzled address
            if (swizzled_addr < 1024) {
                sdata[swizzled_addr] = input[tid];
                __syncthreads();
                
                output[tid] = sdata[swizzled_addr] * 3.0f;
            }
        }
    }
}

// Kernel 4: Matrix transpose with swizzling to avoid conflicts
__global__ void swizzledTranspose(float* input, float* output, int n) {
    // Swizzled shared memory to avoid bank conflicts during transpose
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 padding approach
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read input in coalesced pattern
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    __syncthreads();
    
    // Calculate output coordinates for transpose
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write output in transposed pattern
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Kernel 5: Demonstration of swizzling effectiveness
__global__ void compareAccessPatterns(float* output) {
    // Normal access pattern (may have conflicts)
    __shared__ float normal_mem[32][32];
    
    // Swizzled access pattern (reduces conflicts)
    __shared__ float swizzled_mem[32][33];  // +1 for padding
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (tx < 32 && ty < 32) {
        // Normal access - potential conflicts during transpose
        normal_mem[ty][tx] = tx + ty;
        __syncthreads();
        
        // Swizzled access - reduced conflicts due to padding
        swizzled_mem[ty][tx] = tx * 2 + ty * 2;
        __syncthreads();
        
        // Demonstrate different access patterns
        if (tx == 0) {
            output[threadIdx.x] = normal_mem[tx][ty] + swizzled_mem[tx][ty];
        }
    }
}

// Alternative swizzling using permutation
__device__ __forceinline__ int permute_swizzle(int addr) {
    // Permutation: swap certain bit positions
    int bank = (addr >> 2) & 31;  // Extract bank bits
    int offset = addr & 3;        // Extract offset within bank
    return (addr & ~127) | ((bank << 2) | offset);  // Reconstruct
}

// Kernel 6: Permutation-based swizzling
__global__ void permutationSwizzling(float* input, float* output, int n) {
    __shared__ float sdata[1024];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n && tid < 1024) {
        int permuted_addr = permute_swizzle(tid);
        
        // Bounds check
        if (permuted_addr < 1024) {
            sdata[permuted_addr] = input[tid];
            __syncthreads();
            
            output[tid] = sdata[permuted_addr] * 0.5f;
        }
    }
}

int main() {
    printf("=== CUDA Shared Memory Swizzling Tutorial ===\n\n");
    
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output1, *h_output2, *h_output3, *h_output4, *h_output5, *h_output6;
    h_input = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    h_output3 = (float*)malloc(size);
    h_output4 = (float*)malloc(size);
    h_output5 = (float*)malloc(size);
    h_output6 = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output1, *d_output2, *d_output3, *d_output4, *d_output5, *d_output6;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    cudaMalloc(&d_output3, size);
    cudaMalloc(&d_output4, size);
    cudaMalloc(&d_output5, size);
    cudaMalloc(&d_output6, size);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Example 1: Basic swizzling
    printf("1. Basic Swizzling:\n");
    basicSwizzling<<<(N + 255) / 256, 256>>>(d_input, d_output1, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n\n");
    
    // Example 2: Matrix tiling with swizzling
    printf("2. Matrix Tiling with Swizzling:\n");
    const int MATRIX_SIZE = 64;
    const int MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
    size_t matrix_size = MATRIX_ELEMENTS * sizeof(float);
    
    float *h_matrix_in, *h_matrix_out;
    float *d_matrix_in, *d_matrix_out;
    
    h_matrix_in = (float*)malloc(matrix_size);
    h_matrix_out = (float*)malloc(matrix_size);
    cudaMalloc(&d_matrix_in, matrix_size);
    cudaMalloc(&d_matrix_out, matrix_size);
    
    // Initialize matrix
    for (int i = 0; i < MATRIX_ELEMENTS; i++) {
        h_matrix_in[i] = i * 1.0f;
    }
    cudaMemcpy(d_matrix_in, h_matrix_in, matrix_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, 
                  (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    swizzledTiling<<<gridSize, blockSize>>>(d_matrix_in, d_matrix_out, MATRIX_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matrix_out, d_matrix_out, matrix_size, cudaMemcpyDeviceToHost);
    printf("   Matrix tiling with swizzling completed.\n");
    printf("   First few results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_matrix_out[i]);
    }
    printf("\n\n");
    
    // Example 3: Advanced swizzling
    printf("3. Advanced Swizzling:\n");
    advancedSwizzling<<<(N + 255) / 256, 256>>>(d_input, d_output3, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output3, d_output3, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output3[i]);
    }
    printf("\n\n");
    
    // Example 4: Swizzled transpose
    printf("4. Swizzled Matrix Transpose:\n");
    swizzledTranspose<<<gridSize, blockSize>>>(d_matrix_in, d_matrix_out, MATRIX_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matrix_out, d_matrix_out, matrix_size, cudaMemcpyDeviceToHost);
    printf("   Swizzled transpose completed.\n");
    printf("   First few transposed elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_matrix_out[i]);
    }
    printf("\n\n");
    
    // Example 5: Access pattern comparison
    printf("5. Access Pattern Comparison:\n");
    dim3 compareBlockSize(32, 32);
    dim3 compareGridSize(1, 1);
    compareAccessPatterns<<<compareGridSize, compareBlockSize>>>(d_output5);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output5, d_output5, size, cudaMemcpyDeviceToHost);
    printf("   Comparison results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output5[i]);
    }
    printf("\n\n");
    
    // Example 6: Permutation-based swizzling
    printf("6. Permutation-based Swizzling:\n");
    permutationSwizzling<<<(N + 255) / 256, 256>>>(d_input, d_output6, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output6, d_output6, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output6[i]);
    }
    printf("\n\n");
    
    printf("Note: Swizzling is primarily used to avoid shared memory bank conflicts.\n");
    printf("The performance benefits are most apparent in scenarios with predictable\n");
    printf("access patterns that would otherwise cause systematic bank conflicts.\n\n");
    
    // Cleanup
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    free(h_output4);
    free(h_output5);
    free(h_output6);
    free(h_matrix_in);
    free(h_matrix_out);
    
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    cudaFree(d_output6);
    cudaFree(d_matrix_in);
    cudaFree(d_matrix_out);
    
    printf("Tutorial completed!\n");
    return 0;
}