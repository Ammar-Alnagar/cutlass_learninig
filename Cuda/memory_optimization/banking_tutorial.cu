/*
 * CUDA Shared Memory Banking Tutorial
 * 
 * This tutorial demonstrates shared memory banking concepts and how to avoid conflicts.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: No bank conflicts - accessing different banks
__global__ void noBankConflicts(float* output) {
    __shared__ float sdata[32];  // 32 elements, each in different bank
    int tid = threadIdx.x;
    
    // Each thread accesses a different bank - no conflicts
    sdata[tid] = tid * 2.0f;
    __syncthreads();
    
    // Each thread reads from a different bank - no conflicts
    output[tid] = sdata[tid];
}

// Kernel 2: Broadcast - multiple threads reading same address
__global__ void broadcastAccess(float* output) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    
    if (tid == 0) {
        sdata[0] = 42.0f;  // Only thread 0 writes
    }
    __syncthreads();
    
    // All threads read the same element - efficient broadcast
    output[tid] = sdata[0];
}

// Kernel 3: Bank conflicts - multiple threads accessing same bank
__global__ void bankConflicts(float* output) {
    __shared__ float sdata[128];  // 128 elements
    int tid = threadIdx.x;
    
    // If threads access sdata[0], sdata[32], sdata[64], sdata[96] - all map to bank 0
    // This creates bank conflicts
    if (tid < 4) {
        sdata[tid * 32] = tid * 10.0f;  // 4-way bank conflict
    }
    __syncthreads();
    
    // Reading with potential conflicts
    if (tid < 4) {
        output[tid] = sdata[tid * 32];
    }
}

// Kernel 4: Matrix transpose with bank conflicts (problematic)
#define TILE_SIZE 32
__global__ void transposeWithConflicts(float* input, float* output, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];  // No padding - potential conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read in coalesced pattern (good)
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    __syncthreads();
    
    // Write out with transpose - causes bank conflicts!
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < n && y < n) {
        // This access pattern causes bank conflicts
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Kernel 5: Matrix transpose without bank conflicts (solution with padding)
__global__ void transposeWithoutConflicts(float* input, float* output, int n) {
    // Add padding to avoid bank conflicts: TILE_SIZE + 1 instead of TILE_SIZE
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read in coalesced pattern
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    __syncthreads();
    
    // Write out - no bank conflicts due to padding
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Kernel 6: Demonstrating diagonal access conflicts
__global__ void diagonalAccess(float* output) {
    __shared__ float sdata[32][32];
    int tid = threadIdx.x;
    
    if (tid < 32) {
        // Diagonal access causes bank conflicts
        sdata[tid][tid] = tid * 1.0f;
        __syncthreads();
        
        output[tid] = sdata[tid][tid];
    }
}

// Kernel 7: Proper access pattern to avoid conflicts
__global__ void properAccessPattern(float* output) {
    // Use padding to avoid conflicts
    __shared__ float sdata[32][33];  // 33 instead of 32 adds padding
    int tid = threadIdx.x;
    
    if (tid < 32) {
        // Now each thread accesses a different bank due to padding
        sdata[tid][tid] = tid * 1.0f;
        __syncthreads();
        
        output[tid] = sdata[tid][tid];
    }
}

int main() {
    printf("=== CUDA Shared Memory Banking Tutorial ===\n\n");
    
    const int N = 32;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_output1, *h_output2, *h_output3, *h_output4, *h_output5, *h_output6, *h_output7;
    h_output1 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    h_output3 = (float*)malloc(size);
    h_output4 = (float*)malloc(size);
    h_output5 = (float*)malloc(size);
    h_output6 = (float*)malloc(size);
    h_output7 = (float*)malloc(size);
    
    // Allocate device memory
    float *d_output1, *d_output2, *d_output3, *d_output4, *d_output5, *d_output6, *d_output7;
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    cudaMalloc(&d_output3, size);
    cudaMalloc(&d_output4, size);
    cudaMalloc(&d_output5, size);
    cudaMalloc(&d_output6, size);
    cudaMalloc(&d_output7, size);
    
    // Example 1: No bank conflicts
    printf("1. No Bank Conflicts (Good):\n");
    noBankConflicts<<<1, 32>>>(d_output1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    printf("   Results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n\n");
    
    // Example 2: Broadcast access
    printf("2. Broadcast Access (Efficient):\n");
    broadcastAccess<<<1, 32>>>(d_output2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output2, d_output2, size, cudaMemcpyDeviceToHost);
    printf("   Results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output2[i]);
    }
    printf("\n\n");
    
    // Example 3: Bank conflicts
    printf("3. Bank Conflicts (Inefficient):\n");
    bankConflicts<<<1, 32>>>(d_output3);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output3, d_output3, size, cudaMemcpyDeviceToHost);
    printf("   Results: ");
    for (int i = 0; i < 4; i++) {
        printf("%.1f ", h_output3[i]);
    }
    printf("(others are 0)\n\n");
    
    // Example 4: Matrix transpose with conflicts
    printf("4. Matrix Transpose with Bank Conflicts:\n");
    const int MATRIX_SIZE = 64;
    const int MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
    size_t matrix_size = MATRIX_ELEMENTS * sizeof(float);
    
    float *h_matrix_in, *h_matrix_out_conflict, *h_matrix_out_no_conflict;
    float *d_matrix_in, *d_matrix_out_conflict, *d_matrix_out_no_conflict;
    
    h_matrix_in = (float*)malloc(matrix_size);
    h_matrix_out_conflict = (float*)malloc(matrix_size);
    h_matrix_out_no_conflict = (float*)malloc(matrix_size);
    cudaMalloc(&d_matrix_in, matrix_size);
    cudaMalloc(&d_matrix_out_conflict, matrix_size);
    cudaMalloc(&d_matrix_out_no_conflict, matrix_size);
    
    // Initialize matrix
    for (int i = 0; i < MATRIX_ELEMENTS; i++) {
        h_matrix_in[i] = i * 1.0f;
    }
    cudaMemcpy(d_matrix_in, h_matrix_in, matrix_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, 
                  (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    transposeWithConflicts<<<gridSize, blockSize>>>(d_matrix_in, d_matrix_out_conflict, MATRIX_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matrix_out_conflict, d_matrix_out_conflict, matrix_size, cudaMemcpyDeviceToHost);
    printf("   Transpose with conflicts completed.\n");
    
    // Example 5: Matrix transpose without conflicts
    printf("5. Matrix Transpose without Bank Conflicts (Solution):\n");
    transposeWithoutConflicts<<<gridSize, blockSize>>>(d_matrix_in, d_matrix_out_no_conflict, MATRIX_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matrix_out_no_conflict, d_matrix_out_no_conflict, matrix_size, cudaMemcpyDeviceToHost);
    printf("   Transpose without conflicts completed.\n");
    printf("   First few transposed elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_matrix_out_no_conflict[i]);
    }
    printf("\n\n");
    
    // Example 6: Diagonal access conflicts
    printf("6. Diagonal Access Conflicts:\n");
    diagonalAccess<<<1, 32>>>(d_output6);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output6, d_output6, size, cudaMemcpyDeviceToHost);
    printf("   Results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output6[i]);
    }
    printf("\n\n");
    
    // Example 7: Proper access pattern with padding
    printf("7. Proper Access Pattern (Solution with Padding):\n");
    properAccessPattern<<<1, 32>>>(d_output7);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output7, d_output7, size, cudaMemcpyDeviceToHost);
    printf("   Results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output7[i]);
    }
    printf("\n\n");
    
    // Performance comparison note
    printf("Note: In real applications, you would measure timing differences\n");
    printf("between conflicting and non-conflicting access patterns.\n\n");
    
    // Cleanup
    free(h_output1);
    free(h_output2);
    free(h_output3);
    free(h_output4);
    free(h_output5);
    free(h_output6);
    free(h_output7);
    free(h_matrix_in);
    free(h_matrix_out_conflict);
    free(h_matrix_out_no_conflict);
    
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    cudaFree(d_output6);
    cudaFree(d_output7);
    cudaFree(d_matrix_in);
    cudaFree(d_matrix_out_conflict);
    cudaFree(d_matrix_out_no_conflict);
    
    printf("Tutorial completed!\n");
    return 0;
}