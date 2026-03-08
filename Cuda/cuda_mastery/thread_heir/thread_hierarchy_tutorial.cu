/*
 * CUDA Thread Hierarchy Tutorial
 * 
 * This tutorial demonstrates the fundamental concepts of CUDA's thread hierarchy:
 * Grid, Block, and Thread organization.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Basic thread identification
__global__ void printThreadInfo(int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        printf("Thread %d (local: %d) in Block %d, Global ID: %d\n", 
               threadIdx.x, threadIdx.x, blockIdx.x, tid);
    }
}

// Kernel 2: 2D thread indexing
__global__ void matrixThreadIndexing(float* matrix, int width, int height) {
    // 2D indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        matrix[idx] = idx * 1.0f;  // Assign value based on position
        printf("Thread (%d,%d) in Block (%d,%d) processed element [%d,%d] at global index %d\n",
               threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, idx);
    }
}

// Kernel 3: Grid-stride loop for handling arrays larger than grid size
__global__ void gridStrideLoop(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;  // Process element
        printf("Thread %d processed element %d\n", tid, i);
    }
}

// Kernel 4: Demonstrating warp behavior
__global__ void warpBehavior(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int warpId = tid / 32;  // Each warp has 32 threads
        int laneId = tid % 32;  // Position within warp
        
        data[tid] = warpId * 100 + laneId;
        printf("Thread %d: Warp %d, Lane %d, Value: %d\n", tid, warpId, laneId, data[tid]);
    }
}

int main() {
    printf("=== CUDA Thread Hierarchy Tutorial ===\n\n");
    
    // Example 1: Basic thread info
    printf("1. Basic Thread Information:\n");
    int n1 = 16;
    int blockSize1 = 8;
    int gridSize1 = (n1 + blockSize1 - 1) / blockSize1;
    
    printThreadInfo<<<gridSize1, blockSize1>>>(n1);
    cudaDeviceSynchronize();
    printf("\n");
    
    // Example 2: 2D thread indexing
    printf("2. 2D Thread Indexing:\n");
    int width = 4, height = 3;
    float *h_matrix, *d_matrix;
    size_t matrix_size = width * height * sizeof(float);
    
    h_matrix = (float*)malloc(matrix_size);
    cudaMalloc(&d_matrix, matrix_size);
    
    dim3 blockSize2(2, 2);  // 2x2 threads per block
    dim3 gridSize2((width + blockSize2.x - 1) / blockSize2.x, 
                   (height + blockSize2.y - 1) / blockSize2.y);
    
    matrixThreadIndexing<<<gridSize2, blockSize2>>>(d_matrix, width, height);
    cudaDeviceSynchronize();
    printf("\n");
    
    // Example 3: Grid-stride loop
    printf("3. Grid-Stride Loop:\n");
    int n3 = 20;
    float *h_data3, *d_data3;
    size_t size3 = n3 * sizeof(float);
    
    h_data3 = (float*)malloc(size3);
    cudaMalloc(&d_data3, size3);
    
    // Initialize data
    for (int i = 0; i < n3; i++) {
        h_data3[i] = i * 1.0f;
    }
    cudaMemcpy(d_data3, h_data3, size3, cudaMemcpyHostToDevice);
    
    int blockSize3 = 4;
    int gridSize3 = 2;  // Smaller grid to demonstrate stride
    
    gridStrideLoop<<<gridSize3, blockSize3>>>(d_data3, n3);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data3, d_data3, size3, cudaMemcpyDeviceToHost);
    printf("Results: ");
    for (int i = 0; i < n3; i++) {
        printf("%.1f ", h_data3[i]);
    }
    printf("\n\n");
    
    // Example 4: Warp behavior
    printf("4. Warp Behavior:\n");
    int n4 = 65;  // More than one warp to see multiple warps
    int *h_data4, *d_data4;
    size_t size4 = n4 * sizeof(int);
    
    h_data4 = (int*)malloc(size4);
    cudaMalloc(&d_data4, size4);
    
    int blockSize4 = 32;
    int gridSize4 = (n4 + blockSize4 - 1) / blockSize4;
    
    warpBehavior<<<gridSize4, blockSize4>>>(d_data4, n4);
    cudaDeviceSynchronize();
    printf("\n");
    
    // Cleanup
    free(h_matrix);
    free(h_data3);
    free(h_data4);
    cudaFree(d_matrix);
    cudaFree(d_data3);
    cudaFree(d_data4);
    
    printf("Tutorial completed!\n");
    return 0;
}