// matmul_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_optimized(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < N; t += TILE_SIZE) {
        // Load tiles into shared memory
        As[ty][tx] = (row < N && t+tx < N) ? A[row * N + t + tx] : 0.0f;
        Bs[ty][tx] = (t+ty < N && col < N) ? B[(t + ty) * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}