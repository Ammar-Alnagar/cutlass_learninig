// reduction_initial.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Basic reduction with poor performance
__global__ void reduction_basic(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes one element
    float sum = (idx < n) ? input[idx] : 0.0f;
    
    // Reduction within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __shared__ float temp[1024];  // Assuming max block size
        if (threadIdx.x % (2 * stride) == 0) {
            temp[threadIdx.x] = sum;
        }
        __syncthreads();
        
        if (threadIdx.x % (2 * stride) == 0) {
            sum += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}