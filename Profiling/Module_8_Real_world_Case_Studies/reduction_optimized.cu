// reduction_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

__device__ float warpReduce(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_optimized(float *input, float *output, int n) {
    __shared__ float sdata[32]; // One element per warp (assuming 1024 threads/block -> 32 warps)
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Each thread loads two elements and sums them
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    
    // Reduce within warp using shuffle
    sum = warpReduce(sum);
    
    // Write reduced value to shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    
    __syncthreads();
    
    // Final reduce within block
    if (tid < 32) {
        sum = sdata[tid];
        sum = warpReduce(sum);
        if (tid == 0) output[blockIdx.x] = sum;
    }
}