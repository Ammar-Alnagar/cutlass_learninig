// computational_optimization.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Unoptimized computation kernel
__global__ void unoptimized_computation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Inefficient: using division instead of multiplication
        x = x / 2.0f;
        // Inefficient: using slow math functions unnecessarily
        x = powf(x, 2.0f);
        output[idx] = x;
    }
}

// Optimized computation kernel
__global__ void optimized_computation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Efficient: multiply by reciprocal
        x = x * 0.5f;
        // Efficient: use faster intrinsic if precision allows
        x = x * x;  // Instead of powf(x, 2.0f)
        output[idx] = x;
    }
}

// Kernel with optimized loop unrolling
__global__ void loop_unroll_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Grid-stride loop with unrolling
    for (int i = idx; i < n; i += stride * 4) {
        // Process 4 elements per thread
        if (i < n) output[i] = input[i] * input[i];
        if (i + 1 < n) output[i + 1] = input[i + 1] * input[i + 1];
        if (i + 2 < n) output[i + 2] = input[i + 2] * input[i + 2];
        if (i + 3 < n) output[i + 3] = input[i + 3] * input[i + 3];
    }
}

// Kernel with reduced register usage
__global__ void low_reg_usage_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // Process in pipeline to reduce register pressure
        val *= 2.0f;
        val += 1.0f;
        val = fmaxf(val, 0.0f);  // ReLU activation
        output[idx] = val;
    }
}

// High arithmetic intensity kernel
__global__ void high_arith_intens_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        
        // High computation-to-memory ratio
        for (int i = 0; i < 50; i++) {
            x = x * x + 0.1f;
            x = sqrtf(fmaxf(x, 1e-8f));
            x = x * 0.9f + 0.1f * input[idx];  // Mix with original
        }
        
        output[idx] = x;
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 1000) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch different kernels
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Unoptimized kernel
    unoptimized_computation<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Optimized kernel
    optimized_computation<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Loop unrolling kernel
    loop_unroll_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Low register usage kernel
    low_reg_usage_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // High arithmetic intensity kernel
    high_arith_intens_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("Computational optimization kernels executed successfully!\n");
    return 0;
}