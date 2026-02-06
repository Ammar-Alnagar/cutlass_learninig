// inefficient_kernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void inefficientKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Intentionally inefficient: causes branch divergence
        if (idx % 2 == 0) {
            output[idx] = input[idx] * 2.0f;
        } else {
            // Different computation path
            float temp = input[idx] * 1.5f;
            output[idx] = temp + 0.5f;
        }
        
        // Memory access pattern issue: strided access
        if (idx + 10 < n) {
            output[idx] += input[idx + 10];  // Non-coalesced access pattern
        }
    }
}

__global__ void efficientKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // More efficient: uniform execution
        float val = input[idx];
        output[idx] = val * 2.0f;
        
        // Coalesced memory access
        if (idx + 1 < n) {
            output[idx] += input[idx + 1];
        }
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output1 = (float*)malloc(bytes);
    float *h_output2 = (float*)malloc(bytes);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    // Allocate device memory
    float *d_input, *d_output1, *d_output2;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output1, bytes);
    cudaMalloc(&d_output2, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch inefficient kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    inefficientKernel<<<gridSize, blockSize>>>(d_input, d_output1, N);
    cudaDeviceSynchronize();
    
    // Launch efficient kernel
    efficientKernel<<<gridSize, blockSize>>>(d_input, d_output2, N);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_output1, d_output1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output1); free(h_output2);
    cudaFree(d_input); cudaFree(d_output1); cudaFree(d_output2);
    
    printf("Kernels executed successfully!\n");
    return 0;
}