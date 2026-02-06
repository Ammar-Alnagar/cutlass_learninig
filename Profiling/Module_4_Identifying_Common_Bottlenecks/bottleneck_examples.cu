// bottleneck_examples.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Memory-bound example: lots of memory access, little computation
__global__ void memory_bound_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Multiple memory accesses per computation
        float sum = 0.0f;
        sum += input[idx];
        if (idx + 1 < n) sum += input[idx + 1];
        if (idx + 2 < n) sum += input[idx + 2];
        if (idx + 3 < n) sum += input[idx + 3];
        if (idx + 4 < n) sum += input[idx + 4];
        
        output[idx] = sum / 5.0f;  // Minimal computation
    }
}

// Compute-bound example: lots of computation, minimal memory access
__global__ void compute_bound_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        
        // Heavy computation per memory access
        for (int i = 0; i < 100; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
            x = x * 2.0f + 1.0f;
        }
        
        output[idx] = x;
    }
}

// Occupancy-limited example: high register usage
__global__ void occupancy_limited_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Use many registers to limit occupancy
        float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
        float r10, r11, r12, r13, r14, r15, r16, r17, r18, r19;
        
        r0 = input[idx];
        r1 = r0 * 1.1f; r2 = r1 * 1.2f; r3 = r2 * 1.3f; r4 = r3 * 1.4f;
        r5 = r4 * 1.5f; r6 = r5 * 1.6f; r7 = r6 * 1.7f; r8 = r7 * 1.8f;
        r9 = r8 * 1.9f; r10 = r9 * 2.0f; r11 = r10 * 2.1f; r12 = r11 * 2.2f;
        r13 = r12 * 2.3f; r14 = r13 * 2.4f; r15 = r14 * 2.5f; r16 = r15 * 2.6f;
        r17 = r16 * 2.7f; r18 = r17 * 2.8f; r19 = r18 * 2.9f;
        
        output[idx] = r19;
    }
}

// Branch-divergent example
__global__ void branch_divergent_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // Conditional execution causing divergence
        if (val > 0.5f) {
            // Expensive path
            for (int i = 0; i < 10; i++) {
                val = val * val + 0.1f;
            }
        } else {
            // Cheap path
            val = val * 2.0f;
        }
        
        output[idx] = val;
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
        h_input[i] = (float)(i % 1000) / 1000.0f;  // Values between 0 and 1
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch different kernels and profile each
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Memory-bound kernel
    memory_bound_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Compute-bound kernel
    compute_bound_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Occupancy-limited kernel
    occupancy_limited_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Branch-divergent kernel
    branch_divergent_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("All kernels executed successfully!\n");
    return 0;
}