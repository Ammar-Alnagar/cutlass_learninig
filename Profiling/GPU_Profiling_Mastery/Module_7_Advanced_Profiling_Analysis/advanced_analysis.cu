// advanced_analysis.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Complex kernel with multiple optimization opportunities
__global__ void complex_kernel(float *input, float *output, int n, int stride) {
    // Shared memory for data reuse
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Initialize shared memory
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    __syncthreads();
    
    if (idx < n) {
        float val = shared_data[tid];
        
        // Multiple memory accesses with stride
        if (idx + stride < n) {
            val += input[idx + stride];
        }
        
        // Computational loop
        for (int i = 0; i < 20; i++) {
            val = val * val + 0.1f;
            val = sqrtf(fmaxf(val, 1e-8f));
        }
        
        // Memory write with potential bank conflicts
        output[idx] = val * shared_data[(tid + 1) % blockDim.x];
    }
}

// Kernel with divergent branching
__global__ void divergent_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // Divergent branching based on data
        if (val > 0.5f) {
            // Expensive path
            for (int i = 0; i < 30; i++) {
                val = val * 1.01f + sinf(val);
            }
        } else if (val > 0.25f) {
            // Medium path
            for (int i = 0; i < 15; i++) {
                val = val * 1.02f + cosf(val);
            }
        } else {
            // Cheap path
            val = val * 2.0f;
        }
        
        output[idx] = val;
    }
}

// Optimized version of the complex kernel
__global__ void optimized_complex_kernel(float *input, float *output, int n, int stride) {
    // Use padded shared memory to avoid bank conflicts
    __shared__ float shared_data[256 + 1];  // +1 to avoid bank conflicts
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load with bounds checking
    shared_data[tid] = (idx < n) ? input[idx] : 0.0f;
    
    __syncthreads();
    
    if (idx < n) {
        float val = shared_data[tid];
        
        // Coalesced memory access
        if (idx + 1 < n) {
            val += input[idx + 1];
        }
        
        // Optimized computation with fewer operations
        for (int i = 0; i < 10; i++) {
            val = val * val * 0.9f + 0.1f;
        }
        
        // Avoid shared memory bank conflicts in write
        output[idx] = val * shared_data[tid];  // Use same index to avoid conflicts
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input with varying values to trigger divergence
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / N;  // Values from 0 to 1
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernels with different configurations
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Complex kernel with shared memory
    size_t sharedMemSize = blockSize * sizeof(float);
    complex_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, N, 10);
    cudaDeviceSynchronize();
    
    // Divergent kernel
    divergent_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Optimized kernel
    optimized_complex_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, N, 10);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("Advanced analysis kernels executed successfully!\n");
    return 0;
}