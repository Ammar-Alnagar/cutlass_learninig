/*
 * CUDA Memory Hierarchy Tutorial
 * 
 * This tutorial demonstrates the different memory types in CUDA:
 * Global, Shared, Constant, and Register memory.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel demonstrating global memory usage
__global__ void globalMemoryExample(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Access global memory
        float value = input[idx];
        output[idx] = value * 2.0f;
    }
}

// Kernel demonstrating shared memory usage
__global__ void sharedMemoryExample(float* input, float* output, int n) {
    // Declare shared memory
    __shared__ float sData[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperative loading into shared memory
    if (i < n) {
        sData[tid] = input[i];
    } else {
        sData[tid] = 0.0f;  // Padding for safety
    }
    __syncthreads();  // Synchronize threads in block
    
    // Process data in shared memory
    if (tid > 0) {
        sData[tid] += sData[tid - 1];  // Shared memory access
    }
    __syncthreads();
    
    // Write back to global memory
    if (i < n) {
        output[i] = sData[tid];
    }
}

// Constant memory declaration
__constant__ float constCoeffs[256];

// Kernel demonstrating constant memory usage
__global__ void constantMemoryExample(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Access constant memory - cached and broadcast-efficient
        output[idx] = input[idx] * constCoeffs[idx % 256];
    }
}

// Kernel demonstrating register usage
__global__ void registerUsageExample(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Values stored in registers for fast access
        float a = input[idx];
        float b = a * 1.5f;
        float c = b + 2.0f;
        float d = c * a;
        float result = d - b;
        
        output[idx] = result;
    }
}

// Kernel demonstrating memory coalescing
__global__ void coalescedAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Coalesced access: consecutive threads access consecutive memory
        float val = input[idx];
        output[idx] = val * 2.0f;
    }
}

// Kernel demonstrating shared memory banking
__global__ void sharedMemoryBanking(float* input, float* output, int n) {
    // 32-way banked shared memory
    __shared__ float sMem[32][33];  // +1 to avoid bank conflicts in transpose
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    if (tx < 32 && ty < 32) {
        int globalIdx = (by * 32 + ty) * 32 + (bx * 32 + tx);
        if (globalIdx < n) {
            // Load into shared memory with potential bank conflicts
            sMem[ty][tx] = input[globalIdx];
        }
        __syncthreads();
        
        // Transpose access pattern - could cause bank conflicts without padding
        float val = sMem[tx][ty];  // Transposed access
        if (globalIdx < n) {
            output[globalIdx] = val * 2.0f;
        }
    }
}

int main() {
    printf("=== CUDA Memory Hierarchy Tutorial ===\n\n");
    
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output;
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Example 1: Global memory
    printf("1. Global Memory Example:\n");
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    globalMemoryExample<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");
    
    // Example 2: Shared memory
    printf("2. Shared Memory Example:\n");
    dim3 sharedBlockSize(256);
    dim3 sharedGridSize((N + sharedBlockSize.x - 1) / sharedBlockSize.x);
    
    sharedMemoryExample<<<sharedGridSize, sharedBlockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");
    
    // Example 3: Constant memory
    printf("3. Constant Memory Example:\n");
    float h_coeffs[256];
    for (int i = 0; i < 256; i++) {
        h_coeffs[i] = 1.0f + (i * 0.1f);  // Different coefficients
    }
    
    // Copy coefficients to constant memory
    cudaMemcpyToSymbol(constCoeffs, h_coeffs, 256 * sizeof(float));
    
    constantMemoryExample<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");
    
    // Example 4: Register usage
    printf("4. Register Usage Example:\n");
    registerUsageExample<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");
    
    // Example 5: Coalesced access
    printf("5. Coalesced Memory Access Example:\n");
    coalescedAccess<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");
    
    // Example 6: Shared memory banking
    printf("6. Shared Memory Banking Example:\n");
    dim3 bankingBlockSize(32, 32);  // 2D block for matrix operations
    dim3 bankingGridSize((32 + 31) / 32, (32 + 31) / 32);  // For 32x32 tile
    
    sharedMemoryBanking<<<bankingGridSize, bankingBlockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("Tutorial completed!\n");
    return 0;
}