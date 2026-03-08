/*
 * Warp-Level Primitives Hands-On Exercise
 *
 * Complete the kernel using warp-level primitives like shuffle operations.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Warp Shuffle Operations - STUDENT EXERCISE
__global__ void warpShuffleExample(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;  // Thread's position within its warp
    
    if (tid < n) {
        float value = input[tid];
        
        // TODO: Use warp shuffle to get value from next thread in warp
        // Hint: Use __shfl_down_sync or __shfl_sync
        // Example: float nextValue = __shfl_down_sync(0xFFFFFFFF, value, 1, 32);
        float neighborValue = /* YOUR SHUFFLE CODE */;
        
        // Store the neighbor's value
        output[tid] = neighborValue;
    }
}

// Kernel: Warp Vote Operations - STUDENT EXERCISE
__global__ void warpVoteExample(int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Check if the value is greater than threshold
        bool isGreater = (input[tid] > 50);
        
        // TODO: Use warp vote to check if ANY thread in the warp has isGreater = true
        // Hint: Use __any_sync
        int anyGreater = /* YOUR VOTE CODE */;
        
        // TODO: Use warp vote to check if ALL threads in the warp have isGreater = true
        // Hint: Use __all_sync
        int allGreater = /* YOUR VOTE CODE */;
        
        // Store results
        output[tid] = anyGreater * 100 + allGreater * 10;
    }
}

// Utility function to initialize array
void initArray(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i % 100;  // Values from 0 to 99
    }
}

// Utility function to initialize float array
void initFloatArray(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 1.0f;
    }
}

int main() {
    printf("=== Warp-Level Primitives Hands-On Exercise ===\n");
    printf("Complete the missing code sections in the warp primitive kernels.\n\n");

    // Setup for warp primitives exercise
    const int N = 1024;
    size_t size_int = N * sizeof(int);
    size_t size_float = N * sizeof(float);
    
    // Allocate host memory
    float *h_input_f, *h_output_f;
    int *h_input_i, *h_output_i;
    
    h_input_f = (float*)malloc(size_float);
    h_output_f = (float*)malloc(size_float);
    h_input_i = (int*)malloc(size_int);
    h_output_i = (int*)malloc(size_int);
    
    // Initialize input arrays
    initFloatArray(h_input_f, N);
    initArray(h_input_i, N);
    
    // Allocate device memory
    float *d_input_f, *d_output_f;
    int *d_input_i, *d_output_i;
    
    cudaMalloc(&d_input_f, size_float);
    cudaMalloc(&d_output_f, size_float);
    cudaMalloc(&d_input_i, size_int);
    cudaMalloc(&d_output_i, size_int);
    
    // Copy inputs to device
    cudaMemcpy(d_input_f, h_input_f, size_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_i, h_input_i, size_int, cudaMemcpyHostToDevice);
    
    // Launch warp shuffle kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // This will fail to compile until you complete the kernel
    warpShuffleExample<<<gridSize, blockSize>>>(d_input_f, d_output_f, N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Warp shuffle kernel execution failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Launch warp vote kernel
    warpVoteExample<<<gridSize, blockSize>>>(d_input_i, d_output_i, N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Warp vote kernel execution failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_f, d_output_f, size_float, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_i, d_output_i, size_int, cudaMemcpyDeviceToHost);
    
    // Print some results
    printf("Warp shuffle results - First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output_f[i]);
    }
    printf("\n");
    
    printf("Warp vote results - First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_output_i[i]);
    }
    printf("\n");
    
    // Cleanup
    free(h_input_f); free(h_output_f);
    free(h_input_i); free(h_output_i);
    cudaFree(d_input_f); cudaFree(d_output_f);
    cudaFree(d_input_i); cudaFree(d_output_i);
    
    printf("\nExercise completed! Try the other hands-on exercises.\n");
    
    return 0;
}