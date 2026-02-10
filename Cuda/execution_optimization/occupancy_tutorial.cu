/*
 * CUDA Occupancy Tutorial
 * 
 * This tutorial demonstrates occupancy concepts and optimization techniques.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: High register usage (low occupancy)
__global__ void high_registers(float* data, int n) {
    // Use many variables to increase register usage
    float a = data[threadIdx.x + blockIdx.x * blockDim.x];
    float b = a * 2.0f;
    float c = b + 1.0f;
    float d = c * a;
    float e = d - b;
    float f = e * 0.5f;
    float g = f + a;
    float h = g * b;
    float i = h - c;
    float j = i * d;
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = a + b + c + d + e + f + g + h + i + j;
    }
}

// Kernel 2: Low register usage (high occupancy)
__global__ void low_registers(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = data[idx];
        val = val * 2.0f;
        val = val + 1.0f;
        val = val * data[idx];
        val = val - (data[idx] * 2.0f);
        data[idx] = val;
    }
}

// Kernel 3: Controlled register usage with launch bounds
__global__ 
__launch_bounds__(256, 4)  // Up to 256 threads per block, at least 4 blocks per SM
void controlled_registers(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float a = data[idx];
        float b = a * 2.0f;
        data[idx] = a + b;
    }
}

// Kernel 4: Balanced resource usage
__global__ void balanced_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    // Use moderate amount of shared memory to allow more blocks per SM
    if (tid < n) {
        sdata[local_tid] = input[tid];
        __syncthreads();
        
        // Process data
        sdata[local_tid] *= 2.0f;
        __syncthreads();
        
        output[tid] = sdata[local_tid];
    }
}

// Kernel 5: Occupancy calculation helper
__global__ void occupancy_test_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// Function to print occupancy information
void printOccupancyInfo(void (*kernel)(float*, int), int blockSize, const char* kernelName) {
    int device;
    cudaGetDevice(&device);
    
    int minGridSize, blockSizeFromFunc;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeFromFunc, (const void*)kernel, 0, 0);
    
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, (const void*)kernel, blockSize, 0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int maxActiveBlocks = prop.maxThreadsPerMultiProcessor / blockSize;
    float occupancy = (float)activeBlocks / maxActiveBlocks;
    
    printf("%s:\n", kernelName);
    printf("  Block size: %d\n", blockSize);
    printf("  Active blocks per SM: %d\n", activeBlocks);
    printf("  Max possible blocks per SM: %d\n", maxActiveBlocks);
    printf("  Occupancy: %.2f%%\n", occupancy * 100);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("\n");
}

int main() {
    printf("=== CUDA Occupancy Tutorial ===\n\n");
    
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_data1, *h_data2, *h_data3, *h_data4, *h_data5;
    h_data1 = (float*)malloc(size);
    h_data2 = (float*)malloc(size);
    h_data3 = (float*)malloc(size);
    h_data4 = (float*)malloc(size);
    h_data5 = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_data1[i] = i * 1.0f;
        h_data2[i] = i * 1.0f;
        h_data3[i] = i * 1.0f;
        h_data4[i] = i * 1.0f;
        h_data5[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_data1, *d_data2, *d_data3, *d_data4, *d_data5;
    cudaMalloc(&d_data1, size);
    cudaMalloc(&d_data2, size);
    cudaMalloc(&d_data3, size);
    cudaMalloc(&d_data4, size);
    cudaMalloc(&d_data5, size);
    
    // Copy input data to device
    cudaMemcpy(d_data1, h_data1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data3, h_data3, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data4, h_data4, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data5, h_data5, size, cudaMemcpyHostToDevice);
    
    // Print occupancy information for different kernels
    printf("Occupancy Analysis:\n");
    printOccupancyInfo((void (*)(float*, int))high_registers, 256, "High Register Usage");
    printOccupancyInfo((void (*)(float*, int))low_registers, 256, "Low Register Usage");
    printOccupancyInfo((void (*)(float*, int))controlled_registers, 256, "Controlled Registers (Launch Bounds)");
    printOccupancyInfo((void (*)(float*, int))occupancy_test_kernel, 128, "Standard Kernel (128 threads)");
    printOccupancyInfo((void (*)(float*, int))occupancy_test_kernel, 256, "Standard Kernel (256 threads)");
    printOccupancyInfo((void (*)(float*, int))occupancy_test_kernel, 512, "Standard Kernel (512 threads)");
    
    // Example 1: High register usage
    printf("1. High Register Usage Kernel:\n");
    high_registers<<<(N + 255) / 256, 256>>>(d_data1, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data1, d_data1, size, cudaMemcpyDeviceToHost);
    printf("   Completed with high register usage (lower occupancy)\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data1[i]);
    }
    printf("\n\n");
    
    // Example 2: Low register usage
    printf("2. Low Register Usage Kernel:\n");
    low_registers<<<(N + 255) / 256, 256>>>(d_data2, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data2, d_data2, size, cudaMemcpyDeviceToHost);
    printf("   Completed with low register usage (higher occupancy)\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data2[i]);
    }
    printf("\n\n");
    
    // Example 3: Controlled register usage with launch bounds
    printf("3. Controlled Register Usage (Launch Bounds):\n");
    controlled_registers<<<(N + 255) / 256, 256>>>(d_data3, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data3, d_data3, size, cudaMemcpyDeviceToHost);
    printf("   Completed with launch bounds directive\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data3[i]);
    }
    printf("\n\n");
    
    // Example 4: Balanced resource usage with shared memory
    printf("4. Balanced Resource Usage (Shared Memory):\n");
    int sharedMemSize = 256 * sizeof(float);  // 256 threads * sizeof(float)
    balanced_kernel<<<(N + 255) / 256, 256, sharedMemSize>>>(d_data4, d_data4, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data4, d_data4, size, cudaMemcpyDeviceToHost);
    printf("   Completed with balanced resource usage\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data4[i]);
    }
    printf("\n\n");
    
    // Example 5: Occupancy optimization demonstration
    printf("5. Occupancy Optimization Demonstration:\n");
    printf("   Testing different block sizes for optimal performance:\n");
    
    int blockSizes[] = {64, 128, 256, 512};
    int numSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    for (int i = 0; i < numSizes; i++) {
        int blockSize = blockSizes[i];
        
        // Calculate occupancy for this block size
        int activeBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, 
                                                     (const void*)occupancy_test_kernel, 
                                                     blockSize, 0);
        
        // Time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        occupancy_test_kernel<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_data5, N);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        printf("   BlockSize: %d, Occupancy: %d blocks/SM, Time: %.3f ms\n", 
               blockSize, activeBlocks, milliseconds);
    }
    printf("\n");
    
    // Show how to programmatically get register usage
    printf("6. Getting Register Usage Information:\n");
    struct cudaFuncAttributes attr1, attr2;
    cudaFuncGetAttributes(&attr1, high_registers);
    cudaFuncGetAttributes(&attr2, low_registers);
    
    printf("   High register kernel: %d registers per thread\n", attr1.numRegs);
    printf("   Low register kernel: %d registers per thread\n", attr2.numRegs);
    printf("   Shared memory per block: %zu bytes\n", attr1.sharedSizeBytes);
    printf("   Constant memory usage: %zu bytes\n", attr1.constSizeBytes);
    printf("\n");
    
    // Cleanup
    free(h_data1);
    free(h_data2);
    free(h_data3);
    free(h_data4);
    free(h_data5);
    
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    cudaFree(d_data4);
    cudaFree(d_data5);
    
    printf("Tutorial completed!\n");
    printf("\nKey Takeaways:\n");
    printf("- Occupancy = Active Warps / Max Warps Per SM\n");
    printf("- Higher occupancy helps hide memory latency\n");
    printf("- But maximum occupancy doesn't always mean maximum performance\n");
    printf("- Balance occupancy with computational efficiency\n");
    
    return 0;
}