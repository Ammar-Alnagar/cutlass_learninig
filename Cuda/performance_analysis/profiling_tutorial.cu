/*
 * Profiling with Nsight Compute Tutorial
 * 
 * This tutorial demonstrates how to profile CUDA applications using Nsight Compute.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Low occupancy example
__global__ void low_occupancy_kernel(float* data, int n) {
    // Use many registers to reduce occupancy
    float r0 = data[threadIdx.x + blockIdx.x * blockDim.x];
    float r1 = r0 * 1.1f; float r2 = r1 * 1.1f; float r3 = r2 * 1.1f;
    float r4 = r3 * 1.1f; float r5 = r4 * 1.1f; float r6 = r5 * 1.1f;
    float r7 = r6 * 1.1f; float r8 = r7 * 1.1f; float r9 = r8 * 1.1f;
    float r10 = r9 * 1.1f; float r11 = r10 * 1.1f; float r12 = r11 * 1.1f;
    float r13 = r12 * 1.1f; float r14 = r13 * 1.1f; float r15 = r14 * 1.1f;
    float r16 = r15 * 1.1f; float r17 = r16 * 1.1f; float r18 = r17 * 1.1f;
    float r19 = r18 * 1.1f; float r20 = r19 * 1.1f; float r21 = r20 * 1.1f;
    float r22 = r21 * 1.1f; float r23 = r22 * 1.1f; float r24 = r23 * 1.1f;
    float r25 = r24 * 1.1f; float r26 = r25 * 1.1f; float r27 = r26 * 1.1f;
    float r28 = r27 * 1.1f; float r29 = r28 * 1.1f; float r30 = r29 * 1.1f;
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 +
                    r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 +
                    r20 + r21 + r22 + r23 + r24 + r25 + r26 + r27 + r28 + r29 + r30;
    }
}

// Kernel 2: High occupancy example
__global__ void high_occupancy_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Kernel 3: Memory-bound example (uncoalesced access)
__global__ void uncoalesced_kernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Uncoalesced access pattern: stride of 32
        int access_idx = (idx * 32) % n;
        output[idx] = input[access_idx] * 2.0f;
    }
}

// Kernel 4: Compute-bound example
__global__ void compute_bound_kernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = input[idx];
        // Perform many computations per memory access
        for (int i = 0; i < 100; i++) {
            val = val * 1.01f + 0.01f;
            val = val * val + 0.001f;
        }
        output[idx] = val;
    }
}

// Kernel 5: Example with shared memory bank conflicts
__global__ void bank_conflict_kernel(float* input, float* output, int n) {
    __shared__ float sdata[1024];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    if (tid < n) {
        // Load to shared memory
        sdata[local_tid] = input[tid];
        __syncthreads();
        
        // Create bank conflicts by accessing same bank from multiple threads
        // Every 32nd thread accesses the same bank
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            sum += sdata[local_tid % 32];  // All threads access same bank
        }
        
        output[tid] = sum;
    }
}

// Function to print occupancy information
void print_occupancy_info(const char* kernel_name, void (*kernel)(float*, int)) {
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (const void*)kernel, 0, 0);
    
    int active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, (const void*)kernel, block_size, 0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int max_blocks = prop.maxThreadsPerMultiProcessor / block_size;
    float occupancy = (float)active_blocks / max_blocks;
    
    printf("%s:\n", kernel_name);
    printf("  Block size: %d\n", block_size);
    printf("  Active blocks per SM: %d\n", active_blocks);
    printf("  Max possible blocks per SM: %d\n", max_blocks);
    printf("  Occupancy: %.2f%%\n", occupancy * 100);
    
    struct cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel);
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    printf("\n");
}

int main() {
    printf("=== Profiling with Nsight Compute Tutorial ===\n\n");
    
    const int N = 1024 * 1024;  // 1M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_data1, *h_data2, *h_data3, *h_data4, *h_data5;
    h_data1 = (float*)malloc(size);
    h_data2 = (float*)malloc(size);
    h_data3 = (float*)malloc(size);
    h_data4 = (float*)malloc(size);
    h_data5 = (float*)malloc(size);
    
    // Initialize data
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
    
    // Copy data to device
    cudaMemcpy(d_data1, h_data1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data3, h_data3, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data4, h_data4, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data5, h_data5, size, cudaMemcpyHostToDevice);
    
    // Print occupancy information for different kernels
    printf("Occupancy Analysis:\n");
    print_occupancy_info("Low Occupancy Kernel", (void (*)(float*, int))low_occupancy_kernel);
    print_occupancy_info("High Occupancy Kernel", (void (*)(float*, int))high_occupancy_kernel);
    
    // Run kernels
    printf("Running kernels for profiling...\n\n");
    
    // Example 1: Low occupancy kernel
    printf("1. Low Occupancy Kernel:\n");
    printf("   This kernel uses many registers, reducing occupancy.\n");
    low_occupancy_kernel<<<(N + 255) / 256, 256>>>(d_data1, N);
    cudaDeviceSynchronize();
    printf("   Completed.\n\n");
    
    // Example 2: High occupancy kernel
    printf("2. High Occupancy Kernel:\n");
    printf("   This kernel uses few registers, allowing high occupancy.\n");
    high_occupancy_kernel<<<(N + 255) / 256, 256>>>(d_data2, N);
    cudaDeviceSynchronize();
    printf("   Completed.\n\n");
    
    // Example 3: Uncoalesced memory access
    printf("3. Uncoalesced Memory Access Kernel:\n");
    printf("   This kernel demonstrates poor memory access patterns.\n");
    uncoalesced_kernel<<<(N + 255) / 256, 256>>>(d_data3, d_data3, N);
    cudaDeviceSynchronize();
    printf("   Completed.\n\n");
    
    // Example 4: Compute-bound kernel
    printf("4. Compute-Bound Kernel:\n");
    printf("   This kernel performs many computations per memory access.\n");
    compute_bound_kernel<<<(N + 255) / 256, 256>>>(d_data4, d_data4, N);
    cudaDeviceSynchronize();
    printf("   Completed.\n\n");
    
    // Example 5: Bank conflict kernel
    printf("5. Shared Memory Bank Conflict Kernel:\n");
    printf("   This kernel demonstrates shared memory bank conflicts.\n");
    bank_conflict_kernel<<<(N + 255) / 256, 256>>>(d_data5, d_data5, N);
    cudaDeviceSynchronize();
    printf("   Completed.\n\n");
    
    printf("Profiling Instructions:\n");
    printf("=======================\n");
    printf("To profile these kernels with Nsight Compute, use commands like:\n\n");
    
    printf("1. Basic profiling:\n");
    printf("   ncu --set full ./profiling_tutorial\n\n");
    
    printf("2. Specific metrics:\n");
    printf("   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,achieved_occupancy ./profiling_tutorial\n\n");
    
    printf("3. Memory-focused analysis:\n");
    printf("   ncu --set memory ./profiling_tutorial\n\n");
    
    printf("4. Compute-focused analysis:\n");
    printf("   ncu --set compute ./profiling_tutorial\n\n");
    
    printf("5. Compare different kernels:\n");
    printf("   ncu --kernel-name \"low_occupancy_kernel\" ./profiling_tutorial\n");
    printf("   ncu --kernel-name \"high_occupancy_kernel\" ./profiling_tutorial\n\n");
    
    printf("Key Metrics to Monitor:\n");
    printf("- sm__throughput.avg.pct_of_peak_sustained_elapsed: GPU utilization\n");
    printf("- achieved_occupancy: Thread occupancy\n");
    printf("- dram__throughput: Memory bandwidth utilization\n");
    printf("- gld_efficiency, gst_efficiency: Memory access efficiency\n");
    printf("- shared_replay_overhead: Shared memory bank conflicts\n");
    printf("- branch_efficiency: Divergent branching efficiency\n\n");
    
    printf("Interpreting Results:\n");
    printf("- Low occupancy kernels: Optimize register usage\n");
    printf("- Low memory efficiency: Improve access patterns\n");
    printf("- High bank conflict overhead: Reorganize shared memory access\n");
    printf("- Low branch efficiency: Reduce divergent branching\n\n");
    
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
    printf("Run this program with Nsight Compute to analyze the performance characteristics.\n");
    
    return 0;
}