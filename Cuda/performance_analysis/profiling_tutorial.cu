/*
 * CUDA Profiling with Nsight Compute Tutorial
 *
 * This tutorial demonstrates how to profile CUDA kernels using Nsight Compute.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Kernel 1: Memory-bound example for profiling
__global__ void memory_bound_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple operation with lots of memory access
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// Kernel 2: Compute-bound example for profiling
__global__ void compute_bound_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        
        // Many operations per memory access
        for (int i = 0; i < 50; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
            x = x * 2.0f + 1.0f;
        }
        
        output[idx] = x;
    }
}

// Kernel 3: Example with shared memory for profiling
__global__ void shared_memory_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data cooperatively
    if (gid < n) {
        sdata[tid] = input[gid];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();  // Synchronize after loading
    
    // Process data in shared memory
    float result = sdata[tid];
    if (tid > 0) {
        result += sdata[tid - 1];  // Use neighbor's data
    }
    
    __syncthreads();  // Synchronize before storing
    
    if (gid < n) {
        output[gid] = result;
    }
}

// Kernel 4: Uncoalesced access pattern for profiling
__global__ void uncoalesced_kernel(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Uncoalesced access pattern - stride access
        int access_idx = idx * stride;
        if (access_idx < n * stride) {
            output[idx] = input[access_idx] * 2.0f;
        }
    }
}

// Kernel 5: High occupancy vs low occupancy comparison
__global__ void high_occupancy_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple operation with minimal register usage
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel 6: Low occupancy kernel (high register usage)
__global__ void low_occupancy_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use many registers to reduce occupancy
        float a = input[idx];
        float b = a * 2.0f;
        float c = b + 1.0f;
        float d = c * a;
        float e = d - b;
        float f = e * 0.5f;
        float g = f + a;
        float h = g * b;
        float i = h - c;
        float j = i * d;
        float k = j + e;
        float l = k * f;
        
        output[idx] = a + b + c + d + e + f + g + h + i + j + k + l;
    }
}

// Helper function to measure execution time
float measureKernelTime(void (*kernel)(float*, float*, int), float* input, float* output, int n, size_t sharedMemSize = 0) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (sharedMemSize > 0) {
        kernel<<<(n + 255) / 256, 256, sharedMemSize>>>(input, output, n);
    } else {
        kernel<<<(n + 255) / 256, 256>>>(input, output, n);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0f;  // Return seconds
}

// Function to print profiling guidance
void printProfilingGuidance() {
    printf("\nNsight Compute Profiling Guidance:\n");
    printf("=========================\n");
    printf("To profile these kernels with Nsight Compute, use commands like:\n\n");
    
    printf("Basic profiling:\n");
    printf("  ncu ./profiling_tutorial\n\n");
    
    printf("Profile specific metrics:\n");
    printf("  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gmem__throughput.avg.pct_of_peak_sustained_elapsed ./profiling_tutorial\n\n");
    
    printf("Profile with source correlation:\n");
    printf("  ncu --source ./profiling_tutorial\n\n");
    
    printf("Profile specific kernel:\n");
    printf("  ncu --kernel-name \"memory_bound_kernel\" ./profiling_tutorial\n\n");
    
    printf("Advanced metrics for memory analysis:\n");
    printf("  ncu --metrics gld_efficiency,gst_efficiency,achieved_occupancy,sm__warps_launched_rate.avg ./profiling_tutorial\n\n");
    
    printf("Key metrics to monitor:\n");
    printf("- sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization\n");
    printf("- gmem__throughput.avg.pct_of_peak_sustained_elapsed: Global memory utilization\n");
    printf("- achieved_occupancy: Thread occupancy\n");
    printf("- gld_efficiency/gst_efficiency: Memory access efficiency\n");
    printf("- dram__bytes_per_second.sum: Memory bandwidth\n");
    printf("- smsp__warp_issue_stalled_*: Warp stall reasons\n\n");
}

int main() {
    printf("=== CUDA Profiling with Nsight Compute Tutorial ===\n\n");

    const int N = 1024 * 1024;  // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input, *h_output1, *h_output2, *h_output3, *h_output4, *h_output5, *h_output6;
    h_input = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    h_output3 = (float*)malloc(size);
    h_output4 = (float*)malloc(size);
    h_output5 = (float*)malloc(size);
    h_output6 = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    // Allocate device memory
    float *d_input, *d_output1, *d_output2, *d_output3, *d_output4, *d_output5, *d_output6;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    cudaMalloc(&d_output3, size);
    cudaMalloc(&d_output4, size);
    cudaMalloc(&d_output5, size);
    cudaMalloc(&d_output6, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Example 1: Memory-bound kernel
    printf("1. Memory-Bound Kernel Execution:\n");
    float time_mem_bound = measureKernelTime(memory_bound_kernel, d_input, d_output1, N);
    cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    printf("   Execution time: %.6f seconds\n", time_mem_bound);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n\n");

    // Example 2: Compute-bound kernel
    printf("2. Compute-Bound Kernel Execution:\n");
    float time_compute_bound = measureKernelTime(compute_bound_kernel, d_input, d_output2, N);
    cudaMemcpy(h_output2, d_output2, size, cudaMemcpyDeviceToHost);
    printf("   Execution time: %.6f seconds\n", time_compute_bound);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output2[i]);
    }
    printf("\n\n");

    // Example 3: Shared memory kernel
    printf("3. Shared Memory Kernel Execution:\n");
    size_t sharedMemSize = 256 * sizeof(float);  // For 256 threads
    float time_shared = measureKernelTime(shared_memory_kernel, d_input, d_output3, N, sharedMemSize);
    cudaMemcpy(h_output3, d_output3, size, cudaMemcpyDeviceToHost);
    printf("   Execution time: %.6f seconds\n", time_shared);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output3[i]);
    }
    printf("\n\n");

    // Example 4: Uncoalesced access kernel
    printf("4. Uncoalesced Access Kernel Execution:\n");
    float time_uncoalesced = measureKernelTime(uncoalesced_kernel, d_input, d_output4, N/2, 2);
    cudaMemcpy(h_output4, d_output4, size, cudaMemcpyDeviceToHost);
    printf("   Execution time: %.6f seconds\n", time_uncoalesced);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output4[i]);
    }
    printf("\n\n");

    // Example 5: High occupancy kernel
    printf("5. High Occupancy Kernel Execution:\n");
    float time_high_occ = measureKernelTime(high_occupancy_kernel, d_input, d_output5, N);
    cudaMemcpy(h_output5, d_output5, size, cudaMemcpyDeviceToHost);
    printf("   Execution time: %.6f seconds\n", time_high_occ);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output5[i]);
    }
    printf("\n\n");

    // Example 6: Low occupancy kernel
    printf("6. Low Occupancy Kernel Execution:\n");
    float time_low_occ = measureKernelTime(low_occupancy_kernel, d_input, d_output6, N);
    cudaMemcpy(h_output6, d_output6, size, cudaMemcpyDeviceToHost);
    printf("   Execution time: %.6f seconds\n", time_low_occ);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output6[i]);
    }
    printf("\n\n");

    // Performance comparison
    printf("Performance Comparison:\n");
    printf("- Memory-bound kernel: %.6f seconds\n", time_mem_bound);
    printf("- Compute-bound kernel: %.6f seconds\n", time_compute_bound);
    printf("- Shared memory kernel: %.6f seconds\n", time_shared);
    printf("- Uncoalesced access kernel: %.6f seconds\n", time_uncoalesced);
    printf("- High occupancy kernel: %.6f seconds\n", time_high_occ);
    printf("- Low occupancy kernel: %.6f seconds\n", time_low_occ);
    printf("\n");

    // Print profiling guidance
    printProfilingGuidance();

    printf("Key Profiling Concepts Demonstrated:\n");
    printf("- Memory-bound vs compute-bound kernels\n");
    printf("- Impact of shared memory on performance\n");
    printf("- Effect of memory access patterns\n");
    printf("- Influence of occupancy on performance\n");
    printf("- Register pressure effects\n");

    // Cleanup
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    free(h_output4);
    free(h_output5);
    free(h_output6);

    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    cudaFree(d_output6);

    printf("\nTutorial completed!\n");
    return 0;
}