/*
 * CUDA Register Pressure and Memory Model Tutorial
 *
 * This tutorial demonstrates register pressure concepts and GPU memory model behavior.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: High register usage example
__global__ void high_register_usage(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Use many variables to increase register pressure
        float a = data[tid];
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
        
        data[tid] = a + b + c + d + e + f + g + h + i + j + k + l;
    }
}

// Kernel 2: Low register usage example
__global__ void low_register_usage(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];
        val = val * 2.0f;
        val = val + 1.0f;
        val = val * data[tid];
        val = val - (data[tid] * 2.0f);
        data[tid] = val;
    }
}

// Kernel 3: Controlled register usage with launch bounds
__global__
__launch_bounds__(256, 4)  // Up to 256 threads per block, at least 4 blocks per SM
void controlled_registers(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float a = data[tid];
        float b = a * 2.0f;
        data[tid] = a + b;
    }
}

// Kernel 4: Memory model example with proper synchronization
__global__ void memory_model_example(int* data, int* flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        data[0] = 42;          // Write data
        __threadfence();        // Ensure visibility
        flag[0] = 1;           // Signal other threads
    }
    
    if (tid == 1) {
        while (flag[0] == 0) { /* wait */ }
        __threadfence();       // Ensure we see the data write
        int val = data[0];     // Now guaranteed to see 42
        data[1] = val;
    }
}

// Kernel 5: Atomic operations to prevent race conditions
__global__ void atomic_example(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Atomic operation to prevent race conditions
        atomicAdd(&data[0], tid);
    }
}

// Kernel 6: Shared memory vs register usage comparison
__global__ void shared_vs_register(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    // Using shared memory instead of many registers
    if (tid < n) {
        sdata[local_tid] = input[tid] * 2.0f;
        __syncthreads();
        
        // Process in shared memory
        sdata[local_tid] += 1.0f;
        __syncthreads();
        
        output[tid] = sdata[local_tid];
    }
}

// Kernel 7: Demonstrating false sharing concept
__global__ void false_sharing_example(int* data) {
    int tid = threadIdx.x;
    
    // This could cause false sharing if adjacent threads access adjacent memory
    data[tid]++;  // Each thread accesses different element, no false sharing here
}

// Kernel 8: Proper synchronization for memory ordering
__global__ void proper_sync_example(int* buffer, int* ready_flag, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Write to buffer
        buffer[tid] = tid * 2;
    }
    
    __syncthreads();  // All threads in block finish writing
    
    if (tid == 0) {
        // Signal that buffer is ready
        __threadfence();      // Ensure all writes are visible
        *ready_flag = 1;
    }
}

// Helper function to measure occupancy and register usage
void analyzeKernel(void (*kernel)(float*, int), const char* kernelName, int blockSize) {
    struct cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel);
    
    printf("%s:\n", kernelName);
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Constant memory: %zu bytes\n", attr.constSizeBytes);
    
    // Calculate occupancy
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, (const void*)kernel, blockSize, attr.sharedSizeBytes);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int maxActiveBlocks = prop.maxThreadsPerMultiProcessor / blockSize;
    float occupancy = (float)activeBlocks / maxActiveBlocks;
    
    printf("  Active blocks per SM: %d\n", activeBlocks);
    printf("  Occupancy: %.2f%%\n", occupancy * 100);
    printf("\n");
}

int main() {
    printf("=== CUDA Register Pressure and Memory Model Tutorial ===\n\n");

    const int N = 1024;
    size_t size = N * sizeof(float);
    size_t int_size = N * sizeof(int);

    // Allocate host memory
    float *h_input, *h_output1, *h_output2, *h_output3, *h_output4;
    int *h_data, *h_flag, *h_buffer;
    h_input = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    h_output3 = (float*)malloc(size);
    h_output4 = (float*)malloc(size);
    h_data = (int*)malloc(int_size);
    h_flag = (int*)malloc(sizeof(int));
    h_buffer = (int*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
        h_data[i] = 0;
    }
    *h_flag = 0;

    // Allocate device memory
    float *d_input, *d_output1, *d_output2, *d_output3, *d_output4;
    int *d_data, *d_flag, *d_buffer;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    cudaMalloc(&d_output3, size);
    cudaMalloc(&d_output4, size);
    cudaMalloc(&d_data, int_size);
    cudaMalloc(&d_flag, sizeof(int));
    cudaMalloc(&d_buffer, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, int_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Analyze different kernels for register usage and occupancy
    printf("1. Register Usage Analysis:\n");
    analyzeKernel((void (*)(float*, int))high_register_usage, "High Register Usage", 256);
    analyzeKernel((void (*)(float*, int))low_register_usage, "Low Register Usage", 256);
    analyzeKernel((void (*)(float*, int))controlled_registers, "Controlled Registers", 256);

    // Example 1: High register usage
    printf("2. High Register Usage Kernel:\n");
    high_register_usage<<<(N + 255) / 256, 256>>>(d_input, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output1, d_input, size, cudaMemcpyDeviceToHost);
    printf("   Completed high register usage kernel\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n\n");

    // Reset input for next test
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Example 2: Low register usage
    printf("3. Low Register Usage Kernel:\n");
    low_register_usage<<<(N + 255) / 256, 256>>>(d_input, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output2, d_input, size, cudaMemcpyDeviceToHost);
    printf("   Completed low register usage kernel\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output2[i]);
    }
    printf("\n\n");

    // Reset input for next test
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Example 3: Controlled register usage
    printf("4. Controlled Register Usage Kernel:\n");
    controlled_registers<<<(N + 255) / 256, 256>>>(d_input, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output3, d_input, size, cudaMemcpyDeviceToHost);
    printf("   Completed controlled register usage kernel\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output3[i]);
    }
    printf("\n\n");

    // Example 4: Memory model with synchronization
    printf("5. Memory Model with Synchronization:\n");
    memory_model_example<<<1, 32>>>(d_buffer, d_flag);
    cudaDeviceSynchronize();
    cudaMemcpy(h_buffer, d_buffer, int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Buffer[0] = %d, Buffer[1] = %d, Flag = %d\n", h_buffer[0], h_buffer[1], *h_flag);
    printf("\n");

    // Example 5: Atomic operations
    printf("6. Atomic Operations Example:\n");
    h_data[0] = 0;  // Reset counter
    cudaMemcpy(d_data, h_data, int_size, cudaMemcpyHostToDevice);
    
    atomic_example<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, int_size, cudaMemcpyDeviceToHost);
    printf("   Atomic sum result: %d\n", h_data[0]);
    printf("\n");

    // Example 6: Shared vs register usage
    printf("7. Shared Memory vs Register Usage:\n");
    int sharedMemSize = 256 * sizeof(float);  // For 256 threads
    shared_vs_register<<<(N + 255) / 256, 256, sharedMemSize>>>(d_input, d_output4, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output4, d_output4, size, cudaMemcpyDeviceToHost);
    printf("   Shared memory approach completed\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output4[i]);
    }
    printf("\n\n");

    // Example 8: Proper synchronization
    printf("8. Proper Synchronization Example:\n");
    int ready_flag_val = 0;
    int *d_ready_flag;
    cudaMalloc(&d_ready_flag, sizeof(int));
    cudaMemcpy(d_ready_flag, &ready_flag_val, sizeof(int), cudaMemcpyHostToDevice);
    
    proper_sync_example<<<4, 256>>>(d_buffer, d_ready_flag, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&ready_flag_val, d_ready_flag, sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Ready flag after synchronization: %d\n", ready_flag_val);
    
    cudaFree(d_ready_flag);
    printf("\n");

    printf("Key Concepts Demonstrated:\n");
    printf("- Register pressure affects occupancy and performance\n");
    printf("- High register usage can cause spilling to local memory\n");
    printf("- Proper synchronization is needed for memory ordering\n");
    printf("- Atomic operations prevent race conditions\n");
    printf("- Shared memory can reduce register pressure\n");

    // Cleanup
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    free(h_output4);
    free(h_data);
    free(h_flag);
    free(h_buffer);

    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_data);
    cudaFree(d_flag);
    cudaFree(d_buffer);

    printf("\nTutorial completed!\n");
    return 0;
}