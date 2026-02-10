/*
 * CUDA Memory Fences and Synchronization Tutorial
 *
 * This tutorial demonstrates memory fences and synchronization primitives in CUDA.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Basic synchronization with __syncthreads
__global__ void basic_synchronization(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data cooperatively
    if (gid < n) {
        sdata[tid] = input[gid];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();  // Ensure all threads have loaded data
    
    // Process data - threads depend on each other's data
    float result = sdata[tid];
    if (tid > 0) {
        result += sdata[tid - 1];  // Use neighbor's data
    }
    
    __syncthreads();  // Ensure all computations complete before storing
    
    if (gid < n) {
        output[gid] = result;
    }
}

// Kernel 2: Producer-consumer pattern with memory fences
__global__ void producer_consumer_with_fences(int* buffer, int* flag, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Producer: fill buffer
        for (int i = 0; i < n; i++) {
            buffer[i] = i * 2;
        }
        
        // Ensure all writes to buffer complete before setting flag
        __threadfence();  // Release fence
        
        // Signal consumer that data is ready
        *flag = 1;
    }
    else if (tid == 1) {
        // Consumer: wait for data to be ready
        while (*flag == 0) {
            // Busy wait
        }
        
        // Acquire fence: ensure we see all producer's writes
        __threadfence();
        
        // Now safe to read buffer contents
        for (int i = 0; i < n; i++) {
            buffer[i] *= 3;  // Process the data
        }
    }
}

// Kernel 3: Atomic operations for synchronization
__global__ void atomic_synchronization(int* counters, int* results, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Atomically increment a counter
        int my_count = atomicAdd(&counters[0], 1);
        
        // Use the unique count value
        results[tid] = my_count;
    }
}

// Kernel 4: Reduction with proper synchronization
__global__ void synchronized_reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Synchronize after each step
    }
    
    // Write result to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel 5: Block-level fence example
__global__ void block_fence_example(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Modify data
        data[tid] = tid * 2;
        
        // Block-level fence: ensure memory operations are visible within block
        __threadfence_block();
        
        // Further operations that depend on the modification
        data[tid] += 1;
    }
}

// Kernel 6: Grid-level fence example
__global__ void grid_fence_example(int* counters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each block increments its counter
    atomicAdd(&counters[blockIdx.x], 1);
    
    // Block-level sync to ensure all threads in block complete increment
    __syncthreads();
    
    // Wait for all blocks to reach this point
    __threadfence();  // Ensure all atomic operations are visible globally
    
    // Now all blocks have incremented their counters
    if (tid == 0) {
        // Process global state
        int total = 0;
        for (int i = 0; i < gridDim.x; i++) {
            total += counters[i];
        }
        counters[gridDim.x] = total;  // Store total
    }
}

// Kernel 7: Demonstrating race condition without synchronization
__global__ void race_condition_example(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid > 0 && tid < n) {
        // This is unsafe without synchronization - race condition!
        // Reading data[tid-1] without ensuring it's written
        data[tid] += data[tid-1];
    }
}

// Kernel 8: Corrected version with proper synchronization
__global__ void corrected_sync_example(float* data, float* temp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        temp[tid] = data[tid];  // Copy to temporary location
    }
    __syncthreads();
    
    if (tid > 0 && tid < n) {
        data[tid] += temp[tid-1];  // Safe to read from temp
    }
}

int main() {
    printf("=== CUDA Memory Fences and Synchronization Tutorial ===\n\n");

    const int N = 1024;
    size_t size = N * sizeof(float);
    size_t int_size = N * sizeof(int);

    // Allocate host memory
    float *h_input, *h_output, *h_temp;
    int *h_counters, *h_results, *h_flag, *h_buffer;
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    h_temp = (float*)malloc(size);
    h_counters = (int*)malloc(int_size);
    h_results = (int*)malloc(int_size);
    h_flag = (int*)malloc(sizeof(int));
    h_buffer = (int*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
        h_buffer[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        h_counters[i] = 0;
        h_results[i] = 0;
    }
    *h_flag = 0;

    // Allocate device memory
    float *d_input, *d_output, *d_temp;
    int *d_counters, *d_results, *d_flag, *d_buffer;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_counters, int_size);
    cudaMalloc(&d_results, int_size);
    cudaMalloc(&d_flag, sizeof(int));
    cudaMalloc(&d_buffer, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer, h_buffer, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counters, h_counters, int_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Example 1: Basic synchronization
    printf("1. Basic Synchronization with __syncthreads:\n");
    basic_synchronization<<<4, 256, 256*sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");

    // Example 2: Producer-consumer with fences
    printf("2. Producer-Consumer with Memory Fences:\n");
    producer_consumer_with_fences<<<1, 32>>>(d_buffer, d_flag, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_buffer, d_buffer, size, cudaMemcpyDeviceToHost);
    printf("   First 10 buffer values after producer-consumer: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_buffer[i]);
    }
    printf("\n\n");

    // Example 3: Atomic operations
    printf("3. Atomic Operations for Synchronization:\n");
    h_counters[0] = 0;  // Reset counter
    cudaMemcpy(d_counters, h_counters, int_size, cudaMemcpyHostToDevice);
    
    atomic_synchronization<<<4, 256>>>(d_counters, d_results, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_counters, d_counters, int_size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_results, d_results, int_size, cudaMemcpyDeviceToHost);
    printf("   Counter value after atomics: %d\n", h_counters[0]);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_results[i]);
    }
    printf("\n\n");

    // Example 4: Synchronized reduction
    printf("4. Synchronized Reduction:\n");
    int numBlocks = (N + 255) / 256;
    int *h_reduction_results = (int*)malloc(numBlocks * sizeof(int));
    int *d_reduction_results;
    cudaMalloc(&d_reduction_results, numBlocks * sizeof(int));
    
    synchronized_reduction<<<numBlocks, 256, 256*sizeof(float)>>>(d_input, d_reduction_results, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_reduction_results, d_reduction_results, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Reduction results per block (first 5): ");
    for (int i = 0; i < 5 && i < numBlocks; i++) {
        printf("%d ", h_reduction_results[i]);
    }
    printf("\n\n");

    // Example 5: Block-level fence
    printf("5. Block-Level Fence Example:\n");
    block_fence_example<<<4, 256>>>(d_buffer, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_buffer, d_buffer, size, cudaMemcpyDeviceToHost);
    printf("   First 10 buffer values after block fence: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_buffer[i]);
    }
    printf("\n\n");

    // Example 6: Grid-level fence
    printf("6. Grid-Level Fence Example:\n");
    grid_fence_example<<<4, 256>>>(d_counters, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_counters, d_counters, int_size, cudaMemcpyDeviceToHost);
    printf("   Total across all blocks: %d\n", h_counters[4]);  // Assuming gridDim.x = 4
    printf("\n");

    // Example 7 & 8: Race condition demonstration
    printf("7. Race Condition vs Corrected Version:\n");
    // Reset data
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Race condition version (unsafe)
    race_condition_example<<<4, 256>>>(d_input, N);
    cudaDeviceSynchronize();
    printf("   Race condition kernel executed (unsafe pattern shown)\n");
    
    // Corrected version
    corrected_sync_example<<<4, 256>>>(d_input, d_temp, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_input, size, cudaMemcpyDeviceToHost);
    printf("   Corrected sync kernel executed\n");
    printf("   First 10 results after correction: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n\n");

    printf("Key Concepts Demonstrated:\n");
    printf("- __syncthreads() synchronizes all threads in a block\n");
    printf("- Memory fences ensure visibility of memory operations\n");
    printf("- Atomic operations provide thread-safe updates\n");
    printf("- Proper synchronization prevents race conditions\n");
    printf("- Different fence types provide different scope guarantees\n");

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_temp);
    free(h_counters);
    free(h_results);
    free(h_flag);
    free(h_buffer);
    free(h_reduction_results);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_counters);
    cudaFree(d_results);
    cudaFree(d_flag);
    cudaFree(d_buffer);
    cudaFree(d_reduction_results);

    printf("\nTutorial completed!\n");
    return 0;
}