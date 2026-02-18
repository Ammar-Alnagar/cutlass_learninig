#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive gather operation: output[i] = input[indices[i]]
__global__ void gather_naive(float *input, int *indices, float *output, int n, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int index = indices[idx];
        if (index >= 0 && index < input_size) {
            output[idx] = input[index];
        } else {
            output[idx] = 0.0f;  // Out of bounds
        }
    }
}

// Naive scatter operation: output[indices[i]] = input[i]
__global__ void scatter_naive(float *input, int *indices, float *output, int n, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int index = indices[idx];
        if (index >= 0 && index < output_size) {
            output[index] = input[idx];
        }
    }
}

// Scatter with atomic operations to handle collisions
__global__ void scatter_atomic_naive(float *input, int *indices, float *output, int n, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int index = indices[idx];
        if (index >= 0 && index < output_size) {
            atomicAdd(&output[index], input[idx]);
        }
    }
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

// Timing utility
double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

// CPU gather for verification
void cpu_gather(float *input, int *indices, float *output, int n, int input_size) {
    for (int i = 0; i < n; i++) {
        int index = indices[i];
        if (index >= 0 && index < input_size) {
            output[i] = input[index];
        } else {
            output[i] = 0.0f;  // Out of bounds
        }
    }
}

// CPU scatter for verification
void cpu_scatter(float *input, int *indices, float *output, int n, int output_size) {
    // Initialize output to zero
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
    }
    
    for (int i = 0; i < n; i++) {
        int index = indices[i];
        if (index >= 0 && index < output_size) {
            output[index] = input[i];
        }
    }
}

int main(int argc, char **argv) {
    int n = 1 << 18; // 256K elements by default
    int input_size = 1 << 20; // 1M elements for input array
    
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        input_size = atoi(argv[2]);
    }
    
    printf("Gather/Scatter operations: %d elements, input size %d\n", n, input_size);
    
    size_t input_size_bytes = input_size * sizeof(float);
    size_t output_size_bytes = n * sizeof(float);
    size_t indices_size_bytes = n * sizeof(int);
    
    // Host arrays
    float *h_input = (float*)malloc(input_size_bytes);
    int *h_indices = (int*)malloc(indices_size_bytes);
    float *h_output = (float*)malloc(output_size_bytes);
    float *h_expected = (float*)malloc(output_size_bytes);
    
    // Initialize input with random values
    for (int i = 0; i < input_size; i++) {
        h_input[i] = ((float)(rand() % 10000)) / 100.0f; // Range [0, 100]
    }
    
    // Initialize indices randomly within range
    for (int i = 0; i < n; i++) {
        h_indices[i] = rand() % input_size;  // Valid indices
    }
    
    // Calculate expected result for gather on CPU
    cpu_gather(h_input, h_indices, h_expected, n, input_size);
    
    // Device arrays
    float *d_input, *d_output;
    int *d_indices;
    checkCudaError(cudaMalloc(&d_input, input_size_bytes), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_indices, indices_size_bytes), "cudaMalloc d_indices");
    checkCudaError(cudaMalloc(&d_output, output_size_bytes), "cudaMalloc d_output");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_input to d_input");
    checkCudaError(cudaMemcpy(d_indices, h_indices, indices_size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_indices to d_indices");
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching gather kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    gather_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_indices, d_output, n, input_size);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    gather_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_indices, d_output, n, input_size);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_output to h_output");
    
    // Verification: Check a few random points
    bool success = true;
    int num_checks = min(1000, n);
    
    for (int check = 0; check < num_checks; check++) {
        int idx = rand() % n;
        
        if (abs(h_output[idx] - h_expected[idx]) > 1e-5) {
            printf("Gather verification failed at index %d: expected %f, got %f\n", 
                   idx, h_expected[idx], h_output[idx]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Gather verification PASSED (checked %d random elements)\n", num_checks);
        
        // Calculate performance metrics for gather
        double bytes_moved = n * sizeof(int) + n * sizeof(float) + n * sizeof(float); // indices + input reads + output writes
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Gather - Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Gather - Total execution time: %.3f s\n", total_time_s);
        printf("Gather - Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Gather - Throughput: %.2f GElements/s\n", (n / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Gather verification FAILED\n");
    }
    
    // Now test scatter operation
    printf("\nTesting scatter operation...\n");
    
    // Prepare for scatter test
    cpu_scatter(h_input, h_indices, h_expected, n, input_size);
    checkCudaError(cudaMemset(d_output, 0, input_size * sizeof(float)), "cudaMemset d_output");
    
    // Timing for scatter
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    scatter_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_indices, d_output, n, input_size);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    total_time_s = get_time_diff(start, end);
    
    // Copy scatter result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_output to h_output");
    
    // Verification for scatter: Check a few random points
    success = true;
    num_checks = min(1000, input_size);
    
    for (int check = 0; check < num_checks; check++) {
        int idx = rand() % input_size;
        
        if (abs(h_output[idx] - h_expected[idx]) > 1e-5) {
            printf("Scatter verification failed at index %d: expected %f, got %f\n", 
                   idx, h_expected[idx], h_output[idx]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Scatter verification PASSED (checked %d random elements)\n", num_checks);

        // Calculate performance metrics for scatter
        double bytes_moved_scatter = n * sizeof(int) + n * sizeof(float) + input_size * sizeof(float); // indices + input + output
        double bandwidth_gbs_scatter = (bytes_moved_scatter / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);

        printf("Scatter - Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Scatter - Total execution time: %.3f s\n", total_time_s);
        printf("Scatter - Effective bandwidth: %.2f GB/s\n", bandwidth_gbs_scatter);
        printf("Scatter - Throughput: %.2f GElements/s\n", (n / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Scatter verification FAILED\n");
    }
    
    // Cleanup
    free(h_input); free(h_indices); free(h_output); free(h_expected);
    cudaFree(d_input); cudaFree(d_indices); cudaFree(d_output);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}