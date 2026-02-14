#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive matrix transpose kernel - causes severe bank conflicts
__global__ void transpose_naive(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[y * width + x] = input[x * height + y];  // Transpose operation
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

// CPU transpose for verification
void cpu_transpose(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[y * width + x] = input[x * height + y];
        }
    }
}

int main(int argc, char **argv) {
    int width = 1024, height = 1024;  // Default matrix dimensions
    if (argc >= 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    } else if (argc == 2) {
        width = height = atoi(argv[1]);  // Square matrix
    }
    
    printf("Matrix transpose: %dx%d\n", width, height);
    
    size_t input_size = width * height * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(input_size);
    float *h_expected = (float*)malloc(input_size);
    
    // Initialize input with random values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Calculate expected result on CPU
    cpu_transpose(h_input, h_expected, width, height);
    
    // Device arrays
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, input_size), "cudaMalloc d_output");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "cudaMemcpy h_input to d_input");
    
    // Kernel configuration
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    transpose_naive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    transpose_naive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_output to h_output");
    
    // Verification: Check a few random points
    bool success = true;
    int num_checks = min(1000, width * height / 10);
    
    for (int check = 0; check < num_checks; check++) {
        int x = rand() % width;
        int y = rand() % height;
        int idx = y * width + x;
        
        if (abs(h_output[idx] - h_expected[idx]) > 1e-5) {
            printf("Verification failed at [%d][%d]: expected %f, got %f\n", x, y, h_expected[idx], h_output[idx]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED (checked %d random elements)\n", num_checks);
        
        // Calculate performance metrics
        double bytes_moved = 2.0 * width * height * sizeof(float); // Read + write
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Throughput: %.2f GElements/s\n", ((double)width * height / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_input); free(h_output); free(h_expected);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}