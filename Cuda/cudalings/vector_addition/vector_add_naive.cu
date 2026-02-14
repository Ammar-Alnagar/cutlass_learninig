#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive vector addition kernel
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
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

int main(int argc, char **argv) {
    int N = 1 << 20; // 1M elements by default
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    printf("Vector addition with %d elements\n", N);
    
    size_t size = N * sizeof(float);
    
    // Host arrays
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, size), "cudaMalloc d_C");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C");
    
    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            printf("Verification failed at index %d: expected %f, got %f\n", i, h_A[i] + h_B[i], h_C[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED\n");
        
        // Calculate performance metrics
        double bytes_processed = 3.0 * N * sizeof(float); // 2 reads + 1 write
        double bandwidth_gbs = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Throughput: %.2f GElements/s\n", (N / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}