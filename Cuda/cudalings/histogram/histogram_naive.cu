#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define HISTOGRAM_SIZE 256

// Naive histogram kernel - causes race conditions
__global__ void histogram_naive(unsigned char *input, unsigned int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unsigned char value = input[idx];
        atomicAdd(&(histogram[value]), 1);  // Atomic operation to prevent race conditions
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

// CPU histogram for verification
void cpu_histogram(unsigned char *input, unsigned int *histogram, int n) {
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        histogram[i] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        histogram[input[i]]++;
    }
}

int main(int argc, char **argv) {
    int n = 1 << 22; // 4M elements by default
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Histogram of %d elements (values 0-255)\n", n);
    
    size_t input_size = n * sizeof(unsigned char);
    size_t hist_size = HISTOGRAM_SIZE * sizeof(unsigned int);
    
    // Host arrays
    unsigned char *h_input = (unsigned char*)malloc(input_size);
    unsigned int *h_gpu_hist = (unsigned int*)malloc(hist_size);
    unsigned int *h_cpu_hist = (unsigned int*)malloc(hist_size);
    
    // Initialize input with random values (0-255)
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 256;
    }
    
    // Calculate expected result on CPU
    cpu_histogram(h_input, h_cpu_hist, n);
    
    // Device arrays
    unsigned char *d_input;
    unsigned int *d_histogram;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_histogram, hist_size), "cudaMalloc d_histogram");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "cudaMemcpy h_input to d_input");
    
    // Initialize histogram to zero on device
    checkCudaError(cudaMemset(d_histogram, 0, hist_size), "cudaMemset d_histogram");
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    histogram_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histogram, n);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    histogram_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histogram, n);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_gpu_hist, d_histogram, hist_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_histogram to h_gpu_hist");
    
    // Verification: Compare histograms
    bool success = true;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (h_gpu_hist[i] != h_cpu_hist[i]) {
            printf("Verification failed at bin %d: CPU=%u, GPU=%u\n", i, h_cpu_hist[i], h_gpu_hist[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED\n");
        
        // Calculate performance metrics
        double bytes_processed = n * sizeof(unsigned char); // Read operations
        double bandwidth_gbs = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Throughput: %.2f GElements/s\n", (n / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_input); free(h_gpu_hist); free(h_cpu_hist);
    cudaFree(d_input); cudaFree(d_histogram);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}