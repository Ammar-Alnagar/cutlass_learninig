#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive bubble sort kernel - extremely inefficient on GPU
__global__ void sort_naive(float *data, int n) {
    bool sorted = false;
    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (!sorted) {
        sorted = true;
        
        // Even phase: compare even-indexed neighbors
        if (tid % 2 == 0 && tid + 1 < n) {
            if (data[tid] > data[tid + 1]) {
                float temp = data[tid];
                data[tid] = data[tid + 1];
                data[tid + 1] = temp;
                sorted = false;
            }
        }
        __syncthreads();
        
        // Odd phase: compare odd-indexed neighbors
        if (tid % 2 == 1 && tid + 1 < n) {
            if (data[tid] > data[tid + 1]) {
                float temp = data[tid];
                data[tid] = data[tid + 1];
                data[tid + 1] = temp;
                sorted = false;
            }
        }
        __syncthreads();
    }
}

// Optimized bitonic sort kernel for power-of-2 arrays
__global__ void bitonic_sort_step(float *data, int j, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;

    if (ixj > idx && idx < n && ixj < n) {
        if ((idx & k) == 0) {
            if (data[idx] > data[ixj]) {
                float temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[idx] < data[ixj]) {
                float temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
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

// CPU sort for verification (using qsort)
int compare_floats(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

bool is_sorted(float *data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i-1]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int n = 1 << 12; // 4096 elements by default (small due to inefficiency of naive approach)
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    // Limit size for naive approach
    if (n > 4096) {
        printf("Warning: reducing size to 4096 for naive approach\n");
        n = 4096;
    }
    
    printf("Sorting %d elements with naive approach\n", n);
    
    size_t input_size = n * sizeof(float);
    
    // Host array
    float *h_data = (float*)malloc(input_size);
    float *h_original = (float*)malloc(input_size);
    
    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(rand() % 10000) / 100.0f;  // Random floats between 0 and 100
        h_original[i] = h_data[i];
    }
    
    // Sort copy with CPU for verification
    qsort(h_original, n, sizeof(float), compare_floats);
    
    // Device array
    float *d_data;
    checkCudaError(cudaMalloc(&d_data, input_size), "cudaMalloc d_data");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_data, h_data, input_size, cudaMemcpyHostToDevice), "cudaMemcpy h_data to d_data");
    
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
    sort_naive<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    sort_naive<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_data, d_data, input_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_data to h_data");
    
    // Verification: Check if sorted
    bool success = is_sorted(h_data, n);
    
    if (success) {
        printf("Verification PASSED - Array is sorted\n");
        
        // Additional verification: compare with CPU result
        bool values_match = true;
        for (int i = 0; i < n; i++) {
            if (abs(h_data[i] - h_original[i]) > 1e-5) {
                values_match = false;
                break;
            }
        }
        
        if (values_match) {
            printf("Values match CPU result\n");
            
            // Calculate performance metrics
            double bytes_processed = n * sizeof(float); // Read operations
            double bandwidth_gbs = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
            
            printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
            printf("Total execution time: %.3f s\n", total_time_s);
            printf("Bandwidth: %.2f GB/s\n", bandwidth_gbs);
            printf("Throughput: %.2f KElements/s\n", (n / 1e3) / (kernel_time_ms / 1000.0));
        } else {
            printf("Values do not match CPU result\n");
            success = false;
        }
    } else {
        printf("Verification FAILED - Array is not sorted\n");
    }
    
    // Cleanup
    free(h_data); free(h_original);
    cudaFree(d_data);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}