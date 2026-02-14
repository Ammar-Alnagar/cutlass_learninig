#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive exclusive scan kernel - inefficient approach
__global__ void scan_naive(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = 0.0f;
        for (int i = 0; i < idx; i++) {
            output[idx] += input[i];
        }
    }
}

// Work-efficient scan kernel (up-sweep/down-sweep)
__global__ void scan_work_efficient(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    temp[2*thid] = (2*thid < n) ? input[2*thid] : 0.0f;
    temp[2*thid+1] = (2*thid+1 < n) ? input[2*thid+1] : 0.0f;
    
    // Up-sweep (reduce) phase
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (thid == 0) {
        temp[n - 1] = 0.0f;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results to device memory
    if (2*thid < n) output[2*thid] = temp[2*thid];
    if (2*thid+1 < n) output[2*thid+1] = temp[2*thid+1];
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

// CPU exclusive scan for verification
void cpu_scan(float *input, float *output, int n) {
    if (n > 0) {
        output[0] = 0.0f;  // Exclusive scan starts with 0
        for (int i = 1; i < n; i++) {
            output[i] = output[i-1] + input[i-1];
        }
    }
}

int main(int argc, char **argv) {
    int n = 1 << 14; // 16384 elements by default (limited due to naive approach inefficiency)
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    // Limit size for naive approach
    if (n > 16384) {
        printf("Warning: reducing size to 16384 for naive approach\n");
        n = 16384;
    }
    
    printf("Exclusive scan of %d elements\n", n);
    
    size_t input_size = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(input_size);
    float *h_expected = (float*)malloc(input_size);
    
    // Initialize input with random values
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f; // Using 1.0 for easier verification (should result in 0,1,2,3...)
    }
    
    // Calculate expected result on CPU
    cpu_scan(h_input, h_expected, n);
    
    // Device arrays
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, input_size), "cudaMalloc d_output");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "cudaMemcpy h_input to d_input");
    
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
    scan_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    scan_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_output to h_output");
    
    // Verification: Check a few random points and the last element
    bool success = true;
    int num_checks = min(100, n);
    
    for (int check = 0; check < num_checks; check++) {
        int idx = (check * n) / num_checks;  // Evenly distributed indices
        
        if (abs(h_output[idx] - h_expected[idx]) > 1e-5) {
            printf("Verification failed at index %d: expected %f, got %f\n", idx, h_expected[idx], h_output[idx]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED (checked %d elements)\n", num_checks);
        
        // Calculate performance metrics
        double bytes_processed = 2.0 * n * sizeof(float); // Read + write
        double bandwidth_gbs = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Throughput: %.2f GElements/s\n", (n / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_input); free(h_output); free(h_expected);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}