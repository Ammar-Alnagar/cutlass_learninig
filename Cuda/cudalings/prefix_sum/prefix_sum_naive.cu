#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Macro to avoid shared memory bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// Naive prefix sum (scan) kernel - inefficient sequential approach
__global__ void prefix_sum_naive(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = 0.0f;
        for (int i = 0; i <= idx; i++) {
            output[idx] += input[i];
        }
    }
}

// Efficient inclusive scan kernel using shared memory
__global__ void prefix_sum_work_efficient(float *input, float *output, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    int ai = thid;
    int bi = thid + (blockDim.x);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    if (ai < n) temp[ai + bankOffsetA] = input[ai];
    else temp[ai + bankOffsetA] = 0.0f;
    
    if (bi < n) temp[bi + bankOffsetB] = input[bi];
    else temp[bi + bankOffsetB] = 0.0f;
    
    // Up-sweep (reduce) phase
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (thid == 0) {
        int last = (blockDim.x*2 - 1);
        last += CONFLICT_FREE_OFFSET(last);
        temp[last] = 0.0f;
    }
    
    // Down-sweep phase
    for (int d = 1; d < blockDim.x*2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results to device memory
    if (ai < n) output[ai] = temp[ai + bankOffsetA];
    if (bi < n) output[bi + blockDim.x] = temp[bi + bankOffsetB];
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

// CPU prefix sum for verification
void cpu_prefix_sum(float *input, float *output, int n) {
    if (n > 0) {
        output[0] = input[0];
        for (int i = 1; i < n; i++) {
            output[i] = output[i-1] + input[i];
        }
    }
}

int main(int argc, char **argv) {
    int n = 1 << 16; // 65536 elements by default
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Prefix sum of %d elements\n", n);
    
    size_t input_size = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(input_size);
    float *h_expected = (float*)malloc(input_size);
    
    // Initialize input with random values
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f; // Using 1.0 for easier verification (should result in 1,2,3,4...)
    }
    
    // Calculate expected result on CPU
    cpu_prefix_sum(h_input, h_expected, n);
    
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
    prefix_sum_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    prefix_sum_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    
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