#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive reduction kernel - sums all elements in the array
__global__ void reduction_naive(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    extern __shared__ float sdata[];
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Sequential CPU reduction for verification
float cpu_reduction(float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
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
    int n = 1 << 20; // 1M elements by default
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Reduction of %d elements\n", n);
    
    size_t input_size = n * sizeof(float);
    
    // Host array
    float *h_input = (float*)malloc(input_size);
    
    // Initialize host array with random values
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Calculate expected result on CPU
    float cpu_result = cpu_reduction(h_input, n);

    // Device array
    float *d_input;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "cudaMemcpy h_input to d_input");
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate space for partial results from each block
    float *d_partial_results;
    checkCudaError(cudaMalloc(&d_partial_results, blocksPerGrid * sizeof(float)), "cudaMalloc d_partial_results");
    
    printf("Launching kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    reduction_naive<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_partial_results, n);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    reduction_naive<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_partial_results, n);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy partial results back to host
    float *h_partial_results = (float*)malloc(blocksPerGrid * sizeof(float));
    checkCudaError(cudaMemcpy(h_partial_results, d_partial_results, 
                             blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy d_partial_results to h_partial_results");
    
    // Final reduction on CPU
    float gpu_result = cpu_reduction(h_partial_results, blocksPerGrid);
    
    // Verify result
    bool success = abs(gpu_result - cpu_result) < 1e-3 * abs(cpu_result);
    
    if (success) {
        printf("Verification PASSED\n");
        printf("CPU result: %f\n", cpu_result);
        printf("GPU result: %f\n", gpu_result);
        
        // Calculate performance metrics
        double bytes_processed = n * sizeof(float); // Read operations
        double bandwidth_gbs = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Throughput: %.2f GElements/s\n", (n / 1e9) / (kernel_time_ms / 1000.0));
    } else {
        printf("Verification FAILED\n");
        printf("CPU result: %f\n", cpu_result);
        printf("GPU result: %f\n", gpu_result);
        printf("Difference: %f\n", abs(gpu_result - cpu_result));
    }
    
    // Cleanup
    free(h_input);
    free(h_partial_results);
    cudaFree(d_input);
    cudaFree(d_partial_results);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}