/*
 * Module 5: Performance Optimization Techniques
 * 
 * This example demonstrates various performance optimization techniques for NCCL,
 * including message aggregation, communication-computation overlap, and performance measurement.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/time.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Utility function to get timestamp in microseconds
long long get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
}

// Simple kernel to simulate computation
__global__ void simulate_computation_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.01f + 0.01f;
        }
        data[idx] = val;
    }
}

// Performance test for different message sizes
void performance_test(int nGPUs, ncclComm_t* comms, cudaStream_t* streams, 
                     float** d_sendbufs, float** d_recvbufs, int msg_size) {
    printf("\nPerformance test with message size: %d floats (%.2f MB)\n", 
           msg_size, (msg_size * sizeof(float) * nGPUs) / (1024.0 * 1024.0));
    
    // Warm-up iterations
    for (int i = 0; i < 3; i++) {
        for (int gpu = 0; gpu < nGPUs; gpu++) {
            CUDACHECK(cudaSetDevice(gpu));
            NCCLCHECK(ncclAllReduce((const void*)d_sendbufs[gpu], (void*)d_recvbufs[gpu], 
                                   msg_size, ncclFloat32, ncclSum, comms[gpu], streams[gpu]));
        }
        for (int gpu = 0; gpu < nGPUs; gpu++) {
            CUDACHECK(cudaSetDevice(gpu));
            CUDACHECK(cudaStreamSynchronize(streams[gpu]));
        }
    }
    
    // Actual timing
    long long start_time = get_time_us();
    
    const int num_iterations = 10;
    for (int iter = 0; iter < num_iterations; iter++) {
        for (int gpu = 0; gpu < nGPUs; gpu++) {
            CUDACHECK(cudaSetDevice(gpu));
            NCCLCHECK(ncclAllReduce((const void*)d_sendbufs[gpu], (void*)d_recvbufs[gpu], 
                                   msg_size, ncclFloat32, ncclSum, comms[gpu], streams[gpu]));
        }
        for (int gpu = 0; gpu < nGPUs; gpu++) {
            CUDACHECK(cudaSetDevice(gpu));
            CUDACHECK(cudaStreamSynchronize(streams[gpu]));
        }
    }
    
    long long end_time = get_time_us();
    double elapsed_ms = (end_time - start_time) / 1000.0;
    double avg_time_per_iter_ms = elapsed_ms / num_iterations;
    
    // Calculate bandwidth (bidirectional)
    size_t bytes_per_iter = msg_size * sizeof(float) * nGPUs;
    double bandwidth_gb_s = (bytes_per_iter * 2.0 / (1024.0 * 1024.0 * 1024.0)) / 
                           (avg_time_per_iter_ms / 1000.0);
    
    printf("  Average time per iteration: %.3f ms\n", avg_time_per_iter_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
}

// Demonstrate communication-computation overlap
void overlap_demo(int nGPUs, ncclComm_t* comms, cudaStream_t* comp_streams, 
                 cudaStream_t* comm_streams, float** d_comp_bufs, float** d_comm_bufs, 
                 float** d_result_bufs) {
    printf("\nDemonstrating communication-computation overlap:\n");
    
    // Warm-up
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        CUDACHECK(cudaSetDevice(gpu));
        
        // Launch computation on one stream
        int blockSize = 256;
        int gridSize = (1024 + blockSize - 1) / blockSize;
        simulate_computation_kernel<<<gridSize, blockSize, 0, comp_streams[gpu]>>>(
            d_comp_bufs[gpu], 1024, 100);
        
        // Launch communication on another stream
        NCCLCHECK(ncclAllReduce((const void*)d_comm_bufs[gpu], (void*)d_result_bufs[gpu], 
                               1024, ncclFloat32, ncclSum, comms[gpu], comm_streams[gpu]));
    }
    
    // Synchronize both streams
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        CUDACHECK(cudaSetDevice(gpu));
        CUDACHECK(cudaStreamSynchronize(comp_streams[gpu]));
        CUDACHECK(cudaStreamSynchronize(comm_streams[gpu]));
    }
    
    printf("  Computation and communication executed concurrently\n");
}

// Demonstrate message aggregation
void aggregation_demo(int nGPUs, ncclComm_t* comms, cudaStream_t* streams,
                     float** d_small_msgs, float** d_aggregated_msg, int num_small_msgs, int msg_size) {
    printf("\nDemonstrating message aggregation:\n");
    
    // Method 1: Multiple small messages
    long long start_time = get_time_us();
    
    for (int i = 0; i < num_small_msgs; i++) {
        for (int gpu = 0; gpu < nGPUs; gpu++) {
            CUDACHECK(cudaSetDevice(gpu));
            NCCLCHECK(ncclAllReduce((const void*)(d_small_msgs[gpu] + i * msg_size), 
                                   (void*)(d_small_msgs[gpu] + i * msg_size), 
                                   msg_size, ncclFloat32, ncclSum, comms[gpu], streams[gpu]));
        }
    }
    
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        CUDACHECK(cudaSetDevice(gpu));
        CUDACHECK(cudaStreamSynchronize(streams[gpu]));
    }
    
    long long time_separate = get_time_us() - start_time;
    
    // Method 2: Single aggregated message
    start_time = get_time_us();
    
    NCCLCHECK(ncclGroupStart());
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        CUDACHECK(cudaSetDevice(gpu));
        NCCLCHECK(ncclAllReduce((const void*)d_aggregated_msg[gpu], 
                               (void*)d_aggregated_msg[gpu], 
                               num_small_msgs * msg_size, ncclFloat32, ncclSum, 
                               comms[gpu], streams[gpu]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        CUDACHECK(cudaSetDevice(gpu));
        CUDACHECK(cudaStreamSynchronize(streams[gpu]));
    }
    
    long long time_aggregated = get_time_us() - start_time;
    
    printf("  Time for %d separate messages of size %d: %lld us\n", 
           num_small_msgs, msg_size, time_separate);
    printf("  Time for 1 aggregated message of size %d: %lld us\n", 
           num_small_msgs * msg_size, time_aggregated);
    printf("  Speedup from aggregation: %.2fx\n", (double)time_separate / time_aggregated);
}

int main(int argc, char* argv[]) {
    int nGPUs = 4;  // Number of GPUs to use
    
    // Check available GPUs
    int gpu_count;
    CUDACHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count < nGPUs) {
        nGPUs = gpu_count;
        printf("Only %d GPUs available, using %d\n", gpu_count, nGPUs);
    }
    
    printf("Starting performance optimization demonstration with %d GPUs\n", nGPUs);

    // Initialize NCCL communicators
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGPUs);
    NCCLCHECK(ncclCommInitAll(comms, nGPUs, NULL));

    // Create streams for each GPU
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nGPUs);
    cudaStream_t* comp_streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nGPUs);
    cudaStream_t* comm_streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nGPUs);
    
    // Allocate GPU buffers
    float** d_sendbufs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_recvbufs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_comp_bufs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_comm_bufs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_result_bufs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_small_msgs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_aggregated_msg = (float**)malloc(nGPUs * sizeof(float*));
    
    const int max_msg_size = 1024 * 1024;  // 1M floats
    const int small_msg_size = 1024;       // 1K floats
    const int num_small_msgs = 256;        // Total 256K floats
    
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        // Create streams
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaStreamCreate(&comp_streams[i]));
        CUDACHECK(cudaStreamCreate(&comm_streams[i]));
        
        // Allocate buffers
        CUDACHECK(cudaMalloc(&d_sendbufs[i], sizeof(float) * max_msg_size));
        CUDACHECK(cudaMalloc(&d_recvbufs[i], sizeof(float) * max_msg_size));
        CUDACHECK(cudaMalloc(&d_comp_bufs[i], sizeof(float) * 1024));
        CUDACHECK(cudaMalloc(&d_comm_bufs[i], sizeof(float) * 1024));
        CUDACHECK(cudaMalloc(&d_result_bufs[i], sizeof(float) * 1024));
        CUDACHECK(cudaMalloc(&d_small_msgs[i], sizeof(float) * small_msg_size * num_small_msgs));
        CUDACHECK(cudaMalloc(&d_aggregated_msg[i], sizeof(float) * small_msg_size * num_small_msgs));
        
        // Initialize buffers with random data
        float* h_temp = (float*)malloc(sizeof(float) * max_msg_size);
        for (int j = 0; j < max_msg_size; j++) {
            h_temp[j] = (float)rand() / RAND_MAX;
        }
        CUDACHECK(cudaMemcpy(d_sendbufs[i], h_temp, sizeof(float) * max_msg_size, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recvbufs[i], 0, sizeof(float) * max_msg_size));
        
        // Initialize computation and communication buffers
        for (int j = 0; j < 1024; j++) {
            h_temp[j] = (float)rand() / RAND_MAX;
        }
        CUDACHECK(cudaMemcpy(d_comp_bufs[i], h_temp, sizeof(float) * 1024, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_comm_bufs[i], h_temp, sizeof(float) * 1024, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_result_bufs[i], 0, sizeof(float) * 1024));
        
        // Initialize small messages and aggregated message
        for (int j = 0; j < small_msg_size * num_small_msgs; j++) {
            h_temp[j] = (float)rand() / RAND_MAX;
        }
        CUDACHECK(cudaMemcpy(d_small_msgs[i], h_temp, sizeof(float) * small_msg_size * num_small_msgs, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_aggregated_msg[i], h_temp, sizeof(float) * small_msg_size * num_small_msgs, cudaMemcpyHostToDevice));
        
        free(h_temp);
    }

    printf("\n=== Performance Testing with Different Message Sizes ===");
    
    // Test with different message sizes
    int test_sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        performance_test(nGPUs, comms, streams, d_sendbufs, d_recvbufs, test_sizes[i]);
    }
    
    // Demonstrate communication-computation overlap
    overlap_demo(nGPUs, comms, comp_streams, comm_streams, d_comp_bufs, d_comm_bufs, d_result_bufs);
    
    // Demonstrate message aggregation
    aggregation_demo(nGPUs, comms, streams, d_small_msgs, d_aggregated_msg, num_small_msgs, small_msg_size);

    // Cleanup
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        CUDACHECK(cudaFree(d_sendbufs[i]));
        CUDACHECK(cudaFree(d_recvbufs[i]));
        CUDACHECK(cudaFree(d_comp_bufs[i]));
        CUDACHECK(cudaFree(d_comm_bufs[i]));
        CUDACHECK(cudaFree(d_result_bufs[i]));
        CUDACHECK(cudaFree(d_small_msgs[i]));
        CUDACHECK(cudaFree(d_aggregated_msg[i]));
        
        CUDACHECK(cudaStreamDestroy(streams[i]));
        CUDACHECK(cudaStreamDestroy(comp_streams[i]));
        CUDACHECK(cudaStreamDestroy(comm_streams[i]));
        
        ncclCommDestroy(comms[i]);
    }
    
    free(d_sendbufs);
    free(d_recvbufs);
    free(d_comp_bufs);
    free(d_comm_bufs);
    free(d_result_bufs);
    free(d_small_msgs);
    free(d_aggregated_msg);
    free(streams);
    free(comp_streams);
    free(comm_streams);
    free(comms);

    printf("\nPerformance optimization demonstration completed!\n");
    printf("Key takeaways:\n");
    printf("- Larger messages generally achieve higher bandwidth\n");
    printf("- Communication-computation overlap can improve performance\n");
    printf("- Message aggregation reduces protocol overhead for small messages\n");

    return 0;
}