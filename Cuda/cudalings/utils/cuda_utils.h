// Common CUDA utilities for timing and profiling
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Utility function to check CUDA errors
static inline void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

// Timing utility using gettimeofday
static inline double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

// Timing utility using CUDA events
static inline float get_cuda_event_time(cudaEvent_t start, cudaEvent_t stop) {
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    return time_ms;
}

// Print GPU properties
static inline void print_gpu_properties() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per Dimension: [%d, %d, %d]\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size: [%d, %d, %d]\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Clock Rate: %d kHz\n", prop.clockRate);
    printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Number of MPs: %d\n", prop.multiProcessorCount);
}

// Calculate theoretical occupancy
static inline float calculate_occupancy(int blocks_per_grid, int threads_per_block, int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int max_threads_per_sm;
    cudaOccupancyMaxActiveThreadsPerMultiprocessor(&max_threads_per_sm, 
                                                   (void*)NULL, 
                                                   threads_per_block, 
                                                   0);  // Shared memory per block
    
    int max_blocks_per_sm = max_threads_per_sm / threads_per_block;
    int max_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    
    float occupancy = (float)(blocks_per_grid * threads_per_block) / 
                      (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    
    return fminf(occupancy, 1.0f);
}

#endif // CUDA_UTILS_H