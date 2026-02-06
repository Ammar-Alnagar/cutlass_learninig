/*
 * Module 2: Basic Collective Operations - AllReduce and Broadcast
 * 
 * This example demonstrates the two most fundamental NCCL collective operations:
 * 1. AllReduce - sums values from all GPUs and broadcasts the result to all
 * 2. Broadcast - sends data from root GPU to all other GPUs
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

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

int main(int argc, char* argv[]) {
    int nDevices = 4;  // Number of GPUs to use
    
    // Check available GPUs
    int gpu_count;
    CUDACHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count < nDevices) {
        nDevices = gpu_count;
        printf("Only %d GPUs available, using %d\n", gpu_count, nDevices);
    }
    
    printf("Using %d GPUs for NCCL operations\n", nDevices);

    // Initialize NCCL
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nDevices);
    NCCLCHECK(ncclCommInitAll(comms, nDevices, NULL));

    // Allocate and initialize GPU buffers
    float** d_inputs = (float**)malloc(nDevices * sizeof(float*));
    float** d_outputs = (float**)malloc(nDevices * sizeof(float*));
    
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        // Allocate GPU memory
        CUDACHECK(cudaMalloc(d_inputs + i, sizeof(float) * 4));
        CUDACHECK(cudaMalloc(d_outputs + i, sizeof(float) * 4));
        
        // Initialize input data differently on each GPU
        float h_input[4];
        for (int j = 0; j < 4; j++) {
            h_input[j] = (i + 1) * 10.0f + j;  // GPU 0: [10,11,12,13], GPU 1: [20,21,22,23], etc.
        }
        
        CUDACHECK(cudaMemcpy(d_inputs[i], h_input, sizeof(float) * 4, cudaMemcpyHostToDevice));
        
        // Initialize output to zero
        CUDACHECK(cudaMemset(d_outputs[i], 0, sizeof(float) * 4));
    }

    printf("\n=== AllReduce Example ===\n");
    printf("Before AllReduce:\n");
    for (int i = 0; i < nDevices; i++) {
        float h_data[4];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_data, d_inputs[i], sizeof(float) * 4, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f]\n", i, h_data[0], h_data[1], h_data[2], h_data[3]);
    }

    // Perform AllReduce (sum operation)
    cudaStream_t* streams = (cudaStream_t*)malloc(nDevices * sizeof(cudaStream_t));
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(streams + i));
        NCCLCHECK(ncclAllReduce((const void*)d_inputs[i], (void*)d_outputs[i], 4, ncclFloat32, 
                               ncclSum, comms[i], streams[i]));
    }

    // Wait for all operations to complete
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\nAfter AllReduce (SUM):\n");
    for (int i = 0; i < nDevices; i++) {
        float h_result[4];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_result, d_outputs[i], sizeof(float) * 4, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f]\n", i, h_result[0], h_result[1], h_result[2], h_result[3]);
    }

    // Reset output buffers for broadcast example
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemset(d_outputs[i], 0, sizeof(float) * 4));
    }

    printf("\n=== Broadcast Example (from GPU 0) ===\n");
    
    // Perform Broadcast from GPU 0
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        if (i == 0) {
            // On root GPU (0), broadcast from input
            NCCLCHECK(ncclBcast((void*)d_inputs[i], 4, ncclFloat32, 0, comms[i], streams[i]));
        } else {
            // On other GPUs, receive broadcast
            NCCLCHECK(ncclBcast((void*)d_outputs[i], 4, ncclFloat32, 0, comms[i], streams[i]));
        }
    }

    // Wait for all operations to complete
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\nAfter Broadcast (from GPU 0 -> [10, 11, 12, 13]):\n");
    for (int i = 0; i < nDevices; i++) {
        float h_result[4];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_result, d_outputs[i], sizeof(float) * 4, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f]\n", i, h_result[0], h_result[1], h_result[2], h_result[3]);
    }

    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_inputs[i]));
        CUDACHECK(cudaFree(d_outputs[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    
    free(d_inputs);
    free(d_outputs);
    free(streams);
    free(comms);

    printf("\nBasic collective operations completed successfully!\n");
    return 0;
}