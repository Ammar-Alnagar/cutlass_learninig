/*
 * Module 3: Advanced Collective Operations - Reduce, AllGather, Scatter
 * 
 * This example demonstrates three advanced NCCL collective operations:
 * 1. Reduce - reduces values from all GPUs to a single root GPU
 * 2. AllGather - gathers values from all GPUs and distributes to all
 * 3. Scatter - scatters values from root GPU to all GPUs
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
    
    printf("Using %d GPUs for advanced NCCL operations\n", nDevices);

    // Initialize NCCL
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nDevices);
    NCCLCHECK(ncclCommInitAll(comms, nDevices, NULL));

    // Allocate GPU buffers for different operations
    float** d_sendbufs = (float**)malloc(nDevices * sizeof(float*));
    float** d_recvbufs = (float**)malloc(nDevices * sizeof(float*));
    float** d_tempbufs = (float**)malloc(nDevices * sizeof(float*));
    
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        // For Reduce and AllGather: each GPU sends/receives 4 elements
        CUDACHECK(cudaMalloc(d_sendbufs + i, sizeof(float) * 4));
        CUDACHECK(cudaMalloc(d_recvbufs + i, sizeof(float) * 4));
        
        // For Scatter: root GPU needs nDevices * 4 elements, others need 4
        CUDACHECK(cudaMalloc(d_tempbufs + i, sizeof(float) * 4 * nDevices));
        
        // Initialize send buffers with different values on each GPU
        float h_send[4];
        for (int j = 0; j < 4; j++) {
            h_send[j] = (i + 1) * 10.0f + j;  // GPU 0: [10,11,12,13], GPU 1: [20,21,22,23], etc.
        }
        
        CUDACHECK(cudaMemcpy(d_sendbufs[i], h_send, sizeof(float) * 4, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recvbufs[i], 0, sizeof(float) * 4));
        CUDACHECK(cudaMemset(d_tempbufs[i], 0, sizeof(float) * 4 * nDevices));
    }

    printf("\n=== Reduce Example (to GPU 0) ===\n");
    printf("Before Reduce:\n");
    for (int i = 0; i < nDevices; i++) {
        float h_data[4];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_data, d_sendbufs[i], sizeof(float) * 4, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f]\n", i, h_data[0], h_data[1], h_data[2], h_data[3]);
    }

    // Perform Reduce (sum operation to root GPU 0)
    cudaStream_t* streams = (cudaStream_t*)malloc(nDevices * sizeof(cudaStream_t));
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(streams + i));
    }

    // Initialize receive buffer on root GPU
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemset(d_recvbufs[0], 0, sizeof(float) * 4));

    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclReduce((const void*)d_sendbufs[i], (void*)d_recvbufs[0], 4, ncclFloat32, 
                            ncclSum, 0, comms[i], streams[i]));  // Root is GPU 0
    }

    // Wait for all operations to complete
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\nAfter Reduce (SUM to GPU 0):\n");
    for (int i = 0; i < nDevices; i++) {
        float h_result[4];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_result, d_recvbufs[0], sizeof(float) * 4, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f] %s\n", 
               i, h_result[0], h_result[1], h_result[2], h_result[3],
               (i == 0) ? "(only root GPU has result)" : "(other GPUs unchanged)");
    }

    printf("\n=== AllGather Example ===\n");
    
    // Reset buffers for AllGather
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemset(d_recvbufs[i], 0, sizeof(float) * 4 * nDevices));  // Larger buffer for AllGather
        
        // Reinitialize send buffers
        float h_send[4];
        for (int j = 0; j < 4; j++) {
            h_send[j] = (i + 1) * 10.0f + j;
        }
        CUDACHECK(cudaMemcpy(d_sendbufs[i], h_send, sizeof(float) * 4, cudaMemcpyHostToDevice));
    }

    // Perform AllGather
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        // AllGather: each GPU contributes 4 elements, receives 4*nDevices elements
        NCCLCHECK(ncclAllGather((const void*)d_sendbufs[i], (void*)d_tempbufs[i], 4, ncclFloat32, 
                               comms[i], streams[i]));
    }

    // Wait for all operations to complete
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\nAfter AllGather:\n");
    for (int i = 0; i < nDevices; i++) {
        float h_result[16];  // 4*nDevices = 16 elements
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_result, d_tempbufs[i], sizeof(float) * 16, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
               i, h_result[0], h_result[1], h_result[2], h_result[3],
               h_result[4], h_result[5], h_result[6], h_result[7],
               h_result[8], h_result[9], h_result[10], h_result[11],
               h_result[12], h_result[13], h_result[14], h_result[15]);
        printf("         (Data from GPUs 0-3: [0-3], [4-7], [8-11], [12-15])\n");
    }

    printf("\n=== Scatter Example (from GPU 0) ===\n");
    
    // Prepare data on root GPU for scatter
    CUDACHECK(cudaSetDevice(0));
    float h_scatter_data[16];  // 4 elements per GPU * 4 GPUs
    for (int i = 0; i < 16; i++) {
        h_scatter_data[i] = 100.0f + i;  // [100, 101, ..., 115]
    }
    CUDACHECK(cudaMemcpy(d_tempbufs[0], h_scatter_data, sizeof(float) * 16, cudaMemcpyHostToDevice));

    // Reset other GPUs' temp buffers to zero
    for (int i = 1; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemset(d_tempbufs[i], 0, sizeof(float) * 16));
    }

    // Perform Scatter
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        // Scatter: root GPU scatters 4 elements to each GPU
        NCCLCHECK(ncclScatter((const void*)d_tempbufs[0], (void*)d_recvbufs[i], 4, ncclFloat32, 
                             0, comms[i], streams[i]));  // Root is GPU 0
    }

    // Wait for all operations to complete
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\nAfter Scatter (from GPU 0 -> [100-115] distributed):\n");
    for (int i = 0; i < nDevices; i++) {
        float h_result[4];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(h_result, d_recvbufs[i], sizeof(float) * 4, cudaMemcpyDeviceToHost));
        printf("GPU %d: [%.1f, %.1f, %.1f, %.1f]\n", 
               i, h_result[0], h_result[1], h_result[2], h_result[3]);
    }

    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_sendbufs[i]));
        CUDACHECK(cudaFree(d_recvbufs[i]));
        CUDACHECK(cudaFree(d_tempbufs[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    
    free(d_sendbufs);
    free(d_recvbufs);
    free(d_tempbufs);
    free(streams);
    free(comms);

    printf("\nAdvanced collective operations completed successfully!\n");
    return 0;
}