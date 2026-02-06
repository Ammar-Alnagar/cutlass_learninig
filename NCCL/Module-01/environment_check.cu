/*
 * Module 1: NCCL Environment Check
 * 
 * This simple program checks if NCCL is properly installed and accessible.
 * It initializes NCCL and prints the version information.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

int main() {
    // Check CUDA device count
    int nDevices;
    cudaError_t cudaStatus = cudaGetDeviceCount(&nDevices);
    
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Error: No GPU detected or CUDA not properly installed.\n");
        printf("Error: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    
    printf("Number of CUDA devices available: %d\n", nDevices);
    
    if (nDevices == 0) {
        printf("No CUDA devices found. NCCL requires at least one NVIDIA GPU.\n");
        return -1;
    }
    
    // Print NCCL version
    int version;
    ncclResult_t result = ncclGetVersion(&version);
    if (result != ncclSuccess) {
        printf("Error getting NCCL version: %s\n", ncclGetErrorString(result));
        return -1;
    }
    
    int major = version / 10000;
    int minor = (version % 10000) / 100;
    int patch = version % 100;
    
    printf("NCCL Version: %d.%d.%d\n", major, minor, patch);
    printf("\nEnvironment check completed successfully!\n");
    printf("You have %d GPU(s) and NCCL is properly installed.\n", nDevices);
    printf("\nReady to proceed to Module 2 for hands-on NCCL operations!\n");
    
    return 0;
}