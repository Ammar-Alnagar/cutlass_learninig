/*
 * Module 6: Troubleshooting and Best Practices
 * 
 * This example demonstrates robust error handling, debugging techniques,
 * and best practices for developing reliable NCCL applications.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <signal.h>
#include <unistd.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    fprintf(stderr, "Cuda failure %s:%d '%s'\n",   \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    fprintf(stderr, "NCCL failure %s:%d '%s'\n",   \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Global flag for graceful shutdown
volatile sig_atomic_t shutdown_requested = 0;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    fprintf(stderr, "Received signal %d, initiating graceful shutdown...\n", sig);
    shutdown_requested = 1;
}

// Function to validate data consistency across GPUs (for debugging)
int validate_data_consistency(float** d_buffers, int nGPUs, int count, int gpu_id) {
    float* h_local = (float*)malloc(sizeof(float) * count);
    float* h_reference = (float*)malloc(sizeof(float) * count);
    
    // Copy data from current GPU to host
    CUDACHECK(cudaSetDevice(gpu_id));
    CUDACHECK(cudaMemcpy(h_local, d_buffers[gpu_id], sizeof(float) * count, cudaMemcpyDeviceToHost));
    
    // Copy data from GPU 0 as reference
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(h_reference, d_buffers[0], sizeof(float) * count, cudaMemcpyDeviceToHost));
    
    // Compare values (with small tolerance for floating point)
    int errors = 0;
    for (int i = 0; i < count; i++) {
        if (abs(h_local[i] - h_reference[i]) > 1e-5) {
            errors++;
            if (errors <= 5) {  // Limit error output
                fprintf(stderr, "Data mismatch at index %d: GPU %d has %.6f, GPU 0 has %.6f\n", 
                        i, gpu_id, h_local[i], h_reference[i]);
            }
        }
    }
    
    free(h_local);
    free(h_reference);
    
    return errors == 0;
}

// Safe cleanup function that handles errors during cleanup
void safe_cleanup(ncclComm_t* comms, cudaStream_t* streams, float** d_buffers, 
                  int nGPUs, int nBuffersPerGPU) {
    fprintf(stderr, "Starting cleanup procedure...\n");
    
    // Destroy communicators
    for (int i = 0; i < nGPUs; i++) {
        if (comms[i] != NULL) {
            ncclResult_t ret = ncclCommDestroy(comms[i]);
            if (ret != ncclSuccess) {
                fprintf(stderr, "Warning: Failed to destroy NCCL communicator %d: %s\n", 
                        i, ncclGetErrorString(ret));
            }
        }
    }
    
    // Destroy streams
    for (int i = 0; i < nGPUs; i++) {
        if (streams[i] != NULL) {
            cudaError_t ret = cudaStreamDestroy(streams[i]);
            if (ret != cudaSuccess) {
                fprintf(stderr, "Warning: Failed to destroy CUDA stream %d: %s\n", 
                        i, cudaGetErrorString(ret));
            }
        }
    }
    
    // Free GPU memory
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        for (int buf = 0; buf < nBuffersPerGPU; buf++) {
            if (d_buffers[gpu * nBuffersPerGPU + buf] != NULL) {
                cudaError_t ret = cudaFree(d_buffers[gpu * nBuffersPerGPU + buf]);
                if (ret != cudaSuccess) {
                    fprintf(stderr, "Warning: Failed to free GPU memory on GPU %d, buffer %d: %s\n", 
                            gpu, buf, cudaGetErrorString(ret));
                }
            }
        }
    }
    
    fprintf(stderr, "Cleanup completed.\n");
}

// Function to test NCCL operations with error handling
int test_nccl_operation_with_error_handling(int nGPUs, ncclComm_t* comms, 
                                          cudaStream_t* streams, float** d_sendbufs, 
                                          float** d_recvbufs, int count) {
    fprintf(stderr, "Testing NCCL operation with error handling...\n");
    
    // Group operations for efficiency
    ncclResult_t ret = ncclGroupStart();
    if (ret != ncclSuccess) {
        fprintf(stderr, "Failed to start NCCL group: %s\n", ncclGetErrorString(ret));
        return -1;
    }
    
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        ret = ncclAllReduce((const void*)d_sendbufs[i], (void*)d_recvbufs[i], 
                           count, ncclFloat32, ncclSum, comms[i], streams[i]);
        if (ret != ncclSuccess) {
            fprintf(stderr, "NCCL AllReduce failed on GPU %d: %s\n", i, ncclGetErrorString(ret));
            ncclGroupEnd();  // End group even on error
            return -1;
        }
    }
    
    ret = ncclGroupEnd();
    if (ret != ncclSuccess) {
        fprintf(stderr, "Failed to end NCCL group: %s\n", ncclGetErrorString(ret));
        return -1;
    }
    
    // Synchronize all streams
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        cudaError_t cuda_ret = cudaStreamSynchronize(streams[i]);
        if (cuda_ret != cudaSuccess) {
            fprintf(stderr, "CUDA stream synchronization failed on GPU %d: %s\n", 
                    i, cudaGetErrorString(cuda_ret));
            return -1;
        }
    }
    
    fprintf(stderr, "NCCL operation completed successfully.\n");
    return 0;
}

int main(int argc, char* argv[]) {
    int nGPUs = 4;  // Number of GPUs to use
    int nBuffersPerGPU = 2;  // Send and receive buffers per GPU
    
    // Register signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Check available GPUs
    int gpu_count;
    CUDACHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count < nGPUs) {
        nGPUs = gpu_count;
        printf("Only %d GPUs available, using %d\n", gpu_count, nGPUs);
    }
    
    printf("Starting troubleshooting and best practices demonstration with %d GPUs\n", nGPUs);

    // Initialize NCCL communicators
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGPUs);
    for (int i = 0; i < nGPUs; i++) {
        comms[i] = NULL;  // Initialize to NULL for safe cleanup
    }
    
    ncclResult_t init_ret = ncclCommInitAll(comms, nGPUs, NULL);
    if (init_ret != ncclSuccess) {
        fprintf(stderr, "NCCL initialization failed: %s\n", ncclGetErrorString(init_ret));
        free(comms);
        return -1;
    }

    // Create streams for each GPU
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nGPUs);
    for (int i = 0; i < nGPUs; i++) {
        streams[i] = NULL;  // Initialize to NULL for safe cleanup
    }
    
    // Allocate GPU buffers
    float** d_buffers = (float**)malloc(nGPUs * nBuffersPerGPU * sizeof(float*));
    for (int i = 0; i < nGPUs * nBuffersPerGPU; i++) {
        d_buffers[i] = NULL;  // Initialize to NULL for safe cleanup
    }
    
    const int count = 1024;  // Number of floats per buffer
    
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        CUDACHECK(cudaSetDevice(gpu));
        
        // Create stream
        cudaError_t stream_ret = cudaStreamCreate(&streams[gpu]);
        if (stream_ret != cudaSuccess) {
            fprintf(stderr, "Failed to create CUDA stream for GPU %d: %s\n", 
                    gpu, cudaGetErrorString(stream_ret));
            safe_cleanup(comms, streams, d_buffers, nGPUs, nBuffersPerGPU);
            free(comms);
            free(streams);
            free(d_buffers);
            return -1;
        }
        
        // Allocate send buffer (index 0) and receive buffer (index 1) for this GPU
        for (int buf = 0; buf < nBuffersPerGPU; buf++) {
            int idx = gpu * nBuffersPerGPU + buf;
            cudaError_t malloc_ret = cudaMalloc(&d_buffers[idx], sizeof(float) * count);
            if (malloc_ret != cudaSuccess) {
                fprintf(stderr, "Failed to allocate GPU memory for GPU %d, buffer %d: %s\n", 
                        gpu, buf, cudaGetErrorString(malloc_ret));
                
                // Perform cleanup and exit
                safe_cleanup(comms, streams, d_buffers, nGPUs, nBuffersPerGPU);
                free(comms);
                free(streams);
                free(d_buffers);
                return -1;
            }
            
            // Initialize buffer with GPU-specific data
            float* h_temp = (float*)malloc(sizeof(float) * count);
            for (int i = 0; i < count; i++) {
                h_temp[i] = (gpu + 1) * 1000.0f + i;  // Unique value per GPU
            }
            CUDACHECK(cudaMemcpy(d_buffers[idx], h_temp, sizeof(float) * count, cudaMemcpyHostToDevice));
            free(h_temp);
        }
    }

    // Get pointers to send and receive buffers for convenience
    float** d_sendbufs = (float**)malloc(nGPUs * sizeof(float*));
    float** d_recvbufs = (float**)malloc(nGPUs * sizeof(float*));
    for (int i = 0; i < nGPUs; i++) {
        d_sendbufs[i] = d_buffers[i * nBuffersPerGPU + 0];     // Even indices: send buffers
        d_recvbufs[i] = d_buffers[i * nBuffersPerGPU + 1];     // Odd indices: recv buffers
    }

    printf("\n=== Testing Error Handling ===\n");
    
    // Test the NCCL operation with proper error handling
    if (test_nccl_operation_with_error_handling(nGPUs, comms, streams, 
                                               d_sendbufs, d_recvbufs, count) != 0) {
        fprintf(stderr, "NCCL operation failed with errors\n");
        safe_cleanup(comms, streams, d_buffers, nGPUs, nBuffersPerGPU);
        free(d_sendbufs);
        free(d_recvbufs);
        free(comms);
        free(streams);
        free(d_buffers);
        return -1;
    }
    
    // Validate results after AllReduce (each GPU should have the same sum)
    printf("\nValidating AllReduce results...\n");
    int validation_success = 1;
    for (int gpu = 0; gpu < nGPUs; gpu++) {
        float* h_result = (float*)malloc(sizeof(float) * count);
        CUDACHECK(cudaSetDevice(gpu));
        CUDACHECK(cudaMemcpy(h_result, d_recvbufs[gpu], sizeof(float) * count, cudaMemcpyDeviceToHost));
        
        // Expected sum: each GPU contributed values starting with its ID
        // GPU 0: [1000, 1001, ..., 1023] 
        // GPU 1: [2000, 2001, ..., 2023]
        // After AllReduce SUM: [1000+2000+3000+4000, 1001+2001+3001+4001, ...]
        int errors = 0;
        for (int i = 0; i < count; i++) {
            float expected = 0;
            for (int g = 0; g < nGPUs; g++) {
                expected += (g + 1) * 1000.0f + i;
            }
            
            if (abs(h_result[i] - expected) > 1e-3) {
                if (errors < 5) {  // Limit output
                    fprintf(stderr, "Validation error on GPU %d at index %d: got %.3f, expected %.3f\n", 
                            gpu, i, h_result[i], expected);
                }
                errors++;
            }
        }
        
        if (errors > 0) {
            fprintf(stderr, "Validation failed on GPU %d: %d errors\n", gpu, errors);
            validation_success = 0;
        } else {
            printf("  GPU %d: Validation passed\n", gpu);
        }
        
        free(h_result);
    }
    
    if (!validation_success) {
        fprintf(stderr, "Result validation failed\n");
    } else {
        printf("All validations passed!\n");
    }

    printf("\n=== Testing Graceful Shutdown ===\n");
    printf("Simulating potential long-running operation...\n");
    
    // Simulate a long-running operation where we check for shutdown signals
    for (int i = 0; i < 5; i++) {
        if (shutdown_requested) {
            fprintf(stderr, "Shutdown requested during operation, cleaning up...\n");
            break;
        }
        
        // Perform another AllReduce operation
        if (test_nccl_operation_with_error_handling(nGPUs, comms, streams, 
                                                   d_sendbufs, d_recvbufs, count) != 0) {
            fprintf(stderr, "Operation failed during loop\n");
            break;
        }
        
        printf("  Completed iteration %d/%d\n", i+1, 5);
        sleep(1);  // Simulate work
        
        if (shutdown_requested) {
            fprintf(stderr, "Shutdown requested, exiting loop early\n");
            break;
        }
    }
    
    // Perform safe cleanup
    safe_cleanup(comms, streams, d_buffers, nGPUs, nBuffersPerGPU);
    
    // Free host memory
    free(d_sendbufs);
    free(d_recvbufs);
    free(comms);
    free(streams);
    free(d_buffers);

    printf("\nTroubleshooting and best practices demonstration completed successfully!\n");
    printf("Key lessons:\n");
    printf("- Always check return values from NCCL and CUDA functions\n");
    printf("- Implement proper error handling and cleanup\n");
    printf("- Use ncclGroupStart/ncclGroupEnd for multiple operations\n");
    printf("- Handle signals for graceful shutdown in long-running applications\n");
    printf("- Validate results to catch subtle bugs\n");

    return 0;
}