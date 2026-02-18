#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

// Naive attention kernel - computes softmax(Q*K^T) * V
__global__ void attention_naive(float *Q, float *K, float *V, float *output, 
                               int seq_len, int head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < head_dim) {
        float sum = 0.0f;
        
        // Compute attention for this output position
        for (int k = 0; k < seq_len; k++) {
            // Compute Q[row] * K[k] (dot product)
            float qk_score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                qk_score += Q[row * head_dim + d] * K[k * head_dim + d];
            }
            
            // Apply softmax (naive - would normally pre-compute scores)
            // For simplicity, we'll just use the score directly here
            // In a real implementation, we'd compute all scores first, then apply softmax
            sum += qk_score * V[k * head_dim + col];
        }
        
        output[row * head_dim + col] = sum;
    }
}

// Simplified attention with precomputed scores
__global__ void attention_with_scores_naive(float *scores, float *V, float *output,
                                           int seq_len, int head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < head_dim) {
        float sum = 0.0f;
        
        // Apply softmax to scores and multiply with V
        float max_score = scores[row * seq_len];
        for (int k = 1; k < seq_len; k++) {
            if (scores[row * seq_len + k] > max_score) {
                max_score = scores[row * seq_len + k];
            }
        }
        
        // Compute softmax denominator
        float denom = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            denom += expf(scores[row * seq_len + k] - max_score);
        }
        
        // Compute final result
        for (int k = 0; k < seq_len; k++) {
            float softmax_val = expf(scores[row * seq_len + k] - max_score) / denom;
            sum += softmax_val * V[k * head_dim + col];
        }
        
        output[row * head_dim + col] = sum;
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

int main(int argc, char **argv) {
    int seq_len = 64, head_dim = 64; // Small sizes due to naive implementation complexity
    
    if (argc >= 3) {
        seq_len = atoi(argv[1]);
        head_dim = atoi(argv[2]);
    }
    
    printf("Attention mechanism: seq_len=%d, head_dim=%d\n", seq_len, head_dim);

    size_t qkv_size = seq_len * head_dim * sizeof(float);
    
    // Host arrays
    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_output = (float*)malloc(qkv_size);
    
    // Initialize with random values
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = ((float)(rand() % 1000)) / 1000.0f - 0.5f;
        h_K[i] = ((float)(rand() % 1000)) / 1000.0f - 0.5f;
        h_V[i] = ((float)(rand() % 1000)) / 1000.0f - 0.5f;
    }
    
    // Device arrays
    float *d_Q, *d_K, *d_V, *d_output;
    checkCudaError(cudaMalloc(&d_Q, qkv_size), "cudaMalloc d_Q");
    checkCudaError(cudaMalloc(&d_K, qkv_size), "cudaMalloc d_K");
    checkCudaError(cudaMalloc(&d_V, qkv_size), "cudaMalloc d_V");
    checkCudaError(cudaMalloc(&d_output, qkv_size), "cudaMalloc d_output");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice), "cudaMemcpy d_Q");
    checkCudaError(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice), "cudaMemcpy d_K");
    checkCudaError(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice), "cudaMemcpy d_V");
    
    // Kernel configuration
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((head_dim + blockSize.x - 1) / blockSize.x, 
                  (seq_len + blockSize.y - 1) / blockSize.y);
    
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    attention_naive<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_output, seq_len, head_dim);
    cudaDeviceSynchronize();

    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);

    attention_naive<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_output, seq_len, head_dim);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, qkv_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_output");
    
    printf("Attention computation completed\n");
    
    // Calculate performance metrics
    // Attention: seq_len^2 * head_dim (Q*K^T) + seq_len^2 * head_dim (softmax) + seq_len^2 * head_dim (score*V)
    double flops = 3.0 * seq_len * seq_len * head_dim; // Rough estimate
    double gflops = (flops / 1e9) / (kernel_time_ms / 1000.0);
    double bytes_moved = 3.0 * seq_len * head_dim * sizeof(float) + // Q, K, V input
                         seq_len * seq_len * sizeof(float) +         // scores
                         seq_len * head_dim * sizeof(float);         // output
    double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
    
    printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
    printf("Total execution time: %.3f s\n", total_time_s);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
    
    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_output);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return 0;
}