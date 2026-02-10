/*
 * CUDA Warp-Level Primitives Tutorial
 * 
 * This tutorial demonstrates warp-level operations and primitives.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Basic shuffle operations
__global__ void basic_shuffle(float* input, float* output) {
    int laneId = threadIdx.x % 32;  // Thread ID within warp
    
    float value = input[threadIdx.x];
    
    // Thread gets value from thread at (laneId + 1) % 32
    float shfl_value = __shfl_sync(0xFFFFFFFF, value, (laneId + 1) % 32);
    
    output[threadIdx.x] = shfl_value;
}

// Kernel 2: Shuffle up operations
__global__ void shuffle_up_example(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Each thread gets value from (laneId + 4) thread
    float shifted = __shfl_up_sync(0xFFFFFFFF, value, 4);
    
    output[threadIdx.x] = shifted;
}

// Kernel 3: Shuffle down operations
__global__ void shuffle_down_example(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Each thread gets value from (laneId - 4) thread
    float shifted = __shfl_down_sync(0xFFFFFFFF, value, 4);
    
    output[threadIdx.x] = shifted;
}

// Kernel 4: Shuffle XOR operations
__global__ void shuffle_xor_example(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Each thread exchanges with thread at (laneId ^ 16) - pairs with thread+/-16
    float exchanged = __shfl_xor_sync(0xFFFFFFFF, value, 16);
    
    output[threadIdx.x] = exchanged;
}

// Kernel 5: Warp vote operations
__global__ void vote_operations(int* input, int* result_all, int* result_any, unsigned int* result_ballot) {
    int laneId = threadIdx.x % 32;
    
    // Condition: is the value greater than 5?
    bool condition = (input[threadIdx.x] > 5);
    
    // Check if ALL threads in the warp have the condition true
    int all_result = __all_sync(0xFFFFFFFF, condition);
    
    // Check if ANY thread in the warp has the condition true
    int any_result = __any_sync(0xFFFFFFFF, condition);
    
    // Get bitmask of threads with the condition true
    unsigned int ballot_result = __ballot_sync(0xFFFFFFFF, condition);
    
    // Only the first thread in each warp writes results
    if (laneId == 0) {
        result_all[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = all_result;
        result_any[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = any_result;
        result_ballot[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = ballot_result;
    }
}

// Device function for warp-level reduction
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Kernel 6: Warp-level reduction
__global__ void warp_sum_reduction(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    float sum = (tid < n) ? input[tid] : 0.0f;
    
    // Perform warp-level reduction
    sum = warpReduce(sum);
    
    // First thread in each warp writes partial result
    if (laneId == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warpId] = sum;
    }
}

// Device function for warp-level prefix sum
__device__ float warpPrefixSum(float val) {
    float result = val;
    for (int offset = 1; offset < 32; offset *= 2) {
        float temp = __shfl_up_sync(0xFFFFFFFF, result, offset);
        if ((threadIdx.x % 32) >= offset) {
            result += temp;
        }
    }
    return result;
}

// Kernel 7: Warp-level prefix sum
__global__ void warp_prefix_sum(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    
    if (tid < n) {
        float val = input[tid];
        float prefix_sum = warpPrefixSum(val);
        output[tid] = prefix_sum;
    }
}

// Kernel 8: Broadcast within warp
__global__ void warp_broadcast(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Broadcast value from thread 0 of each warp to all threads in that warp
    float broadcast_val = __shfl_sync(0xFFFFFFFF, value, laneId - (laneId % 32));
    
    output[threadIdx.x] = broadcast_val;
}

// Kernel 9: Match operations
__global__ void match_operations(int* input, unsigned int* result) {
    int laneId = threadIdx.x % 32;
    int value = input[threadIdx.x];
    
    // Returns mask of threads with same value as current thread
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, value);
    
    result[threadIdx.x] = match_mask;
}

int main() {
    printf("=== CUDA Warp-Level Primitives Tutorial ===\n\n");
    
    const int N = 256;  // Must be multiple of 32 for warp operations
    size_t size = N * sizeof(float);
    size_t int_size = N * sizeof(int);
    size_t uint_size = N * sizeof(unsigned int);
    
    // Allocate host memory
    float *h_input, *h_output1, *h_output2, *h_output3, *h_output4, *h_output5, *h_output6, *h_output7, *h_output8;
    int *h_int_input, *h_result_all, *h_result_any;
    unsigned int *h_result_ballot, *h_match_result;
    
    h_input = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    h_output3 = (float*)malloc(size);
    h_output4 = (float*)malloc(size);
    h_output5 = (float*)malloc(size);
    h_output6 = (float*)malloc(size);
    h_output7 = (float*)malloc(size);
    h_output8 = (float*)malloc(size);
    h_int_input = (int*)malloc(int_size);
    h_result_all = (int*)malloc(int_size);
    h_result_any = (int*)malloc(int_size);
    h_result_ballot = (unsigned int*)malloc(uint_size);
    h_match_result = (unsigned int*)malloc(uint_size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
        h_int_input[i] = i % 10;  // Values 0-9 repeating
    }
    
    // Allocate device memory
    float *d_input, *d_output1, *d_output2, *d_output3, *d_output4, *d_output5, *d_output6, *d_output7, *d_output8;
    int *d_int_input, *d_result_all, *d_result_any;
    unsigned int *d_result_ballot, *d_match_result;
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    cudaMalloc(&d_output3, size);
    cudaMalloc(&d_output4, size);
    cudaMalloc(&d_output5, size);
    cudaMalloc(&d_output6, size);
    cudaMalloc(&d_output7, size);
    cudaMalloc(&d_output8, size);
    cudaMalloc(&d_int_input, int_size);
    cudaMalloc(&d_result_all, int_size);
    cudaMalloc(&d_result_any, int_size);
    cudaMalloc(&d_result_ballot, uint_size);
    cudaMalloc(&d_match_result, uint_size);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_int_input, h_int_input, int_size, cudaMemcpyHostToDevice);
    
    // Example 1: Basic shuffle
    printf("1. Basic Shuffle Operations:\n");
    basic_shuffle<<<(N + 255) / 256, 256>>>(d_input, d_output1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n\n");
    
    // Example 2: Shuffle up
    printf("2. Shuffle Up Operations:\n");
    shuffle_up_example<<<(N + 255) / 256, 256>>>(d_input, d_output2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output2, d_output2, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output2[i]);
    }
    printf("\n\n");
    
    // Example 3: Shuffle down
    printf("3. Shuffle Down Operations:\n");
    shuffle_down_example<<<(N + 255) / 256, 256>>>(d_input, d_output3);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output3, d_output3, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output3[i]);
    }
    printf("\n\n");
    
    // Example 4: Shuffle XOR
    printf("4. Shuffle XOR Operations:\n");
    shuffle_xor_example<<<(N + 255) / 256, 256>>>(d_input, d_output4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output4, d_output4, size, cudaMemcpyDeviceToHost);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output4[i]);
    }
    printf("\n\n");
    
    // Example 5: Vote operations
    printf("5. Vote Operations:\n");
    vote_operations<<<(N + 255) / 256, 256>>>(d_int_input, d_result_all, d_result_any, d_result_ballot);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result_all, d_result_all, int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_any, d_result_any, int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_ballot, d_result_ballot, uint_size, cudaMemcpyDeviceToHost);
    
    printf("   All result (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_result_all[i]);
    }
    printf("\n   Any result (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_result_any[i]);
    }
    printf("\n   Ballot result (first 2): ");
    for (int i = 0; i < 2; i++) {
        printf("0x%X ", h_result_ballot[i]);
    }
    printf("\n\n");
    
    // Example 6: Warp-level reduction
    printf("6. Warp-Level Reduction:\n");
    warp_sum_reduction<<<(N + 255) / 256, 256>>>(d_input, d_output5, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output5, d_output5, size, cudaMemcpyDeviceToHost);
    printf("   Partial sums per warp (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_output5[i]);
    }
    printf("\n\n");
    
    // Example 7: Warp-level prefix sum
    printf("7. Warp-Level Prefix Sum:\n");
    warp_prefix_sum<<<(N + 255) / 256, 256>>>(d_input, d_output6, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output6, d_output6, size, cudaMemcpyDeviceToHost);
    printf("   Prefix sums (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output6[i]);
    }
    printf("\n\n");
    
    // Example 8: Broadcast
    printf("8. Warp Broadcast:\n");
    warp_broadcast<<<(N + 255) / 256, 256>>>(d_input, d_output7);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output7, d_output7, size, cudaMemcpyDeviceToHost);
    printf("   Broadcast results (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output7[i]);
    }
    printf("\n\n");
    
    // Example 9: Match operations
    printf("9. Match Operations:\n");
    match_operations<<<(N + 255) / 256, 256>>>(d_int_input, d_match_result);
    cudaDeviceSynchronize();
    cudaMemcpy(h_match_result, d_match_result, uint_size, cudaMemcpyDeviceToHost);
    printf("   Match masks (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("0x%X ", h_match_result[i]);
    }
    printf("\n\n");
    
    // Demonstrate warp divergence
    printf("10. Warp Divergence Example:\n");
    printf("    In a warp, if threads take different paths, the warp executes each path serially.\n");
    printf("    This can reduce performance compared to uniform execution paths.\n\n");
    
    printf("Key Takeaways:\n");
    printf("- Warp primitives enable efficient communication within a warp (32 threads)\n");
    printf("- Shuffles allow threads to exchange data directly\n");
    printf("- Vote operations perform boolean operations across the warp\n");
    printf("- Reductions and scans can be efficiently implemented using warp primitives\n");
    printf("- These operations are extremely fast (single instruction)\n");
    
    // Cleanup
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    free(h_output4);
    free(h_output5);
    free(h_output6);
    free(h_output7);
    free(h_output8);
    free(h_int_input);
    free(h_result_all);
    free(h_result_any);
    free(h_result_ballot);
    free(h_match_result);
    
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    cudaFree(d_output6);
    cudaFree(d_output7);
    cudaFree(d_output8);
    cudaFree(d_int_input);
    cudaFree(d_result_all);
    cudaFree(d_result_any);
    cudaFree(d_result_ballot);
    cudaFree(d_match_result);
    
    printf("\nTutorial completed!\n");
    return 0;
}