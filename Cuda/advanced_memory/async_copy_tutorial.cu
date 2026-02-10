/*
 * CUDA Asynchronous Copy Tutorial
 * 
 * This tutorial demonstrates asynchronous memory operations using cp.async and related techniques.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

// Kernel 1: Manual double-buffered async copy pattern
__global__ void manual_async_copy(float* input, float* output, int n) {
    __shared__ float buffer_a[256];
    __shared__ float buffer_b[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start = bid * 512;  // Each block processes 512 elements
    
    float* current_buffer = buffer_a;
    float* next_buffer = buffer_b;
    
    // Prefetch first chunk
    if (block_start + tid < n) {
        current_buffer[tid] = input[block_start + tid];
    }
    if (block_start + tid + 256 < n && tid < 256) {
        next_buffer[tid] = input[block_start + tid + 256];
    }
    __syncthreads();
    
    // Process first chunk while loading second
    if (block_start + tid < n) {
        float result = current_buffer[tid] * 2.0f + 1.0f;
        output[block_start + tid] = result;
    }
    
    // Continue processing while swapping buffers
    for (int offset = 512; block_start + offset < n; offset += 512) {
        // Swap buffers
        float* temp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
        
        // Load next chunk
        __syncthreads();
        if (block_start + offset + tid < n) {
            current_buffer[tid] = input[block_start + offset + tid];
        }
        if (block_start + offset + tid + 256 < n && tid < 256) {
            current_buffer[tid + 256] = input[block_start + offset + tid + 256];
        }
        __syncthreads();
        
        // Process current chunk
        if (block_start + offset - 256 + tid < n) {
            float result = next_buffer[tid] * 2.0f + 1.0f;
            output[block_start + offset - 256 + tid] = result;
        }
    }
    
    // Process final chunk
    __syncthreads();
    int final_offset = ((n - 1) / 512) * 512;
    if (final_offset + tid < n) {
        float result = current_buffer[tid] * 2.0f + 1.0f;
        output[final_offset + tid] = result;
    }
}

// Kernel 2: Async copy using cooperative groups (requires compute capability 8.0+)
__global__ void coop_async_copy(float* input, float* output, int n) {
    extern __shared__ float shared_mem[];
    
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);  // Warp-level operations
    
    // Process data in chunks
    for (int i = 0; i < n; i += blockDim.x) {
        // Async copy using cooperative groups
        cg::memcpy_async(block, shared_mem, input + i, 
                         sizeof(float) * min(blockDim.x, n - i));
        
        // Wait for async copy to complete
        cg::wait(block);
        
        // Process data
        int idx = i + threadIdx.x;
        if (idx < n) {
            shared_mem[threadIdx.x] = shared_mem[threadIdx.x] * 2.0f + 1.0f;
            output[idx] = shared_mem[threadIdx.x];
        }
    }
}

// Kernel 3: Simple async copy with pipeline
__global__ void pipelined_async_copy(float* input, float* output, int n) {
    __shared__ float shared_data[512];  // Two buffers of 256 each
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = 512;
    int block_start = bid * block_size;
    
    // Phase 1: Load first chunk
    if (block_start + tid < n) {
        shared_data[tid] = input[block_start + tid];
    }
    __syncthreads();
    
    // Process first half while loading second half
    if (tid < 256 && block_start + tid < n) {
        float result = shared_data[tid] * 2.0f + 1.0f;
        output[block_start + tid] = result;
    }
    
    // Load second half while processing first half
    if (block_start + tid + 256 < n && tid < 256) {
        shared_data[tid + 256] = input[block_start + tid + 256];
    }
    __syncthreads();
    
    // Process second half
    if (tid < 256 && block_start + tid + 256 < n) {
        float result = shared_data[tid + 256] * 2.0f + 1.0f;
        output[block_start + tid + 256] = result;
    }
}

// Kernel 4: Async copy with proper synchronization
__global__ void sync_async_copy(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data asynchronously (simulated with syncthreads)
    if (tid < n) {
        sdata[threadIdx.x] = input[tid];
    }
    __syncthreads();
    
    // Process data
    if (tid < n) {
        float val = sdata[threadIdx.x];
        val = val * 2.0f + 1.0f;
        sdata[threadIdx.x] = val;
    }
    __syncthreads();
    
    // Store data
    if (tid < n) {
        output[tid] = sdata[threadIdx.x];
    }
}

// Kernel 5: Double buffering with async copy pattern
__global__ void double_buffer_async(float* input, float* output, int n) {
    __shared__ float buffer_a[256];
    __shared__ float buffer_b[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start = bid * 256;
    
    float* current_buffer = buffer_a;
    float* next_buffer = buffer_b;
    
    // Process data in overlapping chunks
    for (int chunk = 0; chunk < n; chunk += 256) {
        // Start loading next chunk while processing current
        if (chunk + 256 < n && tid < 256) {
            next_buffer[tid] = input[chunk + 256 + tid];
        }
        
        // Process current chunk
        if (chunk + tid < n) {
            current_buffer[tid] = input[chunk + tid];
            float processed = current_buffer[tid] * 2.0f + 1.0f;
            output[chunk + tid] = processed;
        }
        
        __syncthreads();
        
        // Swap buffers
        float* temp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
    }
}

int main() {
    printf("=== CUDA Asynchronous Copy Tutorial ===\n\n");
    
    const int N = 8192;  // Larger dataset to see async benefits
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output1, *h_output2, *h_output3, *h_output4, *h_output5;
    h_input = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    h_output3 = (float*)malloc(size);
    h_output4 = (float*)malloc(size);
    h_output5 = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output1, *d_output2, *d_output3, *d_output4, *d_output5;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    cudaMalloc(&d_output3, size);
    cudaMalloc(&d_output4, size);
    cudaMalloc(&d_output5, size);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Example 1: Manual double-buffered async copy
    printf("1. Manual Double-Buffered Async Copy:\n");
    manual_async_copy<<<(N + 511) / 512, 256>>>(d_input, d_output1, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    printf("   Completed with manual double-buffering\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n\n");
    
    // Example 2: Cooperative async copy
    printf("2. Cooperative Async Copy:\n");
    size_t shared_mem_size = 256 * sizeof(float);
    coop_async_copy<<<(N + 255) / 256, 256, shared_mem_size>>>(d_input, d_output2, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output2, d_output2, size, cudaMemcpyDeviceToHost);
    printf("   Completed with cooperative groups async copy\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output2[i]);
    }
    printf("\n\n");
    
    // Example 3: Pipelined async copy
    printf("3. Pipelined Async Copy:\n");
    pipelined_async_copy<<<(N + 511) / 512, 256>>>(d_input, d_output3, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output3, d_output3, size, cudaMemcpyDeviceToHost);
    printf("   Completed with pipelined approach\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output3[i]);
    }
    printf("\n\n");
    
    // Example 4: Sync async copy (simulated)
    printf("4. Synchronized Async Copy (Simulated):\n");
    sync_async_copy<<<(N + 255) / 256, 256>>>(d_input, d_output4, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output4, d_output4, size, cudaMemcpyDeviceToHost);
    printf("   Completed with simulated async approach\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output4[i]);
    }
    printf("\n\n");
    
    // Example 5: Double buffering async copy
    printf("5. Double Buffering Async Copy:\n");
    double_buffer_async<<<(N + 255) / 256, 256>>>(d_input, d_output5, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output5, d_output5, size, cudaMemcpyDeviceToHost);
    printf("   Completed with double buffering\n");
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output5[i]);
    }
    printf("\n\n");
    
    // Performance comparison note
    printf("Performance Notes:\n");
    printf("- True async copy (cp.async) requires newer GPU architectures (Ampere+)\n");
    printf("- The techniques shown simulate async behavior using double buffering\n");
    printf("- Real async copy can overlap memory transfers with computation\n");
    printf("- This hides memory latency and improves overall performance\n\n");
    
    printf("Key Concepts:\n");
    printf("- Asynchronous copies overlap memory transfers with computation\n");
    printf("- Double buffering enables overlapping of load/compute/store phases\n");
    printf("- Proper synchronization is crucial for correctness\n");
    printf("- Async operations are especially beneficial for memory-bound kernels\n\n");
    
    // Cleanup
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    free(h_output4);
    free(h_output5);
    
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    
    printf("Tutorial completed!\n");
    return 0;
}