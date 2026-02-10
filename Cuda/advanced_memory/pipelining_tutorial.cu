/*
 * CUDA Software Pipelining Tutorial
 *
 * This tutorial demonstrates software pipelining techniques to overlap memory transfers with computation.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Basic pipelined kernel with double buffering
__global__ void basic_pipelined_kernel(float* input, float* output, int n) {
    // Allocate shared memory for double buffering
    extern __shared__ float sMem[];
    float* buffer0 = &sMem[0];
    float* buffer1 = &sMem[blockDim.x];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Prime the pipeline: load first chunk
    if (tid < n && threadIdx.x < blockDim.x) {
        buffer0[threadIdx.x] = input[tid];
    }
    
    __syncthreads();
    
    // Main pipeline loop
    for (int i = 0; i < (n - blockDim.x) / blockDim.x; i++) {
        int next_tid = tid + blockDim.x;
        
        // Load phase: load data for next iteration while computing current
        if (next_tid < n && threadIdx.x < blockDim.x) {
            buffer1[threadIdx.x] = input[next_tid];
        }
        
        __syncthreads(); // Ensure loads are complete
        
        // Compute phase: process current buffer
        float result = buffer0[threadIdx.x] * 2.0f + 1.0f;
        
        // Store phase: write result
        if (tid < n && threadIdx.x < blockDim.x) {
            output[tid] = result;
        }
        
        __syncthreads(); // Ensure stores are complete
        
        // Swap buffers for next iteration
        float* temp = buffer0;
        buffer0 = buffer1;
        buffer1 = temp;
        
        tid = next_tid;
    }
    
    // Handle remaining elements in the last buffer
    if (tid < n && threadIdx.x < blockDim.x) {
        float result = buffer0[threadIdx.x] * 2.0f + 1.0f;
        output[tid] = result;
    }
}

// Kernel 2: Non-pipelined version for comparison
__global__ void non_pipelined_kernel(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Non-overlapped computation
        float value = input[tid];
        float result = value * 2.0f + 1.0f;
        output[tid] = result;
    }
}

// Kernel 3: Pipelined matrix multiplication example
__global__ void pipelined_gemm(float* A, float* B, float* C, int N) {
    __shared__ float As[16][17];  // +1 to avoid bank conflicts
    __shared__ float Bs[16][17];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    
    float result = 0.0f;
    
    // Pipeline the k-dimension loop
    for (int k = 0; k < N; k += 16) {
        // Load next tile while computing previous
        if (row < N && k + tx < N) {
            As[ty][tx] = A[row * N + k + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (k + ty < N && col < N) {
            Bs[ty][tx] = B[(k + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute with loaded data
        for (int i = 0; i < 16; i++) {
            result += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

// Kernel 4: Simple pipelined kernel with async copy simulation
__global__ void simple_pipeline_kernel(float* input, float* output, int n) {
    // Simulate pipeline with two stages
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    // Two-stage pipeline: process current while preparing next
    if (tid < n) {
        // Stage 1: Load data
        shared_mem[local_tid] = input[tid];
        __syncthreads();
        
        // Stage 2: Process data
        float processed = shared_mem[local_tid] * 2.0f + 1.0f;
        __syncthreads();
        
        // Stage 3: Store result
        output[tid] = processed;
    }
}

// Helper function to measure kernel execution time
float measureKernelTime(void (*kernel)(float*, float*, int), float* input, float* output, int n, int sharedMemSize = 0) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (sharedMemSize > 0) {
        kernel<<<(n + 255) / 256, 256, sharedMemSize>>>(input, output, n);
    } else {
        kernel<<<(n + 255) / 256, 256>>>(input, output, n);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    printf("=== CUDA Software Pipelining Tutorial ===\n\n");

    const int N = 1024 * 1024;  // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input, *h_output_basic, *h_output_non_pipelined;
    h_input = (float*)malloc(size);
    h_output_basic = (float*)malloc(size);
    h_output_non_pipelined = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    // Allocate device memory
    float *d_input, *d_output_basic, *d_output_non_pipelined;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output_basic, size);
    cudaMalloc(&d_output_non_pipelined, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Example 1: Non-pipelined kernel
    printf("1. Non-Pipelined Kernel:\n");
    float time_non_pipelined = measureKernelTime(non_pipelined_kernel, d_input, d_output_non_pipelined, N);
    cudaMemcpy(h_output_non_pipelined, d_output_non_pipelined, size, cudaMemcpyDeviceToHost);
    printf("   Time: %.3f ms\n", time_non_pipelined);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output_non_pipelined[i]);
    }
    printf("\n\n");

    // Example 2: Basic pipelined kernel
    printf("2. Basic Pipelined Kernel:\n");
    int sharedMemSize = 2 * 256 * sizeof(float);  // Double buffer for 256 threads
    float time_pipelined = measureKernelTime(basic_pipelined_kernel, d_input, d_output_basic, N, sharedMemSize);
    cudaMemcpy(h_output_basic, d_output_basic, size, cudaMemcpyDeviceToHost);
    printf("   Time: %.3f ms\n", time_pipelined);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output_basic[i]);
    }
    printf("\n\n");

    // Performance comparison
    printf("3. Performance Comparison:\n");
    printf("   Non-pipelined time: %.3f ms\n", time_non_pipelined);
    printf("   Pipelined time:     %.3f ms\n", time_pipelined);
    if (time_non_pipelined > 0) {
        printf("   Speedup:            %.2fx\n", time_non_pipelined / time_pipelined);
    }
    printf("\n");

    // Example 4: Simple pipeline kernel
    printf("4. Simple Pipeline Kernel:\n");
    sharedMemSize = 256 * sizeof(float);  // Single buffer for 256 threads
    float time_simple = measureKernelTime(simple_pipeline_kernel, d_input, d_output_basic, N, sharedMemSize);
    printf("   Time: %.3f ms\n", time_simple);
    printf("\n");

    // Example 5: Matrix multiplication example (small matrices for demo)
    printf("5. Pipelined Matrix Multiplication Example:\n");
    const int MAT_SIZE = 64;
    const int MAT_ELEM = MAT_SIZE * MAT_SIZE;
    size_t mat_size = MAT_ELEM * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(mat_size);
    h_B = (float*)malloc(mat_size);
    h_C = (float*)malloc(mat_size);
    cudaMalloc(&d_A, mat_size);
    cudaMalloc(&d_B, mat_size);
    cudaMalloc(&d_C, mat_size);

    // Initialize matrices
    for (int i = 0; i < MAT_ELEM; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaMemcpy(d_A, h_A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((MAT_SIZE + 15) / 16, (MAT_SIZE + 15) / 16);

    pipelined_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, MAT_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, mat_size, cudaMemcpyDeviceToHost);
    printf("   64x64 matrix multiplication completed.\n");
    printf("   Result at [0,0]: %.1f (expected: %.1f)\n", h_C[0], MAT_SIZE * 2.0f);
    printf("\n");

    printf("Key Concepts Demonstrated:\n");
    printf("- Software pipelining overlaps multiple stages of computation\n");
    printf("- Double/triple buffering enables pipelining in shared memory\n");
    printf("- Pipelining can hide memory latency by overlapping transfers with computation\n");
    printf("- Performance gains depend on the balance between computation and memory access\n");

    // Cleanup
    free(h_input);
    free(h_output_basic);
    free(h_output_non_pipelined);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_input);
    cudaFree(d_output_basic);
    cudaFree(d_output_non_pipelined);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nTutorial completed!\n");
    return 0;
}