/*
 * CUDA Tensor Cores Tutorial
 * 
 * This tutorial demonstrates tensor core operations using WMMA API.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Check if we have tensor core support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;

// Kernel 1: Basic 16x16x16 matrix multiplication using tensor cores
__global__ void wmma_example(half* a, half* b, float* c, float* d, int m, int n, int k) {
    // Tile dimensions for tensor cores (16x16 output, 16x16 input)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate thread's warp position in the matrix
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Check bounds
    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) return;
    
    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c, frag_d;
    
    // Calculate addresses for this warp's tile
    int a_offset = warpM * WMMA_M * k;  // Starting row for matrix A
    int b_offset = warpN * WMMA_N;      // Starting column for matrix B
    int c_offset = warpM * WMMA_M * n + warpN * WMMA_N;  // Starting position for matrix C/D
    
    // Initialize accumulator to zero
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Perform matrix multiplication: d = a*b + c
    // Iterate over the K dimension in chunks of WMMA_K
    for (int i = 0; i < k; i += WMMA_K) {
        // Load fragments from global memory
        wmma::load_matrix_sync(frag_a, a + a_offset + i, k);
        wmma::load_matrix_sync(frag_b, b + i * n + b_offset, n);
        
        // Perform matrix multiply-accumulate: frag_c = frag_a * frag_b + frag_c
        wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);
        
        // Update accumulator for next iteration
        frag_c = frag_d;
    }
    
    // Store result back to global memory
    wmma::store_matrix_sync(d + c_offset, frag_d, n, wmma::mem_row_major);
}

// Kernel 2: Larger GEMM using tensor cores with tiling
__global__ void gemm_tensor_core(half* A, half* B, float* C, float* D, 
                                 int M, int N, int K) {
    // Tile sizes for tensor cores
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate which tile this warp will process
    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int laneId = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    
    // Calculate which tile of the output matrix this block handles
    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    
    // Calculate the starting position for this block
    int blockM = blockRow * WMMA_M;
    int blockN = blockCol * WMMA_N;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Loop over K dimension in chunks
    for (int k = 0; k < K; k += WMMA_K) {
        // Load tiles from A and B
        wmma::load_matrix_sync(frag_a, 
            A + (blockM + laneId / 2) * K + k + (laneId % 2) * 8,  // Complex addressing for A
            K);
        wmma::load_matrix_sync(frag_b, 
            B + (k + laneId / 2) * N + blockN + (laneId % 2) * 8,  // Complex addressing for B
            N);
        
        // Perform MMA operation
        wmma::mma_sync(acc_frag, frag_a, frag_b, acc_frag);
    }
    
    // Store result to D
    wmma::store_matrix_sync(D + blockM * N + blockN, acc_frag, N, wmma::mem_row_major);
}

// Kernel 3: Simple tensor core example with small matrices
__global__ void simple_tensor_core_example(half* a, half* b, float* c) {
    const int M = 16, N = 16, K = 16;
    
    // Only one warp performs the operation
    if (threadIdx.x % 32 != 0) return;
    
    wmma::fragment<wmma::matrix_a, M, N, K, half> frag_a;
    wmma::fragment<wmma::matrix_b, M, N, K, half> frag_b;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_c;
    
    // Load matrices
    wmma::load_matrix_sync(frag_a, a, K, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_b, b, K, wmma::mem_col_major);
    wmma::load_matrix_sync(frag_c, c, N, wmma::mem_row_major);
    
    // Perform C = A * B + C
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    
    // Store result
    wmma::store_matrix_sync(c, frag_c, N, wmma::mem_row_major);
}

#endif // __CUDA_ARCH__ >= 700

// Host function to initialize matrices
void initialize_matrices(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    printf("=== CUDA Tensor Cores Tutorial ===\n\n");
    
    // Check for tensor core support
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 7) {
        printf("Tensor cores not supported on this device (need compute capability 7.0+)\n");
        printf("Tutorial completed (no tensor core operations performed).\n");
        return 0;
    }
    
    printf("Tensor cores are supported on this device.\n\n");
    
    // Matrix dimensions for tensor core operations
    const int M = 64, N = 64, K = 64;  // Must be multiples of 16 for basic tensor cores
    const int size_a = M * K;
    const int size_b = K * N;
    const int size_c = M * N;
    
    size_t size_a_bytes = size_a * sizeof(half);
    size_t size_b_bytes = size_b * sizeof(half);
    size_t size_c_bytes = size_c * sizeof(float);
    
    // Allocate host memory
    float *h_A_f, *h_B_f, *h_C_f;
    half *h_A_h, *h_B_h;
    float *h_D_f;  // Result matrix
    
    h_A_f = (float*)malloc(size_a_bytes);
    h_B_f = (float*)malloc(size_b_bytes);
    h_C_f = (float*)malloc(size_c_bytes);
    h_A_h = (half*)malloc(size_a_bytes);
    h_B_h = (half*)malloc(size_b_bytes);
    h_D_f = (float*)malloc(size_c_bytes);
    
    // Initialize matrices with random values
    srand(2023);  // Fixed seed for reproducible results
    initialize_matrices(h_A_f, h_B_f, h_C_f, M, N, K);
    
    // Convert to half precision for tensor cores
    for (int i = 0; i < size_a; i++) {
        h_A_h[i] = __float2half(h_A_f[i]);
    }
    for (int i = 0; i < size_b; i++) {
        h_B_h[i] = __float2half(h_B_f[i]);
    }
    for (int i = 0; i < size_c; i++) {
        h_C_f[i] = h_C_f[i];  // Keep as float for accumulator
    }
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C, *d_D;
    
    cudaMalloc(&d_A, size_a_bytes);
    cudaMalloc(&d_B, size_b_bytes);
    cudaMalloc(&d_C, size_c_bytes);
    cudaMalloc(&d_D, size_c_bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A_h, size_a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_h, size_b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_f, size_c_bytes, cudaMemcpyHostToDevice);
    
    // Example 1: Basic tensor core operation
    printf("1. Basic Tensor Core Matrix Multiplication:\n");
    printf("   Computing C = A * B + C using tensor cores\n");
    printf("   Matrix dimensions: %dx%d = %dx%d * %dx%d\n", M, N, M, K, K, N);
    
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dim3 block(32, 1);  // One warp per block for basic example
    
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    wmma_example<<<grid, block>>>(d_A, d_B, d_C, d_D, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D_f, d_D, size_c_bytes, cudaMemcpyDeviceToHost);
    
    printf("   Operation completed successfully.\n");
    printf("   First few results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", h_D_f[i]);
    }
    printf("\n\n");
    #else
    printf("   Skipping tensor core operations (compile with appropriate architecture flags)\n\n");
    #endif
    
    // Example 2: Larger GEMM with tensor cores
    printf("2. Larger GEMM using Tensor Cores:\n");
    printf("   Computing larger matrix multiplication using tensor cores\n");
    
    dim3 grid2((M + 15) / 16, (N + 15) / 16);
    dim3 block2(32, 1);
    
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    gemm_tensor_core<<<grid2, block2>>>(d_A, d_B, d_C, d_D, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D_f, d_D, size_c_bytes, cudaMemcpyDeviceToHost);
    
    printf("   GEMM completed successfully.\n");
    printf("   First few results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", h_D_f[i]);
    }
    printf("\n\n");
    #else
    printf("   Skipping tensor core operations (compile with appropriate architecture flags)\n\n");
    #endif
    
    // Example 3: Simple tensor core example
    printf("3. Simple Tensor Core Example:\n");
    const int SMALL_M = 16, SMALL_N = 16, SMALL_K = 16;
    
    half *h_small_a, *h_small_b;
    float *h_small_c;
    half *d_small_a, *d_small_b;
    float *d_small_c;
    
    h_small_a = (half*)malloc(SMALL_M * SMALL_K * sizeof(half));
    h_small_b = (half*)malloc(SMALL_K * SMALL_N * sizeof(half));
    h_small_c = (float*)malloc(SMALL_M * SMALL_N * sizeof(float));
    
    cudaMalloc(&d_small_a, SMALL_M * SMALL_K * sizeof(half));
    cudaMalloc(&d_small_b, SMALL_K * SMALL_N * sizeof(half));
    cudaMalloc(&d_small_c, SMALL_M * SMALL_N * sizeof(float));
    
    // Initialize small matrices
    for (int i = 0; i < SMALL_M * SMALL_K; i++) {
        h_small_a[i] = __float2half(static_cast<float>(i % 5 + 1) / 10.0f);
    }
    for (int i = 0; i < SMALL_K * SMALL_N; i++) {
        h_small_b[i] = __float2half(static_cast<float>(i % 3 + 1) / 10.0f);
    }
    for (int i = 0; i < SMALL_M * SMALL_N; i++) {
        h_small_c[i] = 0.0f;
    }
    
    cudaMemcpy(d_small_a, h_small_a, SMALL_M * SMALL_K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_small_b, h_small_b, SMALL_K * SMALL_N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_small_c, h_small_c, SMALL_M * SMALL_N * sizeof(float), cudaMemcpyHostToDevice);
    
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    simple_tensor_core_example<<<1, 32>>>(d_small_a, d_small_b, d_small_c);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_small_c, d_small_c, SMALL_M * SMALL_N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("   Simple tensor core operation completed.\n");
    printf("   Result matrix (first row): ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", h_small_c[i]);
    }
    printf("\n\n");
    #else
    printf("   Skipping tensor core operations (compile with appropriate architecture flags)\n\n");
    #endif
    
    // Information about tensor cores
    printf("Tensor Core Information:\n");
    printf("- Available on GPUs with compute capability 7.0+ (V100, T4, A100, H100, etc.)\n");
    printf("- Perform 4x4x4, 8x8x16, or 16x16x16 matrix operations in one instruction\n");
    printf("- Support mixed precision: FP16 inputs with FP32 accumulation\n");
    printf("- Much higher throughput than regular CUDA cores for matrix operations\n");
    printf("- Essential for high-performance deep learning workloads\n\n");
    
    // Cleanup
    free(h_A_f);
    free(h_B_f);
    free(h_C_f);
    free(h_A_h);
    free(h_B_h);
    free(h_D_f);
    free(h_small_a);
    free(h_small_b);
    free(h_small_c);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_small_a);
    cudaFree(d_small_b);
    cudaFree(d_small_c);
    
    printf("Tutorial completed!\n");
    return 0;
}