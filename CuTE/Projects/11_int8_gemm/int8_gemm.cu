/**
 * Project 11: INT8 GEMM with Fused Dequantization
 * 
 * Objective: Implement INT8 GEMM with per-channel dequantization
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct INT8GemmConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int TM = 16;
    static constexpr int TN = 16;
};

/**
 * TODO: Implement INT8 GEMM with fused dequantization
 * 
 * Key concepts:
 * - INT8 matrix multiplication
 * - Per-channel scale application
 * - Fused dequantization at output
 */
__global__ void int8_gemm_kernel(
    int8_t* A, int8_t* B, float* C,
    float* scale_A, float* scale_B,
    int M, int K, int N) {
    
    using Config = INT8GemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    // TODO: Implement INT8 GEMM
    // 1. Load INT8 tiles
    // 2. Compute INT8 matrix multiply (accumulate in int32)
    // 3. Apply scales and convert to FP32 output
    
    (void)A; (void)B; (void)C;
    (void)scale_A; (void)scale_B;
    (void)M; (void)K; (void)N;
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

void init_int8_matrix(std::vector<int8_t>& m, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) m[i] = (int8_t)((i % 200) - 100);
}

void int8_gemm_reference(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                          std::vector<float>& C, int M, int K, int N,
                          const std::vector<float>& scale_A, const std::vector<float>& scale_B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = (float)acc * scale_A[i] * scale_B[j];
        }
    }
}

int main() {
    using Config = cute::INT8GemmConfig;
    const int M = 256, K = 128, N = 256;
    
    const dim3 block(Config::TM, Config::TN);
    const dim3 grid((N + Config::BN - 1) / Config::BN,
                    (M + Config::BM - 1) / Config::BM);
    
    std::cout << "=== Project 11: INT8 GEMM with Fused Dequantization ===" << std::endl;
    std::cout << "Matrix: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    
    std::vector<int8_t> h_A(M * K), h_B(K * N);
    std::vector<float> h_C(M * N, 0), h_ref(M * N);
    std::vector<float> h_scale_A(M, 0.01f), h_scale_B(N, 0.01f);
    
    init_int8_matrix(h_A, M, K);
    init_int8_matrix(h_B, K, N);
    
    int8_gemm_reference(h_A, h_B, h_ref, M, K, N, h_scale_A, h_scale_B);
    
    int8_t *d_A, *d_B;
    float *d_C, *d_scale_A, *d_scale_B;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_A, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_B, N * sizeof(float)));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A.data(), M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_B, h_scale_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    cute::int8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, d_scale_A, d_scale_B, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << (max_err < 1.0f ? "[PASS] INT8 GEMM" : "[FAIL]") << std::endl;
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_scale_A); cudaFree(d_scale_B);
    return 0;
}
