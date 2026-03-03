/**
 * Project 08: Pipelined GEMM with Async Copy
 * 
 * Objective: Implement software-pipelined GEMM using async memory operations
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct PipelinedGemmConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 8;
    static constexpr int NumStages = 3;  // Pipeline stages
    static constexpr int TM = 16;
    static constexpr int TN = 16;
};

/**
 * TODO: Implement pipelined GEMM with async copy
 * 
 * Key concepts:
 * - Multiple pipeline stages for overlapping load/compute
 * - cp.async for async memory operations (SM80+)
 * - cp.async.commit_group and cp.async.wait_group
 */
__global__ void pipelined_gemm_kernel(float* A, float* B, float* C,
                                       int M, int K, int N) {
    using Config = PipelinedGemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    // TODO: Implement pipelined GEMM
    // 1. Allocate multi-stage shared memory
    // 2. Prologue: load first stage
    // 3. Main loop: overlap load of stage k with compute of stage k-1
    // 4. Epilogue: final compute
    
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

void init_matrix(std::vector<float>& m, int r, int c) {
    for (int i = 0; i < r * c; ++i) m[i] = (float)(i % 100) / 100.0f;
}

void gemm_reference(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

bool verify(const std::vector<float>& gpu, const std::vector<float>& cpu,
            int M, int N, float tol = 1e-2f) {
    float err = 0;
    for (int i = 0; i < M * N; ++i) err = fmaxf(err, fabsf(gpu[i] - cpu[i]));
    std::cout << "Max error: " << err << std::endl;
    return err <= tol;
}

int main() {
    using Config = cute::PipelinedGemmConfig;
    const int M = 512, K = 256, N = 512;
    
    const dim3 block(Config::TM, Config::TN);
    const dim3 grid((N + Config::BN - 1) / Config::BN,
                    (M + Config::BM - 1) / Config::BM);
    
    std::cout << "=== Project 08: Pipelined GEMM with Async Copy ===" << std::endl;
    std::cout << "Matrix: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    std::cout << "Pipeline stages: " << Config::NumStages << std::endl;
    
    std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N, 0), h_ref(M*N);
    init_matrix(h_A, M, K); init_matrix(h_B, K, N);
    gemm_reference(h_A, h_B, h_ref, M, K, N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);
    
    cute::pipelined_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify(h_C, h_ref, M, N)) std::cout << "[PASS] Pipelined GEMM" << std::endl;
    else { std::cout << "[FAIL]" << std::endl; return 1; }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
