/**
 * Project 08: Pipelined GEMM - Reference Solution
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
    static constexpr int NumStages = 3;
    static constexpr int TM = 16;
    static constexpr int TN = 16;
};

__global__ void pipelined_gemm_kernel(float* A, float* B, float* C, int M, int K, int N) {
    using Config = PipelinedGemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    if (tile_m >= M || tile_n >= N) return;
    
    // Multi-stage shared memory
    __shared__ float As[Config::NumStages][Config::BM * Config::BK];
    __shared__ float Bs[Config::NumStages][Config::BK * Config::BN];
    
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    constexpr int THREAD_M = Config::BM / Config::TM;
    constexpr int THREAD_N = Config::BN / Config::TN;
    float accum[THREAD_M][THREAD_N] = {{0.0f}};
    
    int num_tiles = (K + Config::BK - 1) / Config::BK;
    
    // Prologue: load first stage
    int stage = 0;
    for (int m = 0; m < THREAD_M; m++) {
        int row = tile_m + thread_m * THREAD_M + m;
        for (int bk = 0; bk < Config::BK; bk++) {
            int k_idx = bk;
            if (row < M && k_idx < K)
                As[0][(thread_m * THREAD_M + m) * Config::BK + bk] = tensor_A(row, k_idx);
        }
    }
    for (int n = 0; n < THREAD_N; n++) {
        int col = tile_n + thread_n * THREAD_N + n;
        for (int bk = 0; bk < Config::BK; bk++) {
            int k_idx = bk;
            if (k_idx < K && col < N)
                Bs[0][bk * Config::BN + (thread_n * THREAD_N + n)] = tensor_B(k_idx, col);
        }
    }
    
    __syncthreads();
    
    // Main loop
    for (int k = 1; k < num_tiles; k++) {
        int k_start = k * Config::BK;
        int prev_stage = (stage + Config::NumStages - 1) % Config::NumStages;
        
        // Load next stage
        for (int m = 0; m < THREAD_M; m++) {
            int row = tile_m + thread_m * THREAD_M + m;
            for (int bk = 0; bk < Config::BK; bk++) {
                int k_idx = k_start + bk;
                if (row < M && k_idx < K)
                    As[stage][(thread_m * THREAD_M + m) * Config::BK + bk] = tensor_A(row, k_idx);
            }
        }
        for (int n = 0; n < THREAD_N; n++) {
            int col = tile_n + thread_n * THREAD_N + n;
            for (int bk = 0; bk < Config::BK; bk++) {
                int k_idx = k_start + bk;
                if (k_idx < K && col < N)
                    Bs[stage][bk * Config::BN + (thread_n * THREAD_N + n)] = tensor_B(k_idx, col);
            }
        }
        
        __syncthreads();
        
        // Compute previous stage
        for (int kk = 0; kk < Config::BK; kk++) {
            for (int m = 0; m < THREAD_M; m++) {
                float a = As[prev_stage][(thread_m * THREAD_M + m) * Config::BK + kk];
                for (int n = 0; n < THREAD_N; n++) {
                    float b = Bs[prev_stage][kk * Config::BN + (thread_n * THREAD_N + n)];
                    accum[m][n] += a * b;
                }
            }
        }
        
        __syncthreads();
        stage = (stage + 1) % Config::NumStages;
    }
    
    // Store results
    for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
            int row = tile_m + thread_m * THREAD_M + m;
            int col = tile_n + thread_n * THREAD_N + n;
            if (row < M && col < N)
                tensor_C(row, col) = accum[m][n];
        }
    }
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
    
    std::cout << "=== Project 08: Pipelined GEMM (Solution) ===" << std::endl;
    
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
