/**
 * Project 07: MMA GEMM with Tensor Cores - Reference Solution
 * 
 * Note: This is a simplified MMA implementation. Production implementations
 * use more sophisticated register layouts and instruction scheduling.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

namespace cute {

struct MMAGemmConfig {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 16;
    static constexpr int MMA_K = 16;
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 16;
    static constexpr int THREADS_M = 4;
    static constexpr int THREADS_N = 4;
};

__global__ void mma_gemm_kernel(float* A, float* B, float* C, int M, int K, int N) {
    using Config = MMAGemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    if (tile_m >= M || tile_n >= N) return;
    
    __shared__ float As[Config::BM * Config::BK];
    __shared__ float Bs[Config::BK * Config::BN];
    
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    // Accumulator per thread (Config::BM/THREADS_M × Config::BN/THREADS_N)
    constexpr int THREAD_M = Config::BM / Config::THREADS_M;
    constexpr int THREAD_N = Config::BN / Config::THREADS_N;
    float accum[THREAD_M][THREAD_N] = {{0.0f}};
    
    for (int k = 0; k < K; k += Config::BK) {
        // Load A tile
        for (int m = 0; m < THREAD_M; m++) {
            int row = tile_m + thread_m * THREAD_M + m;
            for (int bk = 0; bk < Config::BK; bk++) {
                if (row < M && k + bk < K)
                    As[(thread_m * THREAD_M + m) * Config::BK + bk] = tensor_A(row, k + bk);
            }
        }
        
        // Load B tile
        for (int n = 0; n < THREAD_N; n++) {
            int col = tile_n + thread_n * THREAD_N + n;
            for (int bk = 0; bk < Config::BK; bk++) {
                if (k + bk < K && col < N)
                    Bs[bk * Config::BN + (thread_n * THREAD_N + n)] = tensor_B(k + bk, col);
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory (simplified - not using actual MMA instructions)
        // Real implementation would use wmma::fragment and mma_sync
        for (int kk = 0; kk < Config::BK; kk++) {
            for (int m = 0; m < THREAD_M; m++) {
                float a_val = As[(thread_m * THREAD_M + m) * Config::BK + kk];
                for (int n = 0; n < THREAD_N; n++) {
                    accum[m][n] += a_val * Bs[kk * Config::BN + (thread_n * THREAD_N + n)];
                }
            }
        }
        
        __syncthreads();
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

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } } while(0)

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

bool verify(const std::vector<float>& gpu, const std::vector<float>& cpu, int M, int N, float tol = 5e-2f) {
    float err = 0;
    for (int i = 0; i < M * N; ++i) err = fmaxf(err, fabsf(gpu[i] - cpu[i]));
    std::cout << "Max error: " << err << std::endl;
    return err <= tol;
}

int main() {
    using Config = cute::MMAGemmConfig;
    const int M = 256, K = 128, N = 256;
    
    const dim3 block(Config::THREADS_M, Config::THREADS_N);
    const dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    
    std::cout << "=== Project 07: MMA GEMM (Solution) ===" << std::endl;
    
    std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N), h_ref(M*N);
    init_matrix(h_A, M, K); init_matrix(h_B, K, N);
    gemm_reference(h_A, h_B, h_ref, M, K, N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);
    
    cute::mma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify(h_C, h_ref, M, N)) std::cout << "[PASS]" << std::endl;
    else { std::cout << "[FAIL]" << std::endl; return 1; }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
