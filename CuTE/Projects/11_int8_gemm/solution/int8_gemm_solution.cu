/**
 * Project 11: INT8 GEMM - Reference Solution
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

__global__ void int8_gemm_kernel(int8_t* A, int8_t* B, float* C,
                                  float* scale_A, float* scale_B,
                                  int M, int K, int N) {
    using Config = INT8GemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    if (tile_m >= M || tile_n >= N) return;
    
    __shared__ int8_t As[Config::BM * Config::BK];
    __shared__ int8_t Bs[Config::BK * Config::BN];
    
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    constexpr int THREAD_M = Config::BM / Config::TM;
    constexpr int THREAD_N = Config::BN / Config::TN;
    int32_t accum[THREAD_M][THREAD_N] = {{0}};
    
    for (int k = 0; k < K; k += Config::BK) {
        // Load INT8 tiles
        for (int m = 0; m < THREAD_M; m++) {
            int row = tile_m + thread_m * THREAD_M + m;
            for (int bk = 0; bk < Config::BK; bk++) {
                if (row < M && k + bk < K)
                    As[(thread_m * THREAD_M + m) * Config::BK + bk] = tensor_A(row, k + bk);
            }
        }
        for (int n = 0; n < THREAD_N; n++) {
            int col = tile_n + thread_n * THREAD_N + n;
            for (int bk = 0; bk < Config::BK; bk++) {
                if (k + bk < K && col < N)
                    Bs[bk * Config::BN + (thread_n * THREAD_N + n)] = tensor_B(k + bk, col);
            }
        }
        
        __syncthreads();
        
        // INT8 multiply, int32 accumulate
        for (int kk = 0; kk < Config::BK; kk++) {
            for (int m = 0; m < THREAD_M; m++) {
                int8_t a = As[(thread_m * THREAD_M + m) * Config::BK + kk];
                for (int n = 0; n < THREAD_N; n++) {
                    int8_t b = Bs[kk * Config::BN + (thread_n * THREAD_N + n)];
                    accum[m][n] += (int32_t)a * (int32_t)b;
                }
            }
        }
        __syncthreads();
    }
    
    // Dequantize and store
    for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
            int row = tile_m + thread_m * THREAD_M + m;
            int col = tile_n + thread_n * THREAD_N + n;
            if (row < M && col < N) {
                tensor_C(row, col) = (float)accum[m][n] * scale_A[row] * scale_B[col];
            }
        }
    }
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

void init_int8_matrix(std::vector<int8_t>& m, int r, int c) {
    for (int i = 0; i < r * c; i++) m[i] = (int8_t)((i % 200) - 100);
}

void int8_gemm_reference(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                          std::vector<float>& C, int M, int K, int N,
                          const std::vector<float>& sA, const std::vector<float>& sB) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            C[i * N + j] = (float)acc * sA[i] * sB[j];
        }
}

int main() {
    using Config = cute::INT8GemmConfig;
    const int M = 256, K = 128, N = 256;
    
    std::cout << "=== Project 11: INT8 GEMM (Solution) ===" << std::endl;
    
    std::vector<int8_t> h_A(M*K), h_B(K*N);
    std::vector<float> h_C(M*N, 0), h_ref(M*N);
    std::vector<float> h_sA(M, 0.01f), h_sB(N, 0.01f);
    
    init_int8_matrix(h_A, M, K); init_int8_matrix(h_B, K, N);
    int8_gemm_reference(h_A, h_B, h_ref, M, K, N, h_sA, h_sB);
    
    int8_t *d_A, *d_B; float *d_C, *d_sA, *d_sB;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sA, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sB, N*sizeof(float)));
    
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sA, h_sA.data(), M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sB, h_sB.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    
    const dim3 block(Config::TM, Config::TN);
    const dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    
    cute::int8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, d_sA, d_sB, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    float err = 0;
    for (int i = 0; i < M*N; i++) err = fmaxf(err, fabsf(h_C[i] - h_ref[i]));
    std::cout << "Max error: " << err << std::endl;
    std::cout << (err < 1.0f ? "[PASS]" : "[FAIL]") << std::endl;
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_sA); cudaFree(d_sB);
    return 0;
}
