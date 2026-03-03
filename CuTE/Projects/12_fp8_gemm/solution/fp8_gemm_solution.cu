/**
 * Project 12: FP8 GEMM - Reference Solution
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct FP8GemmConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int TM = 16;
    static constexpr int TN = 16;
};

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
    return (float)x / 255.0f * 10.0f;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float x) {
    return (uint8_t)fminf(255.0f, fmaxf(0.0f, fabsf(x) / 10.0f * 255.0f));
}

__global__ void fp8_gemm_kernel(uint8_t* A, uint8_t* B, float* C, int M, int K, int N) {
    using Config = FP8GemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    if (tile_m >= M || tile_n >= N) return;
    
    __shared__ uint8_t As[Config::BM * Config::BK];
    __shared__ uint8_t Bs[Config::BK * Config::BN];
    
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    constexpr int THREAD_M = Config::BM / Config::TM;
    constexpr int THREAD_N = Config::BN / Config::TN;
    float accum[THREAD_M][THREAD_N] = {{0.0f}};
    
    for (int k = 0; k < K; k += Config::BK) {
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
        
        for (int kk = 0; kk < Config::BK; kk++) {
            for (int m = 0; m < THREAD_M; m++) {
                float a = fp8_e4m3_to_float(As[(thread_m * THREAD_M + m) * Config::BK + kk]);
                for (int n = 0; n < THREAD_N; n++) {
                    float b = fp8_e4m3_to_float(Bs[kk * Config::BN + (thread_n * THREAD_N + n)]);
                    accum[m][n] += a * b;
                }
            }
        }
        __syncthreads();
    }
    
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

void init_fp8_matrix(std::vector<uint8_t>& m, int r, int c) {
    for (int i = 0; i < r * c; i++) m[i] = cute::float_to_fp8_e4m3((float)(i % 100) / 10.0f);
}

void fp8_gemm_reference(const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                        std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += cute::fp8_e4m3_to_float(A[i*K+k]) * cute::fp8_e4m3_to_float(B[k*N+j]);
            C[i*N+j] = sum;
        }
}

int main() {
    using Config = cute::FP8GemmConfig;
    const int M = 256, K = 128, N = 256;
    
    std::cout << "=== Project 12: FP8 GEMM (Solution) ===" << std::endl;
    
    std::vector<uint8_t> h_A(M*K), h_B(K*N);
    std::vector<float> h_C(M*N, 0), h_ref(M*N);
    
    init_fp8_matrix(h_A, M, K); init_fp8_matrix(h_B, K, N);
    fp8_gemm_reference(h_A, h_B, h_ref, M, K, N);
    
    uint8_t *d_A, *d_B; float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    const dim3 block(Config::TM, Config::TN);
    const dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    
    cute::fp8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    float err = 0;
    for (int i = 0; i < M*N; i++) err = fmaxf(err, fabsf(h_C[i] - h_ref[i]));
    std::cout << "Max error: " << err << std::endl;
    std::cout << (err < 1.0f ? "[PASS]" : "[FAIL]") << std::endl;
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
