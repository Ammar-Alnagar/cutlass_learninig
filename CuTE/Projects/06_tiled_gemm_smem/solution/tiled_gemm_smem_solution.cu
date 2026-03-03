/**
 * Project 06: Tiled GEMM with Shared Memory - Reference Solution
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct TiledGemmConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 8;
    static constexpr int TM = 16;
    static constexpr int TN = 16;
    static constexpr int THREAD_M = BM / TM;
    static constexpr int THREAD_N = BN / TN;
};

__global__ void tiled_gemm_smem_kernel(float* A, float* B, float* C,
                                        int M, int K, int N) {
    using Config = TiledGemmConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    if (tile_m >= M || tile_n >= N) return;
    
    // Allocate shared memory
    __shared__ float As[Config::BM * Config::BK];
    __shared__ float Bs[Config::BK * Config::BN];
    
    // Create layouts and tensors
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    // Initialize accumulator
    float accum[Config::THREAD_M][Config::THREAD_N] = {{0.0f}};
    
    // Main loop
    for (int k = 0; k < K; k += Config::BK) {
        // Cooperative loading - A tile
        for (int m = 0; m < Config::THREAD_M; m++) {
            int row = tile_m + thread_m + m * Config::TM;
            for (int bk = 0; bk < Config::BK; bk++) {
                int col = k + bk;
                if (row < M && col < K) {
                    As[(thread_m + m * Config::TM) * Config::BK + bk] = tensor_A(row, col);
                }
            }
        }
        
        // Cooperative loading - B tile
        for (int n = 0; n < Config::THREAD_N; n++) {
            int col = tile_n + thread_n + n * Config::TN;
            for (int bk = 0; bk < Config::BK; bk++) {
                int row = k + bk;
                if (row < K && col < N) {
                    Bs[bk * Config::BN + (thread_n + n * Config::TN)] = tensor_B(row, col);
                }
            }
        }
        
        __syncthreads();
        
        // Compute with register blocking
        for (int kk = 0; kk < Config::BK; kk++) {
            for (int m = 0; m < Config::THREAD_M; m++) {
                float a_val = As[(thread_m + m * Config::TM) * Config::BK + kk];
                for (int n = 0; n < Config::THREAD_N; n++) {
                    float b_val = Bs[kk * Config::BN + (thread_n + n * Config::TN)];
                    accum[m][n] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int m = 0; m < Config::THREAD_M; m++) {
        for (int n = 0; n < Config::THREAD_N; n++) {
            int row = tile_m + thread_m + m * Config::TM;
            int col = tile_n + thread_n + n * Config::TN;
            if (row < M && col < N) {
                tensor_C(row, col) = accum[m][n];
            }
        }
    }
}

} // namespace cute

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void init_matrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) mat[i] = (float)(i % 100) / 100.0f;
}

void gemm_reference(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

bool verify_gemm(const std::vector<float>& C_gpu, const std::vector<float>& C_cpu,
                 int M, int N, float tolerance = 1e-2f) {
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(C_gpu[i] - C_cpu[i]);
        if (error > max_error) max_error = error;
    }
    std::cout << "Max error: " << max_error << std::endl;
    return max_error <= tolerance;
}

int main() {
    using Config = cute::TiledGemmConfig;
    
    const int M = 512, K = 256, N = 512;
    
    const dim3 block_dim(Config::TM, Config::TN);
    const dim3 grid_dim((N + Config::BN - 1) / Config::BN,
                        (M + Config::BM - 1) / Config::BM);
    
    std::cout << "=== Project 06: Tiled GEMM with Shared Memory (Solution) ===" << std::endl;
    std::cout << "Matrix: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    std::vector<float> h_C_ref(M * N);
    gemm_reference(h_A, h_B, h_C_ref, M, K, N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    cute::tiled_gemm_smem_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, K, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_gemm(h_C, h_C_ref, M, N)) {
        std::cout << "[PASS] Tiled GEMM (Shared Memory)" << std::endl;
    } else {
        std::cout << "[FAIL]" << std::endl;
        return EXIT_FAILURE;
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return EXIT_SUCCESS;
}
