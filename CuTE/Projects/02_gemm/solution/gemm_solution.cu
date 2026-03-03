/**
 * Project 02: GEMM - Reference Solution
 * 
 * This implements tiled matrix multiplication with:
 * - 64x64 output tiles
 * - 16x16 thread blocks
 * - Each thread computes 4x4 elements
 * - Shared memory for A and B tiles
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct GemmConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 8;
    static constexpr int TM = 16;
    static constexpr int TN = 16;
    static constexpr int THREAD_M = BM / TM;
    static constexpr int THREAD_N = BN / TN;
};

__global__ void gemm_cute_kernel(float* A, float* B, float* C,
                                  int M, int K, int N) {
    using Config = GemmConfig;
    
    // Step 1: Calculate this block's tile position in global matrix
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    
    // Step 2: Calculate this thread's position within the tile
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    // Step 3: Create 2D layouts for global matrices (row-major)
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    // Step 4: Create tensors from pointers
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    // Step 5: Allocate shared memory for tiles
    __shared__ float tile_A[Config::BM * Config::BK];
    __shared__ float tile_B[Config::BK * Config::BN];
    
    // Step 6: Initialize accumulator for this thread's 4x4 elements
    float accum[Config::THREAD_M][Config::THREAD_N] = {{0.0f}};
    
    // Step 7: Tiled matmul loop
    for (int k = 0; k < K; k += Config::BK) {
        // Phase 1: Cooperative loading from global to shared memory
        // Each thread loads its assigned elements for A tile
        for (int m = 0; m < Config::THREAD_M; ++m) {
            int row = tile_m + thread_m + m * Config::TM;
            for (int bk = 0; bk < Config::BK; ++bk) {
                int col = k + bk;
                if (row < M && col < K) {
                    tile_A[(thread_m + m * Config::TM) * Config::BK + bk] = 
                        tensor_A(row, col);
                } else {
                    tile_A[(thread_m + m * Config::TM) * Config::BK + bk] = 0.0f;
                }
            }
        }
        
        // Each thread loads its assigned elements for B tile
        for (int n = 0; n < Config::THREAD_N; ++n) {
            int col = tile_n + thread_n + n * Config::TN;
            for (int bk = 0; bk < Config::BK; ++bk) {
                int row = k + bk;
                if (row < K && col < N) {
                    tile_B[bk * Config::BN + (thread_n + n * Config::TN)] = 
                        tensor_B(row, col);
                } else {
                    tile_B[bk * Config::BN + (thread_n + n * Config::TN)] = 0.0f;
                }
            }
        }
        
        // Wait for all threads to finish loading
        __syncthreads();
        
        // Phase 2: Compute matrix multiply for this tile
        // Each thread computes its 4x4 output elements
        for (int bk = 0; bk < Config::BK; ++bk) {
            for (int m = 0; m < Config::THREAD_M; ++m) {
                float a_val = tile_A[(thread_m + m * Config::TM) * Config::BK + bk];
                for (int n = 0; n < Config::THREAD_N; ++n) {
                    float b_val = tile_B[bk * Config::BN + (thread_n + n * Config::TN)];
                    accum[m][n] += a_val * b_val;
                }
            }
        }
        
        // Wait for all threads to finish computing before next load
        __syncthreads();
    }
    
    // Step 8: Write accumulated results to global memory
    for (int m = 0; m < Config::THREAD_M; ++m) {
        for (int n = 0; n < Config::THREAD_N; ++n) {
            int row = tile_m + thread_m + m * Config::TM;
            int col = tile_n + thread_n + n * Config::TN;
            if (row < M && col < N) {
                tensor_C(row, col) = accum[m][n];
            }
        }
    }
}

} // namespace cute

// ============================================================================
// Host Code
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void init_matrix(std::vector<float>& mat, int rows, int cols, 
                 float start = 1.0f, float step = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = start + i * step;
    }
}

void gemm_reference(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void print_matrix(const std::vector<float>& mat, int rows, int cols, 
                  int max_rows = 4, int max_cols = 4) {
    for (int i = 0; i < std::min(rows, max_rows); ++i) {
        std::cout << "  ";
        for (int j = 0; j < std::min(cols, max_cols); ++j) {
            std::cout << std::setw(10) << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

bool verify_gemm(const std::vector<float>& C_gpu,
                 const std::vector<float>& C_cpu,
                 int M, int N,
                 float tolerance = 1e-2f) {
    float max_error = 0.0f;
    int max_error_idx = 0;
    
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(C_gpu[i] - C_cpu[i]);
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }
    
    if (max_error > tolerance) {
        int row = max_error_idx / N;
        int col = max_error_idx % N;
        std::cerr << "Max error at (" << row << ", " << col << "): "
                  << "expected " << C_cpu[max_error_idx]
                  << ", got " << C_gpu[max_error_idx] << std::endl;
        return false;
    }
    
    std::cout << "Max error: " << max_error << " (tolerance: " << tolerance << ")" << std::endl;
    return true;
}

int main() {
    using Config = cute::GemmConfig;
    
    const int M = 256;
    const int K = 128;
    const int N = 256;
    
    const dim3 block_dim(Config::TM, Config::TN);
    const dim3 grid_dim((N + Config::BN - 1) / Config::BN,
                        (M + Config::BM - 1) / Config::BM);
    
    std::cout << "=== Project 02: GEMM with CuTe (Solution) ===" << std::endl;
    std::cout << "Matrix A: " << M << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << N << std::endl;
    std::cout << "Matrix C: " << M << " x " << N << std::endl;
    std::cout << std::endl;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    init_matrix(h_A, M, K, 1.0f, 1.0f);
    init_matrix(h_B, K, N, 1.0f, 0.5f);
    
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> h_C_ref(M * N);
    gemm_reference(h_A, h_B, h_C_ref, M, K, N);
    std::cout << std::endl;
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Launching CuTe GEMM kernel..." << std::endl;
    cute::gemm_cute_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, K, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Verifying result..." << std::endl;
    std::cout << "Matrix C (top-left 4x4):" << std::endl;
    print_matrix(h_C, M, N);
    std::cout << std::endl;
    
    if (verify_gemm(h_C, h_C_ref, M, N)) {
        std::cout << "\n[PASS] GEMM: All elements match" << std::endl;
    } else {
        std::cout << "\n[FAIL] GEMM: Result mismatch!" << std::endl;
        return EXIT_FAILURE;
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n=== Solution Complete! ===" << std::endl;
    
    return EXIT_SUCCESS;
}
