/**
 * Project 06: Tiled GEMM with Shared Memory Optimization
 * 
 * Objective: Implement high-performance GEMM using shared memory tiling
 * 
 * Instructions:
 * 1. Read the README.md for theory and guidance
 * 2. Complete all TODO sections in this file
 * 3. Build with: make project_06_tiled_gemm_smem
 * 4. Run and verify correctness
 * 
 * Key CuTe Concepts:
 * - Shared memory tiling for data reuse
 * - Cooperative thread loading
 * - Register blocking
 * - Bank conflict avoidance
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

// Configuration for shared memory tiling
struct TiledGemmConfig {
    static constexpr int BM = 64;       // Tile rows
    static constexpr int BN = 64;       // Tile columns
    static constexpr int BK = 8;        // Reduction tile size
    
    static constexpr int TM = 16;       // Threads in M dimension
    static constexpr int TN = 16;       // Threads in N dimension
    
    static constexpr int THREAD_M = BM / TM;  // Elements per thread in M (4)
    static constexpr int THREAD_N = BN / TN;  // Elements per thread in N (4)
    
    // Shared memory size: (BM * BK + BK * BN) * 4 bytes
    // = (64*8 + 8*64) * 4 = 4096 bytes = 4 KB per block
};

/**
 * TODO: Implement shared memory tiled GEMM kernel
 * 
 * Algorithm:
 * 1. Each block computes one BM×BN tile of C
 * 2. Load tiles of A and B into shared memory cooperatively
 * 3. Each thread computes THREAD_M×THREAD_N elements using register blocking
 * 4. Accumulate over K dimension in BK-sized chunks
 * 
 * Key optimizations to implement:
 * - Cooperative loading: each thread loads specific elements
 * - Register blocking: accumulate multiple elements per thread
 * - Minimize shared memory accesses by keeping data in registers
 */
__global__ void tiled_gemm_smem_kernel(float* A, float* B, float* C,
                                        int M, int K, int N) {
    using Config = TiledGemmConfig;
    
    // Block and thread indices
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int tid = threadIdx.x;
    int thread_m = threadIdx.x % Config::TM;
    int thread_n = threadIdx.y;
    
    // TODO 1: Check bounds - skip if tile is outside matrix
    // if (tile_m >= M || tile_n >= N) return;
    
    // TODO 2: Allocate shared memory for tiles
    // __shared__ float As[Config::BM * Config::BK];
    // __shared__ float Bs[Config::BK * Config::BN];
    
    // TODO 3: Create 2D layouts for global matrices
    // auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    // auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    // auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    // TODO 4: Create tensors from pointers
    // auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    // auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    // auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    // TODO 5: Initialize register accumulator for this thread's elements
    // float accum[Config::THREAD_M][Config::THREAD_N] = {0.0f};
    
    // TODO 6: Main tiled matmul loop
    // for (int k = 0; k < K; k += Config::BK) {
    //     // Phase 1: Cooperative loading from global to shared memory
    //     // Each thread loads assigned elements for A tile
    //     // Each thread loads assigned elements for B tile
    //     
    //     __syncthreads();
    //     
    //     // Phase 2: Compute using register blocking
    //     // For each kk in [0, BK):
    //     //   For each (m, n) element this thread owns:
    //     //     accum[m][n] += As[local_m + m][kk] * Bs[kk][local_n + n]
    //     
    //     __syncthreads();
    // }
    
    // TODO 7: Write accumulated results to global memory
    // for (int m = 0; m < Config::THREAD_M; m++) {
    //     for (int n = 0; n < Config::THREAD_N; n++) {
    //         int row = tile_m + thread_m + m * Config::TM;
    //         int col = tile_n + thread_n + n * Config::TN;
    //         if (row < M && col < N) {
    //             tensor_C(row, col) = accum[m][n];
    //         }
    //     }
    // }
    
    // Suppress unused parameter warnings (remove after implementing)
    (void)A; (void)B; (void)C;
    (void)M; (void)K; (void)N;
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

void init_matrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)(i % 100) / 100.0f;
    }
}

void gemm_reference(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int M, int K, int N) {
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
    
    std::cout << "=== Project 06: Tiled GEMM with Shared Memory ===" << std::endl;
    std::cout << "Matrix: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    std::cout << "Tile: " << Config::BM << "x" << Config::BN << "x" << Config::BK << std::endl;
    std::cout << std::endl;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> h_C_ref(M * N);
    gemm_reference(h_A, h_B, h_C_ref, M, K, N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Launching kernel..." << std::endl;
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
