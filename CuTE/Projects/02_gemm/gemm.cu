/**
 * Project 02: GEMM - General Matrix Multiply using CuTe
 * 
 * Objective: Implement C = A × B using tiled matrix multiplication
 * 
 * Instructions:
 * 1. Read the README.md for theory and guidance
 * 2. Complete all TODO sections in this file
 * 3. Build with: make project_02_gemm
 * 4. Run and verify correctness
 * 
 * Key CuTe Concepts:
 * - 2D Layouts: Shape + Stride for matrices
 * - Tiled computation: Breaking large matrices into tiles
 * - Thread mapping: 2D thread blocks for 2D data
 * - Shared memory: Fast on-chip memory for data reuse
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

// Tile configuration - adjust for performance tuning
struct GemmConfig {
    static constexpr int BM = 64;  // Tile rows (block M)
    static constexpr int BN = 64;  // Tile columns (block N)
    static constexpr int BK = 8;   // Reduction tile size (block K)
    
    // Thread block configuration
    static constexpr int TM = 16;  // Threads in M dimension
    static constexpr int TN = 16;  // Threads in N dimension
    
    // Elements per thread
    static constexpr int THREAD_M = BM / TM;  // 4
    static constexpr int THREAD_N = BN / TN;  // 4
};

/**
 * TODO: Implement tiled GEMM kernel using CuTe
 * 
 * Algorithm:
 * 1. Each thread block computes one BM×BN tile of C
 * 2. Load tiles of A and B into shared memory
 * 3. Each thread computes THREAD_M×THREAD_N elements
 * 4. Accumulate over K dimension in BK-sized chunks
 * 
 * Hints:
 * - Use 2D layouts: make_layout(make_shape(M, N), make_stride(N, 1))
 * - Thread indices: threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y
 * - Shared memory: __shared__ float tile_A[BM * BK];
 * - Synchronization: __syncthreads() after each load phase
 */
__global__ void gemm_cute_kernel(float* A, float* B, float* C,
                                  int M, int K, int N) {
    using Config = GemmConfig;
    
    // TODO 1: Calculate this block's tile position
    // int tile_m = blockIdx.y * Config::BM;
    // int tile_n = blockIdx.x * Config::BN;
    
    // TODO 2: Calculate this thread's position within the tile
    // int thread_m = threadIdx.y;
    // int thread_n = threadIdx.x;
    
    // TODO 3: Create 2D layouts for global matrices
    // Layout for A: (M, K) with row-major stride
    // Layout for B: (K, N) with row-major stride
    // Layout for C: (M, N) with row-major stride
    
    // TODO 4: Create tensors from pointers
    // auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    // auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    // auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    // TODO 5: Allocate shared memory for tiles
    // __shared__ float tile_A[Config::BM * Config::BK];
    // __shared__ float tile_B[Config::BK * Config::BN];
    
    // TODO 6: Initialize accumulator for this thread's output elements
    // float accum[Config::THREAD_M][Config::THREAD_N] = {0.0f};
    
    // TODO 7: Implement tiled matmul loop
    // for (int k = 0; k < K; k += Config::BK) {
    //     // Phase 1: Load tiles from global to shared memory
    //     // Each thread loads its assigned elements
    //     
    //     __syncthreads();  // Wait for all loads to complete
    //     
    //     // Phase 2: Compute matrix multiply for this tile
    //     // For each k_idx in [0, BK):
    //     //   For each output element (m, n) this thread owns:
    //     //     accum[m][n] += tile_A[m][k_idx] * tile_B[k_idx][n]
    //     
    //     __syncthreads();  // Wait for all computes before next load
    // }
    
    // TODO 8: Write accumulated results to global memory
    // for (int m = 0; m < Config::THREAD_M; ++m) {
    //     for (int n = 0; n < Config::THREAD_N; ++n) {
    //         int row = tile_m + thread_m + m;
    //         int col = tile_n + thread_n + n;
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
// Host Code - Setup, Launch, and Verification
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

/**
 * Initialize matrix with sequential values
 */
void init_matrix(std::vector<float>& mat, int rows, int cols, 
                 float start = 1.0f, float step = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = start + i * step;
    }
}

/**
 * Reference CPU GEMM for verification
 */
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

/**
 * Print submatrix
 */
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

/**
 * Verify GEMM result
 */
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

int main(int argc, char** argv) {
    using Config = cute::GemmConfig;
    
    // Matrix dimensions (must be multiples of tile sizes for simplicity)
    const int M = 256;
    const int K = 128;
    const int N = 256;
    
    // Launch configuration
    const dim3 block_dim(Config::TM, Config::TN);  // 16x16 threads
    const dim3 grid_dim((N + Config::BN - 1) / Config::BN,
                        (M + Config::BM - 1) / Config::BM);
    
    std::cout << "=== Project 02: GEMM with CuTe ===" << std::endl;
    std::cout << "Matrix A: " << M << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << N << std::endl;
    std::cout << "Matrix C: " << M << " x " << N << std::endl;
    std::cout << "Tile size: " << Config::BM << " x " << Config::BN << std::endl;
    std::cout << "Launch config: " << grid_dim.x << "x" << grid_dim.y 
              << " blocks, " << block_dim.x << "x" << block_dim.y << " threads" << std::endl;
    std::cout << std::endl;
    
    // Allocate and initialize host matrices
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    init_matrix(h_A, M, K, 1.0f, 1.0f);
    init_matrix(h_B, K, N, 1.0f, 0.5f);
    
    std::cout << "Matrix A (top-left 4x4):" << std::endl;
    print_matrix(h_A, M, K);
    std::cout << "Matrix B (top-left 4x4):" << std::endl;
    print_matrix(h_B, K, N);
    std::cout << std::endl;
    
    // Compute reference result on CPU
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> h_C_ref(M * N);
    gemm_reference(h_A, h_B, h_C_ref, M, K, N);
    std::cout << "CPU reference complete." << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch GPU kernel
    std::cout << "Launching CuTe GEMM kernel..." << std::endl;
    cute::gemm_cute_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, K, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result
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
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n=== Project 02 Complete! ===" << std::endl;
    std::cout << "Next: Try the shared memory optimization challenge in the README." << std::endl;
    
    return EXIT_SUCCESS;
}
