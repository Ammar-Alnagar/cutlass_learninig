// CONCEPT: Tiled GEMM with shared memory — Timed Interview Exercise
// FORMAT: IMPLEMENT (TIMED)
// TIME_TARGET: 45 minutes
// WHY_THIS_MATTERS: Tiled GEMM is the foundation of all high-performance CUDA kernels.
// CUDA_CONNECTION: Direct implementation — this IS the CUTLASS building block.

// INTERVIEW SIMULATION:
// - Set a timer for 45 minutes
// - Implement a tiled matrix multiply using shared memory
// - No hints allowed (interview setting)
// - Verify with provided test case

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define TILE_SIZE 16
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// TODO: Implement a tiled matrix multiply kernel
// Requirements:
// 1. Load tiles of A and B into shared memory
// 2. Synchronize threads
// 3. Compute partial products
// 4. Loop over all tiles
// 5. Write result to C
//
// TIME TARGET: 45 minutes

__global__ void tiledGEMM(const float* A, const float* B, float* C, 
                          int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Row and column for this thread block
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // TODO: Load tile of A into shared memory
        // A tile: row is fixed, column varies with t
        // Handle boundary: if (row < M && t*TILE_SIZE + tx < K)
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // TODO: Load tile of B into shared memory
        // B tile: row varies with t, column is fixed
        // Handle boundary: if (t*TILE_SIZE + ty < K && col < N)
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // TODO: Synchronize to ensure tile is loaded
        __syncthreads();
        
        // TODO: Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // TODO: Synchronize before loading next tile
        __syncthreads();
    }
    
    // TODO: Write result to C (handle boundary)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch kernel
void matrixMultiply(const float* d_A, const float* d_B, float* d_C,
                    int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    tiledGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Verify result
bool verifyResult(const float* C, const float* A, const float* B, 
                  int M, int N, int K, float tolerance = 1e-3f) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += A[i * K + k] * B[k * N + j];
            }
            float actual = C[i * N + j];
            float diff = fabsf(expected - actual);
            if (diff > tolerance) {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "expected " << expected << ", got " << actual << "\n";
                return false;
            }
        }
    }
    return true;
}

int main() {
    const int M = 64, N = 64, K = 64;
    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);
    
    std::cout << "=== Tiled GEMM Test ===\n";
    std::cout << "Matrix sizes: A(" << M << "x" << K << "), "
              << "B(" << K << "x" << N << "), "
              << "C(" << M << "x" << N << ")\n";
    std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << "\n\n";
    
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    std::cout << "Launching tiled GEMM kernel...\n";
    matrixMultiply(d_A, d_B, d_C, M, N, K);
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // Verify
    std::cout << "Verifying result...\n";
    if (verifyResult(h_C, h_A, h_B, M, N, K)) {
        std::cout << "SUCCESS: All elements match (expected: " << K * 2.0f << ")\n";
        std::cout << "Sample: C[0,0] = " << h_C[0] << "\n";
    } else {
        std::cout << "FAILURE: Mismatch detected\n";
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    std::cout << "\n=== INTERVIEW RUBRIC ===\n";
    std::cout << "[ ] Shared memory declared (__shared__)\n";
    std::cout << "[ ] Tile loading with boundary checks\n";
    std::cout << "[ ] __syncthreads() after loading\n";
    std::cout << "[ ] Dot product computation loop\n";
    std::cout << "[ ] __syncthreads() before next tile\n";
    std::cout << "[ ] Result write with boundary check\n";
    std::cout << "\nTIME CHECK:\n";
    std::cout << "< 30 min: Expert — fluent with CUDA tiling\n";
    std::cout << "30-45 min: Strong — solid CUDA skills\n";
    std::cout << "45-60 min: Acceptable — needs more practice\n";
    std::cout << "> 60 min: Review shared memory patterns\n";
    
    return 0;
}
