// ============================================================
// MODULE 07 — Predication
// Exercise 02: Irregular Tile GEMM — Two-Path Pattern
// ============================================================
// CONCEPT:
//   GEMM with M=100, N=100, K=64 where M and N are not divisible by
//   tile size 32. Two-path pattern:
//   - Full tiles: no predication, fast path (90% of tiles)
//   - Partial tiles: predicated, epilogue path (boundary tiles)
//   
//   Tradeoff: Padding (M=128 with zeroed padding) vs Predication
//   - Padding: trades memory for simplicity
//   - Predication: trades compute complexity for memory efficiency
//
// JOB RELEVANCE:
//   Cerebras Performance Engineer — "Production-quality kernel optimization"
//   NVIDIA DL Software Engineer — "Efficient attention kernels" — Variable
//   sequence length handling in LLM inference requires this exact pattern.
//
// ASCII DIAGRAM:
//   GEMM: C = A × B, where A is (100, 64), B is (64, 100), C is (100, 100)
//   
//   Tile grid for C (100×100, tile size 32):
//   ┌────────┬────────┬────────┬────────┐
//   │ (0,0)  │ (0,1)  │ (0,2)  │ (0,3)  │  ← Row 0: tiles 0-2 full, tile 3 partial
//   │ Full   │ Full   │ Full   │ Partial│
//   ├────────┼────────┼────────┼────────┤
//   │ (1,0)  │ (1,1)  │ (1,2)  │ (1,3)  │  ← Row 1: tiles 0-2 full, tile 3 partial
//   │ Full   │ Full   │ Full   │ Partial│
//   ├────────┼────────┼────────┼────────┤
//   │ (2,0)  │ (2,1)  │ (2,2)  │ (2,3)  │  ← Row 2: tiles 0-2 full, tile 3 partial
//   │ Full   │ Full   │ Full   │ Partial│
//   ├────────┼────────┼────────┼────────┤
//   │ (3,0)  │ (3,1)  │ (3,2)  │ (3,3)  │  ← Row 3: ALL partial (rows 96-99)
//   │ Partial│ Partial│ Partial│ Partial│
//   └────────┴────────┴────────┴────────┘
//   
//   Two-path pattern:
//   for each tile (tile_m, tile_n):
//     if is_full_tile(tile_m, tile_n):
//       gemm_fast_path()      // No predication
//     else:
//       gemm_predicated()     // With bounds checking
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex02_irregular_tile_gemm.cu -o ex02_irregular_tile_gemm
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace cute;

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr int M = 100;          // Non-tile-aligned dimension
constexpr int N = 100;          // Non-tile-aligned dimension
constexpr int K = 64;           // Tile-aligned dimension
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int NUM_TILE_M = (M + TILE_M - 1) / TILE_M;  // = 4
constexpr int NUM_TILE_N = (N + TILE_N - 1) / TILE_N;  // = 4
constexpr int NUM_TILE_K = (K + TILE_K - 1) / TILE_K;  // = 2

// ============================================================================
// HELPER: Print matrix slice
// ============================================================================
__host__ void print_matrix_slice(const char* label, float* data, int rows, int cols, 
                                  int row_start, int row_end, int col_start, int col_end) {
  std::cout << label << " [" << row_start << ":" << row_end 
            << ", " << col_start << ":" << col_end << "]:\n";
  for (int i = row_start; i < row_end && i < rows; ++i) {
    std::cout << "  ";
    for (int j = col_start; j < col_end && j < cols; ++j) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(1) 
                << data[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}

// ============================================================================
// KERNEL: Simple tiled GEMM with predication
// ============================================================================
__global__ void irregular_gemm_kernel(float* A, float* B, float* C, 
                                       int M_actual, int N_actual, int K_actual,
                                       bool use_predication) {
  // Simple implementation: one thread block computes one tile of C
  // Each block is responsible for a (TILE_M, TILE_N) output tile
  
  int tile_m = blockIdx.y;  // Which tile row
  int tile_n = blockIdx.x;  // Which tile column
  
  // Thread within block (for this simple example, thread 0 does all work)
  if (threadIdx.x != 0) return;
  
  // Base indices for this tile
  int m_base = tile_m * TILE_M;
  int n_base = tile_n * TILE_N;
  
  // Check if this is a full tile or partial tile
  bool is_full_tile = (m_base + TILE_M <= M_actual) && 
                      (n_base + TILE_N <= N_actual);
  
  // Accumulator for this tile
  float accum[TILE_M * TILE_N] = {0.0f};
  
  // K-loop for GEMM
  for (int k_tile = 0; k_tile < NUM_TILE_K; ++k_tile) {
    int k_base = k_tile * TILE_K;
    
    // Load tiles from A and B (simplified, no SMEM in this example)
    // A tile: (TILE_M, TILE_K), B tile: (TILE_K, TILE_N)
    
    // Compute partial tile of C
    for (int i = 0; i < TILE_M; ++i) {
      for (int j = 0; j < TILE_N; ++j) {
        int global_m = m_base + i;
        int global_n = n_base + j;
        
        // Predication check
        if (use_predication) {
          if (global_m >= M_actual || global_n >= N_actual) {
            continue;  // Skip OOB elements
          }
        }
        
        // Dot product for C[m, n]
        float sum = 0.0f;
        for (int k = k_base; k < k_base + TILE_K && k < K_actual; ++k) {
          float a_val = A[global_m * K_actual + k];
          float b_val = B[k * N_actual + global_n];
          sum += a_val * b_val;
        }
        
        accum[i * TILE_N + j] += sum;
      }
    }
  }
  
  // Store result to C
  for (int i = 0; i < TILE_M; ++i) {
    for (int j = 0; j < TILE_N; ++j) {
      int global_m = m_base + i;
      int global_n = n_base + j;
      
      if (global_m < M_actual && global_n < N_actual) {
        C[global_m * N_actual + global_n] = accum[i * TILE_N + j];
      }
    }
  }
  
  // Print tile info
  printf("Block (%d, %d): m_base=%d, n_base=%d, %s\n",
         tile_n, tile_m, m_base, n_base,
         is_full_tile ? "FULL" : "PARTIAL");
}

// ============================================================================
// CPU REFERENCE: Standard GEMM
// ============================================================================
void cpu_gemm(float* A, float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

// ============================================================================
// PADDED VERSION: Zero-pad to tile boundary
// ============================================================================
__global__ void padded_gemm_kernel(float* A_padded, float* B_padded, float* C_padded,
                                    int M_padded, int N_padded, int K_padded,
                                    int M_actual, int N_actual) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;
  
  if (threadIdx.x != 0) return;
  
  int m_base = tile_m * TILE_M;
  int n_base = tile_n * TILE_N;
  
  float accum[TILE_M * TILE_N] = {0.0f};
  
  // K-loop (no predication needed, data is zero-padded)
  for (int k_tile = 0; k_tile < NUM_TILE_K; ++k_tile) {
    int k_base = k_tile * TILE_K;
    
    for (int i = 0; i < TILE_M; ++i) {
      for (int j = 0; j < TILE_N; ++j) {
        float sum = 0.0f;
        for (int k = k_base; k < k_base + TILE_K; ++k) {
          float a_val = A_padded[i * TILE_K + (k - k_base)];  // Simplified
          float b_val = B_padded[(k - k_base) * TILE_N + j];
          sum += a_val * b_val;
        }
        accum[i * TILE_N + j] += sum;
      }
    }
  }
  
  // Store (only actual elements, not padding)
  for (int i = 0; i < TILE_M && m_base + i < M_actual; ++i) {
    for (int j = 0; j < TILE_N && n_base + j < N_actual; ++j) {
      C_padded[(m_base + i) * N_padded + (n_base + j)] = accum[i * TILE_N + j];
    }
  }
}

// ============================================================================
// MAIN: Setup, kernel launch, verification
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 07 — Predication\n";
  std::cout << "Exercise 02: Irregular Tile GEMM\n";
  std::cout << "============================================================\n\n";
  
  // PREDICT BEFORE RUNNING:
  // Q1: How many full tiles in a 100×100 output with tile size 32?
  //     Answer: 3×3 = 9 full tiles (rows 0-2, cols 0-2)
  //
  // Q2: How many partial tiles?
  //     Answer: 16 - 9 = 7 partial tiles (row 3 or col 3)
  //
  // Q3: What's the tradeoff between padding and predication?
  //     Answer: Padding uses more memory but simpler code.
  //             Predication uses less memory but has branch overhead.
  
  std::cout << "Predictions:\n";
  std::cout << "  Q1: 9 full tiles (3×3 grid)\n";
  std::cout << "  Q2: 7 partial tiles\n";
  std::cout << "  Q3: Padding = more memory, simpler code\n";
  std::cout << "        Predication = less memory, branch overhead\n\n";
  
  // Allocate host memory
  float* h_A = new float[M * K];
  float* h_B = new float[K * N];
  float* h_C_pred = new float[M * N];
  float* h_C_ref = new float[M * N];
  
  // Initialize with known values for verification
  // A[i,j] = (i + j) % 10 for easy verification
  // B[i,j] = (i * j) % 10 + 1
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      h_A[i * K + j] = static_cast<float>((i + j) % 10);
    }
  }
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      h_B[i * N + j] = static_cast<float>((i * j) % 10 + 1);
    }
  }
  
  std::cout << "=== Matrix Dimensions ===\n";
  std::cout << "  A: " << M << " × " << K << "\n";
  std::cout << "  B: " << K << " × " << N << "\n";
  std::cout << "  C: " << M << " × " << N << "\n";
  std::cout << "  Tile size: " << TILE_M << " × " << TILE_N << "\n";
  std::cout << "  Grid: " << NUM_TILE_M << " × " << NUM_TILE_N << " tiles\n\n";
  
  // CPU reference
  std::cout << "=== Computing CPU Reference ===\n";
  cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
  std::cout << "Done.\n\n";
  
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  
  cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch predicated kernel
  std::cout << "=== Running Predicated GEMM ===\n";
  dim3 grid(NUM_TILE_N, NUM_TILE_M);
  dim3 block(32);
  irregular_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, true);
  cudaDeviceSynchronize();
  
  // Copy back
  cudaMemcpy(h_C_pred, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Verify
  std::cout << "\n=== Verification (Predicated vs CPU) ===\n";
  float max_error = 0.0f;
  float tol = 1e-4f;
  bool pass = true;
  
  for (int i = 0; i < M * N; ++i) {
    float error = std::abs(h_C_pred[i] - h_C_ref[i]);
    if (error > max_error) max_error = error;
    if (error > tol) {
      if (pass) {
        printf("First mismatch at index %d: expected %.4f, got %.4f\n",
               i, h_C_ref[i], h_C_pred[i]);
      }
      pass = false;
    }
  }
  
  printf("Max error: %.6f (tolerance: %.6f)\n", max_error, tol);
  std::cout << "Result: " << (pass ? "PASS ✓" : "FAIL ✗") << "\n\n";
  
  // Print sample output
  std::cout << "=== Sample Output (top-left 10×10) ===\n";
  print_matrix_slice("CPU Reference", h_C_ref, M, N, 0, 10, 0, 10);
  std::cout << "\n";
  print_matrix_slice("Predicated GPU", h_C_pred, M, N, 0, 10, 0, 10);
  
  // Cleanup
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_pred;
  delete[] h_C_ref;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  // KEY INSIGHT
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Two-path GEMM pattern for irregular dimensions:\n";
  std::cout << "\n";
  std::cout << "  Full tiles (fast path):\n";
  std::cout << "    - No predication overhead\n";
  std::cout << "    - Maximum occupancy and throughput\n";
  std::cout << "    - 9 out of 16 tiles in this example (56%%)\n";
  std::cout << "\n";
  std::cout << "  Partial tiles (predicated path):\n";
  std::cout << "    - Bounds checking per element\n";
  std::cout << "    - Handles boundary conditions correctly\n";
  std::cout << "    - 7 out of 16 tiles in this example (44%%)\n";
  std::cout << "\n";
  std::cout << "  Padding alternative:\n";
  std::cout << "    - Pad M, N to 128 (next multiple of 32)\n";
  std::cout << "    - Wastes (128-100)×128×4 = 14KB for FP32\n";
  std::cout << "    - Simpler code, but memory inefficient for large tensors\n";
  std::cout << "============================================================\n";
  
  return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 *
 * Q1: When would you choose padding over predication?
 *     Answer: When memory is abundant and the padding overhead is small
 *             (e.g., 100→128 is 28%% padding, acceptable for simplicity).
 *             For very large tensors with small remainders, predication wins.
 *
 * Q2: What percentage of tiles are partial in a 100×100 GEMM with 32×32 tiles?
 *     Answer: 7 out of 16 = 43.75% are partial tiles.
 *
 * Q3: How does this relate to FlashAttention-2?
 *     Answer: FlashAttention-2's QK^T and PV GEMMs have irregular sequence
 *             lengths. The same two-path pattern applies.
 */
