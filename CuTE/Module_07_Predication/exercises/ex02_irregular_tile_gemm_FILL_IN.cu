// ============================================================
// MODULE 07 — Predication
// Exercise 02: Irregular Tile GEMM — Two-Path Pattern
// FILL-IN VERSION — Complete the TODO sections
// ============================================================
// CONCEPT:
//   GEMM with M=100, N=100, K=64 (not divisible by tile size 32).
//   Two-path pattern: full tiles (fast path) vs partial tiles (predicated).
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex02_irregular_tile_gemm_FILL_IN.cu -o ex02_irregular_tile_gemm_FILL_IN
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <cmath>

using namespace cute;

constexpr int M = 100;
constexpr int N = 100;
constexpr int K = 64;
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int NUM_TILE_M = (M + TILE_M - 1) / TILE_M;
constexpr int NUM_TILE_N = (N + TILE_N - 1) / TILE_N;
constexpr int NUM_TILE_K = (K + TILE_K - 1) / TILE_K;

// ============================================================================
// TODO: Add helper to print matrix slice
// ============================================================================


// ============================================================================
// KERNEL: Tiled GEMM with predication
// ============================================================================
__global__ void irregular_gemm_kernel(float* A, float* B, float* C,
                                       int M_actual, int N_actual, int K_actual,
                                       bool use_predication) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;
  
  if (threadIdx.x != 0) return;
  
  int m_base = tile_m * TILE_M;
  int n_base = tile_n * TILE_N;
  
  // TODO: Check if this is a full tile or partial tile
  bool is_full_tile = /* TODO: (m_base + TILE_M <= M_actual) && (n_base + TILE_N <= N_actual) */;
  
  float accum[TILE_M * TILE_N] = {0.0f};
  
  // K-loop for GEMM
  for (int k_tile = 0; k_tile < NUM_TILE_K; ++k_tile) {
    int k_base = k_tile * TILE_K;
    
    for (int i = 0; i < TILE_M; ++i) {
      for (int j = 0; j < TILE_N; ++j) {
        int global_m = m_base + i;
        int global_n = n_base + j;
        
        // TODO: Apply predication check if use_predication is true
        if (use_predication) {
          if (/* TODO: global_m >= M_actual || global_n >= N_actual */) {
            continue;
          }
        }
        
        // Dot product for C[m, n]
        float sum = 0.0f;
        for (int k = k_base; k < k_base + TILE_K && k < K_actual; ++k) {
          float a_val = A[global_m * K_actual + k];
          float b_val = B[k * N_actual + global_n];
          sum += /* TODO: a_val * b_val */;
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
        C[global_m * N_actual + global_n] = /* TODO: accum[i * TILE_N + j] */;
      }
    }
  }
  
  printf("Block (%d, %d): %s\n", tile_n, tile_m, is_full_tile ? "FULL" : "PARTIAL");
}

// ============================================================================
// TODO: Add CPU reference GEMM function
// Hint: Standard O(M*N*K) matrix multiply
// ============================================================================


// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 07 — Predication\n";
  std::cout << "Exercise 02: Irregular Tile GEMM (FILL-IN)\n";
  std::cout << "============================================================\n\n";
  
  // PREDICTIONS:
  // Q1: How many full tiles in 100×100 output with 32×32 tiles?
  //     Your answer: ___
  //
  // Q2: How many partial tiles?
  //     Your answer: ___
  
  std::cout << "Your predictions:\n";
  std::cout << "  Q1: ___\n";
  std::cout << "  Q2: ___\n\n";
  
  // Allocate and initialize
  float* h_A = new float[M * K];
  float* h_B = new float[K * N];
  float* h_C_pred = new float[M * N];
  float* h_C_ref = new float[M * N];
  
  // TODO: Initialize A and B with known values
  
  std::cout << "=== Matrix Dimensions ===\n";
  std::cout << "  A: " << M << " × " << K << "\n";
  std::cout << "  B: " << K << " × " << N << "\n";
  std::cout << "  C: " << M << " × " << N << "\n";
  std::cout << "  Grid: " << NUM_TILE_M << " × " << NUM_TILE_N << " tiles\n\n";
  
  // TODO: Call CPU reference GEMM
  
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  
  // TODO: Copy A and B to device
  
  // Launch kernel
  dim3 grid(NUM_TILE_N, NUM_TILE_M);
  dim3 block(32);
  irregular_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, true);
  cudaDeviceSynchronize();
  
  // TODO: Copy result back and verify
  
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Full tiles: No predication (fast path)\n";
  std::cout << "  Partial tiles: Bounds checking (correctness)\n";
  std::cout << "  Padding alternative: More memory, simpler code\n";
  std::cout << "============================================================\n";
  
  // Cleanup
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_pred;
  delete[] h_C_ref;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 *
 * Q1: When would you choose padding over predication?
 *     Your answer: ___
 *
 * Q2: What percentage of tiles are partial in 100×100 GEMM?
 *     Your answer: ___
 */
