// ============================================================
// MODULE 07 — Predication
// Exercise 03: FlashAttention-2 with Variable Sequence Length
// FILL-IN VERSION — Complete the TODO sections
// ============================================================
// CONCEPT:
//   FlashAttention-2 attention with seqlen_kv = 97 (not divisible by BLOCK_N = 64).
//   Combined predicate: causal_mask(i, j) AND oob_mask(j).
//   Softmax: masked elements contribute exp(-inf) = 0.
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex03_fa2_variable_seqlen_FILL_IN.cu -o ex03_fa2_variable_seqlen_FILL_IN
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cute;

constexpr int SEQLEN_Q = 8;
constexpr int SEQLEN_KV = 97;
constexpr int D_HEAD = 16;
constexpr int BLOCK_N = 64;
constexpr int NUM_BLOCKS = (SEQLEN_KV + BLOCK_N - 1) / BLOCK_N;

// ============================================================================
// TODO: Add CPU reference FlashAttention function
// Hint: For each query position, compute attention scores, apply causal mask,
//       softmax, and weighted sum of V values
// ============================================================================


// ============================================================================
// KERNEL: FlashAttention-2 inner loop with predication
// ============================================================================
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O,
                                        int seqlen_q, int seqlen_kv, int d_head,
                                        int block_n) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  
  int i = blockIdx.x;  // Query position
  if (i >= seqlen_q) return;
  if (threadIdx.x != 0) return;
  
  // Accumulators for online softmax
  float acc[D_HEAD] = {0.0f};
  float denom = 0.0f;
  float prev_max = -INFINITY;
  
  // KV loop over blocks
  for (int block_idx = 0; block_idx < NUM_BLOCKS; ++block_idx) {
    int kv_base = block_idx * block_n;
    
    printf("Q pos %d: processing KV block %d (base=%d)\n", i, block_idx, kv_base);
    
    float block_scores[BLOCK_N];
    float block_max = -INFINITY;
    int valid_count = 0;
    
    for (int j_local = 0; j_local < block_n; ++j_local) {
      int j = kv_base + j_local;  // Global KV position
      
      // TODO: Combined predicate: causal AND OOB
      bool is_causal = /* TODO: (!causal || j <= i) */;  // Assume causal = true
      bool is_in_bounds = /* TODO: (j < seqlen_kv) */;
      bool is_valid = is_causal && is_in_bounds;
      
      if (is_valid) {
        // Compute Q[i] · K[j]
        float dot = 0.0f;
        for (int k = 0; k < d_head; ++k) {
          dot += /* TODO: Q[i * d_head + k] * K[j * d_head + k] */;
        }
        block_scores[j_local] = dot * scale;
        block_max = std::max(block_max, block_scores[j_local]);
        valid_count++;
      } else {
        block_scores[j_local] = -INFINITY;  // Masked
      }
    }
    
    printf("  Valid KV positions: %d, block_max: %.4f\n", valid_count, block_max);
    
    // Online softmax rescale
    float new_max = std::max(prev_max, block_max);
    float rescale_prev = std::exp(prev_max - new_max);
    float rescale_block = std::exp(block_max - new_max);
    
    // TODO: Rescale accumulator
    for (int k = 0; k < d_head; ++k) {
      acc[k] *= /* TODO: rescale_prev */;
    }
    denom *= rescale_prev;
    
    // Accumulate weighted V
    for (int j_local = 0; j_local < block_n; ++j_local) {
      if (block_scores[j_local] != -INFINITY) {
        float exp_val = std::exp(block_scores[j_local] - block_max);
        float weight = exp_val * rescale_block;
        
        int j = kv_base + j_local;
        if (j < seqlen_kv) {
          for (int k = 0; k < d_head; ++k) {
            acc[k] += /* TODO: weight * V[j * d_head + k] */;
          }
          denom += weight;
        }
      }
    }
    
    prev_max = new_max;
  }
  
  // Normalize output
  float inv_denom = 1.0f / denom;
  for (int k = 0; k < d_head; ++k) {
    O[i * d_head + k] = /* TODO: acc[k] * inv_denom */;
  }
}

// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 07 — Predication\n";
  std::cout << "Exercise 03: FlashAttention-2 (FILL-IN)\n";
  std::cout << "============================================================\n\n";
  
  // PREDICTIONS:
  // Q1: How many KV blocks for seqlen_kv=97, BLOCK_N=64?
  //     Your answer: ___
  //
  // Q2: For query position i=5, how many KV positions are valid (causal)?
  //     Your answer: ___
  
  std::cout << "Your predictions:\n";
  std::cout << "  Q1: ___\n";
  std::cout << "  Q2: ___\n\n";
  
  // Allocate and initialize
  float* h_Q = new float[SEQLEN_Q * D_HEAD];
  float* h_K = new float[SEQLEN_KV * D_HEAD];
  float* h_V = new float[SEQLEN_KV * D_HEAD];
  float* h_O_gpu = new float[SEQLEN_Q * D_HEAD];
  float* h_O_cpu = new float[SEQLEN_Q * D_HEAD];
  
  // TODO: Initialize Q, K, V with known values
  
  std::cout << "=== Configuration ===\n";
  std::cout << "  seqlen_q: " << SEQLEN_Q << "\n";
  std::cout << "  seqlen_kv: " << SEQLEN_KV << "\n";
  std::cout << "  BLOCK_N: " << BLOCK_N << "\n";
  std::cout << "  NUM_BLOCKS: " << NUM_BLOCKS << "\n\n";
  
  // TODO: Call CPU reference
  
  // Allocate device memory
  float *d_Q, *d_K, *d_V, *d_O;
  cudaMalloc(&d_Q, SEQLEN_Q * D_HEAD * sizeof(float));
  cudaMalloc(&d_K, SEQLEN_KV * D_HEAD * sizeof(float));
  cudaMalloc(&d_V, SEQLEN_KV * D_HEAD * sizeof(float));
  cudaMalloc(&d_O, SEQLEN_Q * D_HEAD * sizeof(float));
  
  // TODO: Copy Q, K, V to device
  
  // Launch kernel
  flash_attention_kernel<<<SEQLEN_Q, 32>>>(d_Q, d_K, d_V, d_O,
                                            SEQLEN_Q, SEQLEN_KV, D_HEAD, BLOCK_N);
  cudaDeviceSynchronize();
  
  // TODO: Copy result back and verify
  
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Combined predicate: causal AND OOB\n";
  std::cout << "  Online softmax: rescale when new max found\n";
  std::cout << "  exp(-inf) = 0, so masked positions contribute 0\n";
  std::cout << "============================================================\n";
  
  // Cleanup
  delete[] h_Q;
  delete[] h_K;
  delete[] h_V;
  delete[] h_O_gpu;
  delete[] h_O_cpu;
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
  
  return 0;
}

/*
 * CHECKPOINT: You've completed Module 07!
 *
 * Q1: What's the combined predicate for FlashAttention-2?
 *     Your answer: ___
 *
 * Q2: How does online softmax handle masked positions?
 *     Your answer: ___
 */
