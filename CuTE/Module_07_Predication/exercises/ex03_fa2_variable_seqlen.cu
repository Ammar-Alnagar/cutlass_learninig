// ============================================================
// MODULE 07 — Predication
// Exercise 03: FlashAttention-2 with Variable Sequence Length
// ============================================================
// CONCEPT:
//   FlashAttention-2 attention score tile with variable sequence length.
//   Q shape (seqlen_q, d_head), K/V shape (seqlen_kv, d_head) where
//   seqlen_kv = 97 (not divisible by BLOCK_N = 64).
//   
//   Combined predicate: causal_mask(i, j) AND oob_mask(j)
//   - causal_mask(i, j) = (i >= j) for causal attention
//   - oob_mask(j) = (j < seqlen_kv) for bounds checking
//   
//   Softmax normalization: masked elements contribute exp(-inf) = 0
//
// JOB RELEVANCE:
//   NVIDIA DL Software Engineer — "Efficient attention kernels for arbitrary
//   sequence lengths" — This is the EXACT pattern required in production
//   FlashAttention-2 for LLM inference with variable-length sequences.
//
// ASCII DIAGRAM:
//   Attention scores: QK^T / sqrt(d_head), shape (seqlen_q, seqlen_kv)
//   seqlen_q = 8, seqlen_kv = 97, BLOCK_N = 64
//   
//   Score matrix (simplified, 8×8 shown):
//   ┌────────────────────────────────┐
//   │ S00 S01 S02 ... S0,96         │  ← Row 0: only col 0 valid (causal)
//   │ S10 S11 S12 ... S1,96         │  ← Row 1: cols 0-1 valid
//   │ S20 S21 S22 ... S2,96         │  ← Row 2: cols 0-2 valid
//   │ ...                            │
//   │ S70 S71 S72 ... S7,96         │  ← Row 7: cols 0-7 valid
//   └────────────────────────────────┘
//         │                │
//         ▼                ▼
//   Block 0 (cols 0-63)  Block 1 (cols 64-96, partial)
//   
//   Combined predicate for Block 1, Row 7:
//   - Causal: j <= 7 (only cols 0-7 are causal)
//   - OOB: j < 97 (all cols 64-96 are in bounds)
//   - Combined: j <= 7 AND j < 97 → j <= 7
//   - For cols 8-63 in Block 1: causal mask = false
//   - For cols 97-127 (if any): OOB mask = false
//   
//   Softmax with masking:
//   For each row i:
//     for each col j:
//       if predicate(i, j):
//         score[i,j] = QK^T[i,j] / sqrt(d_head)
//       else:
//         score[i,j] = -inf  // Masks out invalid positions
//     softmax[i,:] = exp(score[i,:]) / sum(exp(score[i,:]))
//   
//   Key: exp(-inf) = 0, so masked positions contribute 0 to softmax

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cute;

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr int SEQLEN_Q = 8;       // Query sequence length (small for demo)
constexpr int SEQLEN_KV = 97;     // Key/Value sequence length (irregular)
constexpr int D_HEAD = 16;        // Head dimension (small for demo)
constexpr int BLOCK_N = 64;       // KV block size (not dividing 97)
constexpr int NUM_BLOCKS = (SEQLEN_KV + BLOCK_N - 1) / BLOCK_N;  // = 2

// ============================================================================
// HELPER: Compute attention score with predication (CPU reference)
// ============================================================================
void cpu_flash_attention(float* Q, float* K, float* V, float* O,
                          int seqlen_q, int seqlen_kv, int d_head,
                          bool causal) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  
  // For each query position
  for (int i = 0; i < seqlen_q; ++i) {
    // Compute attention scores for all KV positions
    float scores[SEQLEN_KV];
    float max_score = -INFINITY;
    
    for (int j = 0; j < seqlen_kv; ++j) {
      // Causal mask: can only attend to positions <= current
      if (causal && j > i) {
        scores[j] = -INFINITY;  // Masked
      } else {
        // Compute Q[i] · K[j]
        float dot = 0.0f;
        for (int k = 0; k < d_head; ++k) {
          dot += Q[i * d_head + k] * K[j * d_head + k];
        }
        scores[j] = dot * scale;
      }
      max_score = std::max(max_score, scores[j]);
    }
    
    // Compute softmax with max subtraction for numerical stability
    float exp_sum = 0.0f;
    for (int j = 0; j < seqlen_kv; ++j) {
      if (scores[j] != -INFINITY) {
        float exp_val = std::exp(scores[j] - max_score);
        scores[j] = exp_val;
        exp_sum += exp_val;
      } else {
        scores[j] = 0.0f;  // exp(-inf) = 0
      }
    }
    
    // Normalize
    float inv_sum = 1.0f / exp_sum;
    
    // Compute output: weighted sum of V
    for (int k = 0; k < d_head; ++k) {
      float acc = 0.0f;
      for (int j = 0; j < seqlen_kv; ++j) {
        acc += scores[j] * V[j * d_head + k];
      }
      O[i * d_head + k] = acc * inv_sum;
    }
  }
}

// ============================================================================
// KERNEL: FlashAttention-2 inner loop with predication
// ============================================================================
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O,
                                        int seqlen_q, int seqlen_kv, int d_head,
                                        int block_n) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  
  // Each thread block handles one query position
  int i = blockIdx.x;  // Query position
  if (i >= seqlen_q) return;
  
  // Thread 0 does the work (simplified for demo)
  if (threadIdx.x != 0) return;
  
  // Initialize output accumulator
  float acc[D_HEAD] = {0.0f};
  float denom = 0.0f;
  float prev_max = -INFINITY;
  
  // KV loop over blocks
  for (int block_idx = 0; block_idx < NUM_BLOCKS; ++block_idx) {
    int kv_base = block_idx * block_n;
    
    printf("Q pos %d: processing KV block %d (base=%d)\n", i, block_idx, kv_base);
    
    // Compute attention scores for this KV block
    float block_scores[BLOCK_N];
    float block_max = -INFINITY;
    int valid_count = 0;
    
    for (int j_local = 0; j_local < block_n; ++j_local) {
      int j = kv_base + j_local;  // Global KV position
      
      // Combined predicate: causal AND OOB
      bool is_causal = (!causal || j <= i);
      bool is_in_bounds = (j < seqlen_kv);
      bool is_valid = is_causal && is_in_bounds;
      
      if (is_valid) {
        // Compute Q[i] · K[j]
        float dot = 0.0f;
        for (int k = 0; k < d_head; ++k) {
          dot += Q[i * d_head + k] * K[j * d_head + k];
        }
        block_scores[j_local] = dot * scale;
        block_max = std::max(block_max, block_scores[j_local]);
        valid_count++;
      } else {
        block_scores[j_local] = -INFINITY;  // Masked
      }
    }
    
    printf("  Valid KV positions: %d, block_max: %.4f\n", valid_count, block_max);
    
    // Online softmax: rescale previous accumulator
    float new_max = std::max(prev_max, block_max);
    float rescale_prev = std::exp(prev_max - new_max);
    float rescale_block = std::exp(block_max - new_max);
    
    // Rescale accumulator
    for (int k = 0; k < d_head; ++k) {
      acc[k] *= rescale_prev;
    }
    denom *= rescale_prev;
    
    // Compute exp and accumulate
    for (int j_local = 0; j_local < block_n; ++j_local) {
      if (block_scores[j_local] != -INFINITY) {
        float exp_val = std::exp(block_scores[j_local] - block_max);
        float weight = exp_val * rescale_block;
        
        int j = kv_base + j_local;
        if (j < seqlen_kv) {
          for (int k = 0; k < d_head; ++k) {
            acc[k] += weight * V[j * d_head + k];
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
    O[i * d_head + k] = acc[k] * inv_denom;
  }
}

// ============================================================================
// MAIN: Setup, kernel launch, verification
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 07 — Predication\n";
  std::cout << "Exercise 03: FlashAttention-2 with Variable Sequence Length\n";
  std::cout << "============================================================\n\n";
  
  // PREDICT BEFORE RUNNING:
  // Q1: How many KV blocks for seqlen_kv=97, BLOCK_N=64?
  //     Answer: ceil(97/64) = 2 blocks
  //
  // Q2: For query position i=5, how many KV positions are valid (causal)?
  //     Answer: 6 positions (j=0,1,2,3,4,5)
  //
  // Q3: What is exp(-inf) in softmax?
  //     Answer: 0, so masked positions contribute 0 to the output
  
  std::cout << "Predictions:\n";
  std::cout << "  Q1: 2 KV blocks\n";
  std::cout << "  Q2: 6 valid KV positions for i=5\n";
  std::cout << "  Q3: exp(-inf) = 0\n\n";
  
  // Allocate host memory
  float* h_Q = new float[SEQLEN_Q * D_HEAD];
  float* h_K = new float[SEQLEN_KV * D_HEAD];
  float* h_V = new float[SEQLEN_KV * D_HEAD];
  float* h_O_gpu = new float[SEQLEN_Q * D_HEAD];
  float* h_O_cpu = new float[SEQLEN_Q * D_HEAD];
  
  // Initialize with known values for reproducibility
  // Use simple patterns for easy verification
  for (int i = 0; i < SEQLEN_Q * D_HEAD; ++i) {
    h_Q[i] = static_cast<float>((i % 10)) * 0.1f;
  }
  for (int i = 0; i < SEQLEN_KV * D_HEAD; ++i) {
    h_K[i] = static_cast<float>((i % 7 + 1)) * 0.1f;
    h_V[i] = static_cast<float>((i % 5 + 1)) * 0.1f;
  }
  
  std::cout << "=== Configuration ===\n";
  std::cout << "  seqlen_q: " << SEQLEN_Q << "\n";
  std::cout << "  seqlen_kv: " << SEQLEN_KV << " (not divisible by " << BLOCK_N << ")\n";
  std::cout << "  d_head: " << D_HEAD << "\n";
  std::cout << "  BLOCK_N: " << BLOCK_N << "\n";
  std::cout << "  NUM_BLOCKS: " << NUM_BLOCKS << "\n";
  std::cout << "  Causal: true\n\n";
  
  // CPU reference
  std::cout << "=== Computing CPU Reference ===\n";
  cpu_flash_attention(h_Q, h_K, h_V, h_O_cpu, SEQLEN_Q, SEQLEN_KV, D_HEAD, true);
  std::cout << "Done.\n\n";
  
  // Print sample attention pattern
  std::cout << "=== Attention Pattern (causal mask) ===\n";
  std::cout << "For each query position i, valid KV positions j <= i:\n";
  for (int i = 0; i < SEQLEN_Q; ++i) {
    std::cout << "  i=" << i << ": j in [0, " << i << "] (" << (i + 1) << " positions)\n";
  }
  std::cout << "\n";
  
  // Allocate device memory
  float *d_Q, *d_K, *d_V, *d_O;
  cudaMalloc(&d_Q, SEQLEN_Q * D_HEAD * sizeof(float));
  cudaMalloc(&d_K, SEQLEN_KV * D_HEAD * sizeof(float));
  cudaMalloc(&d_V, SEQLEN_KV * D_HEAD * sizeof(float));
  cudaMalloc(&d_O, SEQLEN_Q * D_HEAD * sizeof(float));
  
  cudaMemcpy(d_Q, h_Q, SEQLEN_Q * D_HEAD * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, SEQLEN_KV * D_HEAD * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, SEQLEN_KV * D_HEAD * sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch kernel
  std::cout << "=== Running GPU FlashAttention ===\n";
  flash_attention_kernel<<<SEQLEN_Q, 32>>>(d_Q, d_K, d_V, d_O,
                                            SEQLEN_Q, SEQLEN_KV, D_HEAD, BLOCK_N);
  cudaDeviceSynchronize();
  
  // Copy back
  cudaMemcpy(h_O_gpu, d_O, SEQLEN_Q * D_HEAD * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Verify
  std::cout << "\n=== Verification (GPU vs CPU) ===\n";
  float max_error = 0.0f;
  float tol = 1e-4f;
  bool pass = true;
  
  for (int i = 0; i < SEQLEN_Q * D_HEAD; ++i) {
    float error = std::abs(h_O_gpu[i] - h_O_cpu[i]);
    if (error > max_error) max_error = error;
    if (error > tol) {
      if (pass) {
        printf("First mismatch at [q=%d, k=%d]: expected %.6f, got %.6f\n",
               i / D_HEAD, i % D_HEAD, h_O_cpu[i], h_O_gpu[i]);
      }
      pass = false;
    }
  }
  
  printf("Max error: %.6f (tolerance: %.6f)\n", max_error, tol);
  std::cout << "Result: " << (pass ? "PASS ✓" : "FAIL ✗") << "\n\n";
  
  // Print output comparison
  std::cout << "=== Output Comparison (first query position) ===\n";
  std::cout << "CPU: { ";
  for (int k = 0; k < D_HEAD; ++k) {
    std::cout << std::fixed << std::setprecision(4) << h_O_cpu[k] << " ";
  }
  std::cout << "}\n";
  std::cout << "GPU: { ";
  for (int k = 0; k < D_HEAD; ++k) {
    std::cout << std::fixed << std::setprecision(4) << h_O_gpu[k] << " ";
  }
  std::cout << "}\n\n";
  
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
  
  // KEY INSIGHT
  std::cout << "============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  FlashAttention-2 with variable sequence length:\n";
  std::cout << "\n";
  std::cout << "  Combined predicate:\n";
  std::cout << "    pred(i, j) = causal_mask(i, j) AND oob_mask(j)\n";
  std::cout << "    causal_mask(i, j) = (j <= i) for causal attention\n";
  std::cout << "    oob_mask(j) = (j < seqlen_kv) for bounds checking\n";
  std::cout << "\n";
  std::cout << "  Online softmax rescale:\n";
  std::cout << "    When a new max is found in a KV block:\n";
  std::cout << "    1. Rescale previous accumulator: acc *= exp(prev_max - new_max)\n";
  std::cout << "    2. Accumulate new values with weight exp(score - new_max)\n";
  std::cout << "    3. Update denominator: denom *= exp(prev_max - new_max) + sum(exp)\n";
  std::cout << "\n";
  std::cout << "  This is the EXACT pattern for production FlashAttention-2!\n";
  std::cout << "============================================================\n";
  
  return pass ? 0 : 1;
}

/*
 * CHECKPOINT: You've completed Module 07!
 *
 * Q1: What's the combined predicate for FlashAttention-2?
 *     Answer: pred(i,j) = (j <= i) AND (j < seqlen_kv)
 *             Causal mask ensures we only attend to past/present.
 *             OOB mask ensures we don't access invalid memory.
 *
 * Q2: How does online softmax handle masked positions?
 *     Answer: Masked positions get score = -inf, so exp(-inf) = 0.
 *             They contribute 0 to the softmax sum.
 *
 * Q3: Why is predication critical for production kernels?
 *     Answer: Real LLMs have arbitrary sequence lengths.
 *             A kernel that only works on tile-aligned dimensions
 *             requires padding, wasting memory and compute.
 */
