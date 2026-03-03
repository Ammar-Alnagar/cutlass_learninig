/*
 * EXERCISE: GQA Stride with Zero Padding - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Handle GQA where Q and K/V have different head counts
 *   - Use zero padding to align memory for efficient access
 *   - Create strided layouts for GQA attention
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   In GQA, multiple Q heads share one K/V head
 *   We can pad K/V to match Q's head dimension for aligned access
 *   Strided layout skips padded regions when indexing
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: GQA Stride with Zero Padding
// ============================================================================
__global__ void gqa_stride_kernel() {
  // CONCEPT: GQA configuration
  // Q has 16 heads, K/V have 4 heads (4 Q heads per KV head)
  constexpr int SEQLEN = 256;
  constexpr int Q_HEADS = 16;
  constexpr int KV_HEADS = 4;
  constexpr int HEAD_DIM = 64;
  constexpr int GROUP_SIZE = Q_HEADS / KV_HEADS;  // 4 Q heads per KV head

  // CONCEPT: Without padding, K/V would be [seqlen, KV_HEADS, head_dim]
  // With padding to match Q's head count: [seqlen, Q_HEADS, head_dim]
  // But only KV_HEADS * head_dim elements are actual data per row
  
  // TODO 1: Create Q layout (dense, no padding)
  // Shape: [SEQLEN, Q_HEADS, HEAD_DIM]
  auto Q_layout = /* YOUR CODE HERE */;

  printf("=== Q Layout (dense) ===\n");
  print(Q_layout);
  printf("\n");

  // TODO 2: Create K layout with GQA stride
  // K has shape [SEQLEN, KV_HEADS, HEAD_DIM] but we want to index it
  // as if it had Q_HEADS heads (with stride skipping non-existent heads)
  // 
  // Use make_layout with custom stride:
  // Shape: (SEQLEN, KV_HEADS, HEAD_DIM)
  // Stride: (KV_HEADS * HEAD_DIM, HEAD_DIM, 1) - standard row-major for K's actual shape
  auto K_layout = /* YOUR CODE HERE */;

  printf("=== K Layout (GQA, %d heads) ===\n", KV_HEADS);
  print(K_layout);
  printf("\n");

  // CONCEPT: Access pattern for GQA
  // Q head h uses K head (h / GROUP_SIZE)
  // Example: Q heads 0,1,2,3 all use K head 0
  //          Q heads 4,5,6,7 all use K head 1
  
  // TODO 3: Calculate which K head corresponds to Q head 7
  // Hint: k_head = q_head / GROUP_SIZE
  int q_head_7 = 7;
  int k_head_for_q7 = /* YOUR CODE HERE */;
  printf("Q head %d uses K head %d\n", q_head_7, k_head_for_q7);

  // TODO 4: Access Q at (seq=10, head=7, dim=5)
  int q_offset = Q_layout(Int<10>{}, Int<7>{}, Int<5>{});
  printf("Q(10, 7, 5) -> offset %d\n", q_offset);

  // TODO 5: Access K at (seq=10, head=k_head_for_q7, dim=5)
  int k_offset = K_layout(Int<10>{}, Int<k_head_for_q7>{}, Int<5>{});
  printf("K(10, %d, 5) -> offset %d\n", k_head_for_q7, k_offset);

  // CONCEPT: Calculate total sizes
  // Q has Q_HEADS * HEAD_DIM elements per sequence position
  // K has KV_HEADS * HEAD_DIM elements per sequence position
  
  // TODO 6: Get total size of Q tensor
  auto Q_total = size(Q_layout);
  printf("\nQ total elements: %d\n", Q_total);

  // TODO 7: Get total size of K tensor
  auto K_total = size(K_layout);
  printf("K total elements: %d\n", K_total);
  
  // Verify: Q should have 4x more elements than K (since Q_HEADS = 4 * KV_HEADS)
  printf("Q/K size ratio: %.1f (expected %d.0)\n", 
         float(Q_total) / float(K_total), GROUP_SIZE);
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_gqa() {
  printf("\n=== CPU Reference ===\n");
  
  // Q [256, 16, 64]: stride = (16*64, 64, 1) = (1024, 64, 1)
  int q_offset = 10 * 16 * 64 + 7 * 64 + 5;
  printf("Q(10, 7, 5) offset = %d\n", q_offset);
  
  // K [256, 4, 64]: stride = (4*64, 64, 1) = (256, 64, 1)
  int k_head = 7 / 4;  // = 1
  int k_offset = 10 * 4 * 64 + k_head * 64 + 5;
  printf("K(10, %d, 5) offset = %d\n", k_head, k_offset);
  
  printf("Q total: 256 * 16 * 64 = %d\n", 256 * 16 * 64);
  printf("K total: 256 * 4 * 64 = %d\n", 256 * 4 * 64);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  printf("=== GQA Stride Exercise ===\n\n");

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  gqa_stride_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range
  nvtxRangePush("gqa_stride_kernel");
  gqa_stride_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_gqa();

  printf("\n[PASS] GQA stride verified\n");

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to Module 02
 * 
 * Q1: In GQA with 32 Q heads and 8 KV heads, how many Q heads share one KV head?
 *     Answer: _______________
 * 
 * Q2: What is the memory savings of GQA vs standard attention?
 *     Answer: _______________
 * 
 * Q3: Why is GQA popular in modern LLMs (Llama-2-70B, etc.)?
 *     Answer: _______________
 */
