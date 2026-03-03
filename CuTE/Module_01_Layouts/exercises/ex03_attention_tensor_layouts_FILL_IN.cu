/*
 * EXERCISE: Attention Tensor Layouts - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Create layouts for Q, K, V tensors in attention
 *   - Understand GQA (Grouped Query Attention) stride patterns
 *   - Use make_layout for 3D tensors [batch, seqlen, head_dim]
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   Q, K, V tensors are typically [batch, seqlen, head_dim]
 *   In GQA, K/V have fewer heads than Q
 *   GQA stride: Q has N heads, K/V have N//groups heads
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: Attention Tensor Layouts
// ============================================================================
__global__ void attention_layouts_kernel() {
  // CONCEPT: Standard attention tensor shape
  // [batch_size, seqlen, num_heads, head_dim]
  // For simplicity, we'll use [seqlen, num_heads, head_dim]
  
  constexpr int SEQLEN = 512;
  constexpr int NUM_HEADS = 32;
  constexpr int HEAD_DIM = 128;

  // TODO 1: Create row-major layout for Q tensor [seqlen, heads, head_dim]
  // Hint: make_layout(make_shape(Int<SEQLEN>{}, Int<NUM_HEADS>{}, Int<HEAD_DIM>{}))
  auto Q_layout = /* YOUR CODE HERE */;

  printf("=== Q Tensor Layout ===\n");
  print(Q_layout);
  printf("\n");

  // CONCEPT: K and V have same shape as Q in standard attention
  // TODO 2: Create layout for K tensor (same shape as Q)
  auto K_layout = /* YOUR CODE HERE */;
  
  // TODO 3: Create layout for V tensor (same shape as Q)
  auto V_layout = /* YOUR CODE HERE */;

  // CONCEPT: In Grouped Query Attention (GQA), K/V have fewer heads
  // Example: Q has 32 heads, K/V have 8 heads (4 Q heads per KV head)
  constexpr int KV_HEADS = 8;
  constexpr int GROUP_SIZE = NUM_HEADS / KV_HEADS;  // 4 Q heads per KV head

  // TODO 4: Create layout for K_GQA tensor [seqlen, KV_HEADS, head_dim]
  // Hint: Same as Q but with KV_HEADS instead of NUM_HEADS
  auto K_GQA_layout = /* YOUR CODE HERE */;

  printf("=== K_GQA Layout (GQA with %d KV heads) ===\n", KV_HEADS);
  print(K_GQA_layout);
  printf("\n");

  // CONCEPT: Access specific elements
  // Q(seq, head, dim) accesses element at that position
  
  // TODO 5: Access Q at position (seq=100, head=5, dim=10)
  // Hint: Q_layout(Int<100>{}, Int<5>{}, Int<10>{})
  int q_offset = /* YOUR CODE HERE */;
  printf("Q(100, 5, 10) -> offset %d\n", q_offset);

  // TODO 6: Access K_GQA at position (seq=100, head=1, dim=10)
  // Note: head index is 0-7 for K_GQA (only 8 KV heads)
  int k_gqa_offset = K_GQA_layout(Int<100>{}, Int<1>{}, Int<10>{});
  printf("K_GQA(100, 1, 10) -> offset %d\n", k_gqa_offset);

  // CONCEPT: Calculate total size
  // TODO 7: Get the total number of elements in Q layout
  // Hint: Use size(Q_layout) or compute from shape
  auto Q_size = /* YOUR CODE HERE */;
  printf("Q tensor total elements: %d\n", Q_size);
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_attention() {
  printf("\n=== CPU Reference ===\n");
  
  // Row-major [512, 32, 128]: stride = (32*128, 128, 1) = (4096, 128, 1)
  int q_offset = 100 * 32 * 128 + 5 * 128 + 10;
  printf("Q(100, 5, 10) offset = %d\n", q_offset);
  
  // K_GQA [512, 8, 128]: stride = (8*128, 128, 1) = (1024, 128, 1)
  int k_offset = 100 * 8 * 128 + 1 * 128 + 10;
  printf("K_GQA(100, 1, 10) offset = %d\n", k_offset);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  printf("=== Attention Tensor Layouts Exercise ===\n\n");

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  attention_layouts_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range
  nvtxRangePush("attention_layouts_kernel");
  attention_layouts_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_attention();

  printf("\n[PASS] Attention layouts verified\n");

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex04
 * 
 * Q1: What is the stride for Q[512, 32, 128] in row-major?
 *     Answer: _______________
 * 
 * Q2: In GQA with 32 Q heads and 8 KV heads, what is the group size?
 *     Answer: _______________
 * 
 * Q3: Why use GQA instead of standard attention?
 *     Answer: _______________
 */
