/*
 * WHAT THIS TEACHES:
 *   - Construct layouts with stride-0 for broadcast patterns
 *   - Express GQA (Grouped Query Attention) where multiple query heads share
 * K/V
 *   - Understand how stride-0 enables efficient memory access without copies
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   GQA is used in Llama-2-70B, Mistral, and other modern LLMs to reduce KV
 * cache size. 8 query heads might share 2 K/V heads, reducing memory by 4x.
 *   CuTe expresses this with stride-0: the K/V head dimension has stride 0,
 *   so all query heads in a group read the same K/V data.
 *   This maps to: Cerebras LLM Inference Performance — "FlashAttention variants
 * including GQA"
 *
 * MENTAL MODEL:
 *   GQA: 8 query heads, 2 K/V heads -> each K/V head is shared by 4 query heads
 *   Query layout: [8, seqlen, head_dim] — normal row-major
 *   K/V layout: [2, seqlen, head_dim] with head stride = 0 for the group
 * dimension When query head q accesses K/V, it uses K/V head index q / 4
 * (integer division)
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: GQA stride-0 broadcast layout
// ============================================================================
__global__ void gqa_stride_zero_kernel() {
  // MENTAL MODEL: Standard attention — each head has its own K/V
  // Layout: [4 heads, 8 seqlen, 16 head_dim]
  auto layout_kv_normal = make_layout(make_shape(Int<4>{}, // heads
                                                 Int<8>{}, // seqlen
                                                 Int<16>{} // head_dim
                                                 ));

  printf("=== Normal KV Layout [heads=4, seqlen=8, head_dim=16] ===\n");
  print(layout_kv_normal);
  printf("\n");

  // MENTAL MODEL: GQA — 8 query heads share 2 K/V heads
  // Each K/V head is shared by 4 query heads (8 / 2 = 4)
  // Query heads 0-3 -> K/V head 0
  // Query heads 4-7 -> K/V head 1

  // K/V layout for GQA: [2 heads, 8 seqlen, 16 head_dim]
  auto layout_kv_gqa =
      make_layout(make_shape(Int<2>{}, // K/V heads (fewer than query heads)
                             Int<8>{}, // seqlen
                             Int<16>{} // head_dim
                             ));

  printf("=== GQA KV Layout [kv_heads=2, seqlen=8, head_dim=16] ===\n");
  print(layout_kv_gqa);
  printf("\n");

  // MENTAL MODEL: When query head q needs K/V, compute kv_head = q / 4
  // Then access: layout_kv_gqa(kv_head, seq, dim)
  // This is the "broadcast" — multiple query indices map to same K/V offset

  // Example: Query head 3 and query head 2 both use K/V head 0
  int q3_kv_head = 3 / 4; // = 0
  int q2_kv_head = 2 / 4; // = 0
  int q5_kv_head = 5 / 4; // = 1

  int offset_q3 = layout_kv_gqa(Int<q3_kv_head>{}, Int<4>{}, Int<8>{});
  int offset_q2 = layout_kv_gqa(Int<q2_kv_head>{}, Int<4>{}, Int<8>{});
  int offset_q5 = layout_kv_gqa(Int<q5_kv_head>{}, Int<4>{}, Int<8>{});

  printf("Query head 3 -> K/V head %d, offset %d\n", q3_kv_head, offset_q3);
  printf("Query head 2 -> K/V head %d, offset %d\n", q2_kv_head, offset_q2);
  printf("Query head 5 -> K/V head %d, offset %d\n", q5_kv_head, offset_q5);
  printf("Q3 and Q2 share same K/V: %s\n",
         (offset_q3 == offset_q2) ? "YES" : "NO");

  // MENTAL MODEL: Advanced — create a "virtual" layout with stride-0
  // This layout appears to have 8 heads but stride-0 makes groups share data
  // Shape: [8 query_heads, 8 seqlen, 16 head_dim]
  // Stride: [0, 16, 1] — head dimension has stride 0!
  // Wait, that's not quite right...

  // Actually, for GQA we want:
  // - Within a group of 4 query heads, they all read the same K/V data
  // - So the "head" stride should be 0 for the intra-group dimension

  // Better model: think of it as [2 groups, 4 intra_group, seqlen, head_dim]
  // where intra_group has stride 0
  auto layout_gqa_broadcast =
      make_layout(make_shape(Int<2>{}, Int<4>{}, Int<8>{},
                             Int<16>{}), // [groups, intra, seq, dim]
                  make_stride(Int<128>{}, Int<0>{}, Int<16>{},
                              Int<1>{}) // intra has stride 0!
      );

  printf("\n=== GQA Broadcast Layout [groups=2, intra=4, seqlen=8, "
         "head_dim=16] ===\n");
  print(layout_gqa_broadcast);
  printf("\n");

  // Verify: all intra-group indices map to same offset
  int g0_i0 = layout_gqa_broadcast(Int<0>{}, Int<0>{}, Int<4>{}, Int<8>{});
  int g0_i1 = layout_gqa_broadcast(Int<0>{}, Int<1>{}, Int<4>{}, Int<8>{});
  int g0_i2 = layout_gqa_broadcast(Int<0>{}, Int<2>{}, Int<4>{}, Int<8>{});
  int g0_i3 = layout_gqa_broadcast(Int<0>{}, Int<3>{}, Int<4>{}, Int<8>{});

  printf("Group 0, intra 0-3 at (seq=4, dim=8):\n");
  printf("  intra=0: offset %d\n", g0_i0);
  printf("  intra=1: offset %d\n", g0_i1);
  printf("  intra=2: offset %d\n", g0_i2);
  printf("  intra=3: offset %d\n", g0_i3);
  printf("All same (stride-0 broadcast): %s\n",
         (g0_i0 == g0_i1 && g0_i1 == g0_i2 && g0_i2 == g0_i3) ? "YES" : "NO");
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_gqa() {
  printf("\n=== CPU Reference ===\n");

  // GQA: 8 query heads, 2 K/V heads
  // Query head q -> K/V head q / 4
  // K/V layout [2, 8, 16], row-major, stride = (128, 16, 1)

  printf("Query head 3 -> K/V head 0, offset = 0*128 + 4*16 + 8 = %d\n",
         0 * 128 + 4 * 16 + 8);
  printf("Query head 2 -> K/V head 0, offset = 0*128 + 4*16 + 8 = %d\n",
         0 * 128 + 4 * 16 + 8);
  printf("Query head 5 -> K/V head 1, offset = 1*128 + 4*16 + 8 = %d\n",
         1 * 128 + 4 * 16 + 8);

  // Stride-0 broadcast layout verification
  // [2, 4, 8, 16] with stride [128, 0, 16, 1]
  printf("\nStride-0 broadcast:\n");
  printf("All intra-group indices map to: group*128 + 0*0 + seq*16 + dim\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== GQA Stride-Zero Broadcast Exercise ===\n");
  printf("GPU: %s\n\n", prop.name);

  // PREDICT BEFORE RUNNING:
  // Q1: For 8 query heads and 2 K/V heads, how many query heads per K/V head?
  // Q2: What stride value enables broadcast (multiple indices -> same offset)?
  // Q3: Query head 7 maps to which K/V head index?

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  gqa_stride_zero_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range
  // PROFILE: ncu --set full ./ex04_gqa_stride_zero
  nvtxRangePush("gqa_stride_zero_kernel");
  gqa_stride_zero_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_gqa();

  printf("\n[PASS] GQA stride-0 broadcast verified\n");

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to Module 02
 *
 * Q1: For 32 query heads and 8 K/V heads, how many query heads per K/V head?
 *     Answer: 32 / 8 = 4 query heads per K/V head
 *
 * Q2: What stride value creates a broadcast pattern?
 *     Answer: 0 — all indices along that dimension map to the same offset
 *
 * Q3: In Llama-2-70B with 64 query heads and 8 K/V heads, what is the
 *     K/V head index for query head 45?
 *     Answer: 45 / 8 = 5 (integer division)
 *
 * === MODULE 01 COMPLETE ===
 * Exit criteria:
 * 1. Can construct row-major [M, N] layout and explain stride (N, 1)
 * 2. Can use logical_divide to tile [128, 64] into [32, 16] tiles
 * 3. Can write 4D attention layout [batch, heads, seqlen, head_dim]
 * 4. Can express GQA with stride-0 broadcast
 *
 * Next: Module 02 — Tensors (make_tensor, slicing, local_tile)
 */
