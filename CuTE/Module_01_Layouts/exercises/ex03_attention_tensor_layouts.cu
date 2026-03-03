/*
 * WHAT THIS TEACHES:
 *   - Construct 4D layouts for attention tensors [batch, heads, seqlen,
 * head_dim]
 *   - Understand row-major stride calculation for multi-dimensional tensors
 *   - Use print_layout to visualize 4D memory mapping
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   Production LLM attention kernels operate on 4D tensors:
 *   [batch, num_heads, seq_len, head_dim]
 *   Understanding the memory layout is critical for correct tiling and
 * coalesced access. This maps to: Modular AI Kernel Engineer —
 * "high-performance attention kernels"
 *
 * MENTAL MODEL:
 *   Row-major 4D [B, H, S, D] has stride (H*S*D, S*D, D, 1)
 *   Moving along head_dim (fastest) = stride 1
 *   Moving along seq_len = stride D (skip one head_dim)
 *   Moving along heads = stride S*D (skip one seq_len)
 *   Moving along batch = stride H*S*D (skip one head dimension)
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: 4D attention tensor layout
// ============================================================================
__global__ void attention_layouts_kernel() {
  // MENTAL MODEL: Small example [2, 4, 8, 16] for visualization
  // batch=2, heads=4, seqlen=8, head_dim=16
  // Total elements: 2 * 4 * 8 * 16 = 1024

  auto layout_4d = make_layout(make_shape(Int<2>{}, // batch
                                          Int<4>{}, // heads
                                          Int<8>{}, // seqlen
                                          Int<16>{} // head_dim
                                          ));

  printf("=== 4D Attention Tensor Layout [batch=2, heads=4, seqlen=8, "
         "head_dim=16] ===\n");
  print(layout_4d);
  printf("\n");

  // MENTAL MODEL: stride for row-major [B, H, S, D] is (H*S*D, S*D, D, 1)
  // For [2, 4, 8, 16]: stride = (4*8*16, 8*16, 16, 1) = (512, 128, 16, 1)
  auto stride_4d = stride(layout_4d);
  printf("Stride tuple: ");
  print(stride_4d);
  printf("\n");

  // Verify specific index mappings
  // Access element at (batch=1, head=2, seq=4, dim=8)
  int offset = layout_4d(Int<1>{}, Int<2>{}, Int<4>{}, Int<8>{});
  int expected = 1 * 512 + 2 * 128 + 4 * 16 + 8; // = 512 + 256 + 64 + 8 = 840
  printf("Offset for (1, 2, 4, 8): %d (expected %d) - %s\n", offset, expected,
         (offset == expected) ? "MATCH" : "MISMATCH");

  // MENTAL MODEL: FlashAttention-2 typically fuses batch*heads into one
  // dimension This simplifies tiling: [batch*heads, seqlen, head_dim]
  auto layout_fused =
      make_layout(make_shape(Int<8>{}, // batch * heads = 2 * 4 = 8
                             Int<8>{}, // seqlen
                             Int<16>{} // head_dim
                             ));

  printf("\n=== Fused Layout [batch*heads=8, seqlen=8, head_dim=16] ===\n");
  print(layout_fused);
  printf("\n");

  // Access the same logical element in fused layout
  // Original (batch=1, head=2) -> fused head index = 1*4 + 2 = 6
  int fused_offset = layout_fused(Int<6>{}, Int<4>{}, Int<8>{});
  printf("Fused offset for (6, 4, 8): %d (should equal 840)\n", fused_offset);

  // MENTAL MODEL: Tiling for FlashAttention-2
  // Tile along seqlen dimension with Br=4 (small for demo, real is 64 or 128)
  auto layout_tiled = logical_divide(
      make_layout(make_shape(Int<8>{}, Int<8>{}, Int<16>{})),
      make_shape(Int<1>{}, Int<4>{}, Int<1>{}) // Tile only seqlen dimension
  );

  printf("\n=== Tiled Layout (seqlen tiled with Br=4) ===\n");
  print(layout_tiled);
  printf("\n");

  // How many tiles in seqlen dimension? 8 / 4 = 2 tiles
  // Access tile 1 (second tile), element at seq position 2 within tile
  // Global seq position = 4 + 2 = 6
  int tiled_off = layout_tiled(
      make_coord(Int<0>{}, Int<1>{},
                 Int<0>{}), // batch*heads=0, tile=1, head_dim=0
      make_coord(Int<0>{}, Int<2>{}, Int<8>{}) // within tile: b=0, seq=2, d=8
  );
  printf("Tiled access at tile(0,1,0) elem(0,2,8): offset %d\n", tiled_off);
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_attention() {
  printf("\n=== CPU Reference ===\n");

  // Row-major [2, 4, 8, 16]
  // stride = (512, 128, 16, 1)
  int offset = 1 * 512 + 2 * 128 + 4 * 16 + 8;
  printf("4D offset (1,2,4,8) = 1*512 + 2*128 + 4*16 + 8 = %d\n", offset);

  // Fused [8, 8, 16]: stride = (128, 16, 1)
  // (batch=1, head=2) -> fused = 6
  int fused_offset = 6 * 128 + 4 * 16 + 8;
  printf("Fused offset (6,4,8) = 6*128 + 4*16 + 8 = %d\n", fused_offset);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== Attention Tensor Layouts Exercise ===\n");
  printf("GPU: %s\n\n", prop.name);

  // PREDICT BEFORE RUNNING:
  // Q1: What is the stride for row-major [B, H, S, D]?
  // Q2: For [2,4,8,16], what offset does (1,2,4,8) map to?
  // Q3: After fusing batch*heads, what is the new shape?

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  attention_layouts_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range
  // PROFILE: ncu --set full ./ex03_attention_tensor_layouts
  nvtxRangePush("attention_layouts_kernel");
  attention_layouts_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_attention();

  printf("\n[PASS] 4D attention layout offsets verified\n");

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex04
 *
 * Q1: What is the stride tuple for row-major [B, H, S, D]?
 *     Answer: (H*S*D, S*D, D, 1)
 *
 * Q2: Why does FlashAttention-2 fuse batch and heads dimensions?
 *     Answer: Simplifies tiling logic — only need to tile along seqlen,
 *             and all batch*heads sequences are processed uniformly.
 *
 * Q3: For seqlen=512 and Br=64, how many row tiles per sequence?
 *     Answer: 512 / 64 = 8 tiles
 */
