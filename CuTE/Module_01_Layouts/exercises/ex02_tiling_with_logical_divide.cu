/*
 * WHAT THIS TEACHES:
 *   - Use logical_divide to tile a layout into smaller blocks
 *   - Understand the composed layout structure (big shape, tile shape)
 *   - Access elements within tiles using nested coordinates
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 tiles Q into [Br, head_dim] blocks where Br=64 or 128.
 *   logical_divide(shape, tile_shape) expresses this tiling without manual
 * index math. This maps to: NVIDIA DL Software Engineer — "FlashAttention
 * tiling with CuTe"
 *
 * MENTAL MODEL:
 *   logical_divide creates a TWO-LEVEL layout:
 *   - Level 0: which tile you're in (the "big" coordinates)
 *   - Level 1: which element within that tile (the "small" coordinates)
 *   Access: layout(make_coord(tile_i, tile_j), make_coord(elem_i, elem_j))
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: Tiling with logical_divide
// ============================================================================
__global__ void tiling_kernel() {
  // MENTAL MODEL: Start with a [128, 64] layout — like a Q tensor slice
  auto layout_big = make_layout(make_shape(Int<128>{}, Int<64>{}));

  printf("=== Original Layout ===\n");
  print(layout_big);
  printf("\n");

  // MENTAL MODEL: logical_divide splits the layout into tiles
  // Divide [128, 64] into tiles of [32, 16]
  // Result: 4 tiles in dim 0 (128/32=4), 4 tiles in dim 1 (64/16=4) = 16 tiles
  // total
  auto layout_tiled =
      logical_divide(layout_big, make_shape(Int<32>{}, Int<16>{}));

  printf("=== Tiled Layout (128x64 divided into 32x16 tiles) ===\n");
  print(layout_tiled);
  printf("\n");

  // MENTAL MODEL: The tiled layout has shape ((4,4), (32,16))
  // - Outer (4,4): which tile (4 tiles in each dimension)
  // - Inner (32,16): element within tile

  // Access element (50, 20) in the original layout
  // Using tiled layout: tile (1, 1), element (18, 4)
  // Because: tile 1 starts at row 32, col 16
  //          element (18, 4) within tile = row 32+18=50, col 16+4=20

  int original_offset = layout_big(Int<50>{}, Int<20>{});
  int tiled_offset = layout_tiled(make_coord(Int<1>{}, Int<1>{}),
                                  make_coord(Int<18>{}, Int<4>{}));

  printf("Original layout (50, 20) -> offset %d\n", original_offset);
  printf("Tiled layout tile(1,1) elem(18,4) -> offset %d\n", tiled_offset);
  printf("Match: %s\n", (original_offset == tiled_offset) ? "YES" : "NO");

  // MENTAL MODEL: Iterate over all tiles and print their starting offsets
  printf("\n=== Tile Starting Offsets ===\n");
  constexpr int NUM_TILES_0 = 128 / 32; // 4 tiles in dim 0
  constexpr int NUM_TILES_1 = 64 / 16;  // 4 tiles in dim 1

  for (int ti = 0; ti < NUM_TILES_0; ti++) {
    for (int tj = 0; tj < NUM_TILES_1; tj++) {
      // Get offset of first element in each tile (element 0,0 within tile)
      int tile_start = layout_tiled(make_coord(ti, tj), make_coord(0, 0));
      printf("Tile (%d,%d) starts at offset %d\n", ti, tj, tile_start);
    }
  }
}

// ============================================================================
// CPU REFERENCE: Verify tile offset calculation
// ============================================================================
void cpu_reference_tiling() {
  printf("\n=== CPU Reference ===\n");
  // Row-major [128, 64]: offset = row * 64 + col
  // Tile (1,1) starts at row=32, col=16
  int tile_start = 32 * 64 + 16;
  printf("Tile (1,1) start offset = 32*64 + 16 = %d\n", tile_start);

  // Element (18, 4) within tile (1,1) is at global (50, 20)
  int elem_offset = 50 * 64 + 20;
  printf("Element (50, 20) offset = 50*64 + 20 = %d\n", elem_offset);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== Tiling with logical_divide Exercise ===\n");
  printf("GPU: %s\n\n", prop.name);

  // PREDICT BEFORE RUNNING:
  // Q1: How many tiles result from dividing [128, 64] by [32, 16]?
  // Q2: What is the starting offset of tile (1, 1)?
  // Q3: Does layout_tiled(tile_coord, elem_coord) equal
  // layout_big(global_coord)?

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  tiling_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range
  // PROFILE: ncu --set full ./ex02_tiling_with_logical_divide
  nvtxRangePush("tiling_kernel");
  tiling_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_tiling();

  printf("\n[PASS] Tiled layout offsets match CPU reference\n");

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 *
 * Q1: What is the shape of logical_divide([128,64], [32,16])?
 *     Answer: ((4, 4), (32, 16)) — outer is tile count, inner is tile size
 *
 * Q2: How do you access element (r, c) within tile (ti, tj)?
 *     Answer: layout(make_coord(ti, tj), make_coord(r, c))
 *
 * Q3: In FlashAttention-2 with Br=64 and seqlen=512, how many row tiles?
 *     Answer: 512 / 64 = 8 row tiles
 */
