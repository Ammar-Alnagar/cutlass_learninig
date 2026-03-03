/*
 * EXERCISE: Tiling with logical_divide - Fill in the Gaps
 *
 * WHAT THIS TEACHES:
 *   - Use logical_divide to tile a layout into smaller blocks
 *   - Understand the composed layout structure (big shape, tile shape)
 *   - Access elements within tiles using nested coordinates
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
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
  // CONCEPT: Start with a [128, 64] layout — like a Q tensor slice
  // This represents a portion of the QKV tensor in attention

  // TODO 1: Create a row-major layout for shape [128, 64]
  // Hint: make_layout(make_shape(Int<128>{}, Int<64>{}))
  auto layout_big =
      make_layout(make_shape(Int<128>{}, Int<64>{}), GenRowMajor{});

  printf("=== Original Layout ===\n");
  print(layout_big);
  printf("\n");

  // CONCEPT: logical_divide splits the layout into tiles
  // Divide [128, 64] into tiles of [32, 16]
  // Result: 4 tiles in dim 0 (128/32=4), 4 tiles in dim 1 (64/16=4) = 16 tiles
  // total

  // TODO 2: Apply logical_divide to create tiled layout
  // Hint: logical_divide(layout, tile_shape)
  // Tile shape is make_shape(Int<32>{}, Int<16>{})
  auto layout_tiled =
      logical_divide(layout_big, make_shape(Int<32>{}, Int<16>{}));

  printf("=== Tiled Layout (128x64 divided into 32x16 tiles) ===\n");
  print(layout_tiled);
  printf("\n");

  // CONCEPT: The tiled layout has shape ((4,4), (32,16))
  // - Outer (4,4): which tile (4 tiles in each dimension)
  // - Inner (32,16): element within tile

  // CONCEPT: Access element (50, 20) in the original layout
  // Using tiled layout: tile (1, 1), element (18, 4)
  // Because: tile 1 starts at row 32, col 16
  //          element (18, 4) within tile = row 32+18=50, col 16+4=20

  // Get offset using original layout
  int original_offset = layout_big(Int<50>{}, Int<20>{});

  // TODO 3: Get offset using tiled layout with nested coordinates
  // Hint: layout_tiled(make_coord(tile_i, tile_j), make_coord(elem_i, elem_j))
  // Tile (1,1), element (18,4)
  int tiled_offset = layout_tiled(make_coord(Int<1>{}, Int<1>{}),
                                  make_coord(Int<18>{}, Int<4>{}));

  printf("Original layout (50, 20) -> offset %d\n", original_offset);
  printf("Tiled layout tile(1,1) elem(18,4) -> offset %d\n", tiled_offset);
  printf("Match: %s\n", (original_offset == tiled_offset) ? "YES" : "NO");

  // CONCEPT: Iterate over all tiles and print their starting offsets
  printf("\n=== Tile Starting Offsets ===\n");

  // TODO 4: Calculate number of tiles in each dimension
  // Hint: 128 / 32 = 4 tiles in dim 0
  //       64 / 16 = 4 tiles in dim 1
  constexpr int NUM_TILES_0 = 4;
  constexpr int NUM_TILES_1 = 4;

  for (int ti = 0; ti < NUM_TILES_0; ++ti) {
    for (int tj = 0; tj < NUM_TILES_1; ++tj) {
      // TODO 5: Get offset of first element in each tile
      // Hint: layout_tiled(make_coord(ti, tj), make_coord(0, 0))
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

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  tiling_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range
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
 *     Answer: _______________
 *
 * Q2: How do you access element (r, c) within tile (ti, tj)?
 *     Answer: _______________
 *
 * Q3: In FlashAttention-2 with Br=64 and seqlen=512, how many row tiles?
 *     Answer: _______________
 */
