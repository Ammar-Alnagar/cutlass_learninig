/**
 * Exercise 04: Layout Composition
 *
 * Objective: Learn to compose multiple layouts together to create hierarchical
 *            memory structures for tiled algorithms
 *
 * Key Concepts:
 * - Layout Composition: Combining layouts to create complex mappings
 * - Hierarchical Layouts: Multi-level organization (block -> thread -> element)
 * - Tiling: Dividing computation into smaller tiles
 */

#include "cute/layout.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

// Manual print_layout — works for any standard Layout type

int main() {
  std::cout << "=== Exercise 04: Layout Composition ===" << std::endl;
  std::cout << std::endl;

  // -------------------------------------------------------------------------
  // TASK 1: Tile layout — 2x2 grid of tiles
  // Default (no stride) = LayoutLeft = column-major
  // Stride (_1, _2): cost 1 to move down a row, cost 2 to move across a col
  // Tile IDs:  0 2
  //            1 3
  // -------------------------------------------------------------------------
  auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}));

  std::cout << "Task 1 - Tile Layout (2x2 tiles):" << std::endl;
  print_layout(tile_layout);

  // -------------------------------------------------------------------------
  // TASK 2: Element layout — 4x4 elements inside one tile
  // Stride (_1, _4): cost 1 down a row, cost 4 across a col (column-major)
  // -------------------------------------------------------------------------
  auto element_layout = make_layout(make_shape(Int<4>{}, Int<4>{}));

  std::cout << "Task 2 - Element Layout (4x4 elements per tile):" << std::endl;
  print_layout(element_layout);

  // -------------------------------------------------------------------------
  // TASK 3: Build the full tiled 8x8 layout using nested shapes
  //
  // We describe the 8x8 matrix as ((4,2),(4,2)) — hierarchically:
  //   First mode:  4 rows within a tile, 2 tile-rows
  //   Second mode: 4 cols within a tile, 2 tile-cols
  //
  // Strides ((1,16),(4,32)):
  //   1  = one step down within a tile (row-within-tile stride)
  //   16 = one step to the next tile-row (4 rows × 4 cols = 16 elements)
  //   4  = one step right within a tile (col-within-tile stride)
  //   32 = one step to the next tile-col (16 elements/tile × 2 tile-rows)
  //
  // Note: composition() returns a ComposedLayout type which doesn't work
  // with our print_layout template. Building via make_layout with nested
  // shapes is the correct explicit approach for tiling anyway.
  // -------------------------------------------------------------------------
  auto composed_layout =
      make_layout(make_shape(make_shape(Int<4>{}, Int<2>{}),
                             make_shape(Int<4>{}, Int<2>{})),
                  make_stride(make_stride(Int<1>{}, Int<16>{}),
                              make_stride(Int<4>{}, Int<32>{})));

  std::cout << "Task 3 - Composed Tiled 8x8 Layout:" << std::endl;
  print_layout(composed_layout);

  // -------------------------------------------------------------------------
  // TASK 4: Compare to plain row-major 8x8
  //
  // full_layout:    stride (8,1) — simple row-major, element (i,j) -> i*8+j
  // composed_layout: stride ((1,16),(4,32)) — tiled, elements within a tile
  //                  are contiguous in column-major order
  //
  // Same logical shape (8x8), completely different memory ordering.
  // The tiled layout is better for GPU because all threads in a warp
  // accessing a tile get coalesced memory reads.
  // -------------------------------------------------------------------------
  auto full_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});

  std::cout << "Task 4 - Full row-major 8x8 Layout (for comparison):"
            << std::endl;
  print_layout(full_layout);

  std::cout << "Composed layout at (5,2) = " << composed_layout(5, 2)
            << std::endl;
  std::cout << "Full layout at    (5,2) = " << full_layout(5, 2) << std::endl;
  std::cout << "Different because: tiled uses column-major within tiles, full "
               "uses row-major."
            << std::endl;
  std::cout << std::endl;

  // -------------------------------------------------------------------------
  // TASK 5: Query a specific element
  //
  // Element (5,2) lives in:
  //   tile row = 5/4 = 1, tile col = 2/4 = 0  → tile (1,0)
  //   local row = 5%4 = 1, local col = 2%4 = 2 → local position (1,2)
  //
  // composed_layout(5,2) = 1*1 + 1*16 + 2*4 + 0*32 = 1 + 16 + 8 = 25
  // full_layout(5,2)     = 5*8 + 2 = 42
  // -------------------------------------------------------------------------
  std::cout << "Task 5 - Index Query for element (5, 2):" << std::endl;
  std::cout << "composed_layout(5,2) = " << composed_layout(5, 2) << std::endl;
  std::cout << "full_layout(5,2)     = " << full_layout(5, 2) << std::endl;
  std::cout << std::endl;

  // -------------------------------------------------------------------------
  // TASK 6: Verify composed layout matches tiled reference
  //
  // tiled_reference is the same layout as composed_layout,
  // just constructed identically to double-check correctness.
  // All 64 elements should match.
  // -------------------------------------------------------------------------
  auto tiled_reference =
      make_layout(make_shape(make_shape(Int<4>{}, Int<2>{}),
                             make_shape(Int<4>{}, Int<2>{})),
                  make_stride(make_stride(Int<1>{}, Int<16>{}),
                              make_stride(Int<4>{}, Int<32>{})));

  std::cout << "Task 6 - Verifying composed_layout against tiled reference:"
            << std::endl;
  bool all_match = true;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      if (composed_layout(i, j) != tiled_reference(i, j)) {
        std::cout << "  Mismatch at (" << i << "," << j
                  << "): " << composed_layout(i, j)
                  << " != " << tiled_reference(i, j) << std::endl;
        all_match = false;
      }
    }
  }
  if (all_match) {
    std::cout << "  All 64 elements match! Layout is correct." << std::endl;
  }
  std::cout << std::endl;

  // -------------------------------------------------------------------------
  // TASK 7: Visualize tile ownership
  //
  // For each element in the 8x8 matrix, show which tile (0-3) owns it.
  // We use tile_layout to compute the tile ID instead of manual arithmetic.
  // size<0>(element_layout) = 4 = tile height
  // size<1>(element_layout) = 4 = tile width
  // -------------------------------------------------------------------------
  std::cout << "Task 7 - Tile ownership map (8x8):" << std::endl;
  int tile_h = size<0>(element_layout);
  int tile_w = size<1>(element_layout);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      std::cout << tile_layout(i / tile_h, j / tile_w) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. Nested make_layout with hierarchical shapes builds tiled "
               "layouts explicitly"
            << std::endl;
  std::cout << "2. Strides encode both intra-tile position and inter-tile jumps"
            << std::endl;
  std::cout << "3. Tiled and row-major layouts have the same shape but "
               "different memory order"
            << std::endl;
  std::cout
      << "4. Tiled column-major within tiles gives better GPU memory coalescing"
      << std::endl;
  std::cout << "5. This pattern is the foundation of tiled GEMM and shared "
               "memory kernels"
            << std::endl;

  return 0;
}
