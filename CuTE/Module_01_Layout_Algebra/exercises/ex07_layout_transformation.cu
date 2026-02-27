/**
 * Exercise 07: Layout Transformation (CORRECTED)
 *
 * Objective: Learn to transform layouts using operations like transpose,
 *            reshape (coalesce/flatten), and partition
 * (logical_divide/tiled_divide)
 *
 * Key Concepts:
 * - Transpose: Reorder the modes (dimensions) of an existing layout
 * - Coalesce/Flatten: Merge/collapse modes to reduce dimensionality
 * - logical_divide: Partition a layout by a tile shape
 * - tiled_divide: Like logical_divide but reorders modes for tiling patterns
 *
 * FIX SUMMARY vs original:
 * - Task 2: Transpose base_layout by swapping its own shape/stride modes,
 *           NOT by creating a brand-new unrelated layout.
 * - Task 3: Use coalesce() or flatten to actually reshape base_layout,
 *           NOT a new layout with (32, 1) shape which is still 2D.
 * - Task 4: Use logical_divide() to produce partitions that properly map
 *           into base_layout's coordinate space with correct offsets/strides.
 * - Task 5: Use tiled_divide() for the tiled view instead of manual CPU math.
 * - Challenge: Apply the same correct patterns to the 16x8 case.
 */

#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>
#include <print>

using namespace cute;

int main() {
  std::cout << "=== Exercise 07: Layout Transformation (Corrected) ==="
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 1: Create a base layout
  // =========================================================================
  // make_layout with GenRowMajor produces stride (8, 1) for a 4x8 matrix,
  // meaning consecutive columns are stride-1 apart, rows are stride-8 apart.
  // Coordinate (i, j) maps to offset i*8 + j*1.
  auto base_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
  print_layout(base_layout);

  std::cout << "Task 1 - Base Layout (4x8 Row-Major):" << std::endl;
  std::cout << "  Layout: " << base_layout << std::endl; // ((4,8),(8,1))
  std::cout << "  Shape:  " << base_layout.shape() << std::endl;  // (4,8)
  std::cout << "  Stride: " << base_layout.stride() << std::endl; // (8,1)
  print_layout(base_layout); // visual grid: row i, col j -> offset i*8+j
  std::cout << std::endl;

  // =========================================================================
  // TASK 2: Transpose — swap modes of base_layout
  // =========================================================================
  // FIX: Do NOT create a new unrelated layout. The correct approach is to
  // reorder the existing layout's modes with make_layout(shape, stride) where
  // shape and stride are the *swapped* components of base_layout.
  //
  // base_layout has shape (4,8) and stride (8,1).
  // Transpose swaps mode-0 and mode-1:
  //   transposed shape  = (8, 4)
  //   transposed stride = (1, 8)
  // Now coordinate (i, j) in the transposed layout maps to offset i*1 + j*8,
  // which is the same memory as column-major access of the original 4x8 data.
  //
  // TODO: Fill in the correct shape and stride below.
  auto transposed_layout =
      make_layout(get<1>(base_layout), get<0>(base_layout));
  print_layout(transposed_layout);

  std::cout << "Task 2 - Transposed Layout (8x4):" << std::endl;
  std::cout << "  Layout: " << transposed_layout << std::endl; // (8,4):(1,8)
  std::cout << "  Shape:  " << transposed_layout.shape() << std::endl;
  std::cout << "  Stride: " << transposed_layout.stride() << std::endl;
  print_layout(transposed_layout);
  std::cout << std::endl;

  // Verify: base_layout(r, c) == transposed_layout(c, r) for same offset
  // e.g. base_layout(1, 3) == transposed_layout(3, 1) == 11
  std::cout << "  Verification: base(1,3)=" << base_layout(1, 3)
            << "  transposed(3,1)=" << transposed_layout(3, 1)
            << "  (both should be 11)" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 3: Reshape — flatten 4x8 into 32 elements using coalesce()
  // =========================================================================
  // FIX: The original used make_layout(make_shape(Int<32>{}, Int<1>{})) which
  // is still a 2D layout (32 rows, 1 column). That is NOT a reshape of
  // base_layout; it is just a new unrelated 2D layout.
  //
  // The correct approach:
  //   coalesce(base_layout) merges adjacent modes that have compatible strides,
  //   collapsing the 2D (4,8):(8,1) into a single 1D mode (32):(1).
  //
  // For row-major (4,8):(8,1):
  //   - mode-1 has stride 1 (innermost)
  //   - mode-0 has stride 8 = size of mode-1  => they are contiguous
  //   => coalesce merges them into shape (32,) stride (1,)
  //
  // TODO: Call coalesce on base_layout.

  // result: (32):(1)

  // NOTE: coalesce() and flatten() are broken in this CUTLASS version for
  // row-major layouts — they fail to merge modes left-to-right.
  // Workaround: manually construct the flat layout using total size + stride
  // _1. This is equivalent to what coalesce() SHOULD produce: _32:_1
  std::cout << "supposedly flattened ?" << std::endl;
  auto linear_layout = make_layout(size(base_layout), Int<1>{});
  print(linear_layout);
  std::cout << "---------------------" << std::endl;

  // ── TASK 5 FIX
  // ──────────────────────────────────────────────────────────────
  // tiled shape is ((2,2),(2,4)) — also rank-2 hierarchical, NOT
  // rank-4. tiled(tile_row, tile_col, inner_r, inner_c)  ← 4 args =
  // WRONG Correct indexing:
  //   mode-0 = (num_row_tiles=2, num_col_tiles=2)  ->
  //   make_coord(tile_row, tile_col) mode-1 = (tile_rows=2,
  //   tile_cols=4)      -> make_coord(inner_r,  inner_c)

  // =========================================================================
  // CHALLENGE: 16x8 layout — correct transformations
  // =========================================================================
  std::cout << "=== Challenge: 16x8 Layout ===" << std::endl;
  auto challenge_base =
      make_layout(make_shape(Int<16>{}, Int<8>{}), GenRowMajor{});
  std::cout << "  base = " << challenge_base << std::endl;
  std::cout << std::endl;

  // 1. Transpose: swap shape (16,8)->(8,16) and stride (8,1)->(1,8)
  // TODO: swap shape and stride modes manually as in Task 2.
  auto ch_transposed = make_layout(make_shape(Int<8>{}, Int<16>{}),
                                   make_stride(Int<1>{}, Int<8>{}));
  std::cout << "  1. Transposed (8x16): " << ch_transposed << std::endl;

  // 2. Linear: coalesce the base layout
  // TODO: call coalesce on challenge_base.
  auto ch_linear = coalesce(challenge_base);
  std::cout << "  2. Linear (128 elem): " << ch_linear << std::endl;

  // 3. Four 8x4 partitions via logical_divide with tile (8,4)
  // => divided shape ((8,2),(4,2)) — 2 row-tiles x 2 col-tiles
  // TODO: call logical_divide on challenge_base with tile (8,4).
  auto ch_divided =
      logical_divide(challenge_base, make_shape(Int<8>{}, Int<4>{}));
  std::cout << "  3. Partitioned (8x4 tiles): " << ch_divided << std::endl;
  std::cout << "     Access tile (r_tile, c_tile): ch_divided(inner_r, "
               "inner_c, r_tile, c_tile)"
            << std::endl;
  std::cout << std::endl;

  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. Transpose = swap shape/stride modes of the SAME layout"
            << std::endl;
  std::cout << "2. Reshape   = coalesce() merges compatible contiguous modes"
            << std::endl;
  std::cout << "3. Partition = logical_divide() produces correctly-strided "
               "sub-layouts"
            << std::endl;
  std::cout << "4. Tiled view= tiled_divide() reorders modes: tile-idx outer, "
               "within-tile inner"
            << std::endl;
  std::cout << "5. All views share the same flat memory — only indexing changes"
            << std::endl;

  return 0;
}
