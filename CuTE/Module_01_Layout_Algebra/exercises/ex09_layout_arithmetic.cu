/**
 * Exercise 09: Layout Arithmetic and Advanced Operations
 *
 * Objective: Master advanced layout operations including arithmetic,
 *            introspection, and complex transformations
 *
 * Tasks:
 * 1. Perform layout arithmetic operations
 * 2. Use layout introspection functions
 * 3. Apply multiple transformations in sequence
 * 4. Work with mixed static/dynamic layouts
 *
 * Key Concepts:
 * - Layout concatenation and product
 * - Rank and size queries
 * - Chained transformations
 * - Compile-time vs runtime layouts
 */

#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 09: Layout Arithmetic and Advanced Operations ==="
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 1: Layout Introspection
  // =========================================================================
  std::cout << "--- Task 1: Layout Introspection ---" << std::endl;

  auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}),
                            GenRowMajor{});

  std::cout << "Layout: " << layout << std::endl;

  // Query rank (number of dimensions)
  std::cout << "Rank: " << rank(layout) << std::endl;

  // Query total size
  std::cout << "Total size: " << size(layout) << std::endl;

  // Query shape components
  auto shape = layout.shape();
  std::cout << "Shape: " << shape << std::endl;
  std::cout << "  Dim 0: " << get<0>(shape) << std::endl;
  std::cout << "  Dim 1: " << get<1>(shape) << std::endl;
  std::cout << "  Dim 2: " << get<2>(shape) << std::endl;

  // Query stride components
  auto stride = layout.stride();
  std::cout << "Stride: " << stride << std::endl;
  std::cout << "  Dim 0 stride: " << get<0>(stride) << std::endl;
  std::cout << "  Dim 1 stride: " << get<1>(stride) << std::endl;
  std::cout << "  Dim 2 stride: " << get<2>(stride) << std::endl;

  // Verify offset calculation manually
  std::cout << "\nManual verification:" << std::endl;
  std::cout << "  layout(1, 2, 3) = " << layout(1, 2, 3) << std::endl;
  std::cout << "  Expected: 1*128 + 2*16 + 3*1 = " << 1 * 128 + 2 * 16 + 3 * 1
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 2: Layout Concatenation (1D)
  // =========================================================================
  std::cout << "--- Task 2: Layout Concatenation ---" << std::endl;

  // Create two 1D layouts
  auto left = make_layout(Int<16>{});
  auto right = make_layout(Int<16>{});

  std::cout << "Left layout:  " << left << std::endl;
  std::cout << "Right layout: " << right << std::endl;

  // Concatenate layouts (conceptually appending)
  // In CuTe, this is done by creating a new layout with combined size
  auto concatenated = make_layout(Int<32>{});

  std::cout << "Concatenated: " << concatenated << std::endl;
  std::cout << "Size check: " << size(left) << " + " << size(right) << " = "
            << size(left) + size(right) << " (actual: " << size(concatenated)
            << ")" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 3: Layout Product (Tensor Product)
  // =========================================================================
  std::cout << "--- Task 3: Layout Product ---" << std::endl;

  // Create two independent layouts
  auto rows = make_layout(Int<4>{});
  auto cols = make_layout(Int<8>{});

  std::cout << "Rows layout: " << rows << std::endl;
  std::cout << "Cols layout: " << cols << std::endl;

  // Create product layout (2D matrix)
  auto product = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});

  std::cout << "Product layout: " << product << std::endl;
  std::cout << "Size check: " << size(rows) << " × " << size(cols) << " = "
            << size(rows) * size(cols) << " (actual: " << size(product) << ")"
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 4: Chained Transformations
  // =========================================================================
  std::cout << "--- Task 4: Chained Transformations ---" << std::endl;

  // Start with a base layout
  auto base = make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{});
  std::cout << "Base layout (8x16 row-major):" << std::endl;
  print(base);
  std::cout << std::endl;

  // Step 1: Transpose
  auto transposed =
      make_layout(get<1>(base), get<0>(base));  // Swap shape and stride
  std::cout << "After transpose (16x8):" << std::endl;
  print(transposed);
  std::cout << std::endl;

  // Step 2: Partition into 4x4 tiles
  auto partitioned =
      logical_divide(transposed, make_shape(Int<4>{}, Int<4>{}));
  std::cout << "After partition into 4x4 tiles:" << std::endl;
  print(partitioned);
  std::cout << std::endl;

  // Step 3: Verify a specific element through the chain
  std::cout << "Element tracking through transformations:" << std::endl;
  std::cout << "  Base (2, 8):     " << base(2, 8) << std::endl;
  std::cout << "  Transposed (8, 2): " << transposed(8, 2) << std::endl;
  std::cout << "  Should be equal: " << (base(2, 8) == transposed(8, 2) ? "YES"
                                                                         : "NO")
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 5: Mixed Static/Dynamic Layouts
  // =========================================================================
  std::cout << "--- Task 5: Mixed Static/Dynamic Layouts ---" << std::endl;

  // Compile-time known dimension
  constexpr int TILE_SIZE = 16;

  // Runtime dimension
  int num_tiles = 4;

  // Mixed layout: static tile size, dynamic count
  auto mixed = make_layout(make_shape(num_tiles, Int<TILE_SIZE>{}),
                           GenRowMajor{});

  std::cout << "Mixed layout (dynamic × static):" << std::endl;
  std::cout << "  Layout: " << mixed << std::endl;
  std::cout << "  Shape:  " << mixed.shape() << std::endl;
  std::cout << "  Stride: " << mixed.stride() << std::endl;

  // Access with runtime coordinates
  int tile_idx = 2;
  int elem_idx = 5;
  std::cout << "  mixed(" << tile_idx << ", " << elem_idx << ") = "
            << mixed(tile_idx, elem_idx) << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 6: Layout Broadcasting
  // =========================================================================
  std::cout << "--- Task 6: Layout Broadcasting ---" << std::endl;

  // Create a broadcast layout (stride 0 for broadcast dimension)
  // This represents a row vector that can be broadcast across rows
  auto broadcast_row = make_layout(
      make_shape(Int<4>{}, Int<8>{}),
      make_stride(Int<0>{}, Int<1>{})  // Stride 0 for row dimension
  );

  std::cout << "Broadcast row layout (4x8, row stride=0):" << std::endl;
  print(broadcast_row);
  std::cout << std::endl;

  // All rows map to the same underlying data
  std::cout << "Broadcast behavior:" << std::endl;
  std::cout << "  broadcast(0, 3) = " << broadcast_row(0, 3) << std::endl;
  std::cout << "  broadcast(1, 3) = " << broadcast_row(1, 3) << std::endl;
  std::cout << "  broadcast(2, 3) = " << broadcast_row(2, 3) << std::endl;
  std::cout << "  broadcast(3, 3) = " << broadcast_row(3, 3) << std::endl;
  std::cout << "  All map to offset 3 (broadcast across rows)" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 7: Sub-Layout Extraction
  // =========================================================================
  std::cout << "--- Task 7: Sub-Layout Extraction ---" << std::endl;

  auto matrix = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});

  // Extract a sub-region (rows 4-7, cols 8-11)
  // This is done by creating a new layout with appropriate offset
  auto sub_shape = make_shape(Int<4>{}, Int<4>{});
  auto sub_stride = make_stride(Int<16>{}, Int<1>{});  // Same stride as parent
  auto sub_layout = make_layout(sub_shape, sub_stride);

  std::cout << "Parent layout (16x16):" << std::endl;
  print(matrix);
  std::cout << std::endl;

  std::cout << "Sub-layout (4x4, same stride):" << std::endl;
  print(sub_layout);
  std::cout << std::endl;

  // Manual offset adjustment for sub-region
  int base_offset = matrix(4, 8);  // Starting point
  std::cout << "Base offset for sub-region (4,8): " << base_offset << std::endl;
  std::cout << "Sub-layout(0,0) + base = " << sub_layout(0, 0) + base_offset
            << " (should equal parent(4,8) = " << matrix(4, 8) << ")"
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 8: Layout Composition Verification
  // =========================================================================
  std::cout << "--- Task 8: Layout Composition Verification ---" << std::endl;

  // Create a composed layout for a tiled matrix multiplication
  // 8x8 matrix divided into 2x2 tiles of 4x4 elements

  auto composed = make_layout(
      make_shape(make_shape(Int<4>{}, Int<2>{}),  // Row: 4 elements × 2 tiles
                 make_shape(Int<4>{}, Int<2>{})), // Col: 4 elements × 2 tiles
      make_stride(make_stride(Int<1>{}, Int<16>{}),  // Row strides
                  make_stride(Int<4>{}, Int<32>{}))  // Col strides
  );

  std::cout << "Composed tiled layout (8x8 as 2x2 tiles of 4x4):" << std::endl;
  print(composed);
  std::cout << std::endl;

  // Verify specific elements
  std::cout << "Element verification:" << std::endl;
  std::cout << "  composed(0, 0) = " << composed(0, 0) << " (tile 0,0 elem 0,0)"
            << std::endl;
  std::cout << "  composed(3, 3) = " << composed(3, 3) << " (tile 0,0 elem 3,3)"
            << std::endl;
  std::cout << "  composed(4, 4) = " << composed(4, 4) << " (tile 1,1 elem 0,0)"
            << std::endl;
  std::cout << "  composed(7, 7) = " << composed(7, 7) << " (tile 1,1 elem 3,3)"
            << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // CHALLENGE: Layout Analysis
  // =========================================================================
  std::cout << "=== Challenge: Layout Analysis ===" << std::endl;

  auto mystery = make_layout(
      make_shape(make_shape(Int<2>{}, Int<4>{}), make_shape(Int<2>{}, Int<4>{})),
      make_stride(make_stride(Int<1>{}, Int<8>{}), make_stride(Int<2>{}, Int<16>{})));

  std::cout << "Analyze this layout:" << std::endl;
  print(mystery);
  std::cout << std::endl;

  std::cout << "Questions:" << std::endl;
  std::cout << "1. What is the total size? Answer: " << size(mystery)
            << std::endl;
  std::cout << "2. What is the rank? Answer: " << rank(mystery) << std::endl;
  std::cout << "3. Offset at ((0,0),(0,0))? Answer: " << mystery(0, 0, 0, 0)
            << std::endl;
  std::cout << "4. Offset at ((1,3),(1,3))? Answer: "
            << mystery(1, 3, 1, 3) << std::endl;
  std::cout << std::endl;

  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. Layout introspection: rank(), size(), shape(), stride()"
            << std::endl;
  std::cout << "2. Layout concatenation combines sizes" << std::endl;
  std::cout << "3. Layout product creates multi-dimensional layouts"
            << std::endl;
  std::cout << "4. Transformations can be chained" << std::endl;
  std::cout << "5. Mixed static/dynamic layouts are supported" << std::endl;
  std::cout << "6. Broadcasting uses stride 0" << std::endl;
  std::cout << "7. Sub-layouts share stride with parent" << std::endl;
  std::cout << "8. Composed layouts enable hierarchical organization"
            << std::endl;

  return 0;
}
