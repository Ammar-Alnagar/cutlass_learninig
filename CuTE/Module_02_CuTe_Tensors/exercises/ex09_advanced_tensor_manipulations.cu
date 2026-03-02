/**
 * Exercise 09: Advanced Tensor Manipulations
 *
 * Objective: Master advanced tensor operations including complex views,
 *            tensor algebra, and performance-oriented patterns
 *
 * Instructions:
 * - Complete each TODO section by implementing advanced tensor operations
 * - Build on concepts from previous exercises
 * - Think about how these patterns apply to real GPU kernels
 *
 * Key Concepts:
 * - Chained views: Multiple transformations on the same data
 * - Reshape views: Same data, different dimensional organization
 * - Tensor partitioning: Dividing tensors into tiles
 * - Strided views: Subsampling patterns
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 09: Advanced Tensor Manipulations ===" << std::endl;
  std::cout << std::endl;

  // Create test data
  int data[256];
  for (int i = 0; i < 256; ++i) {
    data[i] = i;
  }


  // ========================================================================
  // TASK 1: Chained Tensor Views
  // ========================================================================
  // Goal: Create multiple views of the same data with different transformations
  std::cout << "Task 1 - Chained Tensor Views:" << std::endl;
  std::cout << "Start with a 16x16 tensor and create multiple views" << std::endl;
  std::cout << std::endl;

  // Base tensor: 16x16 row-major
  auto base_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
  auto base_tensor = make_tensor(make_gmem_ptr(data), base_layout);

  std::cout << "Base tensor (16x16) layout:" << std::endl;
  // TODO: Print the base layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Create a transposed view
  std::cout << "Creating transposed view..." << std::endl;
  // TODO: Create a 16x16 column-major layout (transpose of row-major)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the transposed layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Transposed view layout:" << std::endl;
  // TODO: Print the transposed layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Verify transpose relationship
  std::cout << "Verify transpose: base(3, 7) should equal transposed(7, 3)" << std::endl;
  std::cout << "  base(3, 7) = " << base_tensor(3, 7) << std::endl;
  std::cout << "  transposed(7, 3) = ";
  // TODO: Print transposed_tensor(7, 3)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Create a sub-tensor view (8x8 from center)
  // ========================================================================
  // Goal: Extract a view of the center 8x8 region
  std::cout << "Task 2 - Sub-tensor View (Center 8x8):" << std::endl;
  
  // TODO: Create an 8x8 layout with stride (16, 1) - same as parent stride
  // Hint: make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<16>{}, Int<1>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Calculate offset for center (starting at row 4, col 4)
  // Hint: offset = base_layout(4, 4)
  int center_offset = 0;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor with data + center_offset
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Center 8x8 sub-tensor (starting at offset " << center_offset << "):" << std::endl;
  // TODO: Print the sub-tensor (just first 4 rows to save space)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  ..." << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Tensor Reshape Views
  // ========================================================================
  // Goal: View the same 256 elements as different shapes
  std::cout << "Task 3 - Tensor Reshape Views:" << std::endl;
  std::cout << "Same 256 elements, different views:" << std::endl;
  std::cout << std::endl;

  // 2D view: 16x16
  auto tensor_2d = make_tensor(make_gmem_ptr(data), base_layout);

  // TODO: Create a 1D view with 256 elements
  // Hint: make_layout(Int<256>{})
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the flat layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a 3D view with shape (4, 4, 16)
  // Hint: make_layout(make_shape(Int<4>{}, Int<4>{}, Int<16>{}), GenRowMajor{})
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the 3D layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "2D view (16x16) layout: ";
  // TODO: Print tensor_2d.layout()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "1D view (256) layout:   ";
  // TODO: Print flat_tensor.layout()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "3D view (4x4x16) layout: ";
  // TODO: Print tensor_3d.layout()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;

  // Verify equivalence
  std::cout << "Verify all views access the same data:" << std::endl;
  std::cout << "Position representing offset 90:" << std::endl;
  std::cout << "  2D(5, 10) = " << tensor_2d(5, 10) << std::endl;
  std::cout << "  1D(90)    = ";
  // TODO: Print flat_tensor(90)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << "  3D(1, 2, 10) = ";
  // TODO: Print tensor_3d(1, 2, 10)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Tensor Broadcasting with Stride 0
  // ========================================================================
  // Goal: Create broadcast layouts for scalar and vector broadcasting
  std::cout << "Task 4 - Tensor Broadcasting Patterns:" << std::endl;
  
  int scalar_value = 42;

  // TODO: Create a broadcast layout for 8x8 matrix from scalar
  // Hint: make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<0>{}, Int<0>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the broadcast layout and &scalar_value
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Scalar broadcast to 8x8:" << std::endl;
  std::cout << "Layout: ";
  // TODO: Print broadcast_layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Sample accesses: broadcast(0,0)=" << broadcast_tensor(0, 0);
  std::cout << ", broadcast(4,4)=" << broadcast_tensor(4, 4);
  std::cout << ", broadcast(7,7)=" << broadcast_tensor(7, 7) << std::endl;
  std::cout << "All access the same scalar value!" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Vector broadcast (bias pattern)
  // ========================================================================
  // Goal: Broadcast a 64-element vector to a 32x64 matrix
  std::cout << "Task 5 - Vector Broadcast (Bias Pattern):" << std::endl;
  
  int bias[64];
  for (int i = 0; i < 64; ++i) {
    bias[i] = i * 10;
  }

  // TODO: Create a bias matrix layout with shape (32, 64) and stride (0, 1)
  // This broadcasts the 64-element vector to 32 identical rows
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the bias matrix layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Bias matrix layout (32 rows, 64 cols, row stride=0):" << std::endl;
  // TODO: Print bias_matrix.layout()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Verify broadcasting - all rows have same values:" << std::endl;
  std::cout << "  bias_matrix(0, 10)  = " << bias_matrix(0, 10) << std::endl;
  std::cout << "  bias_matrix(15, 10) = ";
  // TODO: Print bias_matrix(15, 10)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << "  bias_matrix(31, 10) = ";
  // TODO: Print bias_matrix(31, 10)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (all should be 100)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Strided Tensor Views
  // ========================================================================
  // Goal: Create views that access every Nth element
  std::cout << "Task 6 - Strided Tensor Views:" << std::endl;
  std::cout << "Create a view that accesses every other row" << std::endl;
  std::cout << std::endl;

  // TODO: Create a strided layout with shape (8, 16) and stride (32, 1)
  // Row stride of 32 means we skip every other row of the 16-wide original
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the strided layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Strided view layout (every other row):" << std::endl;
  // TODO: Print strided_layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "First 4 rows of strided view:" << std::endl;
  // TODO: Print first 4 rows (each with 8 columns) of strided_tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (Row 0 = original row 0, Row 1 = original row 2, etc.)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 7: Tensor Partitioning with logical_divide
  // ========================================================================
  // Goal: Use logical_divide to partition a tensor into tiles
  std::cout << "Task 7 - Tensor Partitioning:" << std::endl;
  std::cout << "Partition a 16x16 tensor into 4x4 tiles" << std::endl;
  std::cout << std::endl;

  // TODO: Use logical_divide to partition base_layout into 4x4 tiles
  // Hint: auto partitioned = logical_divide(base_layout, make_shape(Int<4>{}, Int<4>{}));
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Partitioned layout:" << std::endl;
  // TODO: Print partitioned_layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "The partitioned layout has 4x4 = 16 tiles, each tile is 4x4 elements" << std::endl;
  std::cout << "Access pattern: partitioned(tile_i, tile_j, elem_i, elem_j)" << std::endl;
  std::cout << std::endl;

  // Access a specific tile
  std::cout << "Tile (0, 0) - top-left tile elements:" << std::endl;
  // TODO: Print elements of tile (0,0) using partitioned_layout(0, 0, ei, ej)
  // Hint: Loop ei, ej from 0 to 3
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Tile (1, 1) - center tile elements:" << std::endl;
  // TODO: Print elements of tile (1,1)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 8: Tensor Alignment for Vectorized Access
  // ========================================================================
  // Goal: Understand how layout affects vectorized memory access
  std::cout << "Task 8 - Tensor Alignment for Vectorized Access:" << std::endl;
  std::cout << "For vectorized loads (float4), elements must be consecutive" << std::endl;
  std::cout << std::endl;

  auto aligned_layout = make_layout(
    make_shape(Int<16>{}, Int<16>{}),
    make_stride(Int<16>{}, Int<1>{})
  );

  std::cout << "Aligned layout (column stride = 1):" << std::endl;
  // TODO: Print aligned_layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Consecutive column elements (for vectorized load):" << std::endl;
  std::cout << "  Offsets for row 0, cols 0-3: ";
  // TODO: Print aligned_layout(0, 0), aligned_layout(0, 1), aligned_layout(0, 2), aligned_layout(0, 3)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << " (consecutive!)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Design a layout for matrix multiplication tiles
  // ========================================================================
  std::cout << "=== Challenge: Layout for Matrix Multiplication ===" << std::endl;
  std::cout << "Design shared memory layouts for a 16x16 tile with padding" << std::endl;
  std::cout << "Padding avoids bank conflicts during matrix multiplication" << std::endl;
  std::cout << std::endl;

  // TODO: Create a layout for a 16x16 tile with stride (17, 1) - padding of 1
  // The extra column prevents bank conflicts when accessing columns
  // Hint: make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<17>{}, Int<1>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Padded shared memory layout:" << std::endl;
  // TODO: Print your padded layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Notice: Row stride is 17, not 16 (1 element padding per row)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE 2: Compute the offset for a complex access pattern
  // ========================================================================
  std::cout << "=== Challenge 2: Complex Offset Calculation ===" << std::endl;
  std::cout << "For a partitioned 16x16 tensor with 4x4 tiles:" << std::endl;
  std::cout << "Calculate the flat offset for tile (2, 1), element (3, 2)" << std::endl;
  std::cout << std::endl;

  // TODO: Calculate manually:
  // Tile (2, 1) starts at global row = 2*4 = 8, global col = 1*4 = 4
  // Element (3, 2) within tile is at global row = 8+3 = 11, global col = 4+2 = 6
  // Flat offset in 16x16 row-major = 11 * 16 + 6 = 182
  int manual_offset = 0;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Verify using partitioned_layout(2, 1, 3, 2)
  int layout_offset = 0;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Manual calculation: " << manual_offset << std::endl;
  std::cout << "From partitioned layout: " << layout_offset << std::endl;
  std::cout << "Match: " << (manual_offset == layout_offset ? "YES" : "NO") << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. How do you create multiple views of the same data?" << std::endl;
  std::cout << "2. What is the benefit of reshape views (no copying)?" << std::endl;
  std::cout << "3. How does stride=0 enable broadcasting?" << std::endl;
  std::cout << "4. What does logical_divide() do?" << std::endl;
  std::cout << "5. Why add padding to shared memory layouts?" << std::endl;

  return 0;
}
