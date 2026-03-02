/**
 * Exercise 03: Tensor Slicing Operations
 *
 * Objective: Learn to extract sub-tensors and views from larger tensors
 *
 * Instructions:
 * - Complete each TODO section by creating actual tensor slice views
 * - Use make_tensor() with pointer offsets and sub-layouts
 * - Verify your slices by printing and checking values
 *
 * Key Concepts:
 * - Slicing creates a VIEW (not a copy) - same underlying data
 * - A slice needs: base pointer + offset, and a sub-layout
 * - Row slice: 1D view of a row
 * - Column slice: 1D view with stride = num_rows
 * - Sub-matrix: 2D view of a region
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 03: Tensor Slicing Operations ===" << std::endl;
  std::cout << std::endl;

  // Create an 8x8 tensor to slice
  float data[64];
  for (int i = 0; i < 64; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
  auto tensor = make_tensor(make_gmem_ptr(data), layout);

  std::cout << "Original 8x8 Tensor:" << std::endl;
  print(tensor);
  std::cout << std::endl;

  // ========================================================================
  // TASK 1: Extract a row as a 1D tensor view
  // ========================================================================
  // Goal: Create a 1D tensor that views row 3 of the original tensor
  // Hint: Pointer offset = tensor.layout()(3, 0), shape = (8,), stride = (1,)
  std::cout << "Task 1 - Extract Row 3 as a 1D Tensor:" << std::endl;

  // TODO: Create a 1D layout for 8 elements (a row)
  // START YOUR CODE HERE
  auto row_layout = make_layout(make_shape(Int<8>{}), make_stride(Int<1>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view using data + offset for row 3
  // Hint: Offset = layout(3, 0) gives the starting position of row 3
  // START YOUR CODE HERE
  auto offset_row3 = layout(3, 0);
  auto row3_slice = make_tensor(make_gmem_ptr(data + offset_row3), row_layout);

  // END YOUR CODE HERE

  std::cout << "Row 3 slice:" << std::endl;
  // TODO: Print your row slice tensor
  // START YOUR CODE HERE
  print(row3_slice);

  // END YOUR CODE HERE
  std::cout << "Expected: 24 25 26 27 28 29 30 31" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 2: Extract a column as a 1D tensor view
  // ========================================================================
  // Goal: Create a 1D tensor that views column 5 of the original tensor
  // Hint: Pointer offset = tensor.layout()(0, 5), stride = (8,) for row-major
  std::cout << "Task 2 - Extract Column 5 as a 1D Tensor:" << std::endl;

  // TODO: Create a 1D layout for 8 elements with stride 8 (column in row-major)
  // Hint: make_layout(make_shape(Int<8>{}), make_stride(Int<8>{}))
  // START YOUR CODE HERE
  auto col_layout = make_layout(make_shape(Int<8>{}), make_stride(Int<8>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view using data + offset for column 5
  // Hint: Offset = layout(0, 5) gives the starting position of column 5
  // START YOUR CODE HERE
  auto offset_col5 = layout(0, 5);
  auto col5_slice = make_tensor(make_gmem_ptr(data + offset_col5), col_layout);

  // END YOUR CODE HERE

  std::cout << "Column 5 slice:" << std::endl;
  // TODO: Print your column slice tensor
  // START YOUR CODE HERE
  print(col5_slice);

  // END YOUR CODE HERE
  std::cout << "Expected: 5 13 21 29 37 45 53 61" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 3: Extract a sub-matrix (top-left 4x4)
  // ========================================================================
  // Goal: Create a 4x4 tensor view of the top-left region
  // Hint: Same pointer (no offset), shape (4,4), stride (8,1) - parent stride
  std::cout << "Task 3 - Extract Top-Left 4x4 Sub-Matrix:" << std::endl;

  // TODO: Create a 4x4 layout with the same stride as parent (8,1)
  // Hint: make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<8>{},
  // Int<1>{})) START YOUR CODE HERE
  auto top_left_layout = make_layout(make_shape(Int<4>{}, Int<4>{}),
                                     make_stride(Int<8>{}, Int<1>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view using the same data pointer (offset = 0)
  // START YOUR CODE HERE
  auto top_left = make_tensor(make_gmem_ptr(data), top_left_layout);

  // END YOUR CODE HERE

  std::cout << "Top-left 4x4 sub-matrix:" << std::endl;
  // TODO: Print your sub-matrix tensor
  // START YOUR CODE HERE
  print(top_left);

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // TASK 4: Extract a sub-matrix (bottom-right 4x4)
  // ========================================================================
  // Goal: Create a 4x4 tensor view of the bottom-right region
  // Hint: Pointer offset = tensor.layout()(4, 4), shape (4,4), stride (8,1)
  std::cout << "Task 4 - Extract Bottom-Right 4x4 Sub-Matrix:" << std::endl;

  // TODO: Create a 4x4 layout with stride (8, 1)
  // START YOUR CODE HERE
  auto br_layout = make_layout(make_shape(Int<4>{}, Int<4>{}),
                               make_stride(Int<8>{}, Int<1>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view with offset = layout(4, 4)
  // START YOUR CODE HERE
  auto offset_br = layout(4, 4);
  auto bottom_right = make_tensor(make_gmem_ptr(data + offset_br), br_layout);

  // END YOUR CODE HERE

  std::cout << "Bottom-right 4x4 sub-matrix:" << std::endl;
  // TODO: Print your sub-matrix tensor
  // START YOUR CODE HERE
  print(bottom_right);

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // TASK 5: Extract a strided slice (every other row)
  // ========================================================================
  // Goal: Create a view that accesses rows 0, 2, 4, 6 (every other row)
  // Hint: Shape (4, 8), stride (16, 1) - row stride is doubled
  std::cout << "Task 5 - Extract Every Other Row:" << std::endl;

  // TODO: Create a layout with shape (4, 8) and stride (16, 1)
  // This gives you 4 rows, each with 8 elements, but row stride is 16
  // START YOUR CODE HERE
  auto every_other_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                        make_stride(Int<16>{}, Int<1>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view using the same data pointer
  // START YOUR CODE HERE
  auto every_other = make_tensor(make_gmem_ptr(data), every_other_layout);

  // END YOUR CODE HERE

  std::cout << "Every other row view (rows 0, 2, 4, 6):" << std::endl;
  // TODO: Print your strided tensor
  // START YOUR CODE HERE
  print(every_other);

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // TASK 6: Verify that slices are views (not copies)
  // ========================================================================
  // Goal: Modify an element through the slice and verify it changes the
  // original
  std::cout << "Task 6 - Verify Slices are Views (Not Copies):" << std::endl;

  // TODO: Modify element (0, 2) of your row3_slice to 999
  // START YOUR CODE HERE
  row3_slice(2) = 999.0f;

  // END YOUR CODE HERE

  // TODO: Verify by printing tensor(3, 2) from the original tensor
  std::cout << "Original tensor(3, 2) after modifying slice: ";
  // START YOUR CODE HERE
  std::cout << tensor(3, 2);

  // END YOUR CODE HERE
  std::cout << " (should be 999)" << std::endl;

  // Reset the value
  tensor(3, 2) = 26.0f;
  std::cout << std::endl;

  // ========================================================================
  // TASK 7: Extract a center 4x4 block (rows 2-5, cols 2-5)
  // ========================================================================
  // Goal: Create a view of the center region
  // Hint: Offset = layout(2, 2), shape (4, 4), stride (8, 1)
  std::cout << "Task 7 - Extract Center 4x4 Block:" << std::endl;

  // TODO: Create a 4x4 layout with stride (8, 1)
  // START YOUR CODE HERE
  auto center_layout = make_layout(make_shape(Int<4>{}, Int<4>{}),
                                   make_stride(Int<8>{}, Int<1>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor with offset = layout(2, 2)
  // START YOUR CODE HERE
  auto offset_center = layout(2, 2);
  auto center = make_tensor(make_gmem_ptr(data + offset_center), center_layout);

  // END YOUR CODE HERE

  std::cout << "Center 4x4 block (rows 2-5, cols 2-5):" << std::endl;
  // TODO: Print your center block tensor
  // START YOUR CODE HERE
  print(center);

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // CHALLENGE: Extract the main diagonal as a 1D strided view
  // ========================================================================
  // Goal: Create a 1D tensor that views the diagonal elements
  // Hint: Shape (8,), stride (9,) - because diagonal has stride rows+1
  std::cout << "=== Challenge: Extract Diagonal as 1D View ===" << std::endl;

  // TODO: Create a 1D layout with 8 elements and stride 9
  // START YOUR CODE HERE
  auto diag_layout = make_layout(make_shape(Int<8>{}), make_stride(Int<9>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view starting at data (diagonal starts at 0)
  // START YOUR CODE HERE
  auto diag = make_tensor(make_gmem_ptr(data), diag_layout);

  // END YOUR CODE HERE

  std::cout << "Diagonal view:" << std::endl;
  // TODO: Print your diagonal tensor
  // START YOUR CODE HERE
  print(diag);

  // END YOUR CODE HERE
  std::cout << "Expected: 0 9 18 27 36 45 54 63" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // CHALLENGE 2: Create a transposed view of a 4x4 sub-region
  // ========================================================================
  // Goal: View the top-left 4x4 as if it were transposed
  // Hint: Shape (4, 4), stride (1, 8) - swap the strides
  std::cout << "=== Challenge 2: Transposed 4x4 View ===" << std::endl;

  // TODO: Create a layout with shape (4, 4) and stride (1, 8)
  // This accesses columns as if they were rows (transpose)
  // START YOUR CODE HERE
  auto trans_layout = make_layout(make_shape(Int<4>{}, Int<4>{}),
                                  make_stride(Int<1>{}, Int<8>{}));

  // END YOUR CODE HERE

  // TODO: Create a tensor view using the same data pointer
  // START YOUR CODE HERE
  auto transposed = make_tensor(make_gmem_ptr(data), trans_layout);

  // END YOUR CODE HERE

  std::cout << "Transposed 4x4 view:" << std::endl;
  // TODO: Print your transposed tensor
  // START YOUR CODE HERE
  print(transposed);

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. How do you create a row slice? What's the stride?"
            << std::endl;
  std::cout << "2. How do you create a column slice? What's the stride?"
            << std::endl;
  std::cout << "3. Why are slices called 'views' and not 'copies'?"
            << std::endl;
  std::cout << "4. How do you calculate the offset for a sub-matrix?"
            << std::endl;

  return 0;
}
