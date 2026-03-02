/**
 * Exercise 04: Tensor Transpose and View Operations
 *
 * Objective: Learn to create transposed views of tensors without copying data
 *
 * Instructions:
 * - Complete each TODO section by implementing transpose operations
 * - Understand that transpose is a VIEW operation (zero-copy)
 * - Verify transpose relationships: original(i,j) == transposed(j,i)
 *
 * Key Concepts:
 * - Transpose swaps dimensions: (M,N) becomes (N,M)
 * - Transpose swaps strides: row stride <-> col stride
 * - No data is copied - only the view changes
 * - Double transpose returns to original layout
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 04: Tensor Transpose and View ===" << std::endl;
  std::cout << std::endl;

  // Create a 4x6 tensor
  float data[24];
  for (int i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto layout_orig = make_layout(make_shape(Int<4>{}, Int<6>{}), GenRowMajor{});
  auto tensor_orig = make_tensor(make_gmem_ptr(data), layout_orig);

  std::cout << "Original Tensor (4x6):" << std::endl;
  print(tensor_orig);
  std::cout << std::endl;


  // ========================================================================
  // TASK 1: Create a transposed layout (6x4)
  // ========================================================================
  // Goal: Create a layout that represents the transpose of the original
  // Hint: Transpose swaps shape (4,6) -> (6,4) and strides
  std::cout << "Task 1 - Create Transposed Layout:" << std::endl;
  
  // TODO: Create a 6x4 column-major layout (this is equivalent to transposing 4x6 row-major)
  // Hint: make_layout(make_shape(Int<6>{}, Int<4>{}), GenColMajor{})
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Transposed layout:" << std::endl;
  // TODO: Print your transposed layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Create a transposed tensor view
  // ========================================================================
  // Goal: Create a tensor using the transposed layout that views the same data
  // Hint: Use the same data pointer, but with transposed layout
  std::cout << "Task 2 - Create Transposed Tensor View:" << std::endl;
  
  // TODO: Create a tensor using make_tensor() with the same data and your transposed layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Transposed view (6x4):" << std::endl;
  // TODO: Print your transposed tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Verify transpose relationship
  // ========================================================================
  // Goal: Verify that original(i,j) == transposed(j,i)
  std::cout << "Task 3 - Verify Transpose Relationship:" << std::endl;
  std::cout << "Verify: original(i,j) should equal transposed(j,i)" << std::endl;
  std::cout << std::endl;

  // Test position (0, 1) in original
  std::cout << "Position (0, 1):" << std::endl;
  std::cout << "  original(0, 1) = ";
  // TODO: Print tensor_orig(0, 1)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "  transposed(1, 0) = ";
  // TODO: Print tensor_transposed(1, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (should match above)" << std::endl;
  std::cout << std::endl;

  // Test position (2, 3) in original
  std::cout << "Position (2, 3):" << std::endl;
  std::cout << "  original(2, 3) = ";
  // TODO: Print tensor_orig(2, 3)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "  transposed(3, 2) = ";
  // TODO: Print tensor_transposed(3, 2)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (should match above)" << std::endl;
  std::cout << std::endl;

  // Test position (3, 5) in original
  std::cout << "Position (3, 5):" << std::endl;
  std::cout << "  original(3, 5) = ";
  // TODO: Print tensor_orig(3, 5)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "  transposed(5, 3) = ";
  // TODO: Print tensor_transposed(5, 3)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (should match above)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Double transpose (should return to original)
  // ========================================================================
  // Goal: Create a double-transposed view and verify it matches original
  std::cout << "Task 4 - Double Transpose:" << std::endl;
  
  // TODO: Create a 4x6 row-major layout (transpose of the 6x4 column-major)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor with the double-transposed layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Double transposed tensor:" << std::endl;
  // TODO: Print your double-transposed tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "Should match the original 4x6 tensor above" << std::endl;
  std::cout << std::endl;

  // Verify a few positions
  std::cout << "Verification:" << std::endl;
  std::cout << "  original(1, 2) = " << tensor_orig(1, 2) << std::endl;
  std::cout << "  double_transposed(1, 2) = ";
  // TODO: Print double_transposed(1, 2)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Create a transposed view of a sub-region
  // ========================================================================
  // Goal: Extract top-left 3x4, then view it transposed
  std::cout << "Task 5 - Transposed Sub-Region View:" << std::endl;
  std::cout << "Extract top-left 3x4 region and view it transposed:" << std::endl;
  
  // Step 1: Create a 3x4 sub-layout (with parent stride)
  // TODO: Create layout with shape (3, 4) and stride (6, 1)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // Step 2: Create the sub-tensor (no offset, top-left corner)
  // TODO: Create tensor for the sub-region
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Original 3x4 sub-region:" << std::endl;
  // TODO: Print the sub-region tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Step 3: Create transposed view of the sub-region
  // TODO: Create a 4x3 column-major layout for the transpose
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create the transposed tensor view using the same data
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Transposed 4x3 view of sub-region:" << std::endl;
  // TODO: Print the transposed sub-region
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Understand stride transformation in transpose
  // ========================================================================
  // Goal: Extract and compare strides before and after transpose
  std::cout << "Task 6 - Stride Transformation in Transpose:" << std::endl;
  
  // TODO: Get and print the stride of the original tensor
  std::cout << "Original tensor stride: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Get and print the stride of the transposed tensor
  std::cout << "Transposed tensor stride: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Notice: Transpose swaps the stride values!" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Create a specific view - bottom-right 2x3 transposed
  // ========================================================================
  // Goal: Extract bottom-right 2x3 of original, then view it transposed
  std::cout << "=== Challenge: Transposed Bottom-Right 2x3 ===" << std::endl;
  
  // TODO: Create a 2x3 layout with stride (6, 1)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Calculate offset for bottom-right 2x3 (starts at row 2, col 3)
  // Hint: offset = tensor_orig.layout()(2, 3)
  // TODO: Create the tensor with appropriate offset
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Bottom-right 2x3 sub-region:" << std::endl;
  // TODO: Print the sub-region
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Create a 3x2 column-major layout for the transpose
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create the transposed view with the same offset
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Transposed 3x2 view:" << std::endl;
  // TODO: Print the transposed view
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. How does transpose affect shape? (M,N) -> ?" << std::endl;
  std::cout << "2. How does transpose affect strides?" << std::endl;
  std::cout << "3. What is the relationship between original(i,j) and transposed(?,?)?" << std::endl;
  std::cout << "4. Does transpose copy data? Why or why not?" << std::endl;

  return 0;
}
