/**
 * Exercise 02: Tensor Access Patterns
 *
 * Objective: Understand how different access patterns affect performance
 *            and learn to write coalesced access patterns
 *
 * Instructions:
 * - Complete each TODO section by writing the CuTe code yourself
 * - Observe the memory offsets printed to understand coalescing
 * - Think about how GPU threads would access memory in each pattern
 *
 * Key Concepts:
 * - Coalesced Access: Consecutive elements accessed are consecutive in memory
 * - Uncoalesced Access: Strided or scattered memory access
 * - Row-major: Row access is coalesced (stride=1 in column dimension)
 * - Column-major: Column access is coalesced (stride=1 in row dimension)
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 02: Tensor Access Patterns ===" << std::endl;
  std::cout << std::endl;

  // Create an 8x8 tensor for access pattern experiments
  float data[64];
  for (int i = 0; i < 64; ++i) {
    data[i] = static_cast<float>(i);
  }

  // Create row-major tensor
  auto layout_rm = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
  auto tensor_rm = make_tensor(make_gmem_ptr(data), layout_rm);

  std::cout << "Row-Major 8x8 Tensor Layout:" << std::endl;
  print(tensor_rm.layout());
  std::cout << std::endl;

  // ========================================================================
  // TASK 1: Row-wise access (coalesced for row-major)
  // ========================================================================
  // Goal: Access all elements in row 0 sequentially
  // Observe: Memory offsets should be consecutive (0, 1, 2, 3, ...)
  std::cout << "Task 1 - Row-wise Access (COALESCED):" << std::endl;
  std::cout << "Access all 8 elements in row 0:" << std::endl;

  // TODO: Write a loop to access tensor_rm(0, j) for j = 0 to 7
  // TODO: Print each value
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_rm(0, i);
  }
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Write a loop to print the memory offsets using tensor_rm.layout()(0,
  // j)
  std::cout << "Memory offsets: ";
  // START YOUR CODE HERE
  for (int j = 0; j <= 7; ++j) {
    std::cout << tensor_rm.layout()(0, j);
    ;
  }

  // END YOUR CODE HERE
  std::cout << " <- Should be consecutive (0,1,2,3,4,5,6,7)" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 2: Column-wise access (uncoalesced for row-major)
  // ========================================================================
  // Goal: Access all elements in column 0 sequentially
  // Observe: Memory offsets have stride of 8 (0, 8, 16, 24, ...)
  std::cout << "Task 2 - Column-wise Access (UNCOALESCED):" << std::endl;
  std::cout << "Access all 8 elements in column 0:" << std::endl;

  // TODO: Write a loop to access tensor_rm(i, 0) for i = 0 to 7
  // TODO: Print each value
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_rm(i, 0);
  }

  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Write a loop to print the memory offsets using tensor_rm.layout()(i,
  // 0)
  std::cout << "Memory offsets: ";
  // START YOUR CODE HERE
  for (int j = 0; j <= 7; ++j) {
    std::cout << tensor_rm.layout()(j, 0);
    ;
  }

  // END YOUR CODE HERE
  std::cout << " <- Stride of 8 (not consecutive)" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 3: Create a column-major tensor for comparison
  // ========================================================================
  // Goal: Create the same 8x8 data but with column-major layout
  // Hint: Use GenColMajor{} instead of GenRowMajor{}
  std::cout << "Task 3 - Create Column-Major Tensor:" << std::endl;

  // TODO: Create an 8x8 column-major layout
  // TODO: Create a tensor wrapping the same data array
  // START YOUR CODE HERE
  auto layout_cm = make_layout(make_shape(8, 8), GenColMajor{});
  auto tensor_cm = make_tensor(make_gmem_ptr(data), layout_cm);

  // END YOUR CODE HERE

  std::cout << "Column-Major 8x8 Tensor Layout:" << std::endl;
  // TODO: Print your column-major tensor's layout
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_cm(0, i);
  }
  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // TASK 4: Compare access patterns between row-major and column-major
  // ========================================================================
  // Goal: Access the same positions in both tensors and compare offsets
  std::cout << "Task 4 - Compare Access Patterns:" << std::endl;
  std::cout << std::endl;

  // For row-major tensor: row access should be coalesced
  std::cout << "Row-Major Tensor - Row 0 access offsets:";
  // TODO: Print offsets for tensor_rm(0, j) for j = 0..3
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_rm(0, i) << std::endl;
  }

  // END YOUR CODE HERE
  std::cout << std::endl;

  // For row-major tensor: column access is uncoalesced
  std::cout << "Row-Major Tensor - Column 0 access offsets:";
  // TODO: Print offsets for tensor_rm(i, 0) for i = 0..3
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_rm(i, 0) << std::endl;
  }

  // END YOUR CODE HERE
  std::cout << std::endl;

  // For column-major tensor: row access is uncoalesced
  std::cout << "Column-Major Tensor - Row 0 access offsets:";
  // TODO: Print offsets for tensor_cm(0, j) for j = 0..3
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_cm(0, i) << std::endl;
  }

  // END YOUR CODE HERE
  std::cout << std::endl;

  // For column-major tensor: column access is coalesced
  std::cout << "Column-Major Tensor - Column 0 access offsets:";
  // TODO: Print offsets for tensor_cm(i, 0) for i = 0..3
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_cm(i, 0) << std::endl;
  }

  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 5: Diagonal access pattern
  // ========================================================================
  // Goal: Access diagonal elements (0,0), (1,1), (2,2), ...
  // Observe: Stride is (rows + 1) = 9 in row-major layout
  std::cout << "Task 5 - Diagonal Access:" << std::endl;
  std::cout << "Access diagonal elements:" << std::endl;

  // TODO: Write a loop to access tensor_rm(i, i) for i = 0 to 7
  // TODO: Print each value
  // START YOUR CODE HERE
  for (int i = 0; i <= 7; ++i) {
    std::cout << tensor_rm(i, i) << std::endl;
  }

  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Write a loop to print the memory offsets
  std::cout << "Memory offsets: ";
  // START YOUR CODE HERE
  for (int j = 0; j <= 7; ++j) {
    std::cout << tensor_rm.layout()(j, j);
    ;
  }
  // END YOUR CODE HERE
  std::cout << " <- Stride of 9 (row stride + col stride)" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 6: Tiled access pattern (2x2 tiles)
  // ========================================================================
  // Goal: Access elements in 2x2 tile order
  // This is important for tiled matrix multiplication
  std::cout << "Task 6 - Tiled Access (2x2 tiles):" << std::endl;
  std::cout << "Access the top-left 2x2 tile:" << std::endl;

  // TODO: Write nested loops to access tensor_rm(i, j) for i,j in {0,1}
  // TODO: Print each value in row order
  // START YOUR CODE HERE
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      std::cout << tensor_rm(i, j);
    }
  }
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Print the memory offsets for the 2x2 tile
  std::cout << "Memory offsets for 2x2 tile:" << std::endl;
  // START YOUR CODE HERE
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      std::cout << tensor_rm.layout()(i, j);
    }
  }

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // TASK 7: Compute memory offset manually and verify
  // ========================================================================
  // Goal: Calculate offset for position (3, 5) using shape and stride
  // Formula: offset = row * stride_row + col * stride_col
  std::cout << "Task 7 - Manual Offset Calculation:" << std::endl;
  std::cout << "For position (3, 5) in 8x8 row-major tensor:" << std::endl;

  // TODO: Get the stride from the layout
  // Hint: auto stride = tensor_rm.layout().stride();
  // START YOUR CODE HERE

  // END YOUR CODE HERE

  // TODO: Calculate offset manually: 3 * stride[0] + 5 * stride[1]
  int manual_offset = 0; // TODO: Calculate this
  // START YOUR CODE HERE

  // END YOUR CODE HERE

  // TODO: Get offset from layout using tensor_rm.layout()(3, 5)
  int layout_offset = 0; // TODO: Get this from layout
  // START YOUR CODE HERE

  // END YOUR CODE HERE

  std::cout << "Manual calculation: " << manual_offset << std::endl;
  std::cout << "From layout: " << layout_offset << std::endl;
  std::cout << "Match: " << (manual_offset == layout_offset ? "YES" : "NO")
            << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // CHALLENGE: Implement a strided access pattern
  // ========================================================================
  // Goal: Access every 3rd element in row 0: positions (0,0), (0,3), (0,6)
  std::cout << "=== Challenge: Strided Access ===" << std::endl;
  std::cout << "Access every 3rd element in row 0:" << std::endl;

  // TODO: Write a loop with stride 3 to access elements
  // TODO: Print values and their memory offsets
  // START YOUR CODE HERE

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // CHALLENGE 2: Access pattern for matrix transpose
  // ========================================================================
  // Goal: Read from row-major tensor in column order (simulating transpose
  // read)
  std::cout << "=== Challenge 2: Transpose Read Pattern ===" << std::endl;
  std::cout << "Read 4x4 submatrix in column-major order (for transpose):"
            << std::endl;

  // TODO: Write nested loops where OUTER loop is columns, INNER loop is rows
  // This is the pattern you'd use to read data for a transpose operation
  // START YOUR CODE HERE

  // END YOUR CODE HERE
  std::cout << std::endl;

  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Insights:" << std::endl;
  std::cout << "1. Which access pattern is coalesced for row-major tensors?"
            << std::endl;
  std::cout << "2. Which access pattern is coalesced for column-major tensors?"
            << std::endl;
  std::cout << "3. Why does coalesced access matter for GPU performance?"
            << std::endl;
  std::cout << "4. What is the stride for diagonal access in row-major layout?"
            << std::endl;

  return 0;
}
