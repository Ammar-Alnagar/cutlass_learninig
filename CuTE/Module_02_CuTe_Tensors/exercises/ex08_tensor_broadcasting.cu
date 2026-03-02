/**
 * Exercise 08: Tensor Broadcasting
 *
 * Objective: Learn to broadcast tensors for operations like bias addition
 *            and matrix-vector multiplication
 *
 * Instructions:
 * - Complete each TODO section by implementing broadcast operations
 * - Understand how stride=0 creates true broadcast layouts
 * - Apply broadcasting to common operations (bias addition, etc.)
 *
 * Key Concepts:
 * - Broadcasting: Using same value for multiple output positions
 * - Dimension of Size 1: Can be "stretched" to match other sizes
 * - Stride 0: All indices map to the same memory location (true broadcast)
 * - Efficient: No data duplication, just layout manipulation
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 08: Tensor Broadcasting ===" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 1: Broadcasting a scalar to a vector
  // ========================================================================
  // Goal: Create a scalar tensor and understand how to broadcast it
  std::cout << "Task 1 - Scalar to Vector Broadcasting:" << std::endl;
  
  float scalar_data[1] = {5.0f};
  auto scalar_layout = make_layout(make_shape(Int<1>{}), GenRowMajor{});
  auto scalar_tensor = make_tensor(make_gmem_ptr(scalar_data), scalar_layout);

  std::cout << "Scalar tensor value: " << scalar_tensor(0) << std::endl;
  std::cout << std::endl;

  std::cout << "Broadcast to 8 elements (conceptually - same value for all):" << std::endl;
  // TODO: Write a loop that prints scalar_tensor(0) 8 times
  // This simulates broadcasting the scalar to 8 positions
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Now create a TRUE broadcast layout using stride 0
  std::cout << "True broadcast layout (stride = 0):" << std::endl;
  // TODO: Create a layout with shape (8,) and stride (0,)
  // Hint: make_layout(make_shape(Int<8>{}), make_stride(Int<0>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the broadcast layout and scalar_data pointer
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Broadcast layout: ";
  // TODO: Print the broadcast layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Accessing broadcast tensor at different indices:" << std::endl;
  std::cout << "  broadcast(0) = " << broadcast_tensor(0) << std::endl;
  std::cout << "  broadcast(3) = " << broadcast_tensor(3) << std::endl;
  std::cout << "  broadcast(7) = " << broadcast_tensor(7) << std::endl;
  std::cout << "  All access the same memory location (offset 0)!" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Broadcasting a vector to a matrix (row-wise)
  // ========================================================================
  // Goal: Create a row vector and broadcast it across all rows
  std::cout << "Task 2 - Vector to Matrix Broadcasting (Row-wise):" << std::endl;
  
  float row_vector[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto vector_layout = make_layout(make_shape(Int<4>{}), GenRowMajor{});
  auto vector_tensor = make_tensor(make_gmem_ptr(row_vector), vector_layout);

  std::cout << "Original vector: ";
  for (int j = 0; j < 4; ++j) {
    std::cout << vector_tensor(j) << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Broadcast to 4x4 matrix (same row repeated):" << std::endl;
  // TODO: Create a 2D broadcast layout with shape (4, 4) and stride (0, 1)
  // This makes all rows access the same data (the original vector)
  // Hint: make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<0>{}, Int<1>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the broadcast layout and row_vector pointer
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Broadcast matrix layout: ";
  // TODO: Print the broadcast layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Broadcast matrix values:" << std::endl;
  // TODO: Print the 4x4 broadcast matrix using nested loops
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "Notice: All rows are identical (broadcasting!)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Broadcasting a vector to a matrix (column-wise)
  // ========================================================================
  // Goal: Create a column vector and broadcast it across all columns
  std::cout << "Task 3 - Vector to Matrix Broadcasting (Column-wise):" << std::endl;
  
  float col_vector[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  auto col_vector_layout = make_layout(make_shape(Int<4>{}), GenRowMajor{});
  auto col_vector_tensor = make_tensor(make_gmem_ptr(col_vector), col_vector_layout);

  std::cout << "Original column vector: ";
  for (int i = 0; i < 4; ++i) {
    std::cout << col_vector_tensor(i) << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Broadcast to 4x4 matrix (same column repeated):" << std::endl;
  // TODO: Create a 2D broadcast layout with shape (4, 4) and stride (1, 0)
  // This makes all columns access the same data (the original column vector)
  // Hint: make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<1>{}, Int<0>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the broadcast layout and col_vector pointer
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Broadcast matrix layout: ";
  // TODO: Print the broadcast layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Broadcast matrix values:" << std::endl;
  // TODO: Print the 4x4 broadcast matrix using nested loops
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "Notice: All columns are identical (broadcasting!)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Matrix + Vector (Bias Addition)
  // ========================================================================
  // Goal: Add a bias vector to each row of a matrix using broadcasting
  std::cout << "Task 4 - Matrix + Vector (Bias Addition):" << std::endl;
  
  float matrix_data[12];
  for (int i = 0; i < 12; ++i) {
    matrix_data[i] = static_cast<float>(i);
  }
  auto matrix_layout = make_layout(make_shape(Int<3>{}, Int<4>{}), GenRowMajor{});
  auto matrix_tensor = make_tensor(make_gmem_ptr(matrix_data), matrix_layout);

  float bias[4] = {100.0f, 200.0f, 300.0f, 400.0f};
  auto bias_layout = make_layout(make_shape(Int<4>{}), GenRowMajor{});
  auto bias_tensor = make_tensor(make_gmem_ptr(bias), bias_layout);

  std::cout << "Matrix (3x4):" << std::endl;
  // TODO: Print the 3x4 matrix
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Bias vector: ";
  // TODO: Print the bias vector
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Matrix + Bias (broadcasted):" << std::endl;
  // TODO: Add bias to each row of the matrix
  // Use nested loops: for each element matrix(i,j), add bias(j)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Verify broadcast layout properties
  // ========================================================================
  // Goal: Understand how stride 0 affects memory offsets
  std::cout << "Task 5 - Verify Broadcast Layout Properties:" << std::endl;
  
  // Create a broadcast layout for a 4x4 matrix broadcasting a row vector
  auto bcast_layout = make_layout(
    make_shape(Int<4>{}, Int<4>{}),
    make_stride(Int<0>{}, Int<1>{})
  );

  std::cout << "Broadcast layout (4,4) with stride (0,1):" << std::endl;
  print(bcast_layout);
  std::cout << std::endl;

  std::cout << "Memory offsets for each position:" << std::endl;
  // TODO: Print the offset for each position (i, j) using bcast_layout(i, j)
  // Notice that all positions in the same column have the same offset
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Column 0 offsets: ";
  // TODO: Print bcast_layout(0,0), bcast_layout(1,0), bcast_layout(2,0), bcast_layout(3,0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << " (all should be 0)" << std::endl;

  std::cout << "Column 1 offsets: ";
  // TODO: Print bcast_layout(0,1), bcast_layout(1,1), bcast_layout(2,1), bcast_layout(3,1)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << " (all should be 1)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Broadcasting in batched operations
  // ========================================================================
  // Goal: Understand broadcasting for batched neural network operations
  std::cout << "Task 6 - Batched Broadcasting:" << std::endl;
  std::cout << "Scenario: Batch of 4 samples, each with 3 features" << std::endl;
  std::cout << "          Bias has 3 elements (one per feature)" << std::endl;
  std::cout << std::endl;

  float batch_data[12];
  for (int i = 0; i < 12; ++i) {
    batch_data[i] = static_cast<float>(i);
  }
  // Shape (4, 3) = (batch, features)
  auto batch_layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
  auto batch_tensor = make_tensor(make_gmem_ptr(batch_data), batch_layout);

  float feature_bias[3] = {1000.0f, 2000.0f, 3000.0f};
  auto feature_bias_layout = make_layout(make_shape(Int<3>{}), GenRowMajor{});
  auto feature_bias_tensor = make_tensor(make_gmem_ptr(feature_bias), feature_bias_layout);

  std::cout << "Batch data (4 samples x 3 features):" << std::endl;
  // TODO: Print the 4x3 batch tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Feature bias: ";
  // TODO: Print the bias tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Batch + Bias (bias broadcasted to all 4 samples):" << std::endl;
  // TODO: Add bias to each sample in the batch
  // For each element batch(i, j), add feature_bias(j)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Create a 2D broadcast from a scalar
  // ========================================================================
  std::cout << "=== Challenge: 2D Broadcast from Scalar ===" << std::endl;
  std::cout << "Create a 4x4 matrix where all elements are the same scalar value" << std::endl;
  
  float value[1] = {42.0f};
  
  // TODO: Create a 4x4 layout with stride (0, 0) - both dimensions broadcast
  // This makes ALL positions access the same scalar
  // Hint: make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<0>{}, Int<0>{}))
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor using the layout and value pointer
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "2D broadcast from scalar:" << std::endl;
  // TODO: Print the 4x4 broadcast tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "All 16 elements are the same value (42)!" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE 2: Outer product using broadcasting
  // ========================================================================
  std::cout << "=== Challenge 2: Outer Product Pattern ===" << std::endl;
  std::cout << "Compute outer product of two vectors using broadcasting concept" << std::endl;
  std::cout << "Given: vector A = [1, 2, 3], vector B = [10, 20, 30, 40]" << std::endl;
  std::cout << "Result: 3x4 matrix where result(i,j) = A[i] * B[j]" << std::endl;
  std::cout << std::endl;

  float vec_a[3] = {1.0f, 2.0f, 3.0f};
  float vec_b[4] = {10.0f, 20.0f, 30.0f, 40.0f};

  std::cout << "Outer product result:" << std::endl;
  // TODO: Compute and print the 3x4 outer product
  // For each position (i, j), multiply vec_a[i] * vec_b[j]
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. How does stride=0 create broadcasting?" << std::endl;
  std::cout << "2. What layout do you need to broadcast a row vector to a matrix?" << std::endl;
  std::cout << "3. What layout do you need to broadcast a column vector to a matrix?" << std::endl;
  std::cout << "4. Why is broadcasting efficient (no data copying)?" << std::endl;

  return 0;
}
