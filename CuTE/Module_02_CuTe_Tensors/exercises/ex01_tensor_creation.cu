/**
 * Exercise 01: Tensor Creation from Raw Pointers
 *
 * Objective: Learn to create CuTe tensors by wrapping raw pointers with layouts
 *
 * Instructions:
 * - Complete each TODO section by writing the CuTe code yourself
 * - Do NOT look at solutions - derive them from the hints provided
 * - Compile and run after each task to verify your implementation
 * - Study the printed output to understand layout-to-memory mapping
 *
 * Key Functions to discover/use:
 * - make_tensor(ptr, layout) - Creates a tensor from pointer and layout
 * - make_gmem_ptr(ptr) - Wraps a global memory pointer
 * - make_shape(...) - Creates a shape from dimensions
 * - make_stride(...) - Creates a stride from values
 * - GenRowMajor{}, GenColMajor{} - Layout generators
 * - Int<N>{} - Compile-time integer constant
 * - print(tensor), print_layout(layout) - Debug printing
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 01: Tensor Creation from Raw Pointers ==="
            << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // TASK 1: Create a 1D tensor from a raw pointer
  // ========================================================================
  // Given: A raw array of 16 floats initialized with values 0..15
  // Goal: Wrap it with a CuTe layout and create a tensor
  // Hint: Use make_shape(Int<16>{}) and GenRowMajor{}
  std::cout << "Task 1 - Create a 1D Tensor:" << std::endl;
  
  float data_1d[16];
  for (int i = 0; i < 16; ++i) {
    data_1d[i] = static_cast<float>(i);
  }

  // TODO: Create a 1D layout for 16 elements
  // TODO: Create a tensor by wrapping data_1d with your layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Your 1D tensor:" << std::endl;
  // TODO: Print your tensor using print()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Create a 2D row-major tensor (4x4 matrix)
  // ========================================================================
  // Given: A raw array of 16 floats
  // Goal: Create a 4x4 row-major tensor
  // Hint: make_shape(Int<4>{}, Int<4>{}) with GenRowMajor{}
  std::cout << "Task 2 - Create a 2D Row-Major Tensor (4x4):" << std::endl;
  
  float data_2d[16];
  for (int i = 0; i < 16; ++i) {
    data_2d[i] = static_cast<float>(i * 10);
  }

  // TODO: Create a 4x4 row-major layout
  // TODO: Create a tensor wrapping data_2d
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Your 2D row-major tensor:" << std::endl;
  // TODO: Print your tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Access specific elements from your 2D tensor
  // ========================================================================
  // Goal: Use tensor(row, col) syntax to access elements
  // Verify: Check that your accesses match expected values
  std::cout << "Task 3 - Access Tensor Elements:" << std::endl;
  std::cout << "Access the following positions and print the values:" << std::endl;
  std::cout << "  Position (0, 0): ";
  // TODO: Print tensor_2d(0, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 0)" << std::endl;

  std::cout << "  Position (1, 2): ";
  // TODO: Print tensor_2d(1, 2)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 12)" << std::endl;

  std::cout << "  Position (3, 3): ";
  // TODO: Print tensor_2d(3, 3)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 30)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Create a column-major tensor (4x3 matrix)
  // ========================================================================
  // Given: A raw array of 12 floats
  // Goal: Create a 4x3 column-major tensor
  // Hint: GenColMajor{} creates column-major layout
  std::cout << "Task 4 - Create a Column-Major Tensor (4x3):" << std::endl;
  
  float data_cm[12];
  for (int i = 0; i < 12; ++i) {
    data_cm[i] = static_cast<float>(i + 100);
  }

  // TODO: Create a 4x3 column-major layout
  // TODO: Create a tensor wrapping data_cm
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Your column-major tensor:" << std::endl;
  // TODO: Print your tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Access and print element at position (1, 2)
  // Hint: In column-major, element (1,2) has offset = 1 + 2*4 = 9
  std::cout << "Element at (1, 2): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 109)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Create a tensor with custom stride (padded layout)
  // ========================================================================
  // Given: A raw array of 36 floats (enough for 4x9)
  // Goal: Create a 4x8 tensor with stride=9 (1 element padding per row)
  // Hint: make_stride(Int<9>{}, Int<1>{}) for row stride=9, col stride=1
  std::cout << "Task 5 - Create a Padded Tensor (4x8 with stride=9):" << std::endl;
  
  float data_padded[36];
  for (int i = 0; i < 36; ++i) {
    data_padded[i] = static_cast<float>(i);
  }

  // TODO: Create a layout with shape (4, 8) and stride (9, 1)
  // TODO: Create a tensor wrapping data_padded
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Your padded tensor:" << std::endl;
  // TODO: Print your tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Access element at (2, 5) - what value do you get?
  // Hint: offset = 2*9 + 5*1 = 23
  std::cout << "Element at (2, 5): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 23)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Verify tensor properties (shape and stride)
  // ========================================================================
  // Goal: Extract and print the shape and stride from your tensor's layout
  // Hint: tensor.layout().shape() and tensor.layout().stride()
  std::cout << "Task 6 - Verify Tensor Properties:" << std::endl;
  std::cout << "For the 4x4 row-major tensor from Task 2:" << std::endl;
  
  // TODO: Print the shape of tensor_2d
  std::cout << "  Shape: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: (4, 4))" << std::endl;

  // TODO: Print the stride of tensor_2d
  std::cout << "  Stride: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: (4, 1))" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Create a 3D tensor (2x3x4)
  // ========================================================================
  // Given: A raw array of 24 floats
  // Goal: Create a 3D tensor with shape (2, 3, 4) in row-major order
  // Hint: make_shape(Int<2>{}, Int<3>{}, Int<4>{}) with GenRowMajor{}
  std::cout << "=== Challenge: Create a 3D Tensor (2x3x4) ===" << std::endl;
  
  float data_3d[24];
  for (int i = 0; i < 24; ++i) {
    data_3d[i] = static_cast<float>(i);
  }

  // TODO: Create a 3D row-major layout with shape (2, 3, 4)
  // TODO: Create a tensor wrapping data_3d
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Your 3D tensor:" << std::endl;
  // TODO: Print your 3D tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Access element at position (0, 1, 2)
  // Hint: In row-major (2,3,4), offset = 0*12 + 1*4 + 2 = 6
  std::cout << "Element at (0, 1, 2): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 6)" << std::endl;

  // TODO: Print the layout of your 3D tensor to see its structure
  std::cout << "3D tensor layout:" << std::endl;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Reflect on what you learned:" << std::endl;
  std::cout << "1. How does make_tensor() wrap a raw pointer?" << std::endl;
  std::cout << "2. What is the relationship between shape, stride, and memory offset?" << std::endl;
  std::cout << "3. How does row-major differ from column-major in memory?" << std::endl;
  std::cout << "4. Why would you use a padded/padded layout?" << std::endl;

  return 0;
}
