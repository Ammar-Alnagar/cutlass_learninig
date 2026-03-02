/**
 * Exercise 06: Multi-dimensional Tensors
 *
 * Objective: Work with tensors beyond 2D - 3D, 4D, and higher dimensions
 *
 * Instructions:
 * - Complete each TODO section by creating and manipulating multi-D tensors
 * - Practice stride calculation for higher dimensions
 * - Understand common formats: NCHW (PyTorch), NHWC (TensorFlow)
 *
 * Key Concepts:
 * - 3D tensors: Common for volumes, multi-channel images
 * - 4D tensors: Batched images in deep learning
 * - Stride calculation: stride[i] = product of all smaller dimensions
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 06: Multi-dimensional Tensors ===" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 1: Create a 3D tensor (2x4x4)
  // ========================================================================
  // Goal: Create a 3D tensor representing 2 channels of 4x4 images
  std::cout << "Task 1 - Create a 3D Tensor (2x4x4):" << std::endl;
  
  float data_3d[32];
  for (int i = 0; i < 32; ++i) {
    data_3d[i] = static_cast<float>(i);
  }

  // TODO: Create a 3D row-major layout with shape (2, 4, 4)
  // Hint: make_layout(make_shape(Int<2>{}, Int<4>{}, Int<4>{}), GenRowMajor{})
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor wrapping data_3d
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "3D Layout:" << std::endl;
  // TODO: Print the layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Shape: ";
  // TODO: Print the shape using tensor_3d.layout().shape()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Stride: ";
  // TODO: Print the stride using tensor_3d.layout().stride()
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Access elements in the 3D tensor
  // ========================================================================
  // Goal: Access specific elements using (depth, row, col) coordinates
  std::cout << "Task 2 - Access 3D Tensor Elements:" << std::endl;
  
  // TODO: Access and print element at (0, 0, 0)
  std::cout << "Element at (0, 0, 0): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 0)" << std::endl;

  // TODO: Access and print element at (0, 2, 3)
  std::cout << "Element at (0, 2, 3): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 11)" << std::endl;

  // TODO: Access and print element at (1, 0, 0)
  std::cout << "Element at (1, 0, 0): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 16)" << std::endl;

  // TODO: Access and print element at (1, 3, 3)
  std::cout << "Element at (1, 3, 3): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 31)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Print each "depth slice" (channel) of the 3D tensor
  // ========================================================================
  // Goal: Visualize the 3D tensor as separate 2D slices
  std::cout << "Task 3 - Print Depth Slices:" << std::endl;
  
  std::cout << "Depth slice 0 (channel 0):" << std::endl;
  // TODO: Write nested loops to print tensor_3d(0, i, j) for i,j in 0..3
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Depth slice 1 (channel 1):" << std::endl;
  // TODO: Write nested loops to print tensor_3d(1, i, j) for i,j in 0..3
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Understand 3D stride calculation
  // ========================================================================
  // Goal: Manually calculate stride for 3D row-major layout
  std::cout << "Task 4 - 3D Stride Calculation:" << std::endl;
  std::cout << "For 3D layout (D, H, W) in row-major:" << std::endl;
  std::cout << "  stride_D = H * W" << std::endl;
  std::cout << "  stride_H = W" << std::endl;
  std::cout << "  stride_W = 1" << std::endl;
  std::cout << std::endl;

  // TODO: Calculate expected strides for shape (2, 4, 4)
  int expected_stride_d = 0;  // TODO: Calculate
  int expected_stride_h = 0;  // TODO: Calculate
  int expected_stride_w = 1;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Expected strides: (" << expected_stride_d << ", " 
            << expected_stride_h << ", " << expected_stride_w << ")" << std::endl;

  // TODO: Get actual strides from the layout and compare
  std::cout << "Actual strides:   ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Manual offset calculation in 3D
  // ========================================================================
  // Goal: Calculate offset for position (1, 2, 3) manually
  std::cout << "Task 5 - Manual 3D Offset Calculation:" << std::endl;
  std::cout << "Formula: offset = d * stride_D + h * stride_H + w * stride_W" << std::endl;
  std::cout << std::endl;

  int d = 1, h = 2, w = 3;

  // TODO: Calculate offset manually using the strides
  int manual_offset = 0;  // TODO: Calculate d * 16 + h * 4 + w * 1
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Get offset from layout using tensor_3d.layout()(d, h, w)
  int layout_offset = 0;  // TODO: Get from layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Position (1, 2, 3):" << std::endl;
  std::cout << "  Manual calculation: " << manual_offset << std::endl;
  std::cout << "  From layout: " << layout_offset << std::endl;
  std::cout << "  Match: " << (manual_offset == layout_offset ? "YES" : "NO") << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Create a 4D tensor (2x2x4x4) in NCHW format
  // ========================================================================
  // Goal: Create a 4D tensor representing batch=2, channels=2, height=4, width=4
  std::cout << "Task 6 - Create a 4D Tensor (2x2x4x4) in NCHW format:" << std::endl;
  
  float data_4d[64];
  for (int i = 0; i < 64; ++i) {
    data_4d[i] = static_cast<float>(i);
  }

  // TODO: Create a 4D row-major layout with shape (2, 2, 4, 4)
  // Hint: make_layout(make_shape(Int<2>{}, Int<2>{}, Int<4>{}, Int<4>{}), GenRowMajor{})
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor wrapping data_4d
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "4D Layout:" << std::endl;
  // TODO: Print the layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Shape: ";
  // TODO: Print the shape
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Stride: ";
  // TODO: Print the stride
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 7: Access elements in the 4D tensor
  // ========================================================================
  // Goal: Access specific elements using (batch, channel, row, col) coordinates
  std::cout << "Task 7 - Access 4D Tensor Elements:" << std::endl;
  
  // TODO: Access and print element at (0, 0, 0, 0)
  std::cout << "Element at (0, 0, 0, 0): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 0)" << std::endl;

  // TODO: Access and print element at (0, 0, 2, 3)
  std::cout << "Element at (0, 0, 2, 3): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 11)" << std::endl;

  // TODO: Access and print element at (0, 1, 0, 0)
  std::cout << "Element at (0, 1, 0, 0): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 16)" << std::endl;

  // TODO: Access and print element at (1, 0, 0, 0)
  std::cout << "Element at (1, 0, 0, 0): ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 32)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 8: Print a specific batch and channel from 4D tensor
  // ========================================================================
  // Goal: Print the 2D image at batch 0, channel 0
  std::cout << "Task 8 - Print Batch 0, Channel 0:" << std::endl;
  
  // TODO: Write nested loops to print tensor_4d(0, 0, i, j) for i,j in 0..3
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 9: Calculate 4D strides manually
  // ========================================================================
  // Goal: Understand how 4D strides are calculated
  std::cout << "Task 9 - 4D Stride Calculation:" << std::endl;
  std::cout << "For 4D layout (N, C, H, W) in row-major:" << std::endl;
  std::cout << "  stride_N = C * H * W" << std::endl;
  std::cout << "  stride_C = H * W" << std::endl;
  std::cout << "  stride_H = W" << std::endl;
  std::cout << "  stride_W = 1" << std::endl;
  std::cout << std::endl;

  // TODO: Calculate expected strides for shape (2, 2, 4, 4)
  int expected_stride_n = 0;  // TODO: Calculate 2 * 4 * 4
  int expected_stride_c = 0;  // TODO: Calculate 4 * 4
  int expected_stride_h = 4;
  int expected_stride_w = 1;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Expected strides: (" << expected_stride_n << ", " 
            << expected_stride_c << ", " << expected_stride_h << ", " 
            << expected_stride_w << ")" << std::endl;

  // TODO: Get actual strides and compare
  std::cout << "Actual strides:   ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Calculate 4D offset manually
  // ========================================================================
  std::cout << "=== Challenge: 4D Offset Calculation ===" << std::endl;
  std::cout << "Calculate offset for position (1, 0, 2, 3):" << std::endl;
  
  int n = 1, c = 0, h = 2, w = 3;

  // TODO: Calculate offset manually: n*32 + c*16 + h*4 + w*1
  int manual_offset_4d = 0;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Get offset from layout
  int layout_offset_4d = 0;
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Manual calculation: " << manual_offset_4d << std::endl;
  std::cout << "From layout: " << layout_offset_4d << std::endl;
  std::cout << "Match: " << (manual_offset_4d == layout_offset_4d ? "YES" : "NO") << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE 2: Create HWC format tensor (4x4x3 - RGB image)
  // ========================================================================
  std::cout << "=== Challenge 2: HWC Format (Height, Width, Channels) ===" << std::endl;
  std::cout << "Create a 4x4x3 tensor representing an RGB image:" << std::endl;
  
  float data_hwc[48];
  for (int i = 0; i < 48; ++i) {
    data_hwc[i] = static_cast<float>(i);
  }

  // TODO: Create a 3D row-major layout with shape (4, 4, 3) - HWC order
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Create a tensor wrapping data_hwc
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "HWC Layout: ";
  // TODO: Print the layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Stride: ";
  // TODO: Print the stride
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Access element at (2, 3, 1) - row 2, col 3, channel 1 (green)
  std::cout << "Element at (2, 3, 1) - row 2, col 3, green channel: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. How do you calculate stride for dimension i in N-D row-major?" << std::endl;
  std::cout << "2. What is the difference between NCHW and NHWC formats?" << std::endl;
  std::cout << "3. How do you access elements in a 3D tensor?" << std::endl;
  std::cout << "4. What is the formula for offset in N-D tensors?" << std::endl;

  return 0;
}
