/**
 * Exercise 01: Basic Layout Creation
 *
 * Objective: Learn to create fundamental CuTe layouts using make_layout,
 * make_shape, and make_stride
 *
 * Tasks:
 * 1. Create a 1D layout with 16 elements
 * 2. Create a 2D layout with shape (8, 4) in row-major order
 * 3. Create a 3D layout with shape (4, 4, 2)
 * 4. Print each layout to understand its structure
 *
 * Key Functions:
 * - make_layout(shape, stride) - Creates a layout from shape and stride
 * - make_shape(d0, d1, ...) - Creates a shape tuple
 * - make_stride(s0, s1, ...) - Creates a stride tuple
 * - GenRowMajor{} - Generator for row-major strides
 * - GenColMajor{} - Generator for column-major strides
 */

#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 01: Basic Layout Creation ===" << std::endl;
  std::cout << std::endl;

  // TASK 1: Create a 1D layout with 16 elements
  // Hint: Use make_shape(Int<16>{}) and the layout will have stride 1
  // TODO: Uncomment and complete the following line:
  // auto layout_1d = make_layout(make_shape(Int<16>{}), GenRowMajor{});

  std::cout << "Task 1 - 1D Layout (16 elements):" << std::endl;
  std::cout << "TODO: Create and print a 1D layout" << std::endl;
  // TODO: Print the layout using print(layout_1d) or std::cout << layout_1d
  print(layout_1d);
  std::cout << std::endl;

  // TASK 2: Create a 2D row-major layout with shape (8, 4)
  // In row-major: stride for dim 0 = 4, stride for dim 1 = 1
  // Hint: Use GenRowMajor{} to automatically generate correct strides
  // TODO: Uncomment and complete:
  // auto layout_2d_rm = make_layout(make_shape(Int<8>{}, Int<4>{}),
  // GenRowMajor{});
  auto layout_2d_rm = make_layout(make_shape(Int<8>{}, INt<4>{}), GenRowMajor);

  std::cout << "Task 2 - 2D Row-Major Layout (8x4):" << std::endl;
  std::cout << "TODO: Create and print a 2D row-major layout" << std::endl;
  // TODO: Print the layout
  std::cout << std::endl;

  // TASK 3: Create a 2D column-major layout with shape (8, 4)
  // In column-major: stride for dim 0 = 1, stride for dim 1 = 8
  // Hint: Use GenColMajor{} to automatically generate correct strides
  // TODO: Uncomment and complete:
  // auto layout_2d_cm = make_layout(make_shape(Int<8>{}, Int<4>{}),
  // GenColMajor{});

  auto layout_2d_cm = make_layout(make_shape(Int<8>{}, Int<4>{}), GenColMajor);

  std::cout << "Task 3 - 2D Column-Major Layout (8x4):" << std::endl;
  std::cout << "TODO: Create and print a 2D column-major layout" << std::endl;
  // TODO: Print the layout
  std::cout << std::endl;

  // TASK 4: Create a 3D layout with shape (4, 4, 2)
  // Think about what the strides should be for row-major ordering
  // Hint: For row-major (4,4,2): stride = (8, 2, 1)
  // TODO: Uncomment and complete:
  // auto layout_3d = make_layout(make_shape(Int<4>{}, Int<4>{}, Int<2>{}),
  // GenRowMajor{});

  auto layout_3d =
      make_layout(make_shape(Int<4>{}, Int<4>{}, Int<2>{}), GenRowMajor);

  std::cout << "Task 4 - 3D Layout (4x4x2):" << std::endl;
  std::cout << "TODO: Create and print a 3D layout" << std::endl;
  // TODO: Print the layout
  std::cout << std::endl;

  // VERIFICATION: Manually verify the offset calculations
  std::cout << "=== Verification ===" << std::endl;
  std::cout << "For a 2D row-major layout (8x4):" << std::endl;
  std::cout << "  layout(0,0) should be 0" << std::endl;
  std::cout << "  layout(0,1) should be 1" << std::endl;
  std::cout << "  layout(1,0) should be 4" << std::endl;
  std::cout << "  layout(2,3) should be 11" << std::endl;
  std::cout << std::endl;

  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. Layouts map logical coordinates to memory offsets"
            << std::endl;
  std::cout << "2. Shape defines the dimensions" << std::endl;
  std::cout << "3. Stride defines how to step through each dimension"
            << std::endl;
  std::cout << "4. Row-major: rightmost dimension has stride 1" << std::endl;
  std::cout << "5. Column-major: leftmost dimension has stride 1" << std::endl;

  return 0;
}
