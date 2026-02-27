/**
 * Exercise 02: Row-Major vs Column-Major Layouts
 *
 * Objective: Understand the difference between row-major and column-major
 * layouts and how they affect memory access patterns
 *
 * Tasks:
 * 1. Create both row-major and column-major layouts for a 4x4 matrix
 * 2. Map coordinates to offsets and observe the difference
 * 3. Identify which access pattern is coalesced for each layout
 * 4. Visualize the memory layout in grid format
 *
 * Key Concepts:
 * - Row-Major: Consecutive elements in a row are contiguous in memory
 * - Column-Major: Consecutive elements in a column are contiguous in memory
 * - Coalesced Access: When consecutive threads access consecutive memory
 * addresses
 */

#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

template <typename Layout>
void print_layout_grid(Layout const &layout, const char *name) {
  std::cout << name << " Memory Layout Grid:" << std::endl;
  std::cout << "----------------------------" << std::endl;

  auto shape = layout.shape();
  int rows = get<0>(shape);
  int cols = get<1>(shape);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int offset = layout(i, j);
      printf("%3d ", offset);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "=== Exercise 02: Row-Major vs Column-Major ===" << std::endl;
  std::cout << std::endl;

  // Create a 4x4 row-major layout
  // TODO: Complete the layout creation
  auto layout_rm = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});

  // Create a 4x4 column-major layout
  // TODO: Complete the layout creation
  auto layout_cm = make_layout(make_shape(Int<4>{}, Int<4>{}), GenColMajor{});

  // Print layout structures
  std::cout << "Row-Major Layout Structure:" << std::endl;
  print(layout_rm);
  std::cout << std::endl;

  std::cout << "Column-Major Layout Structure:" << std::endl;
  print(layout_cm);
  std::cout << std::endl;

  // Visualize as grids
  print_layout_grid(layout_rm, "Row-Major (4x4)");
  print_layout_grid(layout_cm, "Column-Major (4x4)");

  // TASK: Compare offset mappings
  std::cout << "=== Offset Comparison ===" << std::endl;
  std::cout << "Coordinate | Row-Major | Column-Major" << std::endl;
  std::cout << "-----------|-----------|-------------" << std::endl;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      int offset_rm = layout_rm(i, j);
      int offset_cm = layout_cm(i, j);
      std::cout << "(" << i << "," << j << ")      | " << offset_rm
                << "         | " << offset_cm << std::endl;
    }
  }
  std::cout << std::endl;

  // TASK: Identify coalesced access patterns
  std::cout << "=== Coalesced Access Analysis ===" << std::endl;
  std::cout << "For Row-Major layout:" << std::endl;
  std::cout << "  Row-wise access (varying j): ";
  for (int j = 0; j < 4; ++j) {
    std::cout << layout_rm(0, j) << " ";
  }
  std::cout << "-> Consecutive? "
            << (layout_rm(0, 1) - layout_rm(0, 0) == 1 ? "YES (COALESCED)"
                                                       : "NO")
            << std::endl;

  std::cout << "  Column-wise access (varying i): ";
  for (int i = 0; i < 4; ++i) {
    std::cout << layout_rm(i, 0) << " ";
  }
  std::cout << "-> Consecutive? "
            << (layout_rm(1, 0) - layout_rm(0, 0) == 1 ? "YES"
                                                       : "NO (NOT COALESCED)")
            << std::endl;
  std::cout << std::endl;

  std::cout << "For Column-Major layout:" << std::endl;
  std::cout << "  Row-wise access (varying j): ";
  for (int j = 0; j < 4; ++j) {
    std::cout << layout_cm(0, j) << " ";
  }
  std::cout << "-> Consecutive? "
            << (layout_cm(0, 1) - layout_cm(0, 0) == 1 ? "YES"
                                                       : "NO (NOT COALESCED)")
            << std::endl;

  std::cout << "  Column-wise access (varying i): ";
  for (int i = 0; i < 4; ++i) {
    std::cout << layout_cm(i, 0) << " ";
  }
  std::cout << "-> Consecutive? "
            << (layout_cm(1, 0) - layout_cm(0, 0) == 1 ? "YES (COALESCED)"
                                                       : "NO")
            << std::endl;
  std::cout << std::endl;

  // CHALLENGE: Predict the offset for specific coordinates
  std::cout << "=== Challenge: Predict the Offsets ===" << std::endl;
  std::cout << "Before checking the output above, predict:" << std::endl;
  std::cout << "  Row-major layout(3, 2) = ? (Answer: " << layout_rm(3, 2)
            << ")" << std::endl;
  std::cout << "  Column-major layout(3, 2) = ? (Answer: " << layout_cm(3, 2)
            << ")" << std::endl;
  std::cout << std::endl;

  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. Row-major layouts have coalesced row access" << std::endl;
  std::cout << "2. Column-major layouts have coalesced column access"
            << std::endl;
  std::cout << "3. Same data, different memory ordering" << std::endl;
  std::cout << "4. Choose layout based on access pattern for best performance"
            << std::endl;

  return 0;
}
