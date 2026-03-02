/**
 * Exercise 05: Tensor Composition with Layouts
 *
 * Objective: Learn to compose tensors with hierarchical layouts
 *            for complex memory organizations (tiling)
 *
 * Instructions:
 * - Complete each TODO section by building composed layouts
 * - Understand how composition creates hierarchical tensor access
 * - Practice converting between global and tile-local coordinates
 *
 * Key Concepts:
 * - Composition: Combining layouts to create complex structures
 * - Tiling: Organizing data as tiles for efficient GPU access
 * - Coordinate mapping: global = tile_coord * tile_size + element_coord
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 05: Tensor Composition with Layouts ===" << std::endl;
  std::cout << std::endl;

  // Create data for a tiled matrix (8x8 organized as 2x2 tiles of 4x4)
  float data[64];
  for (int i = 0; i < 64; ++i) {
    data[i] = static_cast<float>(i);
  }

  // Create the flat 8x8 tensor as reference
  auto flat_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
  auto flat_tensor = make_tensor(make_gmem_ptr(data), flat_layout);

  std::cout << "Reference: Flat 8x8 Tensor" << std::endl;
  print(flat_tensor);
  std::cout << std::endl;


  // ========================================================================
  // TASK 1: Understand the tile structure
  // ========================================================================
  // Goal: Visualize how an 8x8 matrix is divided into 2x2 tiles of 4x4 each
  std::cout << "Task 1 - Understand Tile Structure:" << std::endl;
  std::cout << "An 8x8 matrix divided into 2x2 tiles (each tile is 4x4):" << std::endl;
  std::cout << std::endl;

  // TODO: Print the tile ID for each position
  // Tile ID = (row / 4) * 2 + (col / 4)
  // This shows which tile each element belongs to
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      // TODO: Calculate tile_row = i / 4, tile_col = j / 4, tile_id = tile_row * 2 + tile_col
      // START YOUR CODE HERE
      
      
      // END YOUR CODE HERE
      printf("T%d ", tile_id);
    }
    std::cout << std::endl;
  }
  std::cout << "T0 = top-left, T1 = top-right, T2 = bottom-left, T3 = bottom-right" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Create tile and element layouts separately
  // ========================================================================
  // Goal: Create separate layouts for tile grid and elements within tiles
  std::cout << "Task 2 - Create Tile and Element Layouts:" << std::endl;
  
  // TODO: Create a 2x2 layout for the tile grid (row-major)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Tile layout (2x2 grid of tiles):" << std::endl;
  // TODO: Print the tile layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // TODO: Create a 4x4 layout for elements within each tile (row-major)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Element layout (4x4 elements per tile):" << std::endl;
  // TODO: Print the element layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Access elements using tile + element coordinates
  // ========================================================================
  // Goal: Learn to convert (tile_i, tile_j, elem_i, elem_j) to global coordinates
  std::cout << "Task 3 - Access Using Tile Coordinates:" << std::endl;
  std::cout << "Formula: global_row = tile_i * 4 + elem_i" << std::endl;
  std::cout << "         global_col = tile_j * 4 + elem_j" << std::endl;
  std::cout << std::endl;

  // Example: Access element (1, 2) in tile (0, 1)
  int tile_i = 0, tile_j = 1;
  int elem_i = 1, elem_j = 2;

  // TODO: Calculate global_row = tile_i * 4 + elem_i
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Calculate global_col = tile_j * 4 + elem_j
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Element (1,2) in tile (0,1):" << std::endl;
  std::cout << "  Global position: (" << global_row << ", " << global_col << ")" << std::endl;
  std::cout << "  Value: " << flat_tensor(global_row, global_col) << std::endl;
  std::cout << "  (expected: global position (1,6) = 14)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Extract individual tiles as tensor views
  // ========================================================================
  // Goal: Create a 4x4 tensor view for each of the 4 tiles
  std::cout << "Task 4 - Extract Individual Tiles:" << std::endl;

  // Tile 0 (top-left): rows 0-3, cols 0-3
  std::cout << "Tile 0 (top-left):" << std::endl;
  // TODO: Create a 4x4 layout with stride (8, 1)
  // TODO: Create tensor with offset = flat_layout(0, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  // TODO: Print tile 0
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Tile 1 (top-right): rows 0-3, cols 4-7
  std::cout << "Tile 1 (top-right):" << std::endl;
  // TODO: Create a 4x4 layout with stride (8, 1)
  // TODO: Create tensor with offset = flat_layout(0, 4)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  // TODO: Print tile 1
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Tile 2 (bottom-left): rows 4-7, cols 0-3
  std::cout << "Tile 2 (bottom-left):" << std::endl;
  // TODO: Create a 4x4 layout with stride (8, 1)
  // TODO: Create tensor with offset = flat_layout(4, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  // TODO: Print tile 2
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // Tile 3 (bottom-right): rows 4-7, cols 4-7
  std::cout << "Tile 3 (bottom-right):" << std::endl;
  // TODO: Create a 4x4 layout with stride (8, 1)
  // TODO: Create tensor with offset = flat_layout(4, 4)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  // TODO: Print tile 3
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Convert global coordinates to tile + element coordinates
  // ========================================================================
  // Goal: Given global (5, 6), find which tile and which element within tile
  std::cout << "Task 5 - Global to Tile Mapping:" << std::endl;
  std::cout << "Given global coordinates (5, 6) in 8x8 matrix with 4x4 tiles:" << std::endl;
  
  int global_row = 5, global_col = 6;

  // TODO: Calculate tile_row = global_row / 4
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Calculate tile_col = global_col / 4
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Calculate elem_row = global_row % 4
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  // TODO: Calculate elem_col = global_col % 4
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "  Tile coordinates: (" << tile_row << ", " << tile_col << ")" << std::endl;
  std::cout << "  Element within tile: (" << elem_row << ", " << elem_col << ")" << std::endl;
  std::cout << "  Value at global (5,6): " << flat_tensor(global_row, global_col) << std::endl;
  std::cout << "  Value at tile(" << tile_row << "," << tile_col << ")(elem) = ";
  // TODO: Verify by accessing through the appropriate tile tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Create a composed layout using logical_divide
  // ========================================================================
  // Goal: Use CuTe's logical_divide to create a hierarchical layout
  std::cout << "Task 6 - Composed Layout with logical_divide:" << std::endl;
  std::cout << "Divide the 8x8 layout into 2x2 tiles of 4x4 each:" << std::endl;
  
  // TODO: Use logical_divide to partition the flat_layout into 2x2 tiles
  // Hint: auto composed = logical_divide(flat_layout, make_shape(Int<4>{}, Int<4>{}));
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Composed layout:" << std::endl;
  // TODO: Print the composed layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  // The composed layout has structure: (tile_i, tile_j, elem_i, elem_j)
  // TODO: Access element using composed coordinates
  // Hint: composed(tile_i, tile_j, elem_i, elem_j) gives the flat offset
  std::cout << "Access using composed coordinates:" << std::endl;
  std::cout << "  Position (tile 0, tile 1, elem 1, elem 2):" << std::endl;
  std::cout << "  Offset: ";
  // TODO: Print composed(0, 1, 1, 2)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;
  std::cout << "  (This should equal flat_layout(1, 6) = 14)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Access all elements of Tile 2 using composed coordinates
  // ========================================================================
  std::cout << "=== Challenge: Access Tile 2 Using Composed Coordinates ===" << std::endl;
  std::cout << "Tile 2 is at tile position (1, 0) - bottom-left tile" << std::endl;
  std::cout << "Access all 4x4 elements of Tile 2:" << std::endl;
  
  // TODO: Use nested loops to access composed(1, 0, elem_i, elem_j) for elem_i, elem_j in 0..3
  // Hint: The composed layout from Task 6 can be accessed as composed(tile_i, tile_j, elem_i, elem_j)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. How do you convert global coordinates to tile + element coordinates?" << std::endl;
  std::cout << "2. What is the formula: global_coord = ?" << std::endl;
  std::cout << "3. Why is tiling useful for GPU algorithms?" << std::endl;
  std::cout << "4. What does logical_divide() do?" << std::endl;

  return 0;
}
