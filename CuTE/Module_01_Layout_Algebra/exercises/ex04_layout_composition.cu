/**
 * Exercise 04: Layout Composition
 * 
 * Objective: Learn to compose multiple layouts together to create hierarchical
 *            memory structures for tiled algorithms
 * 
 * Tasks:
 * 1. Create a tiled layout by composing two layouts
 * 2. Understand how hierarchical layouts map threads to data
 * 3. Create a layout for a 2D thread block processing a matrix
 * 4. Visualize the composed layout structure
 * 
 * Key Concepts:
 * - Layout Composition: Combining layouts to create complex mappings
 * - Hierarchical Layouts: Multi-level organization (block -> thread -> element)
 * - Tiling: Dividing computation into smaller tiles
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: Layout Composition ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a simple tile layout (2x2 tiles)
    // This represents how we divide a matrix into tiles
    // TODO: Create a layout representing 2x2 tiles
    auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{});
    
    std::cout << "Task 1 - Tile Layout (2x2 tiles):" << std::endl;
    std::cout << "Tile layout: " << tile_layout << std::endl;
    std::cout << std::endl;

    // TASK 2: Create an element layout (each tile contains 4x4 elements)
    // This represents the internal structure of each tile
    // TODO: Create a layout for elements within a tile
    auto element_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    std::cout << "Task 2 - Element Layout (4x4 elements per tile):" << std::endl;
    std::cout << "Element layout: " << element_layout << std::endl;
    std::cout << std::endl;

    // TASK 3: Compose the layouts using make_composed_layout
    // This creates a hierarchical layout: (2x2 tiles) x (4x4 elements)
    // The composed layout represents an 8x8 matrix organized as 2x2 tiles of 4x4 elements
    // Hint: Use composition to combine tile and element layouts
    // For now, we'll demonstrate the concept by showing how they work together
    
    std::cout << "Task 3 - Composed Layout Concept:" << std::endl;
    std::cout << "Composed structure: (2x2 tiles) containing (4x4 elements each)" << std::endl;
    std::cout << "Total matrix size: 8x8 elements" << std::endl;
    std::cout << std::endl;

    // Create the full 8x8 layout to show the equivalence
    auto full_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    std::cout << "Full 8x8 layout for comparison:" << std::endl;
    print(full_layout);
    std::cout << std::endl;

    // TASK 4: Understand thread-to-data mapping
    // Imagine a 2x2 thread block where each thread handles a 4x4 tile
    std::cout << "=== Thread-to-Data Mapping ===" << std::endl;
    std::cout << "Thread Block: 2x2 threads" << std::endl;
    std::cout << "Each thread processes: 4x4 elements" << std::endl;
    std::cout << "Total coverage: 8x8 matrix" << std::endl;
    std::cout << std::endl;

    std::cout << "Thread Assignment:" << std::endl;
    for (int ti = 0; ti < 2; ++ti) {
        for (int tj = 0; tj < 2; ++tj) {
            std::cout << "Thread (" << ti << "," << tj << ") handles tile at position (" 
                      << ti << "," << tj << ")" << std::endl;
            std::cout << "  Covers elements: rows [" << (ti*4) << "-" << (ti*4+3) << "], "
                      << "cols [" << (tj*4) << "-" << (tj*4+3) << "]" << std::endl;
        }
    }
    std::cout << std::endl;

    // CHALLENGE: Calculate which thread handles which element
    std::cout << "=== Challenge: Element to Thread Mapping ===" << std::endl;
    std::cout << "For an element at position (row, col) in the 8x8 matrix:" << std::endl;
    std::cout << "  Thread row index = row / 4" << std::endl;
    std::cout << "  Thread col index = col / 4" << std::endl;
    std::cout << "  Element within tile: (row % 4, col % 4)" << std::endl;
    std::cout << std::endl;

    // Example calculations
    int test_positions[][2] = {{0, 0}, {3, 7}, {5, 2}, {7, 7}};
    for (auto& pos : test_positions) {
        int row = pos[0];
        int col = pos[1];
        int thread_row = row / 4;
        int thread_col = col / 4;
        int elem_in_tile_row = row % 4;
        int elem_in_tile_col = col % 4;
        std::cout << "  Element (" << row << "," << col << ") -> Thread (" 
                  << thread_row << "," << thread_col << "), local (" 
                  << elem_in_tile_row << "," << elem_in_tile_col << ")" << std::endl;
    }
    std::cout << std::endl;

    // Visualize the tiled structure
    std::cout << "=== Tiled Structure Visualization ===" << std::endl;
    std::cout << "8x8 Matrix divided into 2x2 tiles of 4x4 elements:" << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int tile_id = (i / 4) * 2 + (j / 4);
            printf(" T%d ", tile_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Layouts can be composed hierarchically" << std::endl;
    std::cout << "2. Tiling divides computation among threads" << std::endl;
    std::cout << "3. Each thread handles a tile of the computation" << std::endl;
    std::cout << "4. Hierarchical layouts enable scalable kernel design" << std::endl;

    return 0;
}
