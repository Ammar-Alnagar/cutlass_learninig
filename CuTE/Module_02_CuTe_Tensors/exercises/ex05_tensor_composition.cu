/**
 * Exercise 05: Tensor Composition with Layouts
 * 
 * Objective: Learn to compose tensors with hierarchical layouts
 *            for complex memory organizations
 * 
 * Tasks:
 * 1. Create a tensor with a composed layout
 * 2. Understand how composition affects tensor access
 * 3. Build multi-level tensor hierarchies
 * 4. Apply composition to tiled algorithms
 * 
 * Key Concepts:
 * - Composition: Combining layouts to create complex structures
 * - Hierarchy: Multiple levels of organization
 * - Tiled Access: Accessing data in tiles for efficiency
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 05: Tensor Composition with Layouts ===" << std::endl;
    std::cout << std::endl;

    // Create data for a tiled matrix (8x8 organized as 2x2 tiles of 4x4)
    float data[64];
    for (int i = 0; i < 64; ++i) {
        data[i] = static_cast<float>(i);
    }

    // TASK 1: Create a simple flat layout
    std::cout << "Task 1 - Flat Layout (8x8):" << std::endl;
    auto flat_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto flat_tensor = make_tensor(make_gmem_ptr(data), flat_layout);
    
    std::cout << "Flat layout: " << flat_layout << std::endl;
    std::cout << std::endl;

    // TASK 2: Create a tile layout (2x2 tiles)
    std::cout << "Task 2 - Tile Layout (2x2):" << std::endl;
    auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{});
    std::cout << "Tile layout: " << tile_layout << std::endl;
    std::cout << std::endl;

    // TASK 3: Create an element-within-tile layout (4x4 elements per tile)
    std::cout << "Task 3 - Element Layout (4x4 per tile):" << std::endl;
    auto element_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    std::cout << "Element layout: " << element_layout << std::endl;
    std::cout << std::endl;

    // TASK 4: Understand the composed structure
    std::cout << "Task 4 - Composed Structure:" << std::endl;
    std::cout << "Total structure: (2x2 tiles) × (4x4 elements per tile)" << std::endl;
    std::cout << "Total elements: " << 2 * 4 << " × " << 2 * 4 << " = 8x8" << std::endl;
    std::cout << std::endl;

    // Visualize the tiling
    std::cout << "Tiling visualization (T0-T3 = tile IDs):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int tile_row = i / 4;
            int tile_col = j / 4;
            int tile_id = tile_row * 2 + tile_col;
            printf("T%d ", tile_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Access elements by tile coordinates
    std::cout << "Task 5 - Access by Tile Coordinates:" << std::endl;
    std::cout << "To access element (elem_i, elem_j) in tile (tile_i, tile_j):" << std::endl;
    std::cout << "  Global row = tile_i * 4 + elem_i" << std::endl;
    std::cout << "  Global col = tile_j * 4 + elem_j" << std::endl;
    std::cout << std::endl;

    // Example: Access element (1, 2) in tile (0, 1)
    int tile_i = 0, tile_j = 1;
    int elem_i = 1, elem_j = 2;
    int global_i = tile_i * 4 + elem_i;
    int global_j = tile_j * 4 + elem_j;
    
    std::cout << "Example: Element (1,2) in tile (0,1)" << std::endl;
    std::cout << "  Global position: (" << global_i << ", " << global_j << ")" << std::endl;
    std::cout << "  Value: " << flat_tensor(global_i, global_j) << std::endl;
    std::cout << std::endl;

    // TASK 6: Create a tensor for each tile
    std::cout << "Task 6 - Individual Tile Tensors:" << std::endl;
    
    for (int ti = 0; ti < 2; ++ti) {
        for (int tj = 0; tj < 2; ++tj) {
            std::cout << "Tile (" << ti << "," << tj << "):" << std::endl;
            for (int ei = 0; ei < 4; ++ei) {
                for (int ej = 0; ej < 4; ++ej) {
                    int gi = ti * 4 + ei;
                    int gj = tj * 4 + ej;
                    printf("%5.1f ", flat_tensor(gi, gj));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // CHALLENGE: Calculate tile and element from global coordinates
    std::cout << "=== Challenge: Global to Tile Mapping ===" << std::endl;
    std::cout << "Given global coordinates (5, 6) in an 8x8 matrix with 4x4 tiles:" << std::endl;
    int global_row = 5, global_col = 6;
    int tile_row = global_row / 4;
    int tile_col = global_col / 4;
    int elem_row = global_row % 4;
    int elem_col = global_col % 4;
    
    std::cout << "  Tile coordinates: (" << tile_row << ", " << tile_col << ")" << std::endl;
    std::cout << "  Element within tile: (" << elem_row << ", " << elem_col << ")" << std::endl;
    std::cout << "  Value: " << flat_tensor(global_row, global_col) << std::endl;
    std::cout << std::endl;

    // APPLICATION: Thread block tiling
    std::cout << "=== Application: Thread Block Tiling ===" << std::endl;
    std::cout << "In CUDA kernels, each thread block can process one tile:" << std::endl;
    std::cout << "- Block (0,0) processes Tile 0" << std::endl;
    std::cout << "- Block (0,1) processes Tile 1" << std::endl;
    std::cout << "- Block (1,0) processes Tile 2" << std::endl;
    std::cout << "- Block (1,1) processes Tile 3" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Composition organizes data hierarchically" << std::endl;
    std::cout << "2. Tiles enable parallel processing" << std::endl;
    std::cout << "3. Global coord = tile_coord × tile_size + element_coord" << std::endl;
    std::cout << "4. Composition is fundamental for tiled algorithms" << std::endl;

    return 0;
}
