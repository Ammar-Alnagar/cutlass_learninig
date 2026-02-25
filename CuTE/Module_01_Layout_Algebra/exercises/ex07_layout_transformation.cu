/**
 * Exercise 07: Layout Transformation
 * 
 * Objective: Learn to transform layouts using operations like transpose,
 *            reshape, and partition
 * 
 * Tasks:
 * 1. Create a layout and understand its structure
 * 2. Apply transformations to change the view of data
 * 3. Understand how transformations affect memory access
 * 4. Practice with common transformation patterns
 * 
 * Key Concepts:
 * - Transformation: Changing how data is viewed/organized
 * - Transpose: Swapping dimensions
 * - Reshape: Changing dimensions without changing data
 * - Partition: Dividing a layout into parts
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 07: Layout Transformation ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a base layout for transformation
    auto base_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    
    std::cout << "Task 1 - Base Layout (4x8):" << std::endl;
    std::cout << "Layout: " << base_layout << std::endl;
    std::cout << "Shape: " << base_layout.shape() << std::endl;
    std::cout << "Stride: " << base_layout.stride() << std::endl;
    std::cout << std::endl;

    // Visualize the base layout
    std::cout << "Base Layout Grid:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", base_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 2: Conceptual Transpose
    // In CuTe, transpose is achieved by reordering shape and stride
    // For this exercise, we'll create the transposed layout manually
    auto transposed_layout = make_layout(make_shape(Int<8>{}, Int<4>{}), GenColMajor{});
    
    std::cout << "Task 2 - Transposed Layout (8x4):" << std::endl;
    std::cout << "Layout: " << transposed_layout << std::endl;
    std::cout << "Shape: " << transposed_layout.shape() << std::endl;
    std::cout << "Stride: " << transposed_layout.stride() << std::endl;
    std::cout << std::endl;

    // Visualize the transposed layout
    std::cout << "Transposed Layout Grid:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", transposed_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Reshape - View 4x8 as 32 elements linearly
    auto linear_layout = make_layout(make_shape(Int<32>{}), GenRowMajor{});
    
    std::cout << "Task 3 - Linear Layout (32 elements):" << std::endl;
    std::cout << "Layout: " << linear_layout << std::endl;
    std::cout << "Same total elements as 4x8, but viewed as 1D" << std::endl;
    std::cout << std::endl;

    // Show equivalence
    std::cout << "Equivalence demonstration:" << std::endl;
    std::cout << "  2D layout(1, 4) = " << base_layout(1, 4) << std::endl;
    std::cout << "  1D layout(12) = " << linear_layout(12) << std::endl;
    std::cout << "  (1*8 + 4 = 12, same offset!)" << std::endl;
    std::cout << std::endl;

    // TASK 4: Partition - Divide 4x8 into two 4x4 layouts
    auto left_partition = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    auto right_partition = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    std::cout << "Task 4 - Partitioned Layouts:" << std::endl;
    std::cout << "Left partition (4x4):" << std::endl;
    print(left_partition);
    std::cout << std::endl;
    
    std::cout << "Right partition (4x4):" << std::endl;
    print(right_partition);
    std::cout << std::endl;

    // Show how partitions map to original
    std::cout << "Mapping partitions to original 4x8:" << std::endl;
    std::cout << "  Left partition covers columns 0-3 of original" << std::endl;
    std::cout << "  Right partition covers columns 4-7 of original" << std::endl;
    std::cout << std::endl;

    // TASK 5: Create a tiled view of the layout
    // View 4x8 as 2x2 tiles of 2x4 elements each
    std::cout << "Task 5 - Tiled View:" << std::endl;
    std::cout << "Viewing 4x8 as 2x2 tiles, each tile is 2x4 elements" << std::endl;
    std::cout << std::endl;

    std::cout << "Tile assignment in 4x8 matrix:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            int tile_row = i / 2;
            int tile_col = j / 4;
            int tile_id = tile_row * 2 + tile_col;
            printf("T%d ", tile_id);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // CHALLENGE: Transformation Practice
    std::cout << "=== Challenge: Apply Transformations ===" << std::endl;
    std::cout << "Given a 16x8 layout, create:" << std::endl;
    std::cout << "1. Transposed layout (8x16)" << std::endl;
    std::cout << "2. Linear layout (128 elements)" << std::endl;
    std::cout << "3. Four 8x4 partitions" << std::endl;
    std::cout << std::endl;

    auto challenge_base = make_layout(make_shape(Int<16>{}, Int<8>{}), GenRowMajor{});
    
    // Solution for challenge
    std::cout << "Solutions:" << std::endl;
    auto challenge_transposed = make_layout(make_shape(Int<8>{}, Int<16>{}), GenColMajor{});
    auto challenge_linear = make_layout(make_shape(Int<128>{}), GenRowMajor{});
    auto challenge_partition = make_layout(make_shape(Int<8>{}, Int<4>{}), GenRowMajor{});
    
    std::cout << "1. Transposed: " << challenge_transposed << std::endl;
    std::cout << "2. Linear: " << challenge_linear << std::endl;
    std::cout << "3. Each partition (8x4): " << challenge_partition << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Transformations change the view, not the data" << std::endl;
    std::cout << "2. Transpose swaps dimensions" << std::endl;
    std::cout << "3. Reshape changes dimensionality" << std::endl;
    std::cout << "4. Partition divides into smaller layouts" << std::endl;
    std::cout << "5. Same data can be viewed multiple ways" << std::endl;

    return 0;
}
