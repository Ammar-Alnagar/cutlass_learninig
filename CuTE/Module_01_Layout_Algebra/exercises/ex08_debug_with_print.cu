/**
 * Exercise 08: Debug Layouts with cute::print
 * 
 * Objective: Master the use of cute::print() for debugging and understanding
 *            layout structures and mappings
 * 
 * Tasks:
 * 1. Use cute::print() to visualize different layouts
 * 2. Interpret the output to understand layout structure
 * 3. Debug common layout construction mistakes
 * 4. Develop a systematic debugging approach
 * 
 * Key Functions:
 * - cute::print(layout) - Detailed layout visualization
 * - layout.shape() - Get the shape tuple
 * - layout.stride() - Get the stride tuple
 * - layout(coord) - Calculate offset for a coordinate
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/util/print.hpp"

using namespace cute;

void debug_layout(const char* name, Layout const& layout) {
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "Layout object: " << layout << std::endl;
    std::cout << "Shape: " << layout.shape() << std::endl;
    std::cout << "Stride: " << layout.stride() << std::endl;
    std::cout << "Detailed print:" << std::endl;
    print(layout);
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Exercise 08: Debug Layouts with cute::print ===" << std::endl;
    std::cout << std::endl;

    // Create various layouts to debug
    auto layout1 = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    auto layout2 = make_layout(make_shape(Int<4>{}, Int<8>{}), GenColMajor{});
    auto layout3 = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<10>{}, Int<1>{}));
    auto layout4 = make_layout(make_shape(Int<2>{}, Int<4>{}, Int<4>{}), GenRowMajor{});

    // TASK 1: Debug simple row-major layout
    debug_layout("Task 1: Row-Major 4x8", layout1);

    // TASK 2: Debug column-major layout
    debug_layout("Task 2: Column-Major 4x8", layout2);

    // TASK 3: Debug padded layout
    debug_layout("Task 3: Padded 4x8 (stride=10)", layout3);

    // TASK 4: Debug 3D layout
    debug_layout("Task 4: 3D Layout 2x4x4", layout4);

    // INTERPRETATION GUIDE
    std::cout << "=== Interpreting cute::print Output ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "The print output shows:" << std::endl;
    std::cout << "1. Layout structure with shape and stride" << std::endl;
    std::cout << "2. How coordinates map to offsets" << std::endl;
    std::cout << "3. Hierarchical organization (if any)" << std::endl;
    std::cout << std::endl;

    // COMMON MISTAKES
    std::cout << "=== Common Layout Mistakes ===" << std::endl;
    std::cout << std::endl;

    // Mistake 1: Wrong stride for intended access pattern
    std::cout << "Mistake 1: Row-major layout but accessing column-wise" << std::endl;
    auto rm_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    std::cout << "Row-major 4x4: " << rm_layout << std::endl;
    std::cout << "Column access: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << rm_layout(i, 0) << " ";
    }
    std::cout << "-> Stride of 4 (NOT coalesced)" << std::endl;
    std::cout << std::endl;

    // Mistake 2: Forgetting padding in shared memory
    std::cout << "Mistake 2: No padding causes bank conflicts" << std::endl;
    auto no_padding = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto with_padding = make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<9>{}, Int<1>{}));
    std::cout << "Without padding: " << no_padding.stride() << std::endl;
    std::cout << "With padding: " << with_padding.stride() << std::endl;
    std::cout << std::endl;

    // Mistake 3: Confusing shape with stride
    std::cout << "Mistake 3: Shape vs Stride confusion" << std::endl;
    std::cout << "Shape defines dimensions, Stride defines step sizes" << std::endl;
    std::cout << "Same shape (4,8) can have different strides:" << std::endl;
    std::cout << "  Row-major stride: " << layout1.stride() << std::endl;
    std::cout << "  Column-major stride: " << layout2.stride() << std::endl;
    std::cout << "  Padded stride: " << layout3.stride() << std::endl;
    std::cout << std::endl;

    // DEBUGGING WORKFLOW
    std::cout << "=== Debugging Workflow ===" << std::endl;
    std::cout << "1. Print the layout: print(layout)" << std::endl;
    std::cout << "2. Check shape: layout.shape()" << std::endl;
    std::cout << "3. Check stride: layout.stride()" << std::endl;
    std::cout << "4. Test specific coordinates: layout(i, j)" << std::endl;
    std::cout << "5. Visualize as grid for 2D layouts" << std::endl;
    std::cout << std::endl;

    // PRACTICE: Debug this layout
    std::cout << "=== Practice: Debug This Layout ===" << std::endl;
    auto mystery_layout = make_layout(
        make_shape(Int<4>{}, Int<8>{}),
        make_stride(Int<8>{}, Int<1>{})
    );
    
    std::cout << "Mystery layout:" << std::endl;
    print(mystery_layout);
    std::cout << std::endl;
    
    std::cout << "Questions:" << std::endl;
    std::cout << "1. What is the shape? Answer: " << mystery_layout.shape() << std::endl;
    std::cout << "2. What is the stride? Answer: " << mystery_layout.stride() << std::endl;
    std::cout << "3. Is it row-major? Answer: " 
              << (get<0>(mystery_layout.stride()) == 8 ? "Yes" : "No") << std::endl;
    std::cout << "4. Offset at (2, 3)? Answer: " << mystery_layout(2, 3) << std::endl;
    std::cout << std::endl;

    // VISUALIZATION HELPER
    std::cout << "=== Visualization Helper Function ===" << std::endl;
    std::cout << "Create a helper to visualize any 2D layout:" << std::endl;
    std::cout << std::endl;
    
    auto sample_layout = make_layout(make_shape(Int<6>{}, Int<8>{}), GenRowMajor{});
    std::cout << "6x8 Row-Major Layout:" << std::endl;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", sample_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. cute::print() shows detailed layout structure" << std::endl;
    std::cout << "2. Always verify shape and stride" << std::endl;
    std::cout << "3. Test with specific coordinates" << std::endl;
    std::cout << "4. Visualize as grid for intuition" << std::endl;
    std::cout << "5. Common mistakes: wrong stride, no padding, shape/stride confusion" << std::endl;

    return 0;
}
