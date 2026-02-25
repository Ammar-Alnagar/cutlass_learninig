/**
 * Exercise 03: Custom Strides and Padding
 * 
 * Objective: Learn to create layouts with custom strides for padded matrices
 *            and understand how padding affects memory layout
 * 
 * Tasks:
 * 1. Create a layout with padding to avoid bank conflicts
 * 2. Create a layout with custom strides for submatrix extraction
 * 3. Calculate the memory overhead of padding
 * 4. Understand how padding improves memory access efficiency
 * 
 * Key Concepts:
 * - Padding: Adding extra elements to improve memory access patterns
 * - Bank Conflicts: Multiple threads accessing same memory bank
 * - Custom Strides: Manually specifying stride values
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 03: Custom Strides and Padding ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a standard 8x8 row-major layout
    // This might cause bank conflicts in shared memory
    auto layout_standard = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    std::cout << "Task 1 - Standard 8x8 Layout:" << std::endl;
    std::cout << "Layout: " << layout_standard << std::endl;
    std::cout << "Shape: " << layout_standard.shape() << std::endl;
    std::cout << "Stride: " << layout_standard.stride() << std::endl;
    std::cout << std::endl;

    // TASK 2: Create a padded 8x8 layout with stride 9 (adding 1 element padding per row)
    // This helps avoid bank conflicts in shared memory
    // Hint: Use make_stride(Int<9>{}, Int<1>{}) for row-major with padding
    // TODO: Uncomment and complete:
    // auto layout_padded = make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<9>{}, Int<1>{}));
    
    std::cout << "Task 2 - Padded 8x8 Layout (stride=9):" << std::endl;
    std::cout << "TODO: Create a padded layout" << std::endl;
    // TODO: Print layout, shape, and stride
    std::cout << std::endl;

    // Visualize the difference
    std::cout << "=== Memory Layout Comparison ===" << std::endl;
    std::cout << "Standard 8x8 (first 3 rows):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", layout_standard(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Padded 8x8 with stride 9 (first 3 rows):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 8; ++j) {
            // TODO: Use layout_padded(i, j) here
            printf("  X ");  // Replace with actual offset
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Calculate memory overhead
    std::cout << "=== Memory Overhead Calculation ===" << std::endl;
    int standard_elements = 8 * 8;
    int padded_elements = 8 * 9;  // 8 rows, 9 columns (including padding)
    int overhead = padded_elements - standard_elements;
    float overhead_percent = (float)overhead / standard_elements * 100;
    
    std::cout << "Standard layout: " << standard_elements << " elements" << std::endl;
    std::cout << "Padded layout: " << padded_elements << " elements" << std::endl;
    std::cout << "Overhead: " << overhead << " elements (" << overhead_percent << "%)" << std::endl;
    std::cout << std::endl;

    // TASK 4: Create a layout for accessing a submatrix
    // Given a 16x16 matrix, create a layout that accesses only even rows
    // This simulates strided access patterns
    // Hint: Use make_stride(Int<32>{}, Int<1>{}) to skip every other row
    auto layout_strided = make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<32>{}, Int<1>{}));
    
    std::cout << "Task 4 - Strided Layout (every other row):" << std::endl;
    std::cout << "Layout: " << layout_strided << std::endl;
    std::cout << "This layout accesses rows 0, 2, 4, 6, 8, 10, 12, 14 of a 16-column matrix" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Bank Conflict Analysis
    std::cout << "=== Bank Conflict Challenge ===" << std::endl;
    std::cout << "On sm_89, shared memory has 32 banks." << std::endl;
    std::cout << "When 32 threads access a column in an 8x8 standard layout:" << std::endl;
    std::cout << "  Thread 0 accesses offset " << layout_standard(0, 0) << " -> bank " << (layout_standard(0, 0) % 32) << std::endl;
    std::cout << "  Thread 1 accesses offset " << layout_standard(1, 0) << " -> bank " << (layout_standard(1, 0) % 32) << std::endl;
    std::cout << "  Thread 2 accesses offset " << layout_standard(2, 0) << " -> bank " << (layout_standard(2, 0) % 32) << std::endl;
    std::cout << "  (No conflict - each thread accesses different bank)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "When 32 threads access a row in an 8x8 standard layout:" << std::endl;
    std::cout << "  Thread 0-7 access offsets 0-7 -> banks 0-7" << std::endl;
    std::cout << "  Thread 8-15 access offsets 8-15 -> banks 8-15" << std::endl;
    std::cout << "  (No conflict within a warp)" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Custom strides enable padded layouts" << std::endl;
    std::cout << "2. Padding trades memory for better access patterns" << std::endl;
    std::cout << "3. Padding helps avoid bank conflicts in shared memory" << std::endl;
    std::cout << "4. Strided layouts enable submatrix access" << std::endl;

    return 0;
}
