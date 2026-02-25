/**
 * Exercise 04: Tensor Transpose and View Operations
 * 
 * Objective: Learn to create transposed views of tensors without copying data
 * 
 * Tasks:
 * 1. Understand how transpose changes tensor view
 * 2. Create transposed views using layout manipulation
 * 3. Verify that transpose is a view operation (no copy)
 * 4. Practice with other view operations
 * 
 * Key Concepts:
 * - Transpose: Swapping dimensions of a tensor
 * - View Operation: Changes how data is accessed, not the data itself
 * - Zero-Copy: Transpose doesn't duplicate memory
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: Tensor Transpose and View ===" << std::endl;
    std::cout << std::endl;

    // Create a 4x6 tensor
    float data[24];
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto layout_orig = make_layout(make_shape(Int<4>{}, Int<6>{}), GenRowMajor{});
    auto tensor_orig = make_tensor(make_gmem_ptr(data), layout_orig);

    std::cout << "Original Tensor (4x6):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 6; ++j) {
            printf("%3d ", static_cast<int>(tensor_orig(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 1: Create a transposed layout (6x4)
    // In CuTe, transpose is achieved by creating a new layout with swapped dimensions
    std::cout << "Task 1 - Transposed Layout:" << std::endl;
    auto layout_transposed = make_layout(make_shape(Int<6>{}, Int<4>{}), GenColMajor{});
    std::cout << "Transposed layout: " << layout_transposed << std::endl;
    std::cout << std::endl;

    // Create a transposed view using the same data
    auto tensor_transposed = make_tensor(make_gmem_ptr(data), layout_transposed);

    // TASK 2: Access the transposed view
    std::cout << "Task 2 - Transposed View (6x4):" << std::endl;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(tensor_transposed(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Verify transpose relationship
    std::cout << "Task 3 - Verify Transpose Relationship:" << std::endl;
    std::cout << "Original(0, 1) = " << tensor_orig(0, 1) << std::endl;
    std::cout << "Transposed(1, 0) = " << tensor_transposed(1, 0) << std::endl;
    std::cout << "Should be equal: " << (tensor_orig(0, 1) == tensor_transposed(1, 0) ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    std::cout << "Original(2, 3) = " << tensor_orig(2, 3) << std::endl;
    std::cout << "Transposed(3, 2) = " << tensor_transposed(3, 2) << std::endl;
    std::cout << "Should be equal: " << (tensor_orig(2, 3) == tensor_transposed(3, 2) ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // TASK 4: Double transpose (should return to original)
    std::cout << "Task 4 - Double Transpose:" << std::endl;
    auto layout_double_transpose = make_layout(make_shape(Int<4>{}, Int<6>{}), GenRowMajor{});
    auto tensor_double_transpose = make_tensor(make_gmem_ptr(data), layout_double_transpose);
    
    std::cout << "Double transposed tensor should match original:" << std::endl;
    std::cout << "Original(1, 2) = " << tensor_orig(1, 2) << std::endl;
    std::cout << "Double transposed(1, 2) = " << tensor_double_transpose(1, 2) << std::endl;
    std::cout << std::endl;

    // TASK 5: Create a view of a sub-region
    std::cout << "Task 5 - Sub-region View:" << std::endl;
    std::cout << "View of top-left 3x4 region:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(tensor_orig(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // CHALLENGE: Create specific views
    std::cout << "=== Challenge: Create These Views ===" << std::endl;
    
    std::cout << "1. Bottom-right 2x3 of original:" << std::endl;
    for (int i = 2; i < 4; ++i) {
        for (int j = 3; j < 6; ++j) {
            printf("%3d ", static_cast<int>(tensor_orig(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "2. Every other row:" << std::endl;
    for (int i = 0; i < 4; i += 2) {
        for (int j = 0; j < 6; ++j) {
            printf("%3d ", static_cast<int>(tensor_orig(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // VIEW OPERATIONS SUMMARY
    std::cout << "=== View Operations Summary ===" << std::endl;
    std::cout << "1. Transpose: Swap dimensions, change access pattern" << std::endl;
    std::cout << "2. Slice: Extract sub-region with new layout" << std::endl;
    std::cout << "3. Reshape: Change dimensions, keep same data" << std::endl;
    std::cout << "4. All views share the same underlying data" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Transpose creates a new view without copying data" << std::endl;
    std::cout << "2. Transpose swaps: original(i,j) = transposed(j,i)" << std::endl;
    std::cout << "3. Layout determines how coordinates map to memory" << std::endl;
    std::cout << "4. Multiple views can reference the same data" << std::endl;

    return 0;
}
