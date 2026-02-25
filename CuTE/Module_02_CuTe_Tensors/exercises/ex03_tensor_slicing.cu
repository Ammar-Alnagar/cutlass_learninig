/**
 * Exercise 03: Tensor Slicing Operations
 * 
 * Objective: Learn to extract sub-tensors and views from larger tensors
 * 
 * Tasks:
 * 1. Extract row and column slices from a 2D tensor
 * 2. Create sub-matrix views
 * 3. Understand how slicing creates views (not copies)
 * 4. Practice with multi-dimensional slicing
 * 
 * Key Concepts:
 * - Slicing: Extracting a subset of tensor elements
 * - View: A new tensor that references the same data
 * - No Copy: Slicing doesn't duplicate data
 * 
 * Note: CuTe's slicing syntax has evolved. This exercise demonstrates
 * the conceptual approach using layout composition.
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 03: Tensor Slicing Operations ===" << std::endl;
    std::cout << std::endl;

    // Create an 8x8 tensor to slice
    float data[64];
    for (int i = 0; i < 64; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto tensor = make_tensor(make_gmem_ptr(data), layout);

    std::cout << "Original 8x8 Tensor:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", static_cast<int>(tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 1: Extract a row (conceptually)
    std::cout << "Task 1 - Row Slice (Row 3):" << std::endl;
    std::cout << "Elements in row 3: ";
    for (int j = 0; j < 8; ++j) {
        std::cout << tensor(3, j) << " ";
    }
    std::cout << std::endl;
    std::cout << "Note: In CuTe, create a view with 1D layout for the row" << std::endl;
    std::cout << std::endl;

    // TASK 2: Extract a column (conceptually)
    std::cout << "Task 2 - Column Slice (Column 5):" << std::endl;
    std::cout << "Elements in column 5: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << tensor(i, 5) << " ";
    }
    std::cout << std::endl;
    std::cout << "Note: Column access has stride equal to row count" << std::endl;
    std::cout << std::endl;

    // TASK 3: Extract a sub-matrix (top-left 4x4)
    std::cout << "Task 3 - Sub-matrix (Top-Left 4x4):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Extract a sub-matrix (bottom-right 4x4)
    std::cout << "Task 4 - Sub-matrix (Bottom-Right 4x4):" << std::endl;
    for (int i = 4; i < 8; ++i) {
        for (int j = 4; j < 8; ++j) {
            printf("%3d ", static_cast<int>(tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Create a strided slice (every other row)
    std::cout << "Task 5 - Strided Slice (Every Other Row):" << std::endl;
    for (int i = 0; i < 8; i += 2) {
        for (int j = 0; j < 4; ++j) {  // Show first 4 columns
            printf("%3d ", static_cast<int>(tensor(i, j)));
        }
        std::cout << "..." << std::endl;
    }
    std::cout << "Note: Strided access uses layout with custom stride" << std::endl;
    std::cout << std::endl;

    // TASK 6: Create a view with different layout (conceptual transpose)
    std::cout << "Task 6 - Transposed View:" << std::endl;
    std::cout << "Accessing as if transposed (column-major access):" << std::endl;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            printf("%3d ", static_cast<int>(tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << "Note: Same data, different access order" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Extract a diagonal view
    std::cout << "=== Challenge: Diagonal View ===" << std::endl;
    std::cout << "Extract the main diagonal:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << tensor(i, i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Note: Diagonal has stride (rows + 1) in row-major layout" << std::endl;
    std::cout << std::endl;

    // PRACTICE: Create specific slices
    std::cout << "=== Practice: Create These Slices ===" << std::endl;
    std::cout << "1. First 4 elements of row 2: ";
    for (int j = 0; j < 4; ++j) {
        std::cout << tensor(2, j) << " ";
    }
    std::cout << std::endl;

    std::cout << "2. Last 4 elements of column 7: ";
    for (int i = 4; i < 8; ++i) {
        std::cout << tensor(i, 7) << " ";
    }
    std::cout << std::endl;

    std::cout << "3. Center 4x4 block (rows 2-5, cols 2-5):" << std::endl;
    for (int i = 2; i < 6; ++i) {
        for (int j = 2; j < 6; ++j) {
            printf("%3d ", static_cast<int>(tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // SLICING WITH LAYOUTS
    std::cout << "=== Slicing with Layouts ===" << std::endl;
    std::cout << "To create a proper slice view in CuTe:" << std::endl;
    std::cout << "1. Create a sub-layout for the slice region" << std::endl;
    std::cout << "2. Use make_tensor with the same pointer + offset" << std::endl;
    std::cout << "3. The new tensor views the same data with different layout" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Slicing extracts sub-regions of tensors" << std::endl;
    std::cout << "2. Slices are views, not copies (no data duplication)" << std::endl;
    std::cout << "3. Row slices have stride 1 in row-major layouts" << std::endl;
    std::cout << "4. Column slices have stride = num_rows in row-major" << std::endl;
    std::cout << "5. Layout composition enables complex slicing" << std::endl;

    return 0;
}
