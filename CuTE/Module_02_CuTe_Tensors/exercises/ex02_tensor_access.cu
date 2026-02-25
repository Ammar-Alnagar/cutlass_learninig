/**
 * Exercise 02: Tensor Access Patterns
 * 
 * Objective: Understand how different access patterns affect performance
 *            and learn to write coalesced access patterns
 * 
 * Tasks:
 * 1. Practice row-wise and column-wise access
 * 2. Identify coalesced vs uncoalesced patterns
 * 3. Understand the impact of layout on access efficiency
 * 4. Optimize access patterns for GPU memory
 * 
 * Key Concepts:
 * - Coalesced Access: Consecutive threads access consecutive addresses
 * - Memory Throughput: Efficient access maximizes bandwidth
 * - Access Stride: Distance between consecutive accesses
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

void print_access_pattern(const char* pattern_name, auto const& tensor, 
                          int start_i, int start_j, int count, bool row_wise) {
    std::cout << pattern_name << ": ";
    for (int k = 0; k < count; ++k) {
        int i = row_wise ? start_i : start_i + k;
        int j = row_wise ? start_j + k : start_j;
        std::cout << tensor(i, j) << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Exercise 02: Tensor Access Patterns ===" << std::endl;
    std::cout << std::endl;

    // Create a 8x8 tensor for access pattern experiments
    float data[64];
    for (int i = 0; i < 64; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Create row-major tensor
    auto layout_rm = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto tensor_rm = make_tensor(make_gmem_ptr(data), layout_rm);

    std::cout << "Row-Major 8x8 Tensor Layout:" << std::endl;
    print(tensor_rm.layout());
    std::cout << std::endl;

    // TASK 1: Row-wise access (coalesced for row-major)
    std::cout << "Task 1 - Row-wise Access (COALESCED):" << std::endl;
    std::cout << "Accessing row 0, columns 0-7:" << std::endl;
    for (int j = 0; j < 8; ++j) {
        std::cout << tensor_rm(0, j) << " ";
    }
    std::cout << std::endl;
    std::cout << "Memory offsets: ";
    for (int j = 0; j < 8; ++j) {
        std::cout << tensor_rm.layout()(0, j) << " ";
    }
    std::cout << " <- Consecutive! (stride = 1)" << std::endl;
    std::cout << std::endl;

    // TASK 2: Column-wise access (uncoalesced for row-major)
    std::cout << "Task 2 - Column-wise Access (UNCOALESCED):" << std::endl;
    std::cout << "Accessing column 0, rows 0-7:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << tensor_rm(i, 0) << " ";
    }
    std::cout << std::endl;
    std::cout << "Memory offsets: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << tensor_rm.layout()(i, 0) << " ";
    }
    std::cout << " <- Stride of 8! (not consecutive)" << std::endl;
    std::cout << std::endl;

    // TASK 3: Create column-major tensor for comparison
    auto layout_cm = make_layout(make_shape(Int<8>{}, Int<8>{}), GenColMajor{});
    auto tensor_cm = make_tensor(make_gmem_ptr(data), layout_cm);

    std::cout << "Task 3 - Column-Major 8x8 Tensor Layout:" << std::endl;
    print(tensor_cm.layout());
    std::cout << std::endl;

    // TASK 4: Compare access patterns
    std::cout << "Task 4 - Access Pattern Comparison:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Row-Major Tensor:" << std::endl;
    std::cout << "  Row access: COALESCED (stride 1)" << std::endl;
    std::cout << "  Column access: UNCOALESCED (stride 8)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Column-Major Tensor:" << std::endl;
    std::cout << "  Row access: UNCOALESCED (stride 8)" << std::endl;
    std::cout << "  Column access: COALESCED (stride 1)" << std::endl;
    std::cout << std::endl;

    // TASK 5: Diagonal access pattern
    std::cout << "Task 5 - Diagonal Access:" << std::endl;
    std::cout << "Accessing diagonal elements:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << tensor_rm(i, i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Memory offsets: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << tensor_rm.layout()(i, i) << " ";
    }
    std::cout << " <- Stride of 9 (row + column stride)" << std::endl;
    std::cout << std::endl;

    // TASK 6: Tiled access pattern
    std::cout << "Task 6 - Tiled Access (2x2 tiles):" << std::endl;
    std::cout << "Accessing 2x2 tile at position (0, 0):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << tensor_rm(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // CHALLENGE: Optimize access pattern
    std::cout << "=== Challenge: Optimize This Access ===" << std::endl;
    std::cout << "Given a row-major tensor, which access is better?" << std::endl;
    std::cout << std::endl;
    std::cout << "Option A: for (i) for (j) tensor(i, j)" << std::endl;
    std::cout << "Option B: for (j) for (i) tensor(i, j)" << std::endl;
    std::cout << std::endl;
    std::cout << "Answer: Option A (outer loop over rows, inner loop over columns)" << std::endl;
    std::cout << "Reason: Inner loop accesses consecutive memory locations" << std::endl;
    std::cout << std::endl;

    // PERFORMANCE TIPS
    std::cout << "=== Performance Tips ===" << std::endl;
    std::cout << "1. Match access pattern to layout (row access for row-major)" << std::endl;
    std::cout << "2. Coalesced access maximizes memory throughput" << std::endl;
    std::cout << "3. Uncoalesced access can reduce bandwidth by 32x" << std::endl;
    std::cout << "4. Consider transposing data if access pattern is fixed" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Access pattern must match layout for coalescing" << std::endl;
    std::cout << "2. Row-major: row access is coalesced" << std::endl;
    std::cout << "3. Column-major: column access is coalesced" << std::endl;
    std::cout << "4. Coalesced access is critical for GPU performance" << std::endl;

    return 0;
}
