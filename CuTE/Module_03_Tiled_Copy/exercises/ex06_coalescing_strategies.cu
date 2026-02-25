/**
 * Exercise 06: Coalescing Strategies
 * 
 * Objective: Master memory coalescing strategies for optimal
 *            memory bandwidth utilization
 * 
 * Tasks:
 * 1. Understand what coalescing means
 * 2. Identify coalesced vs uncoalesced patterns
 * 3. Practice with different layouts
 * 4. Optimize access patterns
 * 
 * Key Concepts:
 * - Coalesced Access: Consecutive threads access consecutive addresses
 * - Memory Warp: 32 threads that execute together
 * - Bandwidth: Coalesced = maximum, Uncoalesced = reduced
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 06: Coalescing Strategies ===" << std::endl;
    std::cout << std::endl;

    float data[128];
    for (int i = 0; i < 128; ++i) {
        data[i] = static_cast<float>(i);
    }

    // TASK 1: Coalesced row access with row-major layout
    std::cout << "Task 1 - Coalesced Row Access:" << std::endl;
    auto rm_layout = make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{});
    auto rm_tensor = make_tensor(make_gmem_ptr(data), rm_layout);

    std::cout << "Row-major 8x16 layout: " << rm_layout << std::endl;
    std::cout << "Stride: " << rm_layout.stride() << std::endl;
    std::cout << std::endl;

    std::cout << "Thread access pattern (row 0, threads 0-15):" << std::endl;
    for (int j = 0; j < 16; ++j) {
        int offset = rm_layout(0, j);
        std::cout << "Thread " << j << " -> offset " << offset << std::endl;
    }
    std::cout << "Result: Consecutive offsets (0,1,2,...,15) = COALESCED" << std::endl;
    std::cout << std::endl;

    // TASK 2: Uncoalesced column access with row-major layout
    std::cout << "Task 2 - Uncoalesced Column Access:" << std::endl;
    std::cout << "Thread access pattern (column 0, threads 0-7):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int offset = rm_layout(i, 0);
        std::cout << "Thread " << i << " -> offset " << offset << std::endl;
    }
    std::cout << "Result: Stride of 16 (0,16,32,...,112) = UNCOALESCED" << std::endl;
    std::cout << "Bandwidth reduction: up to 32x slower!" << std::endl;
    std::cout << std::endl;

    // TASK 3: Coalesced column access with column-major layout
    std::cout << "Task 3 - Coalesced Column Access:" << std::endl;
    auto cm_layout = make_layout(make_shape(Int<8>{}, Int<16>{}), GenColMajor{});
    auto cm_tensor = make_tensor(make_gmem_ptr(data), cm_layout);

    std::cout << "Column-major 8x16 layout: " << cm_layout << std::endl;
    std::cout << "Stride: " << cm_layout.stride() << std::endl;
    std::cout << std::endl;

    std::cout << "Thread access pattern (column 0, threads 0-7):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int offset = cm_layout(i, 0);
        std::cout << "Thread " << i << " -> offset " << offset << std::endl;
    }
    std::cout << "Result: Consecutive offsets (0,1,2,...,7) = COALESCED" << std::endl;
    std::cout << std::endl;

    // TASK 4: Coalescing analysis tool
    std::cout << "Task 4 - Coalescing Analysis:" << std::endl;
    
    auto analyze_coalescing = [&](const char* name, Layout const& layout, 
                                   bool row_access, int start, int count) {
        std::cout << name << ":" << std::endl;
        int first_offset = row_access ? layout(start, 0) : layout(0, start);
        int second_offset = row_access ? layout(start, 1) : layout(1, start);
        int stride = second_offset - first_offset;
        
        std::cout << "  Access stride: " << stride << std::endl;
        std::cout << "  Coalesced: " << (stride == 1 ? "YES" : "NO") << std::endl;
        std::cout << std::endl;
    };

    analyze_coalescing("Row-major, row access", rm_layout, true, 0, 16);
    analyze_coalescing("Row-major, column access", rm_layout, false, 0, 8);
    analyze_coalescing("Col-major, row access", cm_layout, true, 0, 16);
    analyze_coalescing("Col-major, column access", cm_layout, false, 0, 8);

    // TASK 5: Padded layout for bank conflict avoidance
    std::cout << "Task 5 - Padded Layout:" << std::endl;
    auto padded_layout = make_layout(make_shape(Int<8>{}, Int<16>{}), 
                                      make_stride(Int<17>{}, Int<1>{}));
    
    std::cout << "Padded layout (stride 17 instead of 16):" << std::endl;
    std::cout << "  Layout: " << padded_layout << std::endl;
    std::cout << "  Purpose: Avoid bank conflicts in shared memory" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Optimize this access pattern
    std::cout << "=== Challenge: Optimize Access Pattern ===" << std::endl;
    std::cout << "Problem: Need to access data column-wise, but data is row-major" << std::endl;
    std::cout << std::endl;
    std::cout << "Solutions:" << std::endl;
    std::cout << "1. Transpose data to column-major before processing" << std::endl;
    std::cout << "2. Use shared memory with column-major layout" << std::endl;
    std::cout << "3. Restructure algorithm to access row-wise" << std::endl;
    std::cout << std::endl;

    // COALESCING BEST PRACTICES
    std::cout << "=== Coalescing Best Practices ===" << std::endl;
    std::cout << "1. Match access pattern to layout (row access for row-major)" << std::endl;
    std::cout << "2. Use vectorized loads when possible (128-bit)" << std::endl;
    std::cout << "3. Avoid column access in row-major layouts" << std::endl;
    std::cout << "4. Consider data transposition for fixed access patterns" << std::endl;
    std::cout << "5. Use shared memory to reorganize data" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Coalesced access = consecutive threads, consecutive addresses" << std::endl;
    std::cout << "2. Row-major: row access is coalesced" << std::endl;
    std::cout << "3. Column-major: column access is coalesced" << std::endl;
    std::cout << "4. Uncoalesced access can reduce bandwidth 32x" << std::endl;
    std::cout << "5. Always analyze and optimize access patterns" << std::endl;

    return 0;
}
