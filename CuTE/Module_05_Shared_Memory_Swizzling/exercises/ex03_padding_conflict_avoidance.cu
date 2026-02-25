/**
 * Exercise 03: Padding for Conflict Avoidance
 * 
 * Objective: Learn to use padding to avoid bank conflicts
 *            in shared memory access patterns
 * 
 * Tasks:
 * 1. Understand how padding works
 * 2. Calculate padding requirements
 * 3. Implement padded layouts
 * 4. Analyze effectiveness
 * 
 * Key Concepts:
 * - Padding: Extra elements to change stride
 * - Bank Mapping: Changes with padded stride
 * - Trade-off: Memory overhead for performance
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 03: Padding for Conflict Avoidance ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Padding concept
    std::cout << "Task 1 - Padding Concept:" << std::endl;
    std::cout << "Add extra elements to change memory stride" << std::endl;
    std::cout << "Breaks the pattern that causes conflicts" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Example: 32x32 matrix" << std::endl;
    std::cout << "  Without padding: stride = 32 (conflicts!)" << std::endl;
    std::cout << "  With padding: stride = 33 (no conflicts)" << std::endl;
    std::cout << std::endl;

    // TASK 2: Compare padded vs non-padded
    std::cout << "Task 2 - Padded vs Non-Padded:" << std::endl;
    std::cout << std::endl;

    // Non-padded layout
    auto unpadded_layout = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
    std::cout << "Non-padded 32x32 layout:" << std::endl;
    std::cout << "  Stride: " << unpadded_layout.stride() << std::endl;
    
    // Padded layout (add 1 element per row)
    auto padded_layout = make_layout(make_shape(Int<32>{}, Int<32>{}), 
                                      make_stride(Int<33>{}, Int<1>{}));
    std::cout << "Padded 32x32 layout (stride=33):" << std::endl;
    std::cout << "  Stride: " << padded_layout.stride() << std::endl;
    std::cout << std::endl;

    // Analyze column access for both
    std::cout << "Column access bank analysis:" << std::endl;
    std::cout << std::endl;

    std::cout << "Non-padded (first 8 threads):" << std::endl;
    for (int t = 0; t < 8; ++t) {
        int addr = t * 32;  // Column stride = 32
        int bank = addr % 32;
        std::cout << "  Thread " << t << " -> Bank " << bank << std::endl;
    }
    std::cout << "  Result: All threads access bank 0 (32-way conflict!)" << std::endl;
    std::cout << std::endl;

    std::cout << "Padded (first 8 threads):" << std::endl;
    for (int t = 0; t < 8; ++t) {
        int addr = t * 33;  // Column stride = 33
        int bank = addr % 32;
        std::cout << "  Thread " << t << " -> Bank " << bank << std::endl;
    }
    std::cout << "  Result: Different banks (no conflict!)" << std::endl;
    std::cout << std::endl;

    // TASK 3: Calculate padding overhead
    std::cout << "Task 3 - Padding Overhead:" << std::endl;
    
    int rows = 32;
    int cols = 32;
    int padded_cols = 33;
    
    int unpadded_elements = rows * cols;
    int padded_elements = rows * padded_cols;
    int overhead_elements = padded_elements - unpadded_elements;
    float overhead_percent = (float)overhead_elements / unpadded_elements * 100;
    
    std::cout << "Original size: " << unpadded_elements << " elements" << std::endl;
    std::cout << "Padded size: " << padded_elements << " elements" << std::endl;
    std::cout << "Overhead: " << overhead_elements << " elements (" << overhead_percent << "%)" << std::endl;
    std::cout << std::endl;

    // TASK 4: Different padding amounts
    std::cout << "Task 4 - Different Padding Amounts:" << std::endl;
    
    struct PaddingConfig {
        int cols;
        int padded_cols;
        const char* description;
    };
    
    PaddingConfig configs[] = {
        {32, 32, "No padding"},
        {32, 33, "Minimal padding (+1)"},
        {32, 34, "Extra padding (+2)"},
        {32, 40, "Large padding (+8)"},
    };
    
    std::cout << "| Config      | Elements | Overhead |" << std::endl;
    std::cout << "|-------------|----------|----------|" << std::endl;
    for (auto& cfg : configs) {
        int overhead = 32 * (cfg.padded_cols - cfg.cols);
        float pct = (float)overhead / (32 * cfg.cols) * 100;
        printf("| %-11s | %8d | %7.1f%% |\n", cfg.description, 
               32 * cfg.padded_cols, pct);
    }
    std::cout << std::endl;

    // TASK 5: Padding for matrix transpose
    std::cout << "Task 5 - Padding for Matrix Transpose:" << std::endl;
    std::cout << "Transpose requires both row and column access" << std::endl;
    std::cout << "Padding helps both phases:" << std::endl;
    std::cout << "  - Read phase: Row access (usually fine)" << std::endl;
    std::cout << "  - Write phase: Column access (needs padding)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate optimal padding
    std::cout << "=== Challenge: Optimal Padding ===" << std::endl;
    std::cout << "For a 64x64 matrix with 32 banks:" << std::endl;
    std::cout << "  Minimum padding to avoid conflicts: +1 (stride = 65)" << std::endl;
    std::cout << "  Overhead: " << 64 << " elements (" << (64.0 / (64*64) * 100) << "%)" << std::endl;
    std::cout << std::endl;

    std::cout << "For a 128x128 matrix:" << std::endl;
    std::cout << "  Minimum padding: +1 (stride = 129)" << std::endl;
    std::cout << "  Overhead: " << 128 << " elements (" << (128.0 / (128*128) * 100) << "%)" << std::endl;
    std::cout << std::endl;

    // PADDING IMPLEMENTATION
    std::cout << "=== Padding Implementation ===" << std::endl;
    std::cout << R"(
// Padded shared memory declaration
__shared__ float smem[32 * 33];  // 32 rows, 33 columns (padded)

// Access macro
#define SMEM(row, col) smem[(row) * 33 + (col)]

// In kernel:
// Load from global (coalesced)
for (int j = 0; j < 32; ++j) {
    SMEM(threadIdx.y, j) = global[threadIdx.y * 32 + j];
}

__syncthreads();

// Read from shared (may be column access)
float val = SMEM(j, i);  // Padded stride avoids conflicts
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Padding changes stride to avoid conflicts" << std::endl;
    std::cout << "2. Minimal padding (+1) often sufficient" << std::endl;
    std::cout << "3. Trade memory for performance" << std::endl;
    std::cout << "4. Essential for transpose and column access" << std::endl;

    return 0;
}
