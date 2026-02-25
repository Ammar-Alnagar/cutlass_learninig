/**
 * Exercise 07: Swizzle Pattern Design
 * 
 * Objective: Learn to design custom swizzle patterns for specific
 *            access patterns and hardware configurations
 * 
 * Tasks:
 * 1. Analyze access patterns
 * 2. Design appropriate swizzle
 * 3. Verify conflict avoidance
 * 4. Optimize for specific cases
 * 
 * Key Concepts:
 * - Pattern Analysis: Understand access characteristics
 * - Bit Selection: Choose which bits to XOR
 * - Verification: Test swizzle effectiveness
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 07: Swizzle Pattern Design ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Access pattern analysis
    std::cout << "Task 1 - Access Pattern Analysis:" << std::endl;
    std::cout << "Steps to design swizzle:" << std::endl;
    std::cout << "  1. Identify access stride" << std::endl;
    std::cout << "  2. Find which bits are constant" << std::endl;
    std::cout << "  3. Choose XOR bits to vary constant bits" << std::endl;
    std::cout << "  4. Verify distribution across banks" << std::endl;
    std::cout << std::endl;

    // TASK 2: Design for column access (stride = 32)
    std::cout << "Task 2 - Column Access Swizzle Design:" << std::endl;
    std::cout << "Addresses: 0, 32, 64, 96, 128, ..." << std::endl;
    std::cout << "Binary: 000000, 100000, 1000000, 1100000, 10000000, ..." << std::endl;
    std::cout << std::endl;
    
    std::cout << "Observation: Lower 5 bits are always 0" << std::endl;
    std::cout << "Solution: XOR with upper bits to vary lower bits" << std::endl;
    std::cout << "  Swizzle: addr XOR (addr >> 5)" << std::endl;
    std::cout << std::endl;

    // Verify the swizzle
    std::cout << "Verification:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int addr = i * 32;
        int swizzled = addr ^ (addr >> 5);
        int bank = swizzled % 32;
        std::cout << "  Addr " << addr << " -> Swizzled " << swizzled 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "  All different banks! (SUCCESS)" << std::endl;
    std::cout << std::endl;

    // TASK 3: Design for diagonal access (stride = 33)
    std::cout << "Task 3 - Diagonal Access Swizzle Design:" << std::endl;
    std::cout << "Addresses: 0, 33, 66, 99, 132, ..." << std::endl;
    std::cout << std::endl;

    // Analyze bank distribution without swizzle
    std::cout << "Without swizzle:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int addr = i * 33;
        int bank = addr % 32;
        std::cout << "  Addr " << addr << " -> Bank " << bank << std::endl;
    }
    std::cout << std::endl;

    // Try different swizzle patterns
    std::cout << "With swizzle (addr XOR (addr >> 3)):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int addr = i * 33;
        int swizzled = addr ^ (addr >> 3);
        int bank = swizzled % 32;
        std::cout << "  Addr " << addr << " -> Swizzled " << swizzled 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Design for 2D thread block access
    std::cout << "Task 4 - 2D Thread Block Swizzle Design:" << std::endl;
    std::cout << "8x8 thread block accessing 8x8 tile:" << std::endl;
    std::cout << std::endl;

    auto analyze_2d_access = [&](const char* pattern, int rows, int cols, int stride) {
        int bank_counts[32] = {0};
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int addr = i * stride + j;
                int swizzled = addr ^ (addr >> 5);
                int bank = swizzled % 32;
                bank_counts[bank]++;
            }
        }
        
        int max_conflict = 0;
        for (int b = 0; b < 32; ++b) {
            if (bank_counts[b] > max_conflict) {
                max_conflict = bank_counts[b];
            }
        }
        
        std::cout << pattern << ":" << std::endl;
        std::cout << "  Max conflict: " << max_conflict << "-way" << std::endl;
        std::cout << std::endl;
    };

    analyze_2d_access("Row-major 8x8", 8, 8, 64);
    analyze_2d_access("Column-major 8x8", 8, 8, 1);
    analyze_2d_access("Transposed 8x8", 8, 8, 65);

    // TASK 5: Optimize for specific hardware
    std::cout << "Task 5 - Hardware-Specific Optimization:" << std::endl;
    std::cout << "sm_80+ (A100): 32 banks, 4-byte words" << std::endl;
    std::cout << "sm_90 (H100): 32 banks, enhanced swizzle support" << std::endl;
    std::cout << std::endl;

    std::cout << "For A100:" << std::endl;
    std::cout << "  Use XOR with bit 5 for 32-element stride" << std::endl;
    std::cout << "  Consider CuTe's built-in swizzle layouts" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Design for complex pattern
    std::cout << "=== Challenge: Complex Pattern ===" << std::endl;
    std::cout << "Access pattern: 4x4 threads, each accessing 4 consecutive elements" << std::endl;
    std::cout << "Stride between thread rows: 64 elements" << std::endl;
    std::cout << std::endl;
    std::cout << "Design steps:" << std::endl;
    std::cout << "  1. Calculate all addresses" << std::endl;
    std::cout << "  2. Find bank distribution" << std::endl;
    std::cout << "  3. Identify conflicts" << std::endl;
    std::cout << "  4. Design XOR pattern" << std::endl;
    std::cout << "  5. Verify improvement" << std::endl;
    std::cout << std::endl;

    // SWIZZLE DESIGN TEMPLATE
    std::cout << "=== Swizzle Design Template ===" << std::endl;
    std::cout << R"(
// Template for designing swizzle patterns
__device__ __forceinline__ int design_swizzle(int addr, int stride) {
    // Step 1: Identify constant bits in stride
    // Step 2: Choose shift amount
    int shift = 5;  // For 32 banks
    
    // Step 3: Apply XOR
    return addr ^ (addr >> shift);
}

// Example: Column access swizzle
__device__ __forceinline__ int column_swizzle(int addr) {
    return addr ^ (addr >> 5);
}

// Example: Diagonal access swizzle  
__device__ __forceinline__ int diagonal_swizzle(int addr) {
    return addr ^ (addr >> 3);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Analyze access pattern first" << std::endl;
    std::cout << "2. Identify constant bits in addresses" << std::endl;
    std::cout << "3. XOR with appropriate shift" << std::endl;
    std::cout << "4. Always verify with test cases" << std::endl;

    return 0;
}
