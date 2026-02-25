/**
 * Exercise 05: XOR-Based Swizzling
 * 
 * Objective: Master XOR-based swizzling patterns for shared memory
 * 
 * Tasks:
 * 1. Understand XOR properties for swizzling
 * 2. Implement common XOR patterns
 * 3. Analyze swizzle effectiveness
 * 4. Practice with different configurations
 * 
 * Key Concepts:
 * - XOR: Exclusive OR operation
 * - Reversible: XOR is its own inverse
 * - Bit Manipulation: Operates on individual bits
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 05: XOR-Based Swizzling ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: XOR basics for swizzling
    std::cout << "Task 1 - XOR Basics:" << std::endl;
    std::cout << "XOR properties useful for swizzling:" << std::endl;
    std::cout << "  - A XOR A = 0" << std::endl;
    std::cout << "  - A XOR 0 = A" << std::endl;
    std::cout << "  - (A XOR B) XOR B = A (reversible)" << std::endl;
    std::cout << "  - Spreads bits uniformly" << std::endl;
    std::cout << std::endl;

    // Show XOR truth table
    std::cout << "XOR truth table:" << std::endl;
    std::cout << "  0 XOR 0 = 0" << std::endl;
    std::cout << "  0 XOR 1 = 1" << std::endl;
    std::cout << "  1 XOR 0 = 1" << std::endl;
    std::cout << "  1 XOR 1 = 0" << std::endl;
    std::cout << std::endl;

    // TASK 2: Common XOR swizzle patterns
    std::cout << "Task 2 - Common XOR Patterns:" << std::endl;
    std::cout << std::endl;

    auto xor_swizzle = [](int addr, int shift) {
        return addr ^ (addr >> shift);
    };

    std::cout << "Pattern 1: addr XOR (addr >> 5) for 32 banks" << std::endl;
    for (int addr = 0; addr < 16; ++addr) {
        int swizzled = xor_swizzle(addr, 5);
        std::cout << "  " << addr << " -> " << swizzled << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Pattern 2: addr XOR (addr >> 3) for 8 banks" << std::endl;
    for (int addr = 0; addr < 16; ++addr) {
        int swizzled = xor_swizzle(addr, 3);
        std::cout << "  " << addr << " -> " << swizzled << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Analyze bank distribution
    std::cout << "Task 3 - Bank Distribution Analysis:" << std::endl;
    
    auto analyze_distribution = [&](const char* pattern, int stride, int num_threads, int shift) {
        int bank_counts[32] = {0};
        
        for (int t = 0; t < num_threads; ++t) {
            int addr = t * stride;
            int swizzled = addr ^ (addr >> shift);
            int bank = swizzled % 32;
            bank_counts[bank]++;
        }
        
        int max_conflict = 0;
        int banks_used = 0;
        for (int b = 0; b < 32; ++b) {
            if (bank_counts[b] > 0) banks_used++;
            if (bank_counts[b] > max_conflict) {
                max_conflict = bank_counts[b];
            }
        }
        
        std::cout << pattern << " (stride=" << stride << ", shift=" << shift << "):" << std::endl;
        std::cout << "  Banks used: " << banks_used << "/32" << std::endl;
        std::cout << "  Max conflict: " << max_conflict << "-way" << std::endl;
        std::cout << std::endl;
    };

    analyze_distribution("Column access", 32, 32, 5);
    analyze_distribution("Column access", 32, 32, 3);
    analyze_distribution("Diagonal access", 33, 32, 5);

    // TASK 4: Multi-bit XOR swizzling
    std::cout << "Task 4 - Multi-bit XOR Swizzling:" << std::endl;
    std::cout << "Combine multiple XOR operations:" << std::endl;
    std::cout << std::endl;

    auto multi_xor_swizzle = [](int addr) {
        int result = addr;
        result ^= (addr >> 5);  // XOR with bit 5
        result ^= (addr >> 3);  // XOR with bit 3
        return result;
    };

    std::cout << "Multi-bit swizzle: addr XOR (addr>>5) XOR (addr>>3)" << std::endl;
    for (int addr = 0; addr < 16; ++addr) {
        int swizzled = multi_xor_swizzle(addr);
        std::cout << "  " << addr << " -> " << swizzled 
                  << " (bank " << (swizzled % 32) << ")" << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Reversibility demonstration
    std::cout << "Task 5 - Swizzle Reversibility:" << std::endl;
    std::cout << "XOR swizzle is reversible:" << std::endl;
    std::cout << std::endl;

    for (int addr = 0; addr < 8; ++addr) {
        int swizzled = addr ^ (addr >> 5);
        int reversed = swizzled ^ (swizzled >> 5);
        std::cout << "  " << addr << " -> " << swizzled << " -> " << reversed << std::endl;
    }
    std::cout << "Original = Reversed (XOR is self-inverse)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Design optimal swizzle
    std::cout << "=== Challenge: Optimal Swizzle Design ===" << std::endl;
    std::cout << "For column access with stride 64:" << std::endl;
    std::cout << "  Problem: Addresses are 0, 64, 128, 192, ..." << std::endl;
    std::cout << "  All have bits 0-5 = 0" << std::endl;
    std::cout << "  Solution: XOR with bits 6+ to spread" << std::endl;
    std::cout << "  Try: addr XOR (addr >> 6)" << std::endl;
    std::cout << std::endl;

    // CUDA SWIZZLE IMPLEMENTATION
    std::cout << "=== CUDA Swizzle Implementation ===" << std::endl;
    std::cout << R"(
// XOR swizzle function
__device__ __forceinline__ int swizzle_bank(int addr) {
    return addr ^ (addr >> 5);
}

__device__ __forceinline__ int swizzle_offset(int addr) {
    // Keep lower 5 bits (bank index), swizzle upper bits
    return (addr & 0x1F) | (swizzle_bank(addr >> 5) << 5);
}

// In kernel:
__shared__ float smem[1024];

int logical_addr = threadIdx.y * 32 + threadIdx.x;
int physical_addr = swizzle_offset(logical_addr);

smem[physical_addr] = value;  // Swizzled write
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. XOR spreads bits uniformly" << std::endl;
    std::cout << "2. XOR swizzle is reversible" << std::endl;
    std::cout << "3. Shift amount affects distribution" << std::endl;
    std::cout << "4. Multi-bit XOR provides better spreading" << std::endl;

    return 0;
}
