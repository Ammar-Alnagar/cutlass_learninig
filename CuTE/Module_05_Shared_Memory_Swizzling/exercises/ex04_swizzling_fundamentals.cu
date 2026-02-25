/**
 * Exercise 04: Swizzling Fundamentals
 * 
 * Objective: Understand swizzling concepts for bank conflict avoidance
 * 
 * Tasks:
 * 1. Learn what swizzling is
 * 2. Understand XOR-based address transformation
 * 3. See how swizzling spreads accesses
 * 4. Compare with padding
 * 
 * Key Concepts:
 * - Swizzling: XOR-based address remapping
 * - No Overhead: Unlike padding
 * - Conflict Distribution: Spreads accesses across banks
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: Swizzling Fundamentals ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Swizzling concept
    std::cout << "Task 1 - Swizzling Concept:" << std::endl;
    std::cout << "Swizzling applies XOR transformation to addresses" << std::endl;
    std::cout << "Purpose: Spread accesses across banks" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Basic swizzle formula:" << std::endl;
    std::cout << "  swizzled_addr = addr XOR (addr >> shift)" << std::endl;
    std::cout << "  Common: XOR bits to redistribute bank mapping" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simple XOR swizzle demonstration
    std::cout << "Task 2 - Simple XOR Swizzle:" << std::endl;
    std::cout << "Using XOR with bit 5 (for 32 banks):" << std::endl;
    std::cout << std::endl;

    std::cout << "Address -> Bank (no swizzle) vs Bank (swizzled):" << std::endl;
    for (int addr = 0; addr < 32; ++addr) {
        int bank_normal = addr % 32;
        int swizzled = addr ^ (addr >> 2);  // Simple XOR swizzle
        int bank_swizzled = swizzled % 32;
        
        std::cout << "  Addr " << addr << ": Bank " << bank_normal 
                  << " -> Swizzled Bank " << bank_swizzled << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Column access with swizzling
    std::cout << "Task 3 - Column Access with Swizzling:" << std::endl;
    std::cout << "32 threads accessing column (stride = 32):" << std::endl;
    std::cout << std::endl;

    std::cout << "Without swizzle:" << std::endl;
    for (int t = 0; t < 8; ++t) {
        int addr = t * 32;
        int bank = addr % 32;
        std::cout << "  Thread " << t << " -> Addr " << addr << " -> Bank " << bank << std::endl;
    }
    std::cout << "  All threads -> Bank 0 (CONFLICT!)" << std::endl;
    std::cout << std::endl;

    std::cout << "With swizzle (XOR bit 5):" << std::endl;
    for (int t = 0; t < 8; ++t) {
        int addr = t * 32;
        int swizzled = addr ^ (addr >> 5);
        int bank = swizzled % 32;
        std::cout << "  Thread " << t << " -> Addr " << addr 
                  << " -> Swizzled " << swizzled << " -> Bank " << bank << std::endl;
    }
    std::cout << "  Different banks (NO CONFLICT!)" << std::endl;
    std::cout << std::endl;

    // TASK 4: CuTe swizzle layouts
    std::cout << "Task 4 - CuTe Swizzle Layouts:" << std::endl;
    std::cout << "CuTe provides swizzled layout constructors" << std::endl;
    std::cout << std::endl;
    
    // Note: Actual CuTe swizzle API may vary
    std::cout << "Common swizzle modes:" << std::endl;
    std::cout << "  Swizzle<3, 3, 3>: 3-bit XOR for specific patterns" << std::endl;
    std::cout << "  Swizzle<2, 3, 3>: 2-bit XOR variant" << std::endl;
    std::cout << "  Custom: Define your own swizzle pattern" << std::endl;
    std::cout << std::endl;

    // TASK 5: Swizzle vs Padding comparison
    std::cout << "Task 5 - Swizzle vs Padding:" << std::endl;
    
    std::cout << std::endl;
    std::cout << "| Aspect        | Padding      | Swizzling    |" << std::endl;
    std::cout << "|---------------|--------------|--------------|" << std::endl;
    std::cout << "| Memory        | Extra needed | No overhead  |" << std::endl;
    std::cout << "| Complexity    | Simple       | Moderate     |" << std::endl;
    std::cout << "| Effectiveness | High         | High         |" << std::endl;
    std::cout << "| Address calc  | Simple       | XOR required |" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Design swizzle pattern
    std::cout << "=== Challenge: Design Swizzle Pattern ===" << std::endl;
    std::cout << "For column access with stride 32:" << std::endl;
    std::cout << "  Problem: All addresses have bit 5 = 0" << std::endl;
    std::cout << "  Solution: XOR with higher bits to spread" << std::endl;
    std::cout << "  Example: addr XOR (addr >> 5)" << std::endl;
    std::cout << std::endl;

    // SWIZZLE IMPLEMENTATION
    std::cout << "=== Swizzle Implementation ===" << std::endl;
    std::cout << R"(
// Simple XOR swizzle function
__device__ __forceinline__ int swizzle_addr(int addr) {
    return addr ^ (addr >> 5);
}

// In kernel:
__shared__ float smem[1024];

int addr = threadIdx.y * 32 + threadIdx.x;
int swizzled_addr = swizzle_addr(addr);

smem[swizzled_addr] = value;  // Swizzled write
value = smem[swizzled_addr];  // Swizzled read
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Swizzling uses XOR to remap addresses" << std::endl;
    std::cout << "2. No memory overhead (unlike padding)" << std::endl;
    std::cout << "3. Spreads accesses across banks" << std::endl;
    std::cout << "4. CuTe provides swizzle layout helpers" << std::endl;

    return 0;
}
