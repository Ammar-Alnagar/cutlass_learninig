/**
 * Exercise 02: Bank Conflict Analysis
 * 
 * Objective: Learn to identify and analyze bank conflicts
 *            in shared memory access patterns
 * 
 * Tasks:
 * 1. Understand bank conflict causes
 * 2. Identify conflicting access patterns
 * 3. Calculate conflict severity
 * 4. Practice conflict detection
 * 
 * Key Concepts:
 * - Bank Conflict: Multiple threads access same bank
 * - Serialization: Conflicts cause sequential access
 * - 32-way Banked: sm_80+ has 32 banks
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 02: Bank Conflict Analysis ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Bank conflict basics
    std::cout << "Task 1 - Bank Conflict Basics:" << std::endl;
    std::cout << "When multiple threads access addresses in the same bank:" << std::endl;
    std::cout << "  - Accesses are serialized" << std::endl;
    std::cout << "  - Performance reduced proportionally" << std::endl;
    std::cout << "  - 2-way conflict = 2x slower" << std::endl;
    std::cout << "  - 32-way conflict = 32x slower" << std::endl;
    std::cout << std::endl;

    // TASK 2: Analyze row access pattern
    std::cout << "Task 2 - Row Access Analysis:" << std::endl;
    std::cout << "32 threads accessing row of 32 floats (128 bytes):" << std::endl;
    std::cout << std::endl;

    std::cout << "Thread -> Address -> Bank:" << std::endl;
    for (int t = 0; t < 32; ++t) {
        int addr = t * 4;  // Each thread accesses consecutive 4-byte word
        int bank = (addr / 4) % 32;
        std::cout << "  Thread " << t << " -> Addr " << addr 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "Result: NO CONFLICT (each thread accesses different bank)" << std::endl;
    std::cout << std::endl;

    // TASK 3: Analyze column access pattern (conflict!)
    std::cout << "Task 3 - Column Access Analysis:" << std::endl;
    std::cout << "32 threads accessing column of 32x32 matrix:" << std::endl;
    std::cout << std::endl;

    int matrix_stride = 32;  // 32 elements per row
    
    std::cout << "Thread -> Address -> Bank:" << std::endl;
    for (int t = 0; t < 32; ++t) {
        int addr = t * matrix_stride * 4;  // Column access: stride = row_width
        int bank = (addr / 4) % 32;
        std::cout << "  Thread " << t << " -> Addr " << addr 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "Result: 32-WAY CONFLICT (all threads access bank 0)!" << std::endl;
    std::cout << "Performance: 32x slower than no-conflict case" << std::endl;
    std::cout << std::endl;

    // TASK 4: Analyze diagonal access
    std::cout << "Task 4 - Diagonal Access Analysis:" << std::endl;
    std::cout << "32 threads accessing diagonal of 32x32 matrix:" << std::endl;
    std::cout << std::endl;

    std::cout << "Thread -> Address -> Bank:" << std::endl;
    for (int t = 0; t < 32; ++t) {
        int addr = t * (matrix_stride + 1) * 4;  // Diagonal: stride = row_width + 1
        int bank = (addr / 4) % 32;
        std::cout << "  Thread " << t << " -> Addr " << addr 
                  << " -> Bank " << bank << std::endl;
    }
    std::cout << "Result: Pattern depends on stride" << std::endl;
    std::cout << std::endl;

    // TASK 5: Conflict severity calculation
    std::cout << "Task 5 - Conflict Severity:" << std::endl;
    
    auto analyze_conflicts = [&](const char* pattern, int stride, int num_threads) {
        int bank_counts[32] = {0};
        
        for (int t = 0; t < num_threads; ++t) {
            int addr = t * stride * 4;
            int bank = (addr / 4) % 32;
            bank_counts[bank]++;
        }
        
        int max_conflict = 0;
        for (int b = 0; b < 32; ++b) {
            if (bank_counts[b] > max_conflict) {
                max_conflict = bank_counts[b];
            }
        }
        
        std::cout << pattern << ":" << std::endl;
        std::cout << "  Max conflict: " << max_conflict << "-way" << std::endl;
        std::cout << "  Performance: " << max_conflict << "x slower than optimal" << std::endl;
        std::cout << std::endl;
    };

    analyze_conflicts("Row access (stride=1)", 1, 32);
    analyze_conflicts("Column access (stride=32)", 32, 32);
    analyze_conflicts("Diagonal access (stride=33)", 33, 32);
    analyze_conflicts("Padded row (stride=33)", 33, 32);

    // CHALLENGE: Identify conflicts
    std::cout << "=== Challenge: Identify Conflicts ===" << std::endl;
    std::cout << "Pattern A: 32 threads, stride = 16" << std::endl;
    std::cout << "  Answer: 2-way conflict (banks repeat every 32 threads)" << std::endl;
    std::cout << std::endl;

    std::cout << "Pattern B: 32 threads, stride = 17" << std::endl;
    std::cout << "  Answer: No conflict (17 is coprime with 32)" << std::endl;
    std::cout << std::endl;

    std::cout << "Pattern C: 32 threads, stride = 64" << std::endl;
    std::cout << "  Answer: 32-way conflict (all same bank)" << std::endl;
    std::cout << std::endl;

    // CONFLICT AVOIDANCE STRATEGIES
    std::cout << "=== Conflict Avoidance Strategies ===" << std::endl;
    std::cout << "1. Padding: Add extra elements to change stride" << std::endl;
    std::cout << "2. Swizzling: XOR-based address transformation" << std::endl;
    std::cout << "3. Access pattern: Change algorithm to avoid conflicts" << std::endl;
    std::cout << "4. Data layout: Organize data for conflict-free access" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Bank conflicts serialize memory access" << std::endl;
    std::cout << "2. Column access in row-major causes conflicts" << std::endl;
    std::cout << "3. Stride determines conflict pattern" << std::endl;
    std::cout << "4. Avoid strides that are multiples of 32" << std::endl;

    return 0;
}
