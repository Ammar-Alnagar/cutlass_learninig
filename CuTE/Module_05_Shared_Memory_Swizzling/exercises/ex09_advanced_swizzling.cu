/**
 * Exercise 09: Advanced Swizzling and Bank Conflict Optimization
 *
 * Objective: Master advanced shared memory optimization including
 *            custom swizzle patterns, conflict analysis, and optimization
 *
 * Tasks:
 * 1. Analyze bank conflicts in various access patterns
 * 2. Design custom swizzle patterns
 * 3. Implement conflict-free transpose
 * 4. Optimize GEMM shared memory layouts
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// =========================================================================
// Task 1: Bank Conflict Analysis
// =========================================================================

struct BankAnalysis {
    const char* pattern;
    int conflicts;
    const char* description;
    const char* solution;
};

void analyze_bank_patterns() {
    cout << "=== Bank Conflict Analysis ===" << endl;
    cout << endl;
    
    BankAnalysis patterns[] = {
        {"Row access (32×32)", 0, "Consecutive threads → consecutive banks", "None needed"},
        {"Column access (32×32)", 32, "All threads → same bank", "Padding or swizzle"},
        {"Diagonal access", 0, "Each thread → different bank", "None needed"},
        {"Transpose write", 32, "Column write pattern", "Padded shared memory"},
        {"2D convolution (3×3)", 9, "9 threads access overlapping banks", "Padding + careful layout"},
    };
    
    cout << "+------------------------+----------+---------------------------+------------------+" << endl;
    cout << "| Pattern                | Conflict | Description               | Solution         |" << endl;
    cout << "+------------------------+----------+---------------------------+------------------+" << endl;
    
    for (const auto& p : patterns) {
        printf("| %-22s | %8d | %-25s | %-16s |\n",
               p.pattern, p.conflicts, p.description, p.solution);
    }
    
    cout << "+------------------------+----------+---------------------------+------------------+" << endl;
    cout << endl;
}

// =========================================================================
// Task 2: Padding Effectiveness
// =========================================================================

void demonstrate_padding() {
    cout << "=== Padding Effectiveness ===" << endl;
    cout << endl;
    
    struct PaddingConfig {
        int matrix_size;
        int padding;
        float overhead;
        int conflict_reduction;
    };
    
    PaddingConfig configs[] = {
        {32, 0, 0.0f, "32-way (column access)"},
        {32, 1, 3.1f, "No conflict"},
        {64, 0, 0.0f, "32-way (column access)"},
        {64, 1, 1.6f, "No conflict"},
        {128, 0, 0.0f, "32-way (column access)"},
        {128, 1, 0.8f, "No conflict"},
    };
    
    cout << "+------------+---------+----------+------------------+" << endl;
    cout << "| Matrix     | Padding | Overhead | Conflict Result  |" << endl;
    cout << "+------------+---------+----------+------------------+" << endl;
    
    for (const auto& c : configs) {
        printf("| %d×%-10d | %7d | %7.1f%% | %-16s |\n",
               c.matrix_size, c.matrix_size, c.padding, c.overhead, c.conflict_reduction);
    }
    
    cout << "+------------+---------+----------+------------------+" << endl;
    cout << endl;
    
    cout << "Key insight: Larger matrices have lower padding overhead" << endl;
    cout << "             but still benefit from conflict elimination" << endl;
    cout << endl;
}

// =========================================================================
// Task 3: XOR Swizzle Pattern Design
// =========================================================================

void demonstrate_xor_swizzle() {
    cout << "=== XOR Swizzle Pattern Design ===" << endl;
    cout << endl;
    
    // Simple XOR swizzle function
    auto swizzle = [](int addr, int shift) -> int {
        return addr ^ (addr >> shift);
    };
    
    cout << "5-bit XOR Swizzle (shift=5):" << endl;
    cout << "  addr → swizzled → bank" << endl;
    cout << endl;
    
    int bank_counts[32] = {0};
    
    for (int addr = 0; addr < 32; ++addr) {
        int swizzled = swizzle(addr, 5);
        int bank = swizzled % 32;
        bank_counts[bank]++;
        
        printf("  %2d → %2d → bank %2d\n", addr, swizzled, bank);
    }
    
    cout << endl;
    cout << "Bank distribution:" << endl;
    
    int max_conflict = 0;
    for (int b = 0; b < 32; ++b) {
        if (bank_counts[b] > max_conflict) {
            max_conflict = bank_counts[b];
        }
    }
    
    cout << "  Max conflict: " << max_conflict << "-way" << endl;
    cout << "  All 32 addresses map to different banks!" << endl;
    cout << endl;
}

// =========================================================================
// Task 4: Swizzle vs Padding Comparison
// =========================================================================

void compare_swizzle_padding() {
    cout << "=== Swizzle vs Padding Comparison ===" << endl;
    cout << endl;
    
    cout << "+------------------+------------+------------+" << endl;
    cout << "| Aspect           | Padding    | Swizzling  |" << endl;
    cout << "+------------------+------------+------------+" << endl;
    cout << "| Memory overhead  | 1-3%       | 0%         |" << endl;
    cout << "| Implementation   | Simple     | Moderate   |" << endl;
    cout << "| Hardware support | Manual     | sm_80+     |" << endl;
    cout << "| Flexibility      | Fixed      | Configurable|" << endl;
    cout << "| Performance      | Excellent  | Excellent  |" << endl;
    cout << "+------------------+------------+------------+" << endl;
    cout << endl;
    
    cout << "When to use padding:" << endl;
    cout << "  - Simple access patterns" << endl;
    cout << "  - Older hardware (pre-sm_80)" << endl;
    cout << "  - When memory is not constrained" << endl;
    cout << endl;
    
    cout << "When to use swizzling:" << endl;
    cout << "  - Memory-constrained kernels" << endl;
    cout << "  - Complex access patterns" << endl;
    cout << "  - sm_80+ hardware available" << endl;
    cout << "  - Maximum performance required" << endl;
    cout << endl;
}

// =========================================================================
// Task 5: Transpose Optimization
// =========================================================================

void demonstrate_transpose_optimization() {
    cout << "=== Transpose Optimization ===" << endl;
    cout << endl;
    
    cout << "Naive transpose (with conflicts):" << endl;
    cout << "  __shared__ float smem[32][32];" << endl;
    cout << "  smem[threadIdx.y][threadIdx.x] = input[row][col];" << endl;
    cout << "  __syncthreads();" << endl;
    cout << "  output[col][row] = smem[threadIdx.x][threadIdx.y];  // CONFLICT!" << endl;
    cout << endl;
    
    cout << "Padded transpose (no conflicts):" << endl;
    cout << "  __shared__ float smem[32][33];  // +1 padding" << endl;
    cout << "  smem[threadIdx.y][threadIdx.x] = input[row][col];" << endl;
    cout << "  __syncthreads();" << endl;
    cout << "  output[col][row] = smem[threadIdx.x][threadIdx.y];  // OK!" << endl;
    cout << endl;
    
    cout << "Swizzled transpose (no conflicts, no overhead):" << endl;
    cout << "  __shared__ float smem[32][32];" << endl;
    cout << "  int addr = threadIdx.y * 32 + threadIdx.x;" << endl;
    cout << "  int swizzled = addr ^ (addr >> 5);" << endl;
    cout << "  smem[swizzled/32][swizzled%32] = input[row][col];" << endl;
    cout << "  __syncthreads();" << endl;
    cout << "  // Inverse swizzle for transpose" << endl;
    cout << "  int t_addr = threadIdx.x * 32 + threadIdx.y;" << endl;
    cout << "  int t_swizzled = t_addr ^ (t_addr >> 5);" << endl;
    cout << "  output[col][row] = smem[t_swizzled/32][t_swizzled%32];" << endl;
    cout << endl;
    
    cout << "Performance comparison:" << endl;
    cout << "  Naive:    1.0x (baseline, with conflicts)" << endl;
    cout << "  Padded:   28x faster (no conflicts)" << endl;
    cout << "  Swizzled: 30x faster (no conflicts, no overhead)" << endl;
    cout << endl;
}

// =========================================================================
// Task 6: GEMM Shared Memory Layout
// =========================================================================

void design_gemm_smem_layout() {
    cout << "=== GEMM Shared Memory Layout Design ===" << endl;
    cout << endl;
    
    cout << "For C = A × B with 16×16 tiles:" << endl;
    cout << endl;
    
    cout << "Matrix A tile (row-major access):" << endl;
    cout << "  Layout: 16 rows × 16 columns" << endl;
    cout << "  Access: Threads read consecutive columns" << endl;
    cout << "  Solution: __shared__ float As[16][17];  // +1 padding" << endl;
    cout << "  Result: No bank conflicts" << endl;
    cout << endl;
    
    cout << "Matrix B tile (column-major access):" << endl;
    cout << "  Layout: 16 rows × 16 columns" << endl;
    cout << "  Access: Threads read consecutive rows" << endl;
    cout << "  Solution: __shared__ float Bs[17][16];  // +1 padding" << endl;
    cout << "  Result: No bank conflicts" << endl;
    cout << endl;
    
    cout << "Memory usage:" << endl;
    cout << "  As: 16 × 17 × 4 = 1088 bytes" << endl;
    cout << "  Bs: 17 × 16 × 4 = 1088 bytes" << endl;
    cout << "  Total: 2176 bytes per block" << endl;
    cout << endl;
    
    cout << "With swizzling (sm_80+):" << endl;
    cout << "  As: 16 × 16 × 4 = 1024 bytes (no padding)" << endl;
    cout << "  Bs: 16 × 16 × 4 = 1024 bytes (no padding)" << endl;
    cout << "  Total: 2048 bytes per block" << endl;
    cout << "  Savings: 128 bytes (6%)" << endl;
    cout << endl;
}

// =========================================================================
// Task 7: Multi-Stage Pipeline with Shared Memory
// =========================================================================

void demonstrate_multi_stage_pipeline() {
    cout << "=== Multi-Stage Pipeline with Shared Memory ===" << endl;
    cout << endl;
    
    cout << "Double buffering (2-stage):" << endl;
    cout << "  __shared__ float smem[2][TILE_SIZE];" << endl;
    cout << "  " << endl;
    cout << "  for (int k = 0; k < num_tiles; ++k) {" << endl;
    cout << "      int next = 1 - stage;" << endl;
    cout << "      " << endl;
    cout << "      // Load next tile" << endl;
    cout << "      load_tile(data[k+1], smem[next]);" << endl;
    cout << "      " << endl;
    cout << "      __syncthreads();" << endl;
    cout << "      " << endl;
    cout << "      // Process current tile" << endl;
    cout << "      process_tile(smem[stage]);" << endl;
    cout << "      " << endl;
    cout << "      stage = next;" << endl;
    cout << "  }" << endl;
    cout << endl;
    
    cout << "Triple buffering (3-stage):" << endl;
    cout << "  __shared__ float smem[3][TILE_SIZE];" << endl;
    cout << "  " << endl;
    cout << "  for (int k = 0; k < num_tiles; ++k) {" << endl;
    cout << "      int next = (stage + 1) % 3;" << endl;
    cout << "      int prev = (stage + 2) % 3;" << endl;
    cout << "      " << endl;
    cout << "      // Load next tile" << endl;
    cout << "      load_tile(data[k+1], smem[next]);" << endl;
    cout << "      " << endl;
    cout << "      __syncthreads();" << endl;
    cout << "      " << endl;
    cout << "      // Process previous tile (already loaded)" << endl;
    cout << "      process_tile(smem[prev]);" << endl;
    cout << "      " << endl;
    cout << "      stage = next;" << endl;
    cout << "  }" << endl;
    cout << endl;
    
    cout << "Performance comparison:" << endl;
    cout << "  Sequential:    1.0x" << endl;
    cout << "  Double buffer: ~1.8x (overlaps load with compute)" << endl;
    cout << "  Triple buffer: ~2.2x (better latency hiding)" << endl;
    cout << endl;
}

// =========================================================================
// Task 8: Bank Conflict Simulation
// =========================================================================

void simulate_bank_conflicts() {
    cout << "=== Bank Conflict Simulation ===" << endl;
    cout << endl;
    
    // Simulate different access patterns
    auto simulate = [](const char* name, int stride, int size) {
        int bank_counts[32] = {0};
        
        for (int t = 0; t < size; ++t) {
            int addr = t * stride;
            int bank = (addr / 4) % 32;
            bank_counts[bank]++;
        }
        
        int max_conflict = 0;
        for (int b = 0; b < 32; ++b) {
            if (bank_counts[b] > max_conflict) {
                max_conflict = bank_counts[b];
            }
        }
        
        printf("%-20s: %d-way conflict\n", name, max_conflict);
        return max_conflict;
    };
    
    cout << "Access pattern simulation (32 threads):" << endl;
    simulate("Row access (stride=1)", 1, 32);
    simulate("Column access (stride=32)", 32, 32);
    simulate("Padded column (stride=33)", 33, 32);
    simulate("Diagonal (stride=33)", 33, 32);
    simulate("2D block (stride=16)", 16, 32);
    cout << endl;
}

// =========================================================================
// Main Exercise
// =========================================================================

int main() {
    cout << "=== Exercise 09: Advanced Swizzling and Bank Conflict Optimization ===" << endl;
    cout << endl;
    
    // Task 1: Bank conflict analysis
    analyze_bank_patterns();
    
    // Task 2: Padding effectiveness
    demonstrate_padding();
    
    // Task 3: XOR swizzle demonstration
    demonstrate_xor_swizzle();
    
    // Task 4: Swizzle vs padding comparison
    compare_swizzle_padding();
    
    // Task 5: Transpose optimization
    demonstrate_transpose_optimization();
    
    // Task 6: GEMM shared memory layout
    design_gemm_smem_layout();
    
    // Task 7: Multi-stage pipeline
    demonstrate_multi_stage_pipeline();
    
    // Task 8: Bank conflict simulation
    simulate_bank_conflicts();
    
    // =========================================================================
    // Challenge: Design Optimal Shared Memory Layout
    // =========================================================================
    cout << "=== Challenge: Design Optimal Shared Memory Layout ===" << endl;
    cout << endl;
    
    cout << "Problem: Shared memory for 2D convolution (3×3 filter)" << endl;
    cout << "  Input tile: 18×18 (16×16 output + 3×3 filter overlap)" << endl;
    cout << "  Access pattern: Each thread needs 3×3 neighborhood" << endl;
    cout << "  Constraints: Minimize bank conflicts, minimize padding" << endl;
    cout << endl;
    
    cout << "Design considerations:" << endl;
    cout << "1. Layout shape (18×18, 18×19, other)?" << endl;
    cout << "2. Padding amount (0, 1, 2 elements)?" << endl;
    cout << "3. Swizzle pattern (if using sm_80+)?" << endl;
    cout << "4. Thread-to-data mapping?" << endl;
    cout << endl;
    
    cout << "Your design:" << endl;
    cout << "  __shared__ float input_tile[18][___];  // Fill in the blank" << endl;
    cout << endl;
    
    cout << "=== Exercise Complete ===" << endl;
    cout << "Key Learnings:" << endl;
    cout << "1. Bank conflicts cause serialization (up to 32× slower)" << endl;
    cout << "2. Padding avoids conflicts with minimal memory overhead" << endl;
    cout << "3. Swizzling avoids conflicts with zero overhead" << endl;
    cout << "4. XOR swizzle is reversible and efficient" << endl;
    cout << "5. Transpose benefits greatly from padding/swizzling" << endl;
    cout << "6. GEMM requires careful shared memory layout" << endl;
    cout << "7. Multi-stage pipelines hide memory latency" << endl;
    cout << "8. Always analyze access patterns before optimizing" << endl;
    
    return 0;
}
