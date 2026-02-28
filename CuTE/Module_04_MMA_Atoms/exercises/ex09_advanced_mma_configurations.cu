/**
 * Exercise 09: Advanced MMA Configurations and GEMM
 *
 * Objective: Master advanced MMA operations including different
 *            configurations, complete GEMM implementation, and optimization
 *
 * Tasks:
 * 1. Configure MMA atoms for different precisions
 * 2. Implement complete GEMM with MMA
 * 3. Optimize register usage and occupancy
 * 4. Profile and tune MMA performance
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Note: Full CuTe MMA requires CUTLASS installation
// This exercise demonstrates concepts and patterns

using namespace std;

// =========================================================================
// Task 1: MMA Configuration Selection
// =========================================================================

struct MMAConfig {
    const char* name;
    int M, N, K;
    const char* a_type;
    const char* b_type;
    const char* c_type;
    float peak_tflops;
    const char* use_case;
};

MMAConfig configs[] = {
    {"SM80_16x16x16_F16", 16, 16, 16, "FP16", "FP16", "FP32", 312.0f, "General GEMM"},
    {"SM80_16x16x32_F16", 16, 16, 32, "FP16", "FP16", "FP32", 312.0f, "Large K GEMM"},
    {"SM80_16x16x8_BF16", 16, 16, 8,  "BF16", "BF16", "FP32", 156.0f, "ML Training"},
    {"SM80_8x8x4_FP64",   8,  8,  4,  "FP64", "FP64", "FP64", 9.7f,  "Scientific"},
    {"SM80_16x16x32_INT8",16, 16, 32, "INT8", "INT8", "INT32", 624.0f, "Inference"},
};

void print_mma_configs() {
    cout << "=== MMA Configurations for sm_80 (A100) ===" << endl;
    cout << endl;
    
    cout << "+------------------------+-------+------+------+------+---------+----------------+" << endl;
    cout << "| Configuration          | M×N×K | A    | B    | C/D  | TFLOPS  | Use Case       |" << endl;
    cout << "+------------------------+-------+------+------+------+---------+----------------+" << endl;
    
    for (const auto& cfg : configs) {
        printf("| %-22s | %d×%d×%2d | %-4s | %-4s | %-4s | %7.1f | %-14s |\n",
               cfg.name, cfg.M, cfg.N, cfg.K,
               cfg.a_type, cfg.b_type, cfg.c_type,
               cfg.peak_tflops, cfg.use_case);
    }
    
    cout << "+------------------------+-------+------+------+------+---------+----------------+" << endl;
    cout << endl;
}

// =========================================================================
// Task 2: Thread to Tensor Core Mapping
// =========================================================================

void demonstrate_thread_mapping() {
    cout << "=== Thread to Tensor Core Mapping ===" << endl;
    cout << endl;
    
    // For 16×16×16 MMA with 32 threads (1 warp)
    const int WARP_SIZE = 32;
    const int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    
    cout << "MMA Atom: " << MMA_M << "×" << MMA_N << "×" << MMA_K << endl;
    cout << "Threads: " << WARP_SIZE << " (1 warp)" << endl;
    cout << endl;
    
    // Each thread's responsibility
    int elements_per_thread = (MMA_M * MMA_N) / WARP_SIZE;
    cout << "Each thread computes: " << elements_per_thread << " elements of output" << endl;
    cout << endl;
    
    // Thread layout (4 rows × 8 columns)
    cout << "Thread layout (4×8 = 32 threads):" << endl;
    cout << "  Thread (row, col) -> Output elements" << endl;
    
    for (int t_row = 0; t_row < 4; ++t_row) {
        for (int t_col = 0; t_col < 8; ++t_col) {
            int thread_id = t_row * 8 + t_col;
            int elem_row_start = t_row * 4;  // 16/4 = 4 rows per thread row
            int elem_col_start = t_col * 2;  // 16/8 = 2 cols per thread col
            
            cout << "  Thread " << thread_id << " (" << t_row << "," << t_col << ") -> "
                 << "elements (" << elem_row_start << "-" << elem_row_start+3
                 << ", " << elem_col_start << "-" << elem_col_start+1 << ")" << endl;
        }
    }
    cout << endl;
}

// =========================================================================
// Task 3: Register Allocation Analysis
// =========================================================================

void analyze_register_usage() {
    cout << "=== Register Allocation Analysis ===" << endl;
    cout << endl;
    
    struct RegisterBudget {
        const char* config;
        int a_regs;
        int b_regs;
        int c_regs;
        int total;
        int max_threads_per_sm;
    };
    
    RegisterBudget budgets[] = {
        {"FP16 16×16×16", 4, 4, 8, 16, 64},
        {"FP16 16×16×32", 8, 8, 8, 24, 42},
        {"BF16 16×16×8",  2, 2, 8, 12, 64},
        {"FP32 8×8×4",    8, 8, 16, 32, 32},
        {"INT8 16×16×32", 8, 8, 8, 24, 42},
    };
    
    cout << "+------------------+-----+-----+-----+-------+------------------+" << endl;
    cout << "| Configuration    | A   | B   | C   | Total | Max Threads/SM   |" << endl;
    cout << "+------------------+-----+-----+-----+-------+------------------+" << endl;
    
    for (const auto& b : budgets) {
        printf("| %-16s | %3d | %3d | %3d | %5d | %16d |\n",
               b.config, b.a_regs, b.b_regs, b.c_regs, b.total, b.max_threads_per_sm);
    }
    
    cout << "+------------------+-----+-----+-----+-------+------------------+" << endl;
    cout << endl;
    
    cout << "Note: A100 has 65536 registers per SM" << endl;
    cout << "      Max threads limited by registers: 65536 / total_regs" << endl;
    cout << endl;
}

// =========================================================================
// Task 4: Multi-Step Accumulation Pattern
// =========================================================================

void demonstrate_accumulation_pattern() {
    cout << "=== Multi-Step Accumulation Pattern ===" << endl;
    cout << endl;
    
    int K = 1024;
    int tile_K = 16;
    int num_tiles = K / tile_K;
    
    cout << "GEMM: C = A × B" << endl;
    cout << "  K dimension: " << K << endl;
    cout << "  MMA tile K:  " << tile_K << endl;
    cout << "  Number of MMA operations: " << num_tiles << endl;
    cout << endl;
    
    cout << "Accumulation loop:" << endl;
    cout << "  // Initialize accumulator" << endl;
    cout << "  float accum[16][16] = {0};" << endl;
    cout << endl;
    cout << "  for (int k = 0; k < " << num_tiles << "; ++k) {" << endl;
    cout << "      // Load A tile (16×16)" << endl;
    cout << "      load_A_tile(A, k * " << tile_K << ");" << endl;
    cout << endl;
    cout << "      // Load B tile (16×16)" << endl;
    cout << "      load_B_tile(B, k * " << tile_K << ");" << endl;
    cout << endl;
    cout << "      // MMA with accumulation" << endl;
    cout << "      mma_sync(accum, A_tile, B_tile);" << endl;
    cout << "      // accum = A_tile × B_tile + accum" << endl;
    cout << "  }" << endl;
    cout << endl;
    
    cout << "Total operations: " << num_tiles << " MMA instructions" << endl;
    cout << "Each MMA: " << MMA_M << "×" << MMA_N << "×" << MMA_K 
         << " = " << (MMA_M * MMA_N * MMA_K * 2) << " FLOPs" << endl;
    cout << "Total FLOPs: " << (num_tiles * MMA_M * MMA_N * MMA_K * 2) << endl;
    cout << endl;
}

// =========================================================================
// Task 5: GEMM Performance Estimation
// =========================================================================

void estimate_gemm_performance() {
    cout << "=== GEMM Performance Estimation ===" << endl;
    cout << endl;
    
    int M = 4096, N = 4096, K = 4096;
    float peak_tflops = 312.0f;  // A100 FP16 peak
    
    long long total_flops = 2LL * M * N * K;
    float theoretical_time_ms = (total_flops / 1e12f) / peak_tflops * 1000.0f;
    float practical_time_ms = theoretical_time_ms * 1.5f;  // Assume 67% efficiency
    
    cout << "Problem: C[" << M << "][" << N << "] = A[" << M << "][" << K 
         << "] × B[" << K << "][" << N << "]" << endl;
    cout << endl;
    
    cout << "Computational complexity:" << endl;
    cout << "  Total FLOPs: " << total_flops << " (" << (total_flops / 1e9) << " GFLOPs)" << endl;
    cout << "  Peak performance: " << peak_tflops << " TFLOPS" << endl;
    cout << endl;
    
    cout << "Time estimation:" << endl;
    cout << "  Theoretical minimum: " << theoretical_time_ms << " ms" << endl;
    cout << "  Practical (67% eff): " << practical_time_ms << " ms" << endl;
    cout << "  Expected efficiency: 60-80%" << endl;
    cout << endl;
    
    cout << "Memory requirements:" << endl;
    cout << "  Matrix A: " << (M * K * 2 / 1e6) << " MB (FP16)" << endl;
    cout << "  Matrix B: " << (K * N * 2 / 1e6) << " MB (FP16)" << endl;
    cout << "  Matrix C: " << (M * N * 4 / 1e6) << " MB (FP32)" << endl;
    cout << "  Total: " << ((M * K * 2 + K * N * 2 + M * N * 4) / 1e6) << " MB" << endl;
    cout << endl;
}

// =========================================================================
// Task 6: Occupancy Calculation
// =========================================================================

void calculate_occupancy() {
    cout << "=== Occupancy Calculation ===" << endl;
    cout << endl;
    
    // A100 SM resources
    const int MAX_REGISTERS_PER_SM = 65536;
    const int MAX_SHARED_MEM_PER_SM = 195 * 1024;  // 195 KB
    const int MAX_THREADS_PER_SM = 2048;
    const int MAX_WARPS_PER_SM = 64;
    
    struct KernelConfig {
        const char* name;
        int regs_per_thread;
        int smem_per_block;
        int threads_per_block;
    };
    
    KernelConfig kernels[] = {
        {"Lightweight MMA", 16, 32 * 1024, 128},
        {"Standard GEMM",   24, 64 * 1024, 256},
        {"Heavy GEMM",      32, 96 * 1024, 256},
        {"Large Tile",      48, 128 * 1024, 512},
    };
    
    cout << "+------------------+------+--------+-------+---------+-----------+" << endl;
    cout << "| Kernel           | Regs | SMEM   | Thds  | Occupancy | Active    |" << endl;
    cout << "|                  | /thd | /block | /blk  |           | Warps     |" << endl;
    cout << "+------------------+------+--------+-------+---------+-----------+" << endl;
    
    for (const auto& k : kernels) {
        // Limiting factors
        int blocks_by_regs = MAX_REGISTERS_PER_SM / (k.regs_per_thread * k.threads_per_block);
        int blocks_by_smem = MAX_SHARED_MEM_PER_SM / k.smem_per_block;
        int blocks_by_threads = MAX_THREADS_PER_SM / k.threads_per_block;
        int blocks_by_warps = (MAX_WARPS_PER_SM * 32) / k.threads_per_block;
        
        int max_blocks = min({blocks_by_regs, blocks_by_smem, blocks_by_threads, blocks_by_warps});
        int active_threads = max_blocks * k.threads_per_block;
        int active_warps = active_threads / 32;
        float occupancy = (float)active_warps / MAX_WARPS_PER_SM * 100.0f;
        
        printf("| %-16s | %4d | %6d | %5d | %7.1f%% | %9d |\n",
               k.name, k.regs_per_thread, k.smem_per_block, k.threads_per_block,
               occupancy, active_warps);
    }
    
    cout << "+------------------+------+--------+-------+---------+-----------+" << endl;
    cout << endl;
    
    cout << "A100 SM Resources:" << endl;
    cout << "  Max registers: " << MAX_REGISTERS_PER_SM << endl;
    cout << "  Max shared memory: " << (MAX_SHARED_MEM_PER_SM / 1024) << " KB" << endl;
    cout << "  Max threads: " << MAX_THREADS_PER_SM << endl;
    cout << "  Max warps: " << MAX_WARPS_PER_SM << endl;
    cout << endl;
}

// =========================================================================
// Task 7: Complete GEMM Structure
// =========================================================================

void demonstrate_gemm_structure() {
    cout << "=== Complete GEMM Structure ===" << endl;
    cout << endl;
    
    cout << "Hierarchical organization:" << endl;
    cout << endl;
    
    cout << "Level 1: Grid (Multiple thread blocks)" << endl;
    cout << "  └── Each block computes a tile of C" << endl;
    cout << "      Example: 32×32 blocks for 4096×4096 matrix" << endl;
    cout << endl;
    
    cout << "Level 2: Thread Block (128-512 threads)" << endl;
    cout << "  ├── Shared memory for A and B tiles" << endl;
    cout << "  ├── Cooperative loading from global memory" << endl;
    cout << "  └── MMA loop over K dimension" << endl;
    cout << "      Example: 128 threads, 64 KB shared memory" << endl;
    cout << endl;
    
    cout << "Level 3: Warp (32 threads)" << endl;
    cout << "  └── Executes MMA atom (16×16×16)" << endl;
    cout << "      Each warp computes 16×16 output elements" << endl;
    cout << endl;
    
    cout << "Level 4: Thread (1 thread)" << endl;
    cout << "  └── Handles subset of MMA elements" << endl;
    cout << "      Example: 2×4 = 8 elements per thread" << endl;
    cout << endl;
    
    cout << "Data flow:" << endl;
    cout << "  Global Memory (A, B)" << endl;
    cout << "       ↓ (coalesced load)" << endl;
    cout << "  Shared Memory (tiles)" << endl;
    cout << "       ↓ (vectorized load)" << endl;
    cout << "  Registers (MMA inputs)" << endl;
    cout << "       ↓ (MMA operation)" << endl;
    cout << "  Registers (accumulators)" << endl;
    cout << "       ↓ (coalesced store)" << endl;
    cout << "  Global Memory (C)" << endl;
    cout << endl;
}

// =========================================================================
// Main Exercise
// =========================================================================

int main() {
    cout << "=== Exercise 09: Advanced MMA Configurations and GEMM ===" << endl;
    cout << endl;
    
    // Task 1: MMA configurations
    print_mma_configs();
    
    // Task 2: Thread mapping
    demonstrate_thread_mapping();
    
    // Task 3: Register analysis
    analyze_register_usage();
    
    // Task 4: Accumulation pattern
    demonstrate_accumulation_pattern();
    
    // Task 5: Performance estimation
    estimate_gemm_performance();
    
    // Task 6: Occupancy calculation
    calculate_occupancy();
    
    // Task 7: GEMM structure
    demonstrate_gemm_structure();
    
    // =========================================================================
    // Challenge: Design Optimal GEMM Configuration
    // =========================================================================
    cout << "=== Challenge: Design Optimal GEMM Configuration ===" << endl;
    cout << endl;
    
    cout << "Problem: Matrix multiplication for Transformer attention" << endl;
    cout << "  Q × K^T: 32×128 × 128×32 = 32×32" << endl;
    cout << "  Attention × V: 32×32 × 32×128 = 32×128" << endl;
    cout << "  Batch size: 64 sequences" << endl;
    cout << endl;
    
    cout << "Design considerations:" << endl;
    cout << "1. MMA configuration (M×N×K)?" << endl;
    cout << "   Hint: Small matrices, consider 16×16×16 or 16×16×8" << endl;
    cout << endl;
    
    cout << "2. Precision (FP16, BF16, FP32)?" << endl;
    cout << "   Hint: Attention needs good numerical stability" << endl;
    cout << endl;
    
    cout << "3. Thread block size (128, 256, 512)?" << endl;
    cout << "   Hint: Balance occupancy and work per block" << endl;
    cout << endl;
    
    cout << "4. Shared memory allocation?" << endl;
    cout << "   Hint: Double buffering needs 2× tile size" << endl;
    cout << endl;
    
    cout << "5. Number of pipeline stages?" << endl;
    cout << "   Hint: 2-stage (double buffer) vs 3-stage" << endl;
    cout << endl;
    
    cout << "=== Exercise Complete ===" << endl;
    cout << "Key Learnings:" << endl;
    cout << "1. MMA configurations trade off tile size and throughput" << endl;
    cout << "2. 32 threads (1 warp) cooperate on each MMA atom" << endl;
    cout << "3. Register allocation limits occupancy" << endl;
    cout << "4. Multi-step accumulation for K > tile_K" << endl;
    cout << "5. Performance depends on problem size and configuration" << endl;
    cout << "6. Occupancy is limited by registers, shared memory, or threads" << endl;
    cout << "7. Complete GEMM requires hierarchical organization" << endl;
    
    return 0;
}
