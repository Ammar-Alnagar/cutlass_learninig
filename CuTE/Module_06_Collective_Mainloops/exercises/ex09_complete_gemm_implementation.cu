/**
 * Exercise 09: Complete GEMM Implementation and Performance Tuning
 *
 * Objective: Master complete GEMM implementation including full mainloop,
 *            performance tuning, and optimization strategies
 *
 * Tasks:
 * 1. Design complete GEMM kernel structure
 * 2. Implement multi-stage pipeline
 * 3. Tune performance parameters
 * 4. Profile and optimize kernel
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// =========================================================================
// Task 1: Complete GEMM Kernel Structure
// =========================================================================

void demonstrate_gemm_structure() {
    cout << "=== Complete GEMM Kernel Structure ===" << endl;
    cout << endl;
    
    cout << "Hierarchical organization:" << endl;
    cout << endl;
    
    cout << "Level 1: Grid" << endl;
    cout << "  dim3 grid((N + BLOCK_N - 1) / BLOCK_N," << endl;
    cout << "            (M + BLOCK_M - 1) / BLOCK_M);" << endl;
    cout << "  // Each block computes one tile of C" << endl;
    cout << endl;
    
    cout << "Level 2: Thread Block" << endl;
    cout << "  dim3 block(128);  // 128 threads" << endl;
    cout << "  extern __shared__ uint8_t smem[];" << endl;
    cout << "  // Shared memory for A and B tiles" << endl;
    cout << endl;
    
    cout << "Level 3: Warp" << endl;
    cout << "  int warp_id = threadIdx.x / 32;" << endl;
    cout << "  // Each warp executes MMA atoms" << endl;
    cout << endl;
    
    cout << "Level 4: Thread" << endl;
    cout << "  int lane_id = threadIdx.x % 32;" << endl;
    cout << "  // Each thread handles subset of elements" << endl;
    cout << endl;
    
    cout << "Kernel phases:" << endl;
    cout << "  1. Prologue: Load first tiles, init accumulators" << endl;
    cout << "  2. Mainloop: Iterate over K-dimension" << endl;
    cout << "  3. Epilogue: Final MMA, store results" << endl;
    cout << endl;
}

// =========================================================================
// Task 2: Multi-Stage Pipeline Implementation
// =========================================================================

struct PipelineConfig {
    const char* name;
    int stages;
    float speedup;
    int smem_overhead;
    const char* complexity;
};

void demonstrate_pipeline_implementation() {
    cout << "=== Multi-Stage Pipeline Implementation ===" << endl;
    cout << endl;
    
    PipelineConfig configs[] = {
        {"Sequential", 1, 1.0f, 0, "Low"},
        {"Double Buffer", 2, 1.8f, 2, "Medium"},
        {"Triple Buffer", 3, 2.3f, 3, "High"},
        {"Quad Buffer", 4, 2.6f, 4, "Very High"},
    };
    
    cout << "+------------------+--------+----------+---------------+------------+" << endl;
    cout << "| Configuration    | Stages | Speedup  | SMEM Overhead | Complexity |" << endl;
    cout << "+------------------+--------+----------+---------------+---------------+" << endl;
    
    for (const auto& c : configs) {
        printf("| %-16s | %6d | %7.1fx | %13dx | %-10s |\n",
               c.name, c.stages, c.speedup, c.stages, c.complexity);
    }
    
    cout << "+------------------+--------+----------+---------------+---------------+" << endl;
    cout << endl;
    
    cout << "2-Stage Pipeline Pattern:" << endl;
    cout << "  load(0);" << endl;
    cout << "  for (k = 1; k < K; ++k) {" << endl;
    cout << "      compute(k-1);" << endl;
    cout << "      load(k);" << endl;
    cout << "  }" << endl;
    cout << "  compute(K-1);" << endl;
    cout << endl;
    
    cout << "3-Stage Pipeline Pattern:" << endl;
    cout << "  load(0); load(1);" << endl;
    cout << "  for (k = 2; k < K; ++k) {" << endl;
    cout << "      compute(k-2);" << endl;
    cout << "      load(k);" << endl;
    cout << "  }" << endl;
    cout << "  compute(K-2); compute(K-1);" << endl;
    cout << endl;
}

// =========================================================================
// Task 3: Performance Parameter Tuning
// =========================================================================

struct TuningParameter {
    const char* name;
    const char* typical_values;
    const char* impact;
    const char* recommendation;
};

void demonstrate_parameter_tuning() {
    cout << "=== Performance Parameter Tuning ===" << endl;
    cout << endl;
    
    TuningParameter params[] = {
        {"Block Size", "128, 256, 512", "Occupancy, work per block", "Start with 128 or 256"},
        {"Tile Size (M,N)", "64, 128, 256", "Shared memory, reuse", "128 for A100"},
        {"Tile Size (K)", "8, 16, 32", "Mainloop iterations", "8 or 16"},
        {"Registers", "16-64", "Occupancy, spilling", "Minimize, avoid spill"},
        {"Shared Memory", "32-192 KB", "Blocks per SM", "Use what's needed"},
        {"Pipeline Stages", "2, 3, 4", "Latency hiding", "2-3 for most cases"},
    };
    
    cout << "+------------------+-------------------+----------------------+------------------+" << endl;
    cout << "| Parameter        | Typical Values    | Impact               | Recommendation   |" << endl;
    cout << "+------------------+-------------------+----------------------+------------------+" << endl;
    
    for (const auto& p : params) {
        printf("| %-16s | %-17s | %-20s | %-16s |\n",
               p.name, p.typical_values, p.impact, p.recommendation);
    }
    
    cout << "+------------------+-------------------+----------------------+------------------+" << endl;
    cout << endl;
    
    cout << "Tuning workflow:" << endl;
    cout << "  1. Start with reasonable defaults" << endl;
    cout << "  2. Measure baseline performance" << endl;
    cout << "  3. Tune one parameter at a time" << endl;
    cout << "  4. Profile to identify bottlenecks" << endl;
    cout << "  5. Iterate until satisfied" << endl;
    cout << endl;
}

// =========================================================================
// Task 4: Performance Estimation
// =========================================================================

void estimate_gemm_performance() {
    cout << "=== GEMM Performance Estimation ===" << endl;
    cout << endl;
    
    struct ProblemSize {
        int M, N, K;
        float theoretical_tflops;
        float expected_efficiency;
    };
    
    ProblemSize problems[] = {
        {512, 512, 512, 0.5f, "40-50%"},
        {1024, 1024, 1024, 4.3f, "50-60%"},
        {2048, 2048, 2048, 34.4f, "60-70%"},
        {4096, 4096, 4096, 274.0f, "70-80%"},
        {8192, 8192, 8192, 2194.0f, "75-85%"},
    };
    
    cout << "A100 FP16 GEMM Performance Estimates:" << endl;
    cout << endl;
    
    cout << "+----------------------+------------+------------+----------------+" << endl;
    cout << "| Problem Size (M=N=K) | GFLOPs     | Theoretical| Expected       |" << endl;
    cout << "|                      |            | Time (ms)  | Efficiency     |" << endl;
    cout << "+----------------------+------------+------------+----------------+" << endl;
    
    float peak_tflops = 312.0f;
    
    for (const auto& p : problems) {
        long long flops = 2LL * p.M * p.N * p.K;
        float theoretical_ms = (flops / 1e12f) / peak_tflops * 1000.0f;
        
        printf("| %-6d × %-6d × %-6d | %9.1f  | %9.4f   | %-14s |\n",
               p.M, p.N, p.K, flops / 1e9f, theoretical_ms, p.expected_efficiency);
    }
    
    cout << "+----------------------+------------+------------+----------------+" << endl;
    cout << endl;
    
    cout << "Note: Larger problems achieve better efficiency due to:" << endl;
    cout << "  - More time in compute-bound regime" << endl;
    cout << "  - Better latency hiding" << endl;
    cout << "  - Amortized kernel launch overhead" << endl;
    cout << endl;
}

// =========================================================================
// Task 5: Occupancy Calculation
// =========================================================================

void calculate_optimal_occupancy() {
    cout << "=== Occupancy Calculation ===" << endl;
    cout << endl;
    
    // A100 limits
    const int MAX_REGISTERS_PER_SM = 65536;
    const int MAX_SHARED_MEM_PER_SM = 195 * 1024;
    const int MAX_THREADS_PER_SM = 2048;
    const int MAX_WARPS_PER_SM = 64;
    
    struct KernelConfig {
        const char* name;
        int regs_per_thread;
        int smem_per_block;
        int threads_per_block;
    };
    
    KernelConfig configs[] = {
        {"Lightweight", 16, 32 * 1024, 128},
        {"Balanced", 24, 64 * 1024, 256},
        {"Heavy", 32, 96 * 1024, 256},
        {"Register Heavy", 48, 64 * 1024, 256},
        {"SMEM Heavy", 24, 128 * 1024, 256},
    };
    
    cout << "A100 SM Resources:" << endl;
    cout << "  Max registers: " << MAX_REGISTERS_PER_SM << endl;
    cout << "  Max shared memory: " << (MAX_SHARED_MEM_PER_SM / 1024) << " KB" << endl;
    cout << "  Max threads: " << MAX_THREADS_PER_SM << endl;
    cout << "  Max warps: " << MAX_WARPS_PER_SM << endl;
    cout << endl;
    
    cout << "+------------------+------+--------+-------+----------+-----------+" << endl;
    cout << "| Configuration    | Regs | SMEM   | Thds  | Occupancy| Active    |" << endl;
    cout << "|                  | /thd | /block | /blk  |          | Warps     |" << endl;
    cout << "+------------------+------+--------+-------+----------+-----------+" << endl;
    
    for (const auto& c : configs) {
        int blocks_by_regs = MAX_REGISTERS_PER_SM / (c.regs_per_thread * c.threads_per_block);
        int blocks_by_smem = MAX_SHARED_MEM_PER_SM / c.smem_per_block;
        int blocks_by_threads = MAX_THREADS_PER_SM / c.threads_per_block;
        int blocks_by_warps = (MAX_WARPS_PER_SM * 32) / c.threads_per_block;
        
        int max_blocks = min({blocks_by_regs, blocks_by_smem, blocks_by_threads, blocks_by_warps});
        int active_warps = max_blocks * c.threads_per_block / 32;
        float occupancy = (float)active_warps / MAX_WARPS_PER_SM * 100.0f;
        
        printf("| %-16s | %4d | %6d | %5d | %7.1f%%  | %9d |\n",
               c.name, c.regs_per_thread, c.smem_per_block, c.threads_per_block,
               occupancy, active_warps);
    }
    
    cout << "+------------------+------+--------+-------+----------+-----------+" << endl;
    cout << endl;
}

// =========================================================================
// Task 6: Profiling and Bottleneck Analysis
// =========================================================================

void demonstrate_profiling_workflow() {
    cout << "=== Profiling and Bottleneck Analysis ===" << endl;
    cout << endl;
    
    cout << "Nsight Compute profiling workflow:" << endl;
    cout << endl;
    
    cout << "Step 1: Baseline measurement" << endl;
    cout << "  ncu --set full ./gemm_app" << endl;
    cout << "  Look at: Overall kernel time, SM efficiency" << endl;
    cout << endl;
    
    cout << "Step 2: Memory throughput" << endl;
    cout << "  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_app" << endl;
    cout << "  Target: >80% of peak bandwidth" << endl;
    cout << endl;
    
    cout << "Step 3: Tensor Core utilization" << endl;
    cout << "  ncu --metrics sm__inst_executed_pipe_tensor ./gemm_app" << endl;
    cout << "  Target: >80% of theoretical MMA ops" << endl;
    cout << endl;
    
    cout << "Step 4: Occupancy analysis" << endl;
    cout << "  ncu --metrics sm__warps_per_sm ./gemm_app" << endl;
    cout << "  Target: >50% occupancy" << endl;
    cout << endl;
    
    cout << "Step 5: Bottleneck identification" << endl;
    cout << "  ncu --metrics speedometer ./gemm_app" << endl;
    cout << "  Identifies: Compute-bound vs Memory-bound" << endl;
    cout << endl;
    
    cout << "Common bottlenecks and solutions:" << endl;
    cout << "+----------------------+---------------------------+" << endl;
    cout << "| Bottleneck           | Solution                  |" << endl;
    cout << "+----------------------+---------------------------+" << endl;
    cout << "| Low memory throughput| Improve coalescing        |" << endl;
    cout << "|                      | Use vectorized loads      |" << endl;
    cout << "+----------------------+---------------------------+" << endl;
    cout << "| Low Tensor Core util | Check MMA configuration   |" << endl;
    cout << "|                      | Reduce register pressure  |" << endl;
    cout << "+----------------------+---------------------------+" << endl;
    cout << "| Low occupancy        | Reduce registers/thread   |" << endl;
    cout << "|                      | Reduce shared memory      |" << endl;
    cout << "+----------------------+---------------------------+" << endl;
    cout << "| Register spilling    | Reduce register usage     |" << endl;
    cout << "|                      | Increase block size       |" << endl;
    cout << "+----------------------+---------------------------+" << endl;
    cout << endl;
}

// =========================================================================
// Task 7: Complete GEMM Code Structure
// =========================================================================

void show_complete_gemm_code() {
    cout << "=== Complete GEMM Code Structure ===" << endl;
    cout << endl;
    
    cout << "template <typename MMA_Atom, int BLOCK_M, int BLOCK_N, int BLOCK_K>" << endl;
    cout << "__global__ void gemm_kernel(float* C, const half* A, const half* B," << endl;
    cout << "                           int M, int N, int K) {" << endl;
    cout << endl;
    
    cout << "  // === Setup ===" << endl;
    cout << "  int m_start = blockIdx.y * BLOCK_M;" << endl;
    cout << "  int n_start = blockIdx.x * BLOCK_N;" << endl;
    cout << endl;
    
    cout << "  // === Shared Memory ===" << endl;
    cout << "  extern __shared__ uint8_t smem[];" << endl;
    cout << "  half* As = reinterpret_cast<half*>(smem);" << endl;
    cout << "  half* Bs = reinterpret_cast<half*>(&smem[BLOCK_M * BLOCK_K]);" << endl;
    cout << endl;
    
    cout << "  // === Accumulator ===" << endl;
    cout << "  float accum[MMA_M][MMA_N];" << endl;
    cout << "  zero_fill(accum);" << endl;
    cout << endl;
    
    cout << "  // === Prologue ===" << endl;
    cout << "  load_A_tile(A, As, m_start, 0);" << endl;
    cout << "  load_B_tile(B, Bs, n_start, 0);" << endl;
    cout << "  cp_async_fence();" << endl;
    cout << endl;
    
    cout << "  // === Mainloop ===" << endl;
    cout << "  for (int k = 1; k < K / BLOCK_K; ++k) {" << endl;
    cout << "    cp_async_wait<0>();" << endl;
    cout << "    __syncthreads();" << endl;
    cout << "    mma_sync(accum, As, Bs);" << endl;
    cout << "    load_A_tile(A, As, m_start, k);" << endl;
    cout << "    load_B_tile(B, Bs, n_start, k);" << endl;
    cout << "    cp_async_fence();" << endl;
    cout << "  }" << endl;
    cout << endl;
    
    cout << "  // === Epilogue ===" << endl;
    cout << "  cp_async_wait<0>();" << endl;
    cout << "  __syncthreads();" << endl;
    cout << "  mma_sync(accum, As, Bs);" << endl;
    cout << "  store_C_tile(C, accum, m_start, n_start);" << endl;
    cout << "}" << endl;
    cout << endl;
}

// =========================================================================
// Main Exercise
// =========================================================================

int main() {
    cout << "=== Exercise 09: Complete GEMM Implementation and Performance Tuning ===" << endl;
    cout << endl;
    
    // Task 1: GEMM structure
    demonstrate_gemm_structure();
    
    // Task 2: Pipeline implementation
    demonstrate_pipeline_implementation();
    
    // Task 3: Parameter tuning
    demonstrate_parameter_tuning();
    
    // Task 4: Performance estimation
    estimate_gemm_performance();
    
    // Task 5: Occupancy calculation
    calculate_optimal_occupancy();
    
    // Task 6: Profiling workflow
    demonstrate_profiling_workflow();
    
    // Task 7: Code structure
    show_complete_gemm_code();
    
    // =========================================================================
    // Challenge: Design Optimal GEMM Configuration
    // =========================================================================
    cout << "=== Challenge: Design Optimal GEMM Configuration ===" << endl;
    cout << endl;
    
    cout << "Target: GEMM for Transformer MLP layer" << endl;
    cout << "  Matrix A: 4096 × 16384 (input × hidden)" << endl;
    cout << "  Matrix B: 16384 × 4096 (hidden × output)" << endl;
    cout << "  Matrix C: 4096 × 4096 (output)" << endl;
    cout << "  Precision: FP16 for compute, FP32 accumulation" << endl;
    cout << "  Hardware: A100 (sm_80)" << endl;
    cout << endl;
    
    cout << "Design your configuration:" << endl;
    cout << "1. Block tile size (BLOCK_M, BLOCK_N, BLOCK_K)?" << endl;
    cout << "   Hint: Consider shared memory capacity" << endl;
    cout << endl;
    
    cout << "2. Thread block size (128, 256, 512)?" << endl;
    cout << "   Hint: Balance occupancy and work per block" << endl;
    cout << endl;
    
    cout << "3. Pipeline stages (2, 3, 4)?" << endl;
    cout << "   Hint: Consider K dimension size" << endl;
    cout << endl;
    
    cout << "4. MMA configuration?" << endl;
    cout << "   Hint: FP16 for throughput" << endl;
    cout << endl;
    
    cout << "5. Expected performance (TFLOPS)?" << endl;
    cout << "   Hint: 70-80% of peak is achievable" << endl;
    cout << endl;
    
    cout << "=== Exercise Complete ===" << endl;
    cout << "Key Learnings:" << endl;
    cout << "1. Complete GEMM has prologue, mainloop, and epilogue phases" << endl;
    cout << "2. Multi-stage pipelines overlap load and compute" << endl;
    cout << "3. Performance tuning requires systematic parameter search" << endl;
    cout << "4. Occupancy is limited by registers, shared memory, or threads" << endl;
    cout << "5. Profiling identifies real bottlenecks" << endl;
    cout << "6. Larger problems achieve better efficiency" << endl;
    cout << "7. Complete kernel requires careful coordination of all components" << endl;
    
    return 0;
}
