/**
 * Exercise 08: Performance Optimization
 * 
 * Objective: Learn performance optimization strategies
 *            for CuTe-based GEMM kernels
 * 
 * Tasks:
 * 1. Understand performance metrics
 * 2. Learn optimization strategies
 * 3. Practice profiling techniques
 * 4. Apply optimization workflow
 * 
 * Key Concepts:
 * - TFLOPS: Trillion floating-point operations per second
 * - Bandwidth: Memory throughput (GB/s)
 * - Occupancy: Active warps per SM
 * - Roofline Model: Performance bound analysis
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 08: Performance Optimization ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Performance metrics
    std::cout << "Task 1 - Performance Metrics:" << std::endl;
    std::cout << std::endl;

    std::cout << "TFLOPS (Compute Performance):" << std::endl;
    std::cout << "  Formula: (2 × M × N × K) / (time × 10^12)" << std::endl;
    std::cout << "  A100 peak FP16: 312 TFLOPS (Tensor Core)" << std::endl;
    std::cout << "  Target: >80% of peak for good kernels" << std::endl;
    std::cout << std::endl;

    std::cout << "Bandwidth (Memory Performance):" << std::endl;
    std::cout << "  Formula: bytes_transferred / time" << std::endl;
    std::cout << "  A100 peak: 1555 GB/s (HBM2e)" << std::endl;
    std::cout << "  Target: >70% of peak for memory-bound" << std::endl;
    std::cout << std::endl;

    std::cout << "Occupancy:" << std::endl;
    std::cout << "  Formula: active_warps / max_warps" << std::endl;
    std::cout << "  A100 max: 64 warps per SM" << std::endl;
    std::cout << "  Target: >50% for good latency hiding" << std::endl;
    std::cout << std::endl;

    // TASK 2: Roofline model
    std::cout << "Task 2 - Roofline Model:" << std::endl;
    std::cout << std::endl;

    std::cout << "Arithmetic Intensity (AI):" << std::endl;
    std::cout << "  AI = FLOPs / bytes" << std::endl;
    std::cout << "  High AI = Compute bound" << std::endl;
    std::cout << "  Low AI = Memory bound" << std::endl;
    std::cout << std::endl;

    std::cout << "For GEMM:" << std::endl;
    std::cout << "  FLOPs = 2 × M × N × K" << std::endl;
    std::cout << "  Bytes = (M × K + K × N + M × N) × element_size" << std::endl;
    std::cout << "  AI ≈ 2 × K / element_size (for large M, N)" << std::endl;
    std::cout << std::endl;

    // Example calculation
    int M = 4096, N = 4096, K = 4096;
    long long flops = 2LL * M * N * K;
    long long bytes = 2LL * (M * K + K * N + M * N) * 2;  // FP16
    float ai = (float)flops / bytes;
    
    std::cout << "Example: " << M << "x" << N << "x" << K << " GEMM (FP16)" << std::endl;
    std::cout << "  FLOPs: " << flops / 1e12 << " trillion" << std::endl;
    std::cout << "  Bytes: " << bytes / 1e9 << " billion" << std::endl;
    std::cout << "  AI: " << ai << " FLOPs/byte" << std::endl;
    std::cout << "  Bound: " << (ai > 100 ? "Compute" : "Memory") << std::endl;
    std::cout << std::endl;

    // TASK 3: Optimization strategies
    std::cout << "Task 3 - Optimization Strategies:" << std::endl;
    std::cout << std::endl;

    std::cout << "Memory-Bound Optimization:" << std::endl;
    std::cout << "  1. Increase coalescing" << std::endl;
    std::cout << "  2. Use vectorized loads (128-bit)" << std::endl;
    std::cout << "  3. Optimize shared memory access" << std::endl;
    std::cout << "  4. Reduce redundant loads" << std::endl;
    std::cout << std::endl;

    std::cout << "Compute-Bound Optimization:" << std::endl;
    std::cout << "  1. Increase pipeline depth" << std::endl;
    std::cout << "  2. Maximize Tensor Core utilization" << std::endl;
    std::cout << "  3. Balance occupancy" << std::endl;
    std::cout << "  4. Reduce instruction overhead" << std::endl;
    std::cout << std::endl;

    // TASK 4: Profiling workflow
    std::cout << "Task 4 - Profiling Workflow:" << std::endl;
    std::cout << std::endl;

    std::cout << "Step 1: Baseline measurement" << std::endl;
    std::cout << "  - Run kernel, measure time" << std::endl;
    std::cout << "  - Calculate TFLOPS" << std::endl;
    std::cout << std::endl;

    std::cout << "Step 2: Identify bottleneck" << std::endl;
    std::cout << "  - Use Nsight Compute" << std::endl;
    std::cout << "  - Check occupancy, bandwidth, compute" << std::endl;
    std::cout << std::endl;

    std::cout << "Step 3: Apply optimization" << std::endl;
    std::cout << "  - Target identified bottleneck" << std::endl;
    std::cout << "  - One change at a time" << std::endl;
    std::cout << std::endl;

    std::cout << "Step 4: Measure improvement" << std::endl;
    std::cout << "  - Compare with baseline" << std::endl;
    std::cout << "  - Iterate if needed" << std::endl;
    std::cout << std::endl;

    // TASK 5: Common optimizations
    std::cout << "Task 5 - Common Optimizations:" << std::endl;
    std::cout << std::endl;

    struct Optimization {
        const char* name;
        const char* benefit;
        const char* tradeoff;
    };

    Optimization opts[] = {
        {"Larger tiles", "More compute per load", "More registers/smem"},
        {"Deeper pipeline", "Better overlap", "More complexity"},
        {"Vectorized loads", "4x bandwidth", "Alignment required"},
        {"Swizzling", "No bank conflicts", "Address calculation"},
        {"Register tiling", "Faster access", "Register pressure"},
    };

    std::cout << "| Optimization    | Benefit           | Tradeoff          |" << std::endl;
    std::cout << "|-----------------|-------------------|-------------------|" << std::endl;
    for (auto& opt : opts) {
        printf("| %-15s | %-17s | %-17s |\n", opt.name, opt.benefit, opt.tradeoff);
    }
    std::cout << std::endl;

    // CHALLENGE: Optimize for target
    std::cout << "=== Challenge: Optimization Strategy ===" << std::endl;
    std::cout << "Scenario: GEMM achieving 50 TFLOPS on A100 (target: 200+)" << std::endl;
    std::cout << std::endl;

    std::cout << "Profiling shows:" << std::endl;
    std::cout << "  - Occupancy: 25% (low)" << std::endl;
    std::cout << "  - Memory bandwidth: 80% (good)" << std::endl;
    std::cout << "  - Tensor Core utilization: 40% (low)" << std::endl;
    std::cout << std::endl;

    std::cout << "Recommended optimizations:" << std::endl;
    std::cout << "  1. Reduce register usage (increase occupancy)" << std::endl;
    std::cout << "  2. Increase pipeline depth (better utilization)" << std::endl;
    std::cout << "  3. Check for shared memory bank conflicts" << std::endl;
    std::cout << std::endl;

    // OPTIMIZATION CHECKLIST
    std::cout << "=== Optimization Checklist ===" << std::endl;
    std::cout << R"(
Performance Optimization Checklist:

Memory Access:
[ ] Coalesced global memory access
[ ] Vectorized loads (128-bit)
[ ] No shared memory bank conflicts
[ ] Minimal redundant loads

Compute:
[ ] Tensor Core utilization >80%
[ ] Pipeline depth 2-4 stages
[ ] Loop unrolling where beneficial
[ ] Minimal instruction overhead

Resources:
[ ] Occupancy >50%
[ ] Register usage <64 per thread
[ ] Shared memory <192 KB per SM
[ ] No register spilling

Correctness:
[ ] Results match reference
[ ] No race conditions
[ ] Proper synchronization
[ ] Edge cases handled
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Measure before optimizing" << std::endl;
    std::cout << "2. Use roofline model to identify bounds" << std::endl;
    std::cout << "3. Profile to find bottlenecks" << std::endl;
    std::cout << "4. Optimize iteratively, one change at a time" << std::endl;

    return 0;
}
