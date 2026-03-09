/*
 * Project 05 — Benchmarks Master
 * CUTLASS 3.x vs ThunderKittens Comparison
 *
 * Side-by-side benchmark of the same kernels.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#include "../../utils/benchmark.cuh"

// ============================================================================
// BENCHMARK RESULTS STRUCTURE
// ============================================================================

struct ComparisonResult {
    std::string kernel;
    int M, N, K;
    double tk_tflops;
    double cutlass_tflops;
    double speedup;
};

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Project 05: CUTLASS 3.x vs ThunderKittens ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    std::vector<ComparisonResult> results;
    
    // TODO: Run same benchmarks in both frameworks
    // Record TFLOPS for each
    
    /*
    results.push_back({
        "Dense GEMM FP16", 4096, 4096, 4096,
        tk_tflops, cutlass_tflops, cutlass_tflops / tk_tflops
    });
    */
    
    // Print comparison table
    std::cout << std::left << std::setw(25) << "Kernel"
              << std::right << std::setw(15) << "ThunderKittens"
              << std::setw(15) << "CUTLASS 3.x"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(25) << r.kernel
                  << std::right << std::setw(15) << r.tk_tflops
                  << std::setw(15) << r.cutlass_tflops
                  << std::setw(15) << r.speedup << "×" << std::endl;
    }
    
    std::cout << "\nTODO: Implement ThunderKittens benchmarks for comparison" << std::endl;
    std::cout << "Save results to projects/05_benchmarks_master/results/" << std::endl;
    
    return 0;
}
