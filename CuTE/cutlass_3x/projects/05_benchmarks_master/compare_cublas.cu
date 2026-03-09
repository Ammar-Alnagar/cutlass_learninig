/*
 * Project 05 — Benchmarks Master
 * CUTLASS 3.x vs cuBLAS Comparison
 *
 * Side-by-side benchmark against NVIDIA reference.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#include "../../utils/benchmark.cuh"
#include "../../utils/reference.cuh"

// ============================================================================
// BENCHMARK RESULTS STRUCTURE
// ============================================================================

struct CublasComparisonResult {
    std::string kernel;
    int M, N, K;
    double cublas_tflops;
    double cutlass_tflops;
    double percent_cublas;
};

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Project 05: CUTLASS 3.x vs cuBLAS ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    std::vector<CublasComparisonResult> results;
    
    // TODO: Run cuBLAS and CUTLASS benchmarks
    // Compute % of cuBLAS performance
    
    /*
    // Example for each kernel:
    // 1. Run cuBLAS GEMM, measure TFLOPS
    // 2. Run CUTLASS GEMM, measure TFLOPS
    // 3. Compute percent_cublas = cutlass / cublas * 100
    
    results.push_back({
        "Dense GEMM FP16", 4096, 4096, 4096,
        cublas_tflops, cutlass_tflops, (cutlass_tflops / cublas_tflops) * 100
    });
    */
    
    // Print comparison table
    std::cout << std::left << std::setw(25) << "Kernel"
              << std::right << std::setw(15) << "cuBLAS"
              << std::setw(15) << "CUTLASS 3.x"
              << std::setw(15) << "% cuBLAS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(25) << r.kernel
                  << std::right << std::setw(15) << r.cublas_tflops
                  << std::setw(15) << r.cutlass_tflops
                  << std::setw(15) << r.percent_cublas << "%" << std::endl;
    }
    
    std::cout << "\nTargets:" << std::endl;
    std::cout << "  Ampere (SM80): >85% cuBLAS" << std::endl;
    std::cout << "  Hopper (SM90): >90% cuBLAS" << std::endl;
    std::cout << "  Blackwell (SM100): >95% cuBLAS" << std::endl;
    std::cout << std::endl;
    std::cout << "TODO: Fill in benchmark results" << std::endl;
    std::cout << "Save results to projects/05_benchmarks_master/results/" << std::endl;
    
    return 0;
}
