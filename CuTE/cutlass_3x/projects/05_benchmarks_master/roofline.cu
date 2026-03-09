/*
 * Project 05 — Benchmarks Master
 * Roofline Analysis
 *
 * Generate roofline charts for all benchmark kernels.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#include "../../utils/benchmark.cuh"
#include "../../utils/roofline.cuh"

using namespace cutlass_roofline;

// ============================================================================
// KERNEL DEFINITIONS
// ============================================================================

struct KernelInfo {
    std::string name;
    int M, N, K;
    double achieved_tflops;
};

// TODO: Add your benchmark results here
std::vector<KernelInfo> kernels = {
    {"Dense GEMM FP16", 4096, 4096, 4096, 0.0},    // Fill in achieved TFLOPS
    {"Dense GEMM FP8", 4096, 4096, 4096, 0.0},
    {"Warp-spec GEMM", 8192, 8192, 8192, 0.0},
    {"FA2 Attention", 32768, 4096, 128, 0.0},
    {"FA3 Attention", 32768, 4096, 128, 0.0},
};

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Project 05: Roofline Analysis ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Get device specs
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Estimate theoretical peak (simplified)
    double peak_tflops = 0.0;
    double peak_bandwidth = prop.memoryBusWidth * prop.memoryClockRate * 2 / 8000.0;
    
    if (prop.major == 8) {
        // A100: ~312 TFLOPS FP16 Tensor Core
        peak_tflops = 312.0;
    } else if (prop.major == 9) {
        // H100: ~989 TFLOPS FP16 Tensor Core
        peak_tflops = 989.0;
    } else if (prop.major >= 10) {
        // B200: ~2250 TFLOPS FP8 Tensor Core
        peak_tflops = 2250.0;
    }
    
    std::cout << "Theoretical peak: " << peak_tflops << " TFLOPS" << std::endl;
    std::cout << "Peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;
    
    // Create roofline model
    RooflineModel model(peak_tflops, peak_bandwidth);
    
    // Compute arithmetic intensity for each kernel
    std::vector<KernelAnalysis> analyses;
    for (const auto& k : kernels) {
        double ai = compute_gemm_arithmetic_intensity(k.M, k.N, k.K);
        std::string region = model.bound_region(ai);
        double util = model.utilization(ai, k.achieved_tflops);
        
        analyses.push_back({
            k.name, k.M, k.N, k.K,
            ai, k.achieved_tflops, region, util
        });
    }
    
    // Print roofline table
    print_roofline_table(analyses, model);
    
    // Print ASCII roofline chart
    print_roofline_chart(analyses, model);
    
    std::cout << "\nTODO: Fill in achieved TFLOPS from your benchmarks" << std::endl;
    std::cout << "Save results to projects/05_benchmarks_master/results/" << std::endl;
    
    return 0;
}
