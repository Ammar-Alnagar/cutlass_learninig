#pragma once
/**
 * CUTLASS 3.x Roofline Analysis Utilities
 * 
 * Arithmetic intensity helpers and roofline chart generation.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace cutlass_roofline {

// Device memory bandwidth (GB/s) - query from device
inline float get_memory_bandwidth_gbps() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Memory bandwidth in GB/s
    return float(prop.memoryBusWidth) * float(prop.memoryClockRate) * 2 / 8000.0f;
}

// Compute arithmetic intensity for GEMM: FLOPS / bytes
// For C = A * B: 
//   FLOPS = 2 * M * N * K
//   Bytes = M*K + K*N + M*N (reading A, B, writing C)
inline double compute_gemm_arithmetic_intensity(int M, int N, int K) {
    double flops = 2.0 * M * N * K;
    double bytes = M * K + K * N + M * N;
    return flops / bytes;
}

// Compute arithmetic intensity for batched GEMM
inline double compute_batched_gemm_arithmetic_intensity(int M, int N, int K, int batch) {
    double flops = 2.0 * M * N * K * batch;
    double bytes = (M * K + K * N + M * N) * batch;
    return flops / bytes;
}

// Compute arithmetic intensity for attention (QK^V)
// Q: [B, H, S, D], K: [B, H, S, D], V: [B, H, S, D]
// QK: [B, H, S, S], QKV: [B, H, S, D]
inline double compute_attention_arithmetic_intensity(int B, int H, int S, int D) {
    // QK^T GEMM
    double qk_flops = 2.0 * B * H * S * S * D;
    double qk_bytes = B * H * S * D * 2 + B * H * S * S;  // Q, K read + QK write
    
    // (QK^T)V GEMM  
    double qkv_flops = 2.0 * B * H * S * S * D;
    double qkv_bytes = B * H * S * S + B * H * S * D * 2;  // QK read, V read, output write
    
    double total_flops = qk_flops + qkv_flops;
    double total_bytes = qk_bytes + qkv_bytes;
    
    return total_flops / total_bytes;
}

// Roofline model: compute achievable TFLOPS given arithmetic intensity
struct RooflineModel {
    double peak_compute_tflops;   // Peak compute throughput
    double peak_bandwidth_gbps;   // Peak memory bandwidth
    double roofline_ai;           // Arithmetic intensity at roofline knee
    
    RooflineModel(double peak_tflops, double peak_bw_gbps)
        : peak_compute_tflops(peak_tflops),
          peak_bandwidth_gbps(peak_bw_gbps) {
        // Knee point: where compute-bound transitions to memory-bound
        roofline_ai = peak_tflops * 1e12 / (peak_bw_gbps * 1e9);
    }
    
    // Compute achievable TFLOPS for given arithmetic intensity
    double achievable_tflops(double ai) const {
        // Memory-bound region: bandwidth * AI
        double mem_bound = (peak_bandwidth_gbps * 1e9) * ai / 1e12;
        
        // Compute-bound region: peak compute
        double compute_bound = peak_compute_tflops;
        
        return min(mem_bound, compute_bound);
    }
    
    // Determine if kernel is memory-bound or compute-bound
    std::string bound_region(double ai) const {
        if (ai < roofline_ai) {
            return "MEMORY-BOUND";
        } else {
            return "COMPUTE-BOUND";
        }
    }
    
    // Compute utilization percentage
    double utilization(double ai, double achieved_tflops) const {
        double theoretical = achievable_tflops(ai);
        return (achieved_tflops / theoretical) * 100.0;
    }
};

// Print roofline analysis for a kernel
struct KernelAnalysis {
    std::string name;
    int M, N, K;
    double arithmetic_intensity;
    double achieved_tflops;
    std::string bound_region;
    double utilization;
};

inline void print_roofline_table(
    const std::vector<KernelAnalysis>& analyses,
    const RooflineModel& model
) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Roofline Analysis ===" << std::endl;
    std::cout << "Peak Compute: " << model.peak_compute_tflops << " TFLOPS" << std::endl;
    std::cout << "Peak Bandwidth: " << model.peak_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "Roofline Knee (AI): " << model.roofline_ai << std::endl;
    std::cout << std::endl;
    
    std::cout << std::left << std::setw(25) << "Kernel"
              << std::right << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(10) << "AI"
              << std::setw(12) << "TFLOPS"
              << std::setw(15) << "Region"
              << std::setw(12) << "Util %" << std::endl;
    
    std::cout << std::string(98, '-') << std::endl;
    
    for (const auto& a : analyses) {
        std::cout << std::left << std::setw(25) << a.name
                  << std::right << std::setw(8) << a.M
                  << std::setw(8) << a.N
                  << std::setw(8) << a.K
                  << std::setw(10) << a.arithmetic_intensity
                  << std::setw(12) << a.achieved_tflops
                  << std::setw(15) << a.bound_region
                  << std::setw(12) << a.utilization << std::endl;
    }
}

// Generate ASCII roofline chart
inline void print_roofline_chart(
    const std::vector<KernelAnalysis>& analyses,
    const RooflineModel& model,
    int width = 60, int height = 20
) {
    std::cout << "\n=== Roofline Chart (ASCII) ===" << std::endl;
    
    // Find max AI and TFLOPS for scaling
    double max_ai = model.roofline_ai * 2;
    for (const auto& a : analyses) {
        if (a.arithmetic_intensity > max_ai) max_ai = a.arithmetic_intensity;
    }
    
    double max_tflops = model.peak_compute_tflops * 1.1;
    
    // Create chart grid
    std::vector<std::string> chart(height, std::string(width, ' '));
    
    // Draw roofline boundary
    for (int x = 0; x < width; ++x) {
        double ai = (double)x / width * max_ai;
        double tflops = model.achievable_tflops(ai);
        int y = height - 1 - (int)(tflops / max_tflops * (height - 1));
        if (y >= 0 && y < height) {
            chart[y][x] = '*';
        }
    }
    
    // Plot kernel points
    for (const auto& a : analyses) {
        int x = (int)(a.arithmetic_intensity / max_ai * (width - 1));
        int y = height - 1 - (int)(a.achieved_tflops / max_tflops * (height - 1));
        if (x >= 0 && x < width && y >= 0 && y < height) {
            chart[y][x] = 'K';
        }
    }
    
    // Print chart
    for (int y = 0; y < height; ++y) {
        if (y == 0) {
            std::cout << std::setw(6) << max_tflops << " |" << chart[y] << std::endl;
        } else if (y == height - 1) {
            std::cout << std::setw(6) << 0.0 << " |" << chart[y] << std::endl;
        } else {
            std::cout << "       |" << chart[y] << std::endl;
        }
    }
    std::cout << "       +" << std::string(width, '-') << std::endl;
    std::cout << "        0" << std::setw(width - 5) << max_ai << std::endl;
    std::cout << "                    Arithmetic Intensity (FLOP/byte)" << std::endl;
    
    std::cout << "\nLegend: * = roofline boundary, K = kernel point" << std::endl;
}

// Compare against theoretical peak
inline void print_performance_summary(
    const std::string& kernel_name,
    double achieved_tflops,
    double arithmetic_intensity,
    const RooflineModel& model
) {
    double theoretical = model.achievable_tflops(arithmetic_intensity);
    double utilization = (achieved_tflops / theoretical) * 100.0;
    std::string region = model.bound_region(arithmetic_intensity);
    
    std::cout << "\n=== Performance Summary: " << kernel_name << " ===" << std::endl;
    std::cout << "  Achieved:       " << std::fixed << std::setprecision(2) 
              << achieved_tflops << " TFLOPS" << std::endl;
    std::cout << "  Theoretical:    " << theoretical << " TFLOPS" << std::endl;
    std::cout << "  Utilization:    " << utilization << "%" << std::endl;
    std::cout << "  Bound Region:   " << region << std::endl;
    std::cout << "  Arithmetic Intensity: " << arithmetic_intensity << std::endl;
    
    if (utilization >= 90) {
        std::cout << "  Status: EXCELLENT (>90% utilization)" << std::endl;
    } else if (utilization >= 75) {
        std::cout << "  Status: GOOD (>75% utilization)" << std::endl;
    } else if (utilization >= 50) {
        std::cout << "  Status: MODERATE (>50% utilization)" << std::endl;
    } else {
        std::cout << "  Status: NEEDS OPTIMIZATION (<50% utilization)" << std::endl;
    }
}

} // namespace cutlass_roofline
