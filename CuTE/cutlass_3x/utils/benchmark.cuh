#pragma once
/**
 * CUTLASS 3.x Benchmark Utilities
 * 
 * Timing, TFLOPS calculation, and performance reporting.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

namespace cutlass_bench {

// CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// High-resolution timer for GPU kernels
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
    }

    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// Warmup + timed runs for stable benchmark
template <typename Kernel, typename... Args>
float benchmark_kernel(Kernel kernel, Args... args, 
                       int warmup_iters = 10, 
                       int timed_iters = 100) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        kernel(args...);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < timed_iters; ++i) {
        kernel(args...);
    }
    timer.stop();

    return timer.elapsed_ms() / timed_iters;
}

// Compute TFLOPS for GEMM: 2*M*N*K / time
inline double compute_gemm_tflops(int M, int N, int K, float elapsed_ms) {
    double flops = 2.0 * M * N * K;
    double seconds = elapsed_ms / 1000.0;
    return (flops / seconds) / 1e12;
}

// Compute TFLOPS for GEMM with batch
inline double compute_batched_gemm_tflops(int M, int N, int K, int batch, 
                                          float elapsed_ms) {
    double flops = 2.0 * M * N * K * batch;
    double seconds = elapsed_ms / 1000.0;
    return (flops / seconds) / 1e12;
}

// Print benchmark results table
struct BenchmarkResult {
    std::string name;
    int M, N, K;
    float elapsed_ms;
    double tflops;
    double percent_peak;
};

inline void print_benchmark_table(const std::vector<BenchmarkResult>& results) {
    std::cout << std::left << std::setw(25) << "Kernel"
              << std::right << std::setw(10) << "M"
              << std::setw(10) << "N"
              << std::setw(10) << "K"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "TFLOPS"
              << std::setw(15) << "% Peak" << std::endl;
    
    std::cout << std::string(94, '-') << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(25) << r.name
                  << std::right << std::setw(10) << r.M
                  << std::setw(10) << r.N
                  << std::setw(10) << r.K
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.elapsed_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.tflops
                  << std::setw(15) << std::fixed << std::setprecision(1) << r.percent_peak
                  << std::endl;
    }
}

// Theoretical peak TFLOPS per architecture
inline double get_theoretical_peak_tflops(const char* arch) {
    // Approximate values for common GPUs
    // These should be calibrated for specific hardware
    if (strcmp(arch, "SM80") == 0) {
        // A100: ~312 TFLOPS FP16 Tensor Core
        return 312.0;
    } else if (strcmp(arch, "SM90") == 0) {
        // H100: ~989 TFLOPS FP16 Tensor Core
        return 989.0;
    } else if (strcmp(arch, "SM100") == 0) {
        // B200: ~2250 TFLOPS FP8 Tensor Core (approx)
        return 2250.0;
    }
    return 0.0;
}

// Get current device architecture string
inline const char* get_device_arch() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    static char arch_str[16];
    snprintf(arch_str, sizeof(arch_str), "SM%d", prop.major * 10 + prop.minor);
    return arch_str;
}

// Print device info
inline void print_device_info() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
    std::cout << std::endl;
}

} // namespace cutlass_bench
