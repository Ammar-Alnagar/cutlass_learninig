/*
 * Roofline Model Tutorial
 * 
 * This tutorial demonstrates the roofline model for performance analysis.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Class to calculate and analyze roofline metrics
class RooflineAnalyzer {
public:
    struct KernelMetrics {
        double flops;
        double bytes;
        double execution_time_ms;
        double performance_gflops;
        double operational_intensity;
    };
    
    static void analyze_kernel(const KernelMetrics& metrics, 
                              double peak_bandwidth_GB_s, 
                              double peak_compute_GFLOPs) {
        double intensity_threshold = peak_compute_GFLOPs / peak_bandwidth_GB_s;
        
        printf("=== Roofline Analysis ===\n");
        printf("Operational Intensity: %.3f FLOP/Byte\n", metrics.operational_intensity);
        printf("Performance: %.2f GFLOP/s\n", metrics.performance_gflops);
        printf("Intensity Threshold: %.3f FLOP/Byte\n", intensity_threshold);
        
        if (metrics.operational_intensity < intensity_threshold) {
            printf("Status: MEMORY-BOUND\n");
            printf("Optimization Strategy: Focus on memory access patterns\n");
        } else {
            printf("Status: COMPUTE-BOUND\n");
            printf("Optimization Strategy: Focus on computational efficiency\n");
        }
        
        // Calculate efficiency
        double memory_efficiency = std::min(1.0, metrics.performance_gflops / 
                                          (peak_bandwidth_GB_s * metrics.operational_intensity)) * 100;
        double compute_efficiency = (metrics.performance_gflops / peak_compute_GFLOPs) * 100;
        
        printf("Memory Efficiency: %.2f%%\n", memory_efficiency);
        printf("Compute Efficiency: %.2f%%\n", compute_efficiency);
        printf("========================\n\n");
    }
};

// Kernel 1: Memory-bound operation (Vector Addition)
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 1 FLOP, 3*4 bytes = 12 bytes per element
    }
}

// Kernel 2: Compute-bound operation (Polynomial evaluation)
__global__ void polynomial_eval(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // 8 FLOPs: 7 multiplications + 1 addition
        float result = x*x*x*x*x*x*x*x + 2.0f*x*x*x*x*x*x*x + 3.0f*x*x*x*x*x*x + 
                      4.0f*x*x*x*x*x + 5.0f*x*x*x*x + 6.0f*x*x*x + 7.0f*x*x + 8.0f*x + 9.0f;
        output[idx] = result;
    }
}

// Kernel 3: Matrix multiplication (balanced)
__global__ void matrix_mult(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // 1 FMA = 2 FLOPs
        }
        C[row * N + col] = sum;
    }
}

// Kernel 4: SAXPY (Scale and Add) - moderately compute-intensive
__global__ void saxpy(float alpha, float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];  // 2 FLOPs, 2*4 bytes = 8 bytes per element
    }
}

// Function to measure kernel execution time
float measure_kernel_time(void (*kernel)(float*, float*, float*, int), 
                         float* a, float* b, float* c, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// Function to measure matrix multiplication time
float measure_matmul_time(float* A, float* B, float* C, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y);
    
    cudaEventRecord(start);
    matrix_mult<<<gridSize, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

int main() {
    printf("=== Roofline Model Tutorial ===\n\n");
    
    // Get device properties to determine peak performance
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Clock Rate: %d kHz\n", prop.clockRate);
    
    // Calculate approximate peak memory bandwidth (simplified)
    double peak_bandwidth_GB_s = (2.0 * prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8) / 1e9;
    printf("Estimated Peak Memory Bandwidth: %.2f GB/s\n", peak_bandwidth_GB_s);
    
    // Estimate peak compute performance (simplified)
    double cuda_cores = prop.multiProcessorCount * 64; // Rough estimate
    double clock_GHz = prop.clockRate / 1e6;
    double peak_compute_GFLOPs = 2 * cuda_cores * clock_GHz; // 2 ops per cycle per core (multiply-add)
    printf("Estimated Peak Compute Performance: %.2f GFLOP/s\n\n", peak_compute_GFLOPs);
    
    const int N = 1024 * 1024;  // 1M elements
    const int MAT_SIZE = 512;    // 512x512 matrix
    const int MAT_ELEMENTS = MAT_SIZE * MAT_SIZE;
    
    size_t size = N * sizeof(float);
    size_t mat_size = MAT_ELEMENTS * sizeof(float);
    
    // Allocate host memory
    float *h_a, *h_b, *h_c, *h_result;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_result = (float*)malloc(size);
    
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(mat_size);
    h_B = (float*)malloc(mat_size);
    h_C = (float*)malloc(mat_size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
        h_c[i] = 0.0f;
    }
    
    for (int i = 0; i < MAT_ELEMENTS; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
        h_C[i] = 0.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_result, size);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, mat_size);
    cudaMalloc(&d_B, mat_size);
    cudaMalloc(&d_C, mat_size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mat_size, cudaMemcpyHostToDevice);
    
    // Example 1: Vector Addition (Memory-bound)
    printf("1. Vector Addition (Memory-bound example):\n");
    printf("   Operation: C[i] = A[i] + B[i]\n");
    printf("   FLOPs per element: 1\n");
    printf("   Bytes per element: 12 (3 arrays * 4 bytes)\n");
    printf("   Operational intensity: 1/12 ≈ 0.08 FLOP/Byte\n");
    
    float time_vector_add = measure_kernel_time(vector_add, d_a, d_b, d_c, N);
    
    // Calculate metrics
    double flops_vector_add = N;  // 1 FLOP per element
    double bytes_vector_add = 3 * N * sizeof(float);  // 3 arrays accessed
    double perf_vector_add = (flops_vector_add / 1e9) / (time_vector_add / 1000.0);
    double intensity_vector_add = flops_vector_add / bytes_vector_add;
    
    RooflineAnalyzer::KernelMetrics metrics1 = {
        flops_vector_add, bytes_vector_add, time_vector_add, perf_vector_add, intensity_vector_add
    };
    
    RooflineAnalyzer::analyze_kernel(metrics1, peak_bandwidth_GB_s, peak_compute_GFLOPs);
    
    // Example 2: Polynomial Evaluation (Compute-bound)
    printf("2. Polynomial Evaluation (Compute-bound example):\n");
    printf("   Operation: High-degree polynomial evaluation\n");
    printf("   FLOPs per element: 8\n");
    printf("   Bytes per element: 8 (2 arrays * 4 bytes)\n");
    printf("   Operational intensity: 8/8 = 1.0 FLOP/Byte\n");
    
    float time_poly = measure_kernel_time(polynomial_eval, d_a, d_result, N);
    
    double flops_poly = N * 8;  // 8 FLOPs per element
    double bytes_poly = 2 * N * sizeof(float);  // 2 arrays accessed
    double perf_poly = (flops_poly / 1e9) / (time_poly / 1000.0);
    double intensity_poly = flops_poly / bytes_poly;
    
    RooflineAnalyzer::KernelMetrics metrics2 = {
        flops_poly, bytes_poly, time_poly, perf_poly, intensity_poly
    };
    
    RooflineAnalyzer::analyze_kernel(metrics2, peak_bandwidth_GB_s, peak_compute_GFLOPs);
    
    // Example 3: Matrix Multiplication (Balanced)
    printf("3. Matrix Multiplication (Balanced example):\n");
    printf("   Operation: C = A * B (512x512 matrices)\n");
    printf("   FLOPs: N^3 * 2 = 512^3 * 2 ≈ 268M FLOPs\n");
    printf("   Bytes: N^2 * 3 * 4 = 512^2 * 3 * 4 ≈ 3MB\n");
    printf("   Operational intensity: 268M/3M ≈ 85 FLOP/Byte\n");
    
    float time_matmul = measure_matmul_time(d_A, d_B, d_C, MAT_SIZE);
    
    double flops_matmul = (double)MAT_SIZE * MAT_SIZE * MAT_SIZE * 2;  // N^3 * 2 FLOPs
    double bytes_matmul = (double)MAT_SIZE * MAT_SIZE * 3 * sizeof(float);  // 3 matrices
    double perf_matmul = (flops_matmul / 1e9) / (time_matmul / 1000.0);
    double intensity_matmul = flops_matmul / bytes_matmul;
    
    RooflineAnalyzer::KernelMetrics metrics3 = {
        flops_matmul, bytes_matmul, time_matmul, perf_matmul, intensity_matmul
    };
    
    RooflineAnalyzer::analyze_kernel(metrics3, peak_bandwidth_GB_s, peak_compute_GFLOPs);
    
    // Example 4: SAXPY (Moderately compute-intensive)
    printf("4. SAXPY (Moderately compute-intensive example):\n");
    printf("   Operation: y[i] = alpha * x[i] + y[i]\n");
    printf("   FLOPs per element: 2\n");
    printf("   Bytes per element: 8 (2 arrays * 4 bytes)\n");
    printf("   Operational intensity: 2/8 = 0.25 FLOP/Byte\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    saxpy<<<(N + 255) / 256, 256>>>(2.0f, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float time_saxpy = 0;
    cudaEventElapsedTime(&time_saxpy, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    double flops_saxpy = N * 2;  // 2 FLOPs per element
    double bytes_saxpy = 2 * N * sizeof(float);  // 2 arrays accessed
    double perf_saxpy = (flops_saxpy / 1e9) / (time_saxpy / 1000.0);
    double intensity_saxpy = flops_saxpy / bytes_saxpy;
    
    RooflineAnalyzer::KernelMetrics metrics4 = {
        flops_saxpy, bytes_saxpy, time_saxpy, perf_saxpy, intensity_saxpy
    };
    
    RooflineAnalyzer::analyze_kernel(metrics4, peak_bandwidth_GB_s, peak_compute_GFLOPs);
    
    // Summary
    printf("Summary:\n");
    printf("- Vector Add: Memory-bound (low operational intensity)\n");
    printf("- Polynomial: Compute-bound (high operational intensity)\n");
    printf("- Matrix Mult: Compute-bound (very high operational intensity)\n");
    printf("- SAXPY: Moderately compute-bound\n");
    printf("\nThe roofline model helps identify whether a kernel is limited by memory bandwidth or compute capacity.\n");
    printf("This guides optimization strategies: memory-bound kernels need better access patterns,\n");
    printf("while compute-bound kernels need more efficient computation.\n");
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_result);
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\nTutorial completed!\n");
    return 0;
}