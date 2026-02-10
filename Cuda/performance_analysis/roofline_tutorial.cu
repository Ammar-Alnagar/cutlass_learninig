/*
 * CUDA Roofline Model Tutorial
 *
 * This tutorial demonstrates how to analyze kernels using the roofline model.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Kernel 1: Memory-bound example (vector addition)
__global__ void vector_add_roofline(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // 1 FLOP, ~12 bytes (3 reads + 1 write)
    }
}

// Kernel 2: Compute-bound example (many operations per memory access)
__global__ void compute_bound_kernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        
        // Many operations per memory access
        for (int i = 0; i < 50; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
            x = x * 2.0f + 1.0f;
        }
        
        output[idx] = x;
    }
}

// Kernel 3: Matrix multiplication (balanced kernel)
__global__ void gemm_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 4: Stream triad (classic roofline example)
__global__ void stream_triad(float* a, float* b, float* c, float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = b[idx] + scalar * c[idx];  // 2 ops, ~12 bytes
    }
}

// Helper function to measure execution time
float measureKernelTime(void (*kernel)(float*, float*, int), float* input, float* output, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<(n + 255) / 256, 256>>>(input, output, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0f;  // Return seconds
}

// Helper function to measure GEMM time
float measureGEMMTime(float* A, float* B, float* C, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    gemm_kernel<<<gridSize, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0f;  // Return seconds
}

// Roofline analysis class
class RooflineAnalyzer {
private:
    double peak_bandwidth;   // GB/s
    double peak_compute;     // GFLOPS
    double boundary_oi;      // Boundary operational intensity

public:
    RooflineAnalyzer(double bw, double comp) 
        : peak_bandwidth(bw), peak_compute(comp) {
        boundary_oi = peak_compute / peak_bandwidth;
    }
    
    void analyze_kernel(const char* name, 
                       double flops, double bytes, double time_sec) {
        double oi = flops / bytes;  // FLOPs/byte
        double perf = flops / time_sec / 1e9;  // GFLOPS
        
        printf("Kernel: %s\n", name);
        printf("  FLOPs: %.2e\n", flops);
        printf("  Bytes: %.2e\n", bytes);
        printf("  Time: %.6f sec\n", time_sec);
        printf("  Operational Intensity: %.3f FLOPs/byte\n", oi);
        printf("  Achieved Performance: %.2f GFLOPS\n", perf);
        printf("  Memory Bound Threshold: %.3f FLOPs/byte\n", boundary_oi);
        
        if (oi < boundary_oi) {
            printf("  Status: MEMORY-BOUND\n");
            printf("  Potential: %.2f GFLOPS at current OI\n", 
                   peak_bandwidth * oi);
        } else {
            printf("  Status: COMPUTE-BOUND\n");
            printf("  Potential: %.2f GFLOPS (compute ceiling)\n", 
                   peak_compute);
        }
        
        double ideal_perf = std::min(peak_bandwidth * oi, peak_compute);
        double efficiency = (perf / ideal_perf) * 100.0;
        printf("  Efficiency: %.2f%% of theoretical peak\n", efficiency);
        printf("\n");
    }
};

int main() {
    printf("=== CUDA Roofline Model Tutorial ===\n\n");

    // Typical GPU specs (adjust based on your hardware)
    // For example, RTX 3090: ~1.4 TB/s bandwidth, ~35 TFLOPS FP32
    double peak_bandwidth_GB_s = 936.0;  // Adjust for your GPU
    double peak_compute_GFLOPS = 35000.0;  // Adjust for your GPU
    
    RooflineAnalyzer analyzer(peak_bandwidth_GB_s, peak_compute_GFLOPS);

    const int N = 1024 * 1024;  // 1M elements
    const int MAT_SIZE = 1024;   // 1024x1024 matrix
    size_t size = N * sizeof(float);
    size_t mat_size = MAT_SIZE * MAT_SIZE * sizeof(float);

    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_input, *h_output;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
        h_C[i] = 0.0f;
        h_input[i] = i * 0.5f;
        h_output[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_input, *d_output;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Matrix memory allocations
    float *d_mat_A, *d_mat_B, *d_mat_C;
    cudaMalloc(&d_mat_A, mat_size);
    cudaMalloc(&d_mat_B, mat_size);
    cudaMalloc(&d_mat_C, mat_size);

    // Copy input data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Example 1: Memory-bound kernel (vector addition)
    printf("1. Memory-Bound Kernel Analysis (Vector Addition):\n");
    float time_vector_add = measureKernelTime(vector_add_roofline, d_A, d_C, N);
    
    // Calculate metrics for vector addition
    // Operations: N additions = N FLOPs
    // Data: 3*N elements * 4 bytes = 12*N bytes
    double flops_vector_add = N;
    double bytes_vector_add = 3.0 * N * sizeof(float);
    
    analyzer.analyze_kernel("Vector Addition", flops_vector_add, bytes_vector_add, time_vector_add);

    // Example 2: Compute-bound kernel
    printf("2. Compute-Bound Kernel Analysis:\n");
    float time_compute = measureKernelTime(compute_bound_kernel, d_input, d_output, N);
    
    // Calculate metrics for compute-bound kernel
    // Operations: N * 50 iterations * 3 ops per iter = 150*N FLOPs
    // Data: 2*N elements * 4 bytes = 8*N bytes
    double flops_compute = N * 50 * 3;  // 50 iterations * 3 ops per iteration
    double bytes_compute = 2.0 * N * sizeof(float);  // 2 reads + 1 write per output
    
    analyzer.analyze_kernel("Compute-Bound Kernel", flops_compute, bytes_compute, time_compute);

    // Example 3: Matrix multiplication (GEMM)
    printf("3. Matrix Multiplication Analysis (GEMM):\n");
    
    // Initialize matrices
    float *h_mat_A = (float*)malloc(mat_size);
    float *h_mat_B = (float*)malloc(mat_size);
    float *h_mat_C = (float*)malloc(mat_size);
    
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        h_mat_A[i] = 1.0f;
        h_mat_B[i] = 2.0f;
        h_mat_C[i] = 0.0f;
    }
    
    cudaMemcpy(d_mat_A, h_mat_A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_B, h_mat_B, mat_size, cudaMemcpyHostToDevice);
    
    float time_gemm = measureGEMMTime(d_mat_A, d_mat_B, d_mat_C, MAT_SIZE);
    
    // Calculate metrics for GEMM
    // Operations: MAT_SIZE^3 multiply-adds = 2*MAT_SIZE^3 FLOPs
    // Data: 3*MAT_SIZE^2 elements * 4 bytes = 12*MAT_SIZE^2 bytes
    double flops_gemm = 2.0 * MAT_SIZE * MAT_SIZE * MAT_SIZE;
    double bytes_gemm = 3.0 * MAT_SIZE * MAT_SIZE * sizeof(float);
    
    analyzer.analyze_kernel("Matrix Multiplication (GEMM)", flops_gemm, bytes_gemm, time_gemm);

    // Example 4: Stream triad
    printf("4. Stream Triad Analysis:\n");
    float time_stream = measureKernelTime(stream_triad, d_A, d_C, N);
    
    // Calculate metrics for stream triad: a[i] = b[i] + scalar * c[i]
    // Operations: N multiply-adds = 2*N FLOPs
    // Data: 3*N elements * 4 bytes = 12*N bytes
    double flops_stream = 2.0 * N;
    double bytes_stream = 3.0 * N * sizeof(float);
    
    analyzer.analyze_kernel("Stream Triad", flops_stream, bytes_stream, time_stream);

    // Example 5: Roofline visualization concepts
    printf("5. Roofline Model Concepts:\n");
    printf("   The roofline model plots performance vs operational intensity\n");
    printf("   - X-axis: Operational Intensity (FLOPs/byte)\n");
    printf("   - Y-axis: Performance (GFLOPS)\n");
    printf("   - Sloped region: Memory-bound (limited by bandwidth)\n");
    printf("   - Flat region: Compute-bound (limited by compute power)\n");
    printf("   - Distance from roof = optimization opportunity\n\n");

    printf("Key Insights:\n");
    printf("- Memory-bound kernels: Improve data reuse, access patterns\n");
    printf("- Compute-bound kernels: Optimize arithmetic, use specialized units\n");
    printf("- Roofline shows theoretical performance limits\n");
    printf("- Use to prioritize optimization efforts\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_input);
    free(h_output);
    free(h_mat_A);
    free(h_mat_B);
    free(h_mat_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mat_A);
    cudaFree(d_mat_B);
    cudaFree(d_mat_C);

    printf("\nTutorial completed!\n");
    return 0;
}