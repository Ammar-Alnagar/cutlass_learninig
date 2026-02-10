# Roofline Model

## Concept Overview

The Roofline model is a performance modeling tool that visualizes the relationship between operational intensity (arithmetic intensity) and performance for a given computing platform. It helps identify whether a kernel is compute-bound or memory-bound, and guides optimization efforts by showing the maximum achievable performance given the hardware's limitations.

## Understanding the Roofline Model

### Key Components
- **Operational Intensity (OI)**: Ratio of floating-point operations to bytes transferred (FLOPs/byte)
- **Performance**: Achieved performance in GFLOPs/sec
- **Memory Bandwidth Ceiling**: Maximum memory bandwidth of the system
- **Compute Ceiling**: Maximum compute throughput of the system

### Mathematical Foundation
```
Operational Intensity = FLOPs / Bytes
Performance = min(Memory_Bandwidth * Operational_Intensity, Compute_Peak)
```

### Roofline Plot Interpretation
- Points on the **slope** are **memory-bound**
- Points on the **flat top** are **compute-bound**
- Distance from the roof indicates optimization potential

## Roofline Model Formulae

### Memory Bound Region
```
Performance = Memory_Bandwidth * Operational_Intensity
```

### Compute Bound Region
```
Performance = Compute_Peak (constant)
```

### Boundary Point
```
Operational_Intensity_Boundary = Compute_Peak / Memory_Bandwidth
```

## Practical Roofline Analysis

### Calculating Operational Intensity
```cuda
// Example: Dense Matrix Multiplication (GEMM)
// C = A * B
// Operations: N^3 multiply-adds = 2*N^3 FLOPs
// Data: 3*N^2 elements * 4 bytes (float) = 12*N^2 bytes
// Operational Intensity = (2*N^3) / (12*N^2) = N/6 FLOPs/byte

__global__ void gemm_roofline(float* A, float* B, float* C, int N) {
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
```

### Memory-Bound Example
```cuda
// Vector Addition - Memory bound
// Operations: N additions = N FLOPs
// Data: 3*N elements * 4 bytes = 12*N bytes
// Operational Intensity = N / (12*N) = 1/12 FLOPs/byte

__global__ void vector_add_roofline(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // 1 FLOP, 12 bytes (3 reads + 1 write)
    }
}
```

### Compute-Bound Example
```cuda
// Compute-intensive kernel - Compute bound
// Operations: many FLOPs per memory access

__global__ void compute_bound_kernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        
        // Many operations per memory access
        for (int i = 0; i < 100; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
            x = x * 2.0f + 1.0f;
        }
        
        output[idx] = x;
    }
}
```

## Using Roofline for Optimization

### 1. Identify Bottleneck
- Calculate operational intensity of your kernel
- Measure achieved performance
- Plot on roofline to see if memory or compute bound

### 2. Memory-Bound Optimizations
- Improve memory access patterns (coalescing)
- Increase data reuse (blocking/tiled algorithms)
- Use faster memory types (shared, registers)

### 3. Compute-Bound Optimizations
- Optimize arithmetic operations
- Improve instruction-level parallelism
- Use specialized instructions (tensor cores)

## Roofline Analysis Example

```cpp
// Pseudo-code for roofline analysis
void perform_roofline_analysis() {
    // Measure performance and operational intensity
    double flops = count_flops_executed();
    double bytes = count_bytes_transferred();
    double operational_intensity = flops / bytes;
    
    double achieved_performance = flops / execution_time_seconds;
    
    // Get hardware specs
    double peak_bandwidth = get_memory_bandwidth_gbs();  // e.g., 800 GB/s
    double peak_compute = get_compute_peak_gflops();     // e.g., 15 TFLOPS
    
    // Calculate boundary
    double boundary_oi = peak_compute / peak_bandwidth;
    
    if (operational_intensity < boundary_oi) {
        printf("Kernel is MEMORY-BOUND\n");
        printf("Focus on improving data reuse and access patterns\n");
    } else {
        printf("Kernel is COMPUTE-BOUND\n");
        printf("Focus on arithmetic optimizations\n");
    }
    
    // Calculate potential improvement
    double optimal_performance = std::min(
        peak_bandwidth * operational_intensity,
        peak_compute
    );
    
    double improvement_potential = (optimal_performance - achieved_performance) / 
                                   achieved_performance * 100.0;
    printf("Potential improvement: %.2f%%\n", improvement_potential);
}
```

## Advanced Roofline Concepts

### 1. Multi-Level Roofline
Consider multiple memory hierarchy levels (L1, L2, DRAM) with different bandwidths.

### 2. Precision-Specific Roofline
Different data types (FP64, FP32, FP16, INT8) have different compute ceilings.

### 3. Architecture-Specific Models
Modern GPUs have specialized units (tensor cores) that create additional compute ceilings.

## Practical Roofline Implementation

```cpp
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
    
    void analyze_kernel(const std::string& name, 
                       double flops, double bytes, double time_sec) {
        double oi = flops / bytes;  // FLOPs/byte
        double perf = flops / time_sec / 1e9;  // GFLOPS
        
        printf("Kernel: %s\n", name.c_str());
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
        printf("\n");
    }
};
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Calculate operational intensity for any kernel
- Use the roofline model to determine if kernels are compute or memory bound
- Identify optimization opportunities based on roofline position
- Quantify potential performance improvements from optimizations

## Hands-on Tutorial

See the `roofline_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.