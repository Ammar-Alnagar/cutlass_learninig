# Roofline Model

## Concept Overview

The Roofline model is a performance modeling tool that visualizes the relationship between operational intensity (arithmetic intensity) and performance for computational kernels. It helps identify whether a kernel is compute-bound or memory-bound and guides optimization strategies.

## Understanding the Roofline Model

### Key Concepts

The Roofline model plots:
- **X-axis**: Operational Intensity (Operational Intensity = FLOPs / Byte of memory traffic)
- **Y-axis**: Performance (GFLOP/s or GB/s)

The model has two main "roofs":
1. **Memory Roof**: Maximum achievable memory bandwidth
2. **Compute Roof**: Maximum achievable compute performance

### Mathematical Foundation

```
Operational Intensity = Total FLOPs / Total Bytes accessed
Performance = min(Memory Roof, Compute Roof * Operational Intensity)
```

## Visual Representation

```
Performance (GFLOP/s)
     ^
     |          / 
     |         /   <- Compute Roof (compute-bound region)
     |        /
     |       / 
     |      /| 
     |     / | 
     |    /  | 
     |   /   | 
     |  /    | 
     | /     | 
     |/      | <- Memory Roof (memory-bound region)
     +-------------------------->
     0    Intensity Threshold   Operational Intensity (FLOP/Byte)
```

## Calculating Operational Intensity

### Example 1: Vector Addition
```cuda
// C[i] = A[i] + B[i] for i = 0 to N-1
// Operations: N additions
// Memory: 3*N*sizeof(float) bytes (read A, read B, write C)
// Operational Intensity = N FLOPs / (3*N*4 bytes) = 1/12 ≈ 0.08 FLOP/Byte
```

### Example 2: Matrix Multiplication
```cuda
// C = A * B (NxN matrices)
// Operations: N³ multiply-adds = 2*N³ FLOPs
// Memory: 3*N²*sizeof(float) bytes (read A, read B, write C)
// Operational Intensity = 2*N³ / (3*N²*4) = N/6 FLOP/Byte
// As N increases, becomes more compute-intensive
```

### Example 3: SAXPY (Scale and Add)
```cuda
// Y[i] = alpha*X[i] + Y[i] for i = 0 to N-1
// Operations: N multiplications + N additions = 2*N FLOPs
// Memory: 2*N*sizeof(float) + N*sizeof(float) = 3*N*4 bytes
// Operational Intensity = 2*N / (3*N*4) = 1/6 ≈ 0.17 FLOP/Byte
```

## Identifying Performance Bottlenecks

### Memory-Bound Kernels
- Low operational intensity (< intensity threshold)
- Performance limited by memory bandwidth
- Located in the flat portion of the roofline plot
- Optimization strategy: Improve memory access patterns, coalescing, caching

### Compute-Bound Kernels
- High operational intensity (> intensity threshold)
- Performance limited by compute capacity
- Located in the sloped portion of the roofline plot
- Optimization strategy: Improve arithmetic intensity, use tensor cores, optimize instruction mix

## Practical Roofline Analysis

### Step-by-Step Analysis Process

1. **Measure Performance**
   ```bash
   # Using Nsight Compute
   ncu --metrics flop_count_sp,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,bytes_throughput ./your_kernel
   
   # Or using nvprof (legacy)
   nvprof --metrics flop_count_sp,gld_transactions,gst_transactions ./your_kernel
   ```

2. **Calculate Operational Intensity**
   ```cpp
   // Example calculation
   long long flops = /* measured FLOP count */;
   long long bytes = /* measured memory traffic */;
   double operational_intensity = (double)flops / bytes;
   ```

3. **Calculate Performance**
   ```cpp
   // Performance in GFLOP/s
   double elapsed_time_seconds = /* kernel execution time */;
   double performance_gflops = (double)flops / elapsed_time_seconds / 1e9;
   ```

4. **Determine Hardware Limits**
   ```cpp
   // Example for a GPU with 900 GB/s memory bandwidth and 15 TFLOP/s peak FP32
   double memory_roof = 900.0;  // GB/s
   double compute_roof = 15000.0;  // GFLOP/s
   double intensity_threshold = compute_roof / memory_roof;  // 15000/900 = 16.67 FLOP/Byte
   ```

### Example Analysis Code

```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

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
        
        std::cout << "=== Roofline Analysis ===" << std::endl;
        std::cout << "Operational Intensity: " << metrics.operational_intensity << " FLOP/Byte" << std::endl;
        std::cout << "Performance: " << metrics.performance_gflops << " GFLOP/s" << std::endl;
        std::cout << "Intensity Threshold: " << intensity_threshold << " FLOP/Byte" << std::endl;
        
        if (metrics.operational_intensity < intensity_threshold) {
            std::cout << "Status: MEMORY-BOUND" << std::endl;
            std::cout << "Optimization Strategy: Focus on memory access patterns" << std::endl;
        } else {
            std::cout << "Status: COMPUTE-BOUND" << std::endl;
            std::cout << "Optimization Strategy: Focus on computational efficiency" << std::endl;
        }
        
        // Calculate efficiency
        double memory_efficiency = metrics.performance_gflops / peak_bandwidth_GB_s / metrics.operational_intensity * 100;
        double compute_efficiency = metrics.performance_gflops / peak_compute_GFLOPs * 100;
        
        std::cout << "Memory Efficiency: " << memory_efficiency << "%" << std::endl;
        std::cout << "Compute Efficiency: " << compute_efficiency << "%" << std::endl;
    }
};
```

## Optimization Strategies Based on Roofline Position

### For Memory-Bound Kernels
1. **Improve Data Reuse**: Increase arithmetic intensity by reusing data
2. **Optimize Memory Access**: Ensure coalesced access patterns
3. **Use Faster Memory**: Leverage shared memory, registers, or texture memory
4. **Reduce Memory Traffic**: Minimize unnecessary data transfers

### For Compute-Bound Kernels
1. **Increase Arithmetic Intensity**: Pack more computation per memory access
2. **Use Specialized Units**: Leverage tensor cores, specialized instructions
3. **Optimize Instruction Mix**: Reduce divergent branching, improve ILP
4. **Increase Occupancy**: Ensure sufficient parallelism to hide latency

## Real-World Examples

### Example 1: Convolution Layer (Often Memory-Bound)
```cuda
// Convolution typically has low arithmetic intensity
// FLOPs: O(kernel_size² * output_pixels)
// Memory: O(input_pixels + weights + output_pixels)
// Often memory-bound due to large memory footprint relative to computation
```

### Example 2: Matrix Multiplication (Often Compute-Bound for large matrices)
```cuda
// Large GEMM operations have high arithmetic intensity
// FLOPs: O(N³) for NxN matrices
// Memory: O(N²) for data movement
// Becomes compute-bound as N increases
```

## Limitations of Roofline Model

### Simplifications
- Assumes ideal memory and compute behavior
- Doesn't account for cache effects in detail
- Ignores synchronization overhead
- May not capture all architectural complexities

### Extensions
- Multi-roofline models for different data types
- Cache-aware roofline models
- Power-aware roofline models

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Determine if kernels are compute or memory bound using the roofline model
- Calculate operational intensity for different algorithms
- Use the model to identify and prioritize optimization opportunities
- Apply appropriate optimization strategies based on roofline position

## Hands-on Tutorial

See the `roofline_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.