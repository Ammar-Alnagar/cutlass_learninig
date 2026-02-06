# Module 3: Basic Profiling Techniques

## Overview

In this module, we'll explore fundamental GPU profiling techniques and learn how to interpret profiling results. We'll focus on understanding the most important metrics and how they relate to performance.

## Learning Objectives

By the end of this module, you will:
- Understand key GPU profiling metrics
- Interpret profiling results correctly
- Identify basic performance bottlenecks
- Use profiling tools effectively
- Apply basic optimization strategies based on profiling data

## Key GPU Profiling Metrics

### Execution Metrics
- **Kernel Execution Time**: Total time spent executing the kernel
- **Grid Size**: Number of blocks launched
- **Block Size**: Number of threads per block
- **Active Warps**: Average number of active warps per SM

### Memory Metrics
- **DRAM Throughput**: Data transfer rate between GPU and memory
- **L2 Cache Hit Rate**: Percentage of L2 cache hits
- **Shared Memory Bank Conflicts**: Number of bank conflicts causing serialization

### Compute Metrics
- **Occupancy**: Ratio of active warps to maximum possible warps
- **Warp Execution Efficiency**: Percentage of active threads in warps
- **Branch Divergence**: Impact of divergent branches on performance

## Interpreting Profiling Results

Let's look at a sample profiling output and understand what each metric means:

```
==PROF== Profiling "vectorAdd" - section "SpeedOfLight.Collector.MemoryAccess":
==PROF== Event stats:
=="vectorAdd" (thread=28848)
Metric Name                    Metric Description               Min         Max         Avg
DRAM_READ_THROUGHPUT           DRAM Read Throughput (MB/s)      245.6       256.8       251.2
DRAM_WRITE_THROUGHPUT          DRAM Write Throughput (MB/s)     122.8       128.4       125.6
SM_OCCUPANCY                   SM Occupancy (%)                 50.0        50.0        50.0
ACHIEVED_OCCUPANCY             Achieved Occupancy (%)           37.5        37.5        37.5
WARP_EXECUTION_EFFICIENCY      Warp Execution Efficiency (%)    100.0       100.0       100.0
```

### Analysis of Sample Results:
1. **DRAM Throughput**: Both read and write throughput are moderate, suggesting memory isn't fully saturated
2. **Occupancy**: At 37.5%, this kernel has room for improvement (aim for >50%)
3. **Warp Execution Efficiency**: Perfect at 100%, meaning no inactive threads in warps

## Hands-On Exercise: Analyzing a Sample Kernel

Let's examine a kernel with intentional inefficiencies and profile it:

```cuda
// inefficient_kernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void inefficientKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Intentionally inefficient: causes branch divergence
        if (idx % 2 == 0) {
            output[idx] = input[idx] * 2.0f;
        } else {
            // Different computation path
            float temp = input[idx] * 1.5f;
            output[idx] = temp + 0.5f;
        }
        
        // Memory access pattern issue: strided access
        if (idx + 10 < n) {
            output[idx] += input[idx + 10];  // Non-coalesced access pattern
        }
    }
}

__global__ void efficientKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // More efficient: uniform execution
        float val = input[idx];
        output[idx] = val * 2.0f;
        
        // Coalesced memory access
        if (idx + 1 < n) {
            output[idx] += input[idx + 1];
        }
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output1 = (float*)malloc(bytes);
    float *h_output2 = (float*)malloc(bytes);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    // Allocate device memory
    float *d_input, *d_output1, *d_output2;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output1, bytes);
    cudaMalloc(&d_output2, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch inefficient kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    inefficientKernel<<<gridSize, blockSize>>>(d_input, d_output1, N);
    cudaDeviceSynchronize();
    
    // Launch efficient kernel
    efficientKernel<<<gridSize, blockSize>>>(d_input, d_output2, N);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_output1, d_output1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output1); free(h_output2);
    cudaFree(d_input); cudaFree(d_output1); cudaFree(d_output2);
    
    printf("Kernels executed successfully!\n");
    return 0;
}
```

### Profiling Commands

Compile and profile the inefficient kernel:

```bash
nvcc -o inefficient_kernel inefficient_kernel.cu

# Profile with key metrics
nvprof --metrics achieved_occupancy,warp_execution_efficiency,branch_efficiency, dram_read_throughput,dram_write_throughput ./inefficient_kernel

# Compare with efficient version
nvprof --metrics achieved_occupancy,warp_execution_efficiency,branch_efficiency, dram_read_throughput,dram_write_throughput ./efficient_kernel
```

## Identifying Common Issues

### 1. Low Occupancy
- **Symptom**: Occupancy < 50%
- **Cause**: Too few threads per block or too many resources per thread
- **Solution**: Increase block size or reduce register/shared memory usage

### 2. Poor Memory Throughput
- **Symptom**: DRAM throughput significantly below peak
- **Cause**: Non-coalesced memory accesses or insufficient memory-level parallelism
- **Solution**: Reorganize memory access patterns

### 3. Branch Divergence
- **Symptom**: Branch efficiency significantly < 100%
- **Cause**: Threads in a warp taking different execution paths
- **Solution**: Restructure algorithm to minimize divergent branches

## Profiling Best Practices

### 1. Warm-up Runs
Always run your kernel multiple times to account for initialization overhead:

```bash
# Run multiple times to get stable measurements
for i in {1..5}; do
    nvprof --print-gpu-trace ./kernel_executable
done
```

### 2. Consistent Workload
Ensure consistent input sizes and data patterns for reproducible results.

### 3. Isolate Kernels
Profile individual kernels separately when possible to identify specific bottlenecks.

## Hands-On Exercise

1. Compile and profile the inefficient kernel provided above
2. Note the differences in metrics between the inefficient and efficient versions
3. Try adjusting the block size and see how it affects occupancy
4. Experiment with different metrics to understand their relationships
5. Document your findings about how different code patterns affect performance metrics

## Key Takeaways

- Profiling reveals actual bottlenecks, not perceived ones
- Different metrics provide different insights into performance
- Small code changes can have significant performance impacts
- Always profile with realistic workloads

## Next Steps

In the next module, we'll focus specifically on identifying common bottlenecks and understanding their characteristics.