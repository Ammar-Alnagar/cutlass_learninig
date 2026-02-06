# Module 7: Advanced Profiling and Analysis

## Overview

In this module, we'll explore advanced profiling techniques and analysis methods that provide deeper insights into GPU performance. We'll learn to use sophisticated tools and interpret complex profiling data.

## Learning Objectives

By the end of this module, you will:
- Master advanced profiling tools like Nsight Compute and Nsight Systems
- Understand how to analyze complex profiling reports
- Learn to identify subtle performance bottlenecks
- Apply advanced optimization strategies based on profiling insights
- Create custom profiling workflows for specific use cases

## Advanced Profiling Tools

### Nsight Compute (NCU)

Nsight Compute provides detailed kernel-level profiling with extensive metrics:

```bash
# Basic profiling with Nsight Compute
ncu --target-processes all ./my_cuda_program

# Profile specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.bytes_per_second ./my_cuda_program

# Profile specific kernels
ncu --kernel-name "my_kernel" ./my_cuda_program

# Export results for detailed analysis
ncu --set full --export my_profile_report ./my_cuda_program
```

### Nsight Systems (NSYS)

Nsight Systems provides system-wide profiling and timeline analysis:

```bash
# Basic system profiling
nsys profile --trace=cuda,nvtx ./my_cuda_program

# Profile with specific options
nsys profile --trace-fork-before-exec=true --sample=cpu ./my_cuda_program

# Generate detailed report
nsys profile --export=sqlite --output=my_system_profile ./my_cuda_program
```

## Interpreting Advanced Profiling Data

### Understanding the Roofline Model

The roofline model helps visualize the relationship between arithmetic intensity and performance:

```
Performance (GFLOP/s)
     ^
     |        roof
     |       /
     |      /   compute-bound region
     |     /
     |    /
     |   /-------------------
     |  /   memory-bound region
     | /
     |/_____________________> 
     0                    Arithmetic Intensity (FLOP/byte)
```

### Key Metrics to Monitor

#### Compute Metrics:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: How much of theoretical compute capacity is utilized
- `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed`: SM activity percentage
- `issue_slot_utilization`: How effectively instructions are issued

#### Memory Metrics:
- `dram__throughput.avg.bytes_per_second`: Actual DRAM bandwidth utilization
- `lts__t_sectors_op_read.sum.per_second`: L2 cache read performance
- `gld_efficiency`: Global load coalescing efficiency
- `gst_efficiency`: Global store coalescing efficiency

#### Warp Execution Metrics:
- `warp_execution_efficiency`: Percentage of active threads in warps
- `branch_efficiency`: Efficiency of branch execution
- `sm__inst_executed_per_warp`: Instructions per warp (indicates divergence)

## Advanced Analysis Techniques

### 1. Correlation Analysis

Look for correlations between different metrics to identify root causes:

```bash
# Example: If occupancy is low, check resource usage
ncu --metrics achieved_occupancy,active_warps_per_active_cycle,registers_per_thread,shared_mem_per_block ./program

# If memory throughput is low, check access patterns
ncu --metrics dram__throughput.avg.bytes_per_second,gld_co_transactions_per_request,gst_co_transactions_per_request ./program
```

### 2. Bottleneck Identification Pipeline

```bash
# Step 1: Overall performance
ncu --metrics smsp__cycles_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed ./program

# Step 2: Identify bottleneck type
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second ./program

# Step 3: Drill down into specific areas
ncu --metrics gld_efficiency,gst_efficiency,shared_efficiency ./program

# Step 4: Examine resource usage
ncu --metrics registers_per_thread,shared_mem_per_block,achieved_occupancy ./program
```

### 3. Custom Metric Sets

Create custom metric sets for specific analysis:

```bash
# Memory access pattern analysis
ncu --metrics gld_efficiency,gst_efficiency,gld_throttle_reasons,gst_throttle_reasons ./program

# Compute intensity analysis
ncu --metrics flop_count_sp,dram__bytes_read.sum,dram__bytes_write.sum ./program

# Latency analysis
ncu --metrics smsp__inst_executed_per_second,smsp__inst_executed_per_warp ./program
```

## Hands-On Exercise: Advanced Analysis

Let's create a complex kernel that demonstrates various optimization opportunities:

```cuda
// advanced_analysis.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Complex kernel with multiple optimization opportunities
__global__ void complex_kernel(float *input, float *output, int n, int stride) {
    // Shared memory for data reuse
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Initialize shared memory
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    __syncthreads();
    
    if (idx < n) {
        float val = shared_data[tid];
        
        // Multiple memory accesses with stride
        if (idx + stride < n) {
            val += input[idx + stride];
        }
        
        // Computational loop
        for (int i = 0; i < 20; i++) {
            val = val * val + 0.1f;
            val = sqrtf(fmaxf(val, 1e-8f));
        }
        
        // Memory write with potential bank conflicts
        output[idx] = val * shared_data[(tid + 1) % blockDim.x];
    }
}

// Kernel with divergent branching
__global__ void divergent_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // Divergent branching based on data
        if (val > 0.5f) {
            // Expensive path
            for (int i = 0; i < 30; i++) {
                val = val * 1.01f + sinf(val);
            }
        } else if (val > 0.25f) {
            // Medium path
            for (int i = 0; i < 15; i++) {
                val = val * 1.02f + cosf(val);
            }
        } else {
            // Cheap path
            val = val * 2.0f;
        }
        
        output[idx] = val;
    }
}

// Optimized version of the complex kernel
__global__ void optimized_complex_kernel(float *input, float *output, int n, int stride) {
    // Use padded shared memory to avoid bank conflicts
    __shared__ float shared_data[256 + 1];  // +1 to avoid bank conflicts
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load with bounds checking
    shared_data[tid] = (idx < n) ? input[idx] : 0.0f;
    
    __syncthreads();
    
    if (idx < n) {
        float val = shared_data[tid];
        
        // Coalesced memory access
        if (idx + 1 < n) {
            val += input[idx + 1];
        }
        
        // Optimized computation with fewer operations
        for (int i = 0; i < 10; i++) {
            val = val * val * 0.9f + 0.1f;
        }
        
        // Avoid shared memory bank conflicts in write
        output[idx] = val * shared_data[tid];  // Use same index to avoid conflicts
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input with varying values to trigger divergence
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / N;  // Values from 0 to 1
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernels with different configurations
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Complex kernel with shared memory
    size_t sharedMemSize = blockSize * sizeof(float);
    complex_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, N, 10);
    cudaDeviceSynchronize();
    
    // Divergent kernel
    divergent_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Optimized kernel
    optimized_complex_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, N, 10);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("Advanced analysis kernels executed successfully!\n");
    return 0;
}
```

### Advanced Profiling Commands

```bash
# Compile the example
nvcc -o advanced_analysis advanced_analysis.cu

# Comprehensive profiling with Nsight Compute
ncu --set full --page detail --csv --log-file advanced_profile.csv ./advanced_analysis

# Focus on specific bottlenecks
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second,achieved_occupancy,branch_efficiency ./advanced_analysis

# Profile with timeline information
nsys profile --trace=cuda,nvtx --output=advanced_timeline ./advanced_analysis
```

## Creating Custom Profiling Workflows

### 1. Automated Profiling Script

```bash
#!/bin/bash
# advanced_profiling_script.sh

PROGRAM=$1
KERNEL_NAME=${2:-"*"}

echo "Starting advanced profiling for: $PROGRAM"

# Basic performance metrics
echo "=== Basic Performance ==="
ncu --metrics smsp__cycles_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second --kernel-name "$KERNEL_NAME" "$PROGRAM"

echo -e "\n=== Memory Analysis ==="
ncu --metrics gld_efficiency,gst_efficiency,shared_efficiency,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum --kernel-name "$KERNEL_NAME" "$PROGRAM"

echo -e "\n=== Compute Analysis ==="
ncu --metrics flop_count_sp,issue_slot_utilization,warp_execution_efficiency,branch_efficiency --kernel-name "$KERNEL_NAME" "$PROGRAM"

echo -e "\n=== Resource Usage ==="
ncu --metrics registers_per_thread,shared_mem_per_block,achieved_occupancy,requested occupancy --kernel-name "$KERNEL_NAME" "$PROGRAM"

echo -e "\n=== Exporting Full Report ==="
ncu --set full --export advanced_full_report --kernel-name "$KERNEL_NAME" "$PROGRAM"

echo "Profiling completed. Reports saved."
```

### 2. Performance Comparison Script

```bash
#!/bin/bash
# performance_comparison.sh

OLD_VERSION=$1
NEW_VERSION=$2

echo "Comparing performance between versions..."

# Profile old version
echo "Profiling old version..."
OLD_RESULT=$(ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second --csv --log-file old_performance.csv "$OLD_VERSION" 2>&1 | grep -A 10 "sm__throughput.avg.pct_of_peak_sustained_elapsed")

# Profile new version
echo "Profiling new version..."
NEW_RESULT=$(ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second --csv --log-file new_performance.csv "$NEW_VERSION" 2>&1 | grep -A 10 "sm__throughput.avg.pct_of_peak_sustained_elapsed")

echo "Old version performance: $OLD_RESULT"
echo "New version performance: $NEW_RESULT"

# Calculate improvement
# (Additional calculation logic would go here)
```

## Advanced Analysis Techniques

### 1. Statistical Analysis of Performance Data

When analyzing profiling results, look for statistical patterns:

- Mean vs. median performance across multiple runs
- Variance in performance metrics
- Correlation between different metrics

### 2. Scalability Analysis

Test how performance changes with problem size:

```bash
# Test different problem sizes
for size in 1024 2048 4096 8192; do
    echo "Testing size: $size"
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second ./program -- $size
done
```

### 3. Sensitivity Analysis

Test how performance changes with different parameters:

```bash
# Test different block sizes
for block_size in 64 128 256 512; do
    echo "Testing block size: $block_size"
    ncu --metrics achieved_occupancy,sm__throughput.avg.pct_of_peak_sustained_elapsed ./program -- $block_size
done
```

## Hands-On Exercise

1. Compile and run the advanced analysis kernels
2. Profile them using both Nsight Compute and Nsight Systems
3. Compare the complex and optimized kernels using the profiling data
4. Identify specific bottlenecks in the unoptimized version
5. Create a custom profiling workflow for your specific use case
6. Document your findings and optimization strategies

## Key Takeaways

- Advanced profiling tools provide deep insights into GPU performance
- Understanding metric correlations helps identify root causes
- Custom profiling workflows can automate performance analysis
- Statistical analysis of profiling data reveals important patterns

## Next Steps

In the final module, we'll work through real-world case studies and practice applying all the techniques learned throughout the course.