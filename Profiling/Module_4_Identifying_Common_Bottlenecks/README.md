# Module 4: Identifying Common Bottlenecks

## Overview

In this module, we'll focus on recognizing and diagnosing the most common performance bottlenecks in GPU kernels. Understanding these patterns is crucial for effective optimization.

## Learning Objectives

By the end of this module, you will:
- Recognize the most common GPU performance bottlenecks
- Understand the symptoms of each bottleneck type
- Learn diagnostic techniques for identifying bottlenecks
- Practice using profiling tools to detect bottlenecks
- Apply basic fixes for common issues

## Common GPU Bottleneck Types

### 1. Memory-Bound Bottlenecks

**Characteristics:**
- Performance limited by memory bandwidth rather than computation
- DRAM throughput is close to theoretical maximum
- Little improvement from algorithmic optimizations

**Symptoms:**
- High memory utilization metrics
- Low compute-to-global-memory-access ratio (GMEM ratio)
- Performance scales with memory bandwidth

**Detection:**
```bash
# Check memory throughput
nvprof --metrics dram_read_throughput,dram_write_throughput,gld_throughput,gst_throughput ./kernel

# Calculate GMEM ratio (compute intensity)
nvprof --metrics flop_count_sp,global_load_throughput,global_store_throughput ./kernel
```

### 2. Compute-Bound Bottlenecks

**Characteristics:**
- Performance limited by computational throughput
- Arithmetic intensity is high
- Performance improves with faster compute units

**Symptoms:**
- High arithmetic instruction counts
- Low memory pressure
- SM utilization is high

**Detection:**
```bash
# Check compute metrics
nvprof --metrics flop_count_sp,sm_efficiency,achieved_occupancy ./kernel
```

### 3. Occupancy-Limited Bottlenecks

**Characteristics:**
- Not enough concurrent threads to hide memory latency
- Low occupancy reduces ability to overlap memory and computation

**Symptoms:**
- Occupancy significantly below 100%
- Potential for more active warps per SM

**Detection:**
```bash
# Check occupancy metrics
nvprof --metrics achieved_occupancy,active_warps_per_active_cycle,inst_per_warp ./kernel
```

### 4. Branch Divergence Bottlenecks

**Characteristics:**
- Threads in a warp take different execution paths
- Causes serialization of divergent paths

**Symptoms:**
- Branch efficiency significantly below 100%
- Higher execution time than expected

**Detection:**
```bash
# Check branch metrics
nvprof --metrics branch_efficiency,branch,divergent_branch ./kernel
```

## Diagnostic Patterns

### Pattern 1: Memory-Bound Detection
If your kernel shows:
- DRAM utilization > 80% of peak
- GMEM ratio < 0.5 (operations per global memory access)
- Little improvement from algorithmic changes

Then you likely have a memory-bound bottleneck.

### Pattern 2: Compute-Bound Detection
If your kernel shows:
- High FLOP counts
- Low memory pressure
- SM utilization near 100%

Then you likely have a compute-bound bottleneck.

### Pattern 3: Occupancy-Limited Detection
If your kernel shows:
- Occupancy < 50%
- Active warps per cycle much lower than theoretical maximum
- Register or shared memory usage limiting occupancy

Then you likely have an occupancy-limited bottleneck.

## Hands-On Exercise: Bottleneck Identification

Let's create kernels that exhibit different bottleneck types:

```cuda
// bottleneck_examples.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Memory-bound example: lots of memory access, little computation
__global__ void memory_bound_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Multiple memory accesses per computation
        float sum = 0.0f;
        sum += input[idx];
        if (idx + 1 < n) sum += input[idx + 1];
        if (idx + 2 < n) sum += input[idx + 2];
        if (idx + 3 < n) sum += input[idx + 3];
        if (idx + 4 < n) sum += input[idx + 4];
        
        output[idx] = sum / 5.0f;  // Minimal computation
    }
}

// Compute-bound example: lots of computation, minimal memory access
__global__ void compute_bound_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        
        // Heavy computation per memory access
        for (int i = 0; i < 100; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
            x = x * 2.0f + 1.0f;
        }
        
        output[idx] = x;
    }
}

// Occupancy-limited example: high register usage
__global__ void occupancy_limited_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Use many registers to limit occupancy
        float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
        float r10, r11, r12, r13, r14, r15, r16, r17, r18, r19;
        
        r0 = input[idx];
        r1 = r0 * 1.1f; r2 = r1 * 1.2f; r3 = r2 * 1.3f; r4 = r3 * 1.4f;
        r5 = r4 * 1.5f; r6 = r5 * 1.6f; r7 = r6 * 1.7f; r8 = r7 * 1.8f;
        r9 = r8 * 1.9f; r10 = r9 * 2.0f; r11 = r10 * 2.1f; r12 = r11 * 2.2f;
        r13 = r12 * 2.3f; r14 = r13 * 2.4f; r15 = r14 * 2.5f; r16 = r15 * 2.6f;
        r17 = r16 * 2.7f; r18 = r17 * 2.8f; r19 = r18 * 2.9f;
        
        output[idx] = r19;
    }
}

// Branch-divergent example
__global__ void branch_divergent_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // Conditional execution causing divergence
        if (val > 0.5f) {
            // Expensive path
            for (int i = 0; i < 10; i++) {
                val = val * val + 0.1f;
            }
        } else {
            // Cheap path
            val = val * 2.0f;
        }
        
        output[idx] = val;
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 1000) / 1000.0f;  // Values between 0 and 1
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch different kernels and profile each
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Memory-bound kernel
    memory_bound_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Compute-bound kernel
    compute_bound_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Occupancy-limited kernel
    occupancy_limited_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Branch-divergent kernel
    branch_divergent_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("All kernels executed successfully!\n");
    return 0;
}
```

### Profiling Commands for Bottleneck Detection

```bash
# Compile the example
nvcc -o bottleneck_examples bottleneck_examples.cu

# Profile memory-bound kernel with memory metrics
nvprof --metrics dram_read_throughput,dram_write_throughput,global_load_throughput,global_store_throughput ./bottleneck_examples

# Profile compute-bound kernel with compute metrics
nvprof --metrics flop_count_sp,flop_count_dp,sm_efficiency,achieved_occupancy ./bottleneck_examples

# Profile occupancy-limited kernel
nvprof --metrics achieved_occupancy,active_warps_per_active_cycle,inst_per_warp,registers_per_thread ./bottleneck_examples

# Profile branch-divergent kernel
nvprof --metrics branch_efficiency,branch,divergent_branch,instruction_mixed ./bottleneck_examples
```

## Bottleneck Diagnosis Workflow

### Step 1: High-Level Assessment
1. Run a basic profile to get execution time and basic metrics
2. Look at SM efficiency and achieved occupancy
3. Check memory throughput values

### Step 2: Deep Dive Analysis
Based on initial assessment, focus on relevant metrics:
- If occupancy is low → investigate resource usage
- If memory throughput is high → investigate memory access patterns
- If compute metrics are high → investigate arithmetic intensity

### Step 3: Verification
Test hypotheses by making targeted changes and re-profiling.

## Common Bottleneck Fixes

### Memory-Bound Solutions:
- Improve memory access patterns (coalescing)
- Use shared memory for data reuse
- Reduce memory footprint

### Compute-Bound Solutions:
- Optimize algorithms
- Use more efficient math functions
- Consider precision requirements

### Occupancy-Limited Solutions:
- Reduce register usage
- Adjust block size
- Reduce shared memory usage

### Branch-Divergence Solutions:
- Restructure algorithms to minimize divergence
- Separate divergent paths into different kernels

## Hands-On Exercise

1. Compile and profile the bottleneck examples provided
2. Identify which metrics indicate each type of bottleneck
3. Try adjusting block sizes and see how it affects occupancy
4. Modify one kernel to reduce its bottleneck (e.g., make the memory-bound kernel more compute-efficient)
5. Document your findings about how different bottlenecks manifest in profiling data

## Key Takeaways

- Different bottlenecks show different metric signatures
- Accurate diagnosis requires looking at multiple related metrics
- The same kernel may have different bottlenecks at different scales
- Always verify your diagnosis by making targeted changes

## Next Steps

In the next module, we'll focus specifically on memory optimization techniques to address memory-bound bottlenecks.