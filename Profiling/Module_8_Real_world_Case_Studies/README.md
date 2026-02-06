# Module 8: Real-world Case Studies and Practice

## Overview

In this final module, we'll work through real-world case studies that demonstrate how to apply all the techniques learned throughout the course. We'll tackle complete optimization problems from start to finish.

## Learning Objectives

By the end of this module, you will:
- Apply the complete GPU optimization workflow to real problems
- Analyze and optimize complex kernels from scratch
- Document performance improvements quantitatively
- Create optimization reports for stakeholders
- Develop skills for tackling new optimization challenges independently

## Case Study 1: Image Processing Kernel Optimization

### Problem Statement
Optimize a Gaussian blur kernel for image processing that currently has poor performance.

### Initial Analysis
```cuda
// gaussian_blur_initial.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Naive implementation of Gaussian blur
__global__ void gaussian_blur_naive(float *input, float *output, int width, int height, float *kernel, int kernel_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply convolution with Gaussian kernel
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int px = x + kx;
            int py = y + ky;
            
            // Handle boundary conditions
            if (px >= 0 && px < width && py >= 0 && py < height) {
                int pixel_idx = py * width + px;
                int kernel_idx = (ky + kernel_radius) * (2 * kernel_radius + 1) + (kx + kernel_radius);
                
                sum += input[pixel_idx] * kernel[kernel_idx];
                weight_sum += kernel[kernel_idx];
            }
        }
    }
    
    int idx = y * width + x;
    output[idx] = sum / weight_sum;
}
```

### Profiling the Initial Version
```bash
# Profile the naive implementation
nvprof --metrics dram_read_throughput,dram_write_throughput,gld_efficiency,gst_efficiency,achieved_occupancy ./gaussian_blur_naive

# Check memory access patterns
nvprof --print-gpu-trace ./gaussian_blur_naive
```

### Optimization Strategy
1. **Memory Access Pattern**: Use shared memory to cache image tiles
2. **Boundary Handling**: Pre-load boundary data to shared memory
3. **Coalescing**: Ensure coalesced access patterns

### Optimized Implementation
```cuda
// gaussian_blur_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define FILTER_SIZE (2 * KERNEL_RADIUS + 1)

__global__ void gaussian_blur_optimized(float *input, float *output, int width, int height, float *kernel) {
    // Shared memory with halo for convolution
    __shared__ float tile[TILE_SIZE + 2*KERNEL_RADIUS][TILE_SIZE + 2*KERNEL_RADIUS];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    // Load center tile
    if (x < width && y < height) {
        tile[ty + KERNEL_RADIUS][tx + KERNEL_RADIUS] = input[y * width + x];
    } else {
        tile[ty + KERNEL_RADIUS][tx + KERNEL_RADIUS] = 0.0f;
    }
    
    // Load halo regions
    // Top and bottom halos
    if (ty < KERNEL_RADIUS) {
        int load_y_top = y - KERNEL_RADIUS;
        int load_y_bottom = y + TILE_SIZE;
        
        if (load_y_top >= 0 && x < width) {
            tile[ty][tx + KERNEL_RADIUS] = input[load_y_top * width + x];
        } else {
            tile[ty][tx + KERNEL_RADIUS] = 0.0f;
        }
        
        if (load_y_bottom < height && x < width) {
            tile[ty + TILE_SIZE + KERNEL_RADIUS][tx + KERNEL_RADIUS] = input[load_y_bottom * width + x];
        } else {
            tile[ty + TILE_SIZE + KERNEL_RADIUS][tx + KERNEL_RADIUS] = 0.0f;
        }
    }
    
    // Left and right halos
    if (tx < KERNEL_RADIUS) {
        int load_x_left = x - KERNEL_RADIUS;
        int load_x_right = x + TILE_SIZE;
        
        if (load_x_left >= 0 && y < height) {
            tile[ty + KERNEL_RADIUS][tx] = input[y * width + load_x_left];
        } else {
            tile[ty + KERNEL_RADIUS][tx] = 0.0f;
        }
        
        if (load_x_right < width && y < height) {
            tile[ty + KERNEL_RADIUS][tx + TILE_SIZE + KERNEL_RADIUS] = input[y * width + load_x_right];
        } else {
            tile[ty + KERNEL_RADIUS][tx + TILE_SIZE + KERNEL_RADIUS] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Apply convolution using shared memory
    if (x < width && y < height) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = 0; ky < FILTER_SIZE; ky++) {
            for (int kx = 0; kx < FILTER_SIZE; kx++) {
                int shared_x = tx + kx;
                int shared_y = ty + ky;
                
                int kernel_idx = ky * FILTER_SIZE + kx;
                sum += tile[shared_y][shared_x] * kernel[kernel_idx];
                weight_sum += kernel[kernel_idx];
            }
        }
        
        int idx = y * width + x;
        output[idx] = sum / weight_sum;
    }
}
```

## Case Study 2: Matrix Multiplication Optimization

### Problem Statement
Optimize a matrix multiplication kernel that exhibits poor cache performance.

### Initial Analysis
```cuda
// matmul_initial.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Naive matrix multiplication
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
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

### Optimization Strategy
1. **Tiling**: Use shared memory to cache matrix tiles
2. **Coalescing**: Ensure coalesced access patterns
3. **Loop restructuring**: Optimize memory access order

### Optimized Implementation
```cuda
// matmul_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_optimized(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < N; t += TILE_SIZE) {
        // Load tiles into shared memory
        As[ty][tx] = (row < N && t+tx < N) ? A[row * N + t + tx] : 0.0f;
        Bs[ty][tx] = (t+ty < N && col < N) ? B[(t + ty) * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Case Study 3: Reduction Operation Optimization

### Problem Statement
Optimize a parallel reduction kernel that has poor occupancy and warp divergence.

### Initial Analysis
```cuda
// reduction_initial.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Basic reduction with poor performance
__global__ void reduction_basic(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes one element
    float sum = (idx < n) ? input[idx] : 0.0f;
    
    // Reduction within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __shared__ float temp[1024];  // Assuming max block size
        if (threadIdx.x % (2 * stride) == 0) {
            temp[threadIdx.x] = sum;
        }
        __syncthreads();
        
        if (threadIdx.x % (2 * stride) == 0) {
            sum += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}
```

### Optimization Strategy
1. **Warp-level primitives**: Use warp shuffle operations
2. **Memory coalescing**: Optimize shared memory access
3. **Occupancy**: Improve thread utilization

### Optimized Implementation
```cuda
// reduction_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

__device__ float warpReduce(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_optimized(float *input, float *output, int n) {
    __shared__ float sdata[32]; // One element per warp (assuming 1024 threads/block -> 32 warps)
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Each thread loads two elements and sums them
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    
    // Reduce within warp using shuffle
    sum = warpReduce(sum);
    
    // Write reduced value to shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    
    __syncthreads();
    
    // Final reduce within block
    if (tid < 32) {
        sum = sdata[tid];
        sum = warpReduce(sum);
        if (tid == 0) output[blockIdx.x] = sum;
    }
}
```

## Complete Optimization Workflow

### Step 1: Baseline Measurement
```bash
# Establish baseline performance
nvprof --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second,achieved_occupancy ./baseline_version
```

### Step 2: Bottleneck Identification
```bash
# Identify primary bottlenecks
ncu --metrics gld_efficiency,gst_efficiency,branch_efficiency,achieved_occupancy ./problematic_kernel
```

### Step 3: Targeted Optimization
Apply specific optimizations based on identified bottlenecks.

### Step 4: Validation and Measurement
```bash
# Validate correctness and measure improvement
./optimized_version
nvprof --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second ./optimized_version
```

### Step 5: Iterative Refinement
Continue optimizing until performance goals are met.

## Performance Reporting Template

When documenting optimization results, include:

### Executive Summary
- Performance improvement achieved (% speedup)
- Primary bottlenecks addressed
- Key optimization techniques applied

### Technical Details
- Baseline performance metrics
- Optimized performance metrics
- Specific changes made
- Trade-offs considered

### Validation
- Correctness verification
- Testing methodology
- Edge cases covered

## Hands-On Exercise: Complete Optimization Challenge

Choose one of the following challenges and apply the complete optimization workflow:

### Challenge 1: Histogram Computation
Optimize a histogram computation kernel that currently has race conditions and poor memory access patterns.

### Challenge 2: Prefix Sum (Scan) Operation
Optimize a parallel prefix sum operation with poor inter-warp coordination.

### Challenge 3: Sparse Matrix-Vector Multiplication
Optimize SpMV with irregular memory access patterns and load imbalance.

## Implementation Template

```cuda
// optimization_template.cu
#include <cuda_runtime.h>
#include <stdio.h>

// TODO: Implement your kernel here
// Follow the optimization workflow:
// 1. Profile baseline version
// 2. Identify bottlenecks
// 3. Apply optimizations
// 4. Measure improvement
// 5. Iterate as needed

int main() {
    // TODO: Set up test data
    // TODO: Launch kernel
    // TODO: Validate results
    // TODO: Clean up
    
    printf("Optimization challenge completed!\n");
    return 0;
}
```

## Key Takeaways

- Real-world optimization requires systematic analysis
- Multiple iterations are typically needed for significant improvements
- Always validate correctness after optimization
- Document your optimization process for future reference
- Performance bottlenecks often shift as you optimize

## Congratulations!

You've completed the GPU Kernel Profiling and Optimization Mastery course! You now have the knowledge and skills to:

1. Profile GPU applications effectively
2. Identify performance bottlenecks accurately
3. Apply appropriate optimization techniques
4. Measure and document performance improvements
5. Tackle new optimization challenges independently

Continue practicing with real-world problems to solidify your skills. Remember that optimization is both an art and a science - the more you practice, the better you'll become at identifying and solving performance challenges.