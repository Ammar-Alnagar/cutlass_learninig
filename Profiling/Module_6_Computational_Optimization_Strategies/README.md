# Module 6: Computational Optimization Strategies

## Overview

While memory optimization often provides the biggest performance gains, computational optimization is equally important for maximizing GPU throughput. In this module, we'll explore techniques to optimize arithmetic operations and computational patterns.

## Learning Objectives

By the end of this module, you will:
- Understand arithmetic intensity and its impact on performance
- Learn to optimize mathematical operations for GPU execution
- Apply loop optimization techniques for GPU kernels
- Use built-in GPU functions effectively
- Balance computation and memory access for optimal performance

## Arithmetic Intensity

Arithmetic intensity is the ratio of floating-point operations to memory operations. Higher arithmetic intensity means more computation per memory access, which can help hide memory latency.

### Calculating Arithmetic Intensity
```
Arithmetic Intensity = FLOPs / Bytes accessed
```

### Example:
```cuda
// Low arithmetic intensity (memory bound)
__global__ void low_arith_intensity(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // 1 FLOP, 8 bytes (read+write)
    }
}

// High arithmetic intensity (compute bound)
__global__ void high_arith_intensity(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Many FLOPs per memory access
        for (int i = 0; i < 100; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
            x = x * 2.0f + 1.0f;
        }
        output[idx] = x;
    }
}
```

## Mathematical Function Optimization

### 1. Use Appropriate Precision
- Use single precision (float) when double precision (double) isn't required
- Consider half precision (half) for even better performance when accuracy permits

```cuda
#include <cuda_fp16.h>

// Single precision - faster
__global__ void single_precision(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);  // Single precision sqrt
    }
}

// Half precision - fastest
__global__ void half_precision(half *input, half *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = hsqrt(input[idx]);  // Half precision sqrt
    }
}
```

### 2. Use Intrinsic Functions
CUDA provides intrinsic functions that map directly to hardware operations:

```cuda
// Standard math functions
__global__ void standard_math(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // May use slower library implementations
        output[idx] = sinf(input[idx]) + cosf(input[idx]);
    }
}

// Intrinsic functions - faster but less accurate
__global__ void intrinsic_math(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Hardware-accelerated approximations
        output[idx] = __sinf(input[idx]) + __cosf(input[idx]);
    }
}
```

### 3. Optimize Common Operations
```cuda
// Suboptimal: division is expensive
__global__ void division_example(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / 3.14159f;  // Division operation
    }
}

// Optimized: multiply by reciprocal
__global__ void multiplication_example(float *input, float *output, int n) {
    const float inv_pi = 1.0f / 3.14159f;  // Computed once
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * inv_pi;  // Multiplication is faster
    }
}
```

## Loop Optimization Techniques

### 1. Loop Unrolling
Reduce loop overhead by processing multiple iterations per loop:

```cuda
// Standard loop
__global__ void standard_loop(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            if (idx + i < n) {
                sum += input[idx + i];
            }
        }
        output[idx] = sum;
    }
}

// Unrolled loop
__global__ void unrolled_loop(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        if (idx < n) sum += input[idx];
        if (idx + 1 < n) sum += input[idx + 1];
        if (idx + 2 < n) sum += input[idx + 2];
        if (idx + 3 < n) sum += input[idx + 3];
        output[idx] = sum;
    }
}
```

### 2. Loop Fusion
Combine multiple loops to reduce memory passes:

```cuda
// Separate loops - multiple memory passes
__global__ void separate_loops(float *a, float *b, float *c, float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // First loop
        c[idx] = a[idx] * 2.0f;
    }
    
    // Second loop - requires synchronization
    if (idx < n) {
        d[idx] = c[idx] + 1.0f;
    }
}

// Fused loop - single memory pass
__global__ void fused_loop(float *a, float *b, float *c, float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = a[idx] * 2.0f;
        c[idx] = temp;
        d[idx] = temp + 1.0f;
    }
}
```

## Occupancy and Resource Optimization

### Managing Register Usage
High register usage limits occupancy:

```cuda
// High register usage - limits occupancy
__global__ void high_register_usage(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
        float r10, r11, r12, r13, r14, r15, r16, r17, r18, r19;
        
        r0 = input[idx];
        // ... many register operations
        output[idx] = r19;
    }
}

// Lower register usage - better occupancy
__global__ void low_register_usage(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // Process in-place to reduce register pressure
        val = val * 2.0f;
        val = val + 1.0f;
        output[idx] = val;
    }
}
```

### Using Local Memory Sparingly
Local memory is actually global memory, so avoid unnecessary usage:

```cuda
// Avoid: Large arrays in local memory
__global__ void bad_local_memory(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp_array[100];  // Goes to local memory (slow!)
        temp_array[0] = input[idx];
        // ... process with temp array
        output[idx] = temp_array[0];
    }
}

// Better: Use registers or shared memory
__global__ void good_local_memory(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp_val = input[idx];  // Use register
        // ... process with temp_val
        output[idx] = temp_val;
    }
}
```

## Hands-On Exercise: Computational Optimization

Let's create kernels demonstrating different computational optimization techniques:

```cuda
// computational_optimization.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Unoptimized computation kernel
__global__ void unoptimized_computation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Inefficient: using division instead of multiplication
        x = x / 2.0f;
        // Inefficient: using slow math functions unnecessarily
        x = powf(x, 2.0f);
        output[idx] = x;
    }
}

// Optimized computation kernel
__global__ void optimized_computation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Efficient: multiply by reciprocal
        x = x * 0.5f;
        // Efficient: use faster intrinsic if precision allows
        x = x * x;  // Instead of powf(x, 2.0f)
        output[idx] = x;
    }
}

// Kernel with optimized loop unrolling
__global__ void loop_unroll_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Grid-stride loop with unrolling
    for (int i = idx; i < n; i += stride * 4) {
        // Process 4 elements per thread
        if (i < n) output[i] = input[i] * input[i];
        if (i + 1 < n) output[i + 1] = input[i + 1] * input[i + 1];
        if (i + 2 < n) output[i + 2] = input[i + 2] * input[i + 2];
        if (i + 3 < n) output[i + 3] = input[i + 3] * input[i + 3];
    }
}

// Kernel with reduced register usage
__global__ void low_reg_usage_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // Process in pipeline to reduce register pressure
        val *= 2.0f;
        val += 1.0f;
        val = fmaxf(val, 0.0f);  // ReLU activation
        output[idx] = val;
    }
}

// High arithmetic intensity kernel
__global__ void high_arith_intens_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        
        // High computation-to-memory ratio
        for (int i = 0; i < 50; i++) {
            x = x * x + 0.1f;
            x = sqrtf(fmaxf(x, 1e-8f));
            x = x * 0.9f + 0.1f * input[idx];  // Mix with original
        }
        
        output[idx] = x;
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
        h_input[i] = (float)(i % 1000) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch different kernels
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Unoptimized kernel
    unoptimized_computation<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Optimized kernel
    optimized_computation<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Loop unrolling kernel
    loop_unroll_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Low register usage kernel
    low_reg_usage_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // High arithmetic intensity kernel
    high_arith_intens_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("Computational optimization kernels executed successfully!\n");
    return 0;
}
```

### Profiling Computational Optimizations

```bash
# Compile the example
nvcc -o computational_optimization computational_optimization.cu

# Profile computational metrics
nvprof --metrics flop_count_sp,sm_efficiency,achieved_occupancy,instructions_per_warp ./computational_optimization

# Profile register usage
nvprof --metrics registers_per_thread,achieved_occupancy ./computational_optimization

# Compare arithmetic intensity
nvprof --metrics flop_count_sp,gld_transactions,gst_transactions ./computational_optimization
```

## Computational Optimization Checklist

Before optimizing computation, consider:

1. **Precision Requirements**: Can you use single or half precision instead of double?
2. **Mathematical Functions**: Are you using the most efficient functions for your accuracy needs?
3. **Loop Structure**: Can loops be unrolled or fused to reduce overhead?
4. **Register Pressure**: Are you using more registers than necessary?
5. **Arithmetic Intensity**: Can you increase computation per memory access?

## Hands-On Exercise

1. Compile and profile the unoptimized and optimized computational kernels
2. Compare the FLOP count and occupancy metrics between them
3. Implement a kernel that performs a computationally intensive operation (like matrix multiplication with many operations per element)
4. Profile the kernel and identify opportunities for computational optimization
5. Try different optimization techniques and measure their impact on performance

## Key Takeaways

- Arithmetic intensity affects whether your kernel is compute or memory bound
- Choosing the right mathematical functions can significantly impact performance
- Loop optimization can reduce overhead and improve performance
- Balancing computation and memory access is key to optimal performance

## Next Steps

In the next module, we'll explore advanced profiling techniques and analysis methods to gain deeper insights into GPU performance.