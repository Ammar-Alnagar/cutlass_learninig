# Module 5: Memory Optimization Techniques

## Overview

Memory optimization is often the most impactful area for GPU performance improvements. In this module, we'll explore various techniques to optimize memory access patterns and maximize memory bandwidth utilization.

## Learning Objectives

By the end of this module, you will:
- Understand coalesced vs. uncoalesced memory access patterns
- Learn to use shared memory effectively for data reuse
- Master texture memory for specific access patterns
- Apply memory optimization techniques to real kernels
- Measure the impact of memory optimizations

## Memory Hierarchy in GPUs

GPUs have a complex memory hierarchy that affects performance significantly:

1. **Registers**: Fastest storage, per-thread, limited quantity
2. **Shared Memory**: Fast on-chip memory shared among threads in a block
3. **Global Memory**: Main GPU memory, high bandwidth but higher latency
4. **Constant Memory**: Cached memory for read-only data
5. **Texture Memory**: Cached memory optimized for spatial locality

## Coalesced Memory Access

### What is Coalescing?

Coalescing occurs when threads in a warp access contiguous memory locations. This allows the GPU to combine multiple memory requests into fewer, wider transactions.

### Good Coalescing Example:
```cuda
// Coalesced access - consecutive threads access consecutive memory
__global__ void coalesced_access(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;  // Consecutive threads access consecutive addresses
    }
}
```

### Bad Coalescing Example:
```cuda
// Uncoalesced access - threads access memory with stride
__global__ void uncoalesced_access(float *data, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid * stride] = data[tid * stride] * 2.0f;  // Strided access pattern
    }
}
```

## Shared Memory Optimization

Shared memory is a critical resource for optimizing GPU kernels. It provides much faster access than global memory when used correctly.

### Basic Shared Memory Usage:
```cuda
// Example: Using shared memory for matrix transpose
#define TILE_SIZE 32

__global__ void matrixTranspose(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile into shared memory
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (y+j < height && x < width) {
            tile[threadIdx.y+j][threadIdx.x] = input[(y+j)*width + x];
        }
    }
    
    __syncthreads();
    
    // Transpose and write out
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (y+j < width && x < height) {
            output[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y+j];
        }
    }
}
```

### Shared Memory Bank Conflicts:
- GPUs have 32 shared memory banks
- Accessing different addresses in the same bank serializes the access
- Adding padding (e.g., `[TILE_SIZE+1]`) can prevent conflicts

## Memory Optimization Techniques

### Technique 1: Memory Padding to Prevent Bank Conflicts
```cuda
// Without padding - potential bank conflicts
__shared__ float sdata[32][32];  // 32 columns = 32 banks, potential conflicts

// With padding - prevents bank conflicts
__shared__ float sdata[32][33];  // Extra column prevents conflicts
```

### Technique 2: Tiled Access Patterns
```cuda
// Process data in tiles to maximize data reuse
#define TILE_WIDTH 16

__global__ void tiled_kernel(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;
    
    // Load tile
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    } else {
        tile[ty][tx] = 0.0f;  // Boundary condition
    }
    
    __syncthreads();
    
    // Process tile and write result
    if (x < width && y < height) {
        // Perform computation using tile[ty][tx]
        output[y * width + x] = tile[ty][tx] * 2.0f;
    }
}
```

### Technique 3: Texture Memory for Irregular Access
```cuda
// Texture memory is cached and optimized for spatial locality
texture<float, 1, cudaReadModeElementType> tex_data;

__global__ void texture_kernel(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Texture memory handles irregular access patterns well
        output[idx] = tex1D(tex_data, idx) * 2.0f;
    }
}
```

## Hands-On Exercise: Memory Optimization

Let's create kernels demonstrating different memory optimization techniques:

```cuda
// memory_optimization.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Unoptimized kernel with poor memory access
__global__ void unoptimized_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Strided access pattern - poor coalescing
        int stride_idx = (idx * 3) % n;
        output[idx] = input[stride_idx] * 2.0f;
    }
}

// Optimized kernel with coalesced access
__global__ void optimized_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Sequential access pattern - good coalescing
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel using shared memory for reduction
__global__ void shared_mem_reduction(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    if (i + blockDim.x < n) {
        sdata[tid] += input[i + blockDim.x];
    }
    
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Matrix multiplication with shared memory tiling
#define TILE_SIZE 16

__global__ void matmul_shared_mem(float *A, float *B, float *C, int N) {
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

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
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
    unoptimized_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Optimized kernel
    optimized_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("Memory optimization kernels executed successfully!\n");
    
    // Also demonstrate matrix multiplication
    const int MAT_SIZE = 512;
    const int mat_bytes = MAT_SIZE * MAT_SIZE * sizeof(float);
    
    float *h_A = (float*)malloc(mat_bytes);
    float *h_B = (float*)malloc(mat_bytes);
    float *h_C = (float*)malloc(mat_bytes);
    
    // Initialize matrices
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, mat_bytes);
    cudaMalloc(&d_B, mat_bytes);
    cudaMalloc(&d_C, mat_bytes);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_bytes, cudaMemcpyHostToDevice);
    
    // Launch matrix multiplication kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((MAT_SIZE + TILE_SIZE - 1) / TILE_SIZE, (MAT_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_shared_mem<<<dimGrid, dimBlock, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, d_B, d_C, MAT_SIZE);
    cudaDeviceSynchronize();
    
    // Cleanup matrices
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    printf("Matrix multiplication with shared memory completed!\n");
    
    return 0;
}
```

### Profiling Memory Optimizations

```bash
# Compile the example
nvcc -o memory_optimization memory_optimization.cu

# Profile memory access patterns
nvprof --metrics dram_read_throughput,dram_write_throughput,gld_efficiency,gst_efficiency ./memory_optimization

# Profile shared memory usage
nvprof --metrics shared_load_throughput,shared_store_throughput,shared_efficiency ./memory_optimization

# Compare coalescing efficiency
nvprof --metrics gld_co_transactions_per_request,gst_co_transactions_per_request ./memory_optimization
```

## Memory Optimization Checklist

Before optimizing memory access, consider:

1. **Access Pattern**: Are threads accessing consecutive memory locations?
2. **Data Reuse**: Can shared memory be used to store frequently accessed data?
3. **Bank Conflicts**: Are there conflicts in shared memory access?
4. **Memory Footprint**: Can data structures be packed more efficiently?
5. **Cache Locality**: Would texture memory benefit irregular access patterns?

## Hands-On Exercise

1. Compile and profile the unoptimized and optimized kernels
2. Compare the memory throughput metrics between them
3. Implement a kernel that performs matrix transpose and optimize it using shared memory
4. Profile the shared memory kernel and note improvements in memory efficiency
5. Experiment with different tile sizes in the matrix multiplication example and measure performance

## Key Takeaways

- Memory coalescing is crucial for optimal performance
- Shared memory can dramatically improve performance when used correctly
- Always profile memory metrics to validate optimizations
- Small changes in access patterns can lead to significant performance gains

## Next Steps

In the next module, we'll explore computational optimization strategies to improve arithmetic intensity and computational efficiency.