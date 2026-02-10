# Occupancy

## Concept Overview

Occupancy refers to the ratio of active warps to the maximum number of warps that can reside on a Streaming Multiprocessor (SM) simultaneously. It's a critical factor in achieving good performance on GPUs, as sufficient occupancy is needed to hide memory latency through warp scheduling.

## Understanding Occupancy

### Definition
Occupancy = (Number of active warps per SM) / (Maximum number of warps per SM)

### Why Occupancy Matters
- GPUs rely on having many warps available to schedule during memory stalls
- Higher occupancy generally means better latency hiding
- However, maximum occupancy doesn't always mean maximum performance

## Factors Affecting Occupancy

### 1. Resource Constraints
Three main resources limit occupancy:

#### Registers per Thread
- Each thread reserves registers for its lifetime
- More registers per thread = fewer threads per block = lower occupancy
- Formula: `max_threads_per_SM = (registers_per_SM / registers_per_thread) * 32`

#### Shared Memory per Block
- Each block reserves shared memory for its lifetime
- More shared memory per block = fewer blocks per SM = lower occupancy
- Formula: `max_blocks_per_SM = shared_memory_per_SM / shared_memory_per_block`

#### Threads per Block
- Hardware limits on total threads per SM
- Formula: `max_blocks_per_SM = max_threads_per_SM / threads_per_block`

### 2. Occupancy Calculation Example

For a GPU with:
- 65,536 registers per SM
- 48 KB shared memory per SM
- 2048 max threads per SM
- 32 max warps per SM

If a kernel uses:
- 32 registers per thread
- 16 KB shared memory per block
- 256 threads per block

Then:
- Max blocks limited by registers: `(65536 / (32 * 32)) * 32 = 2048` warps / 256 threads = 8 blocks
- Max blocks limited by shared memory: `49152 / 16384 = 3` blocks
- Max blocks limited by threads: `2048 / 256 = 8` blocks
- Actual limit: `min(8, 3, 8) = 3` blocks per SM
- Active warps: `3 * (256 / 32) = 24` warps
- Occupancy: `24 / 32 = 75%`

## Measuring Occupancy

### Compile-time Information
```bash
# Using nvcc to get register usage
nvcc -Xptxas -v your_kernel.cu
# Output shows register usage per thread
```

### Runtime Calculation
```cuda
// Calculate theoretical occupancy
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, yourKernel, 0, 0);

// Calculate occupancy for specific configuration
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, yourKernel, blockSize, 0);
```

### Using Occupancy Calculator
```cuda
#include <cuda_runtime.h>

void printOccupancyInfo(void (*kernel)(float*, int), int blockSize) {
    int device;
    cudaGetDevice(&device);
    
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, kernel, blockSize, 0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int maxActiveBlocks = prop.maxThreadsPerMultiProcessor / blockSize;
    float occupancy = (float)activeBlocks / maxActiveBlocks;
    
    printf("Active blocks per SM: %d\n", activeBlocks);
    printf("Max possible blocks per SM: %d\n", maxActiveBlocks);
    printf("Occupancy: %.2f%%\n", occupancy * 100);
}
```

## Optimizing Occupancy

### 1. Reduce Register Usage
```cuda
// High register usage
__global__ void high_registers(float* data, int n) {
    float a = data[threadIdx.x];      // Uses many registers
    float b = a * 2.0f;
    float c = b + 1.0f;
    float d = c * a;
    float e = d - b;
    // ... more operations using many variables
    data[threadIdx.x] = e;
}

// Lower register usage - compiler can reuse registers
__global__ void low_registers(float* data, int n) {
    float val = data[threadIdx.x];
    val = val * 2.0f;
    val = val + 1.0f;
    val = val * data[threadIdx.x];
    val = val - (data[threadIdx.x] * 2.0f);
    data[threadIdx.x] = val;
}
```

### 2. Control Register Usage with Launch Bounds
```cuda
// Limit register usage to optimize occupancy
__global__ 
__launch_bounds__(256, 4)  // Up to 256 threads per block, at least 4 blocks per SM
void optimized_kernel(float* data, int n) {
    // Kernel code here
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] *= 2.0f;
    }
}
```

### 3. Balance Resources
```cuda
// Example of balancing shared memory and threads
__global__ void balanced_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    // Use moderate amount of shared memory to allow more blocks per SM
    if (tid < n) {
        sdata[local_tid] = input[tid];
        __syncthreads();
        
        // Process data
        sdata[local_tid] *= 2.0f;
        __syncthreads();
        
        output[tid] = sdata[local_tid];
    }
}
```

## Occupancy vs Performance

### Important Considerations
- Higher occupancy doesn't always mean better performance
- Sometimes fewer, more computationally intensive threads perform better
- Memory-bound kernels benefit more from high occupancy
- Compute-bound kernels may perform well with lower occupancy

### Finding the Sweet Spot
```cuda
// Benchmark different block sizes to find optimal performance
void benchmark_occupancy() {
    const int blockSizes[] = {32, 64, 128, 256, 512};
    const int numSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    for (int i = 0; i < numSizes; i++) {
        int blockSize = blockSizes[i];
        
        // Calculate occupancy
        int activeBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, testKernel, blockSize, 0);
        
        // Time the kernel
        auto start = std::chrono::high_resolution_clock::now();
        
        int gridSize = (N + blockSize - 1) / blockSize;
        testKernel<<<gridSize, blockSize>>>(data, N);
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("BlockSize: %d, Occupancy: %d blocks/SM, Time: %ld μs\n", 
               blockSize, activeBlocks, duration.count());
    }
}
```

## Occupancy Guidelines

### General Rules of Thumb
- Aim for at least 50% occupancy (but not at the expense of algorithm efficiency)
- For memory-bound kernels: higher occupancy is usually better
- For compute-bound kernels: focus on computational efficiency over occupancy
- Use `__launch_bounds__` to guide compiler optimization
- Profile to find the actual performance sweet spot

### When Lower Occupancy is Acceptable
- Compute-intensive kernels with high arithmetic intensity
- Kernels with complex control flow that benefit from more registers
- Cases where memory bandwidth is the limiting factor

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Calculate theoretical occupancy for any kernel configuration
- Balance resource usage to achieve sufficient occupancy for latency hiding
- Use profiling tools to measure actual occupancy
- Understand when high occupancy is beneficial vs. when it's not necessary

## Hands-on Tutorial

See the `occupancy_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.