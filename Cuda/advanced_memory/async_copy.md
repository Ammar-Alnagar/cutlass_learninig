# Asynchronous Copy (cp.async/TMA)

## Concept Overview

Asynchronous copy operations allow data movement between global and shared memory to overlap with computation, significantly improving performance by hiding memory transfer latency. Modern CUDA provides two main approaches: `cp.async` instructions and Tensor Memory Accelerator (TMA).

## Understanding Asynchronous Memory Operations

### The Problem
Traditional memory copies block computation:
```cuda
// Synchronous copy - computation waits for copy to complete
__shared__ float shared_data[256];
global_data_to_shared_memory(global_ptr, shared_data, 256 * sizeof(float));
// Computation cannot start until copy completes
process_data(shared_data);
```

### The Solution
Asynchronous copies allow overlapping:
```cuda
// Asynchronous copy - copy happens in background
cp.async(start_copy, global_ptr, shared_data, 256 * sizeof(float));
// Computation can proceed while copy happens
process_other_data();  // Runs concurrently with copy
cp.async(wait_all);    // Wait for copy to complete before using data
process_data(shared_data);  // Safe to use copied data
```

## cp.async Instructions

### Basic Usage Pattern
```cuda
#include <cuda_pipeline.h>

__global__ void async_copy_example(float* global_input, float* global_output, int n) {
    __shared__ float shared_buffer[256];
    
    // Initialize pipeline
    cuda::pipeline<cuda::thread_scope_block> pipe;
    
    for (int i = 0; i < n; i += 256) {
        // Start asynchronous copy for next iteration
        pipe.producer_acquire();
        cuda::memcpy_async(shared_buffer, global_input + i, 256 * sizeof(float), pipe);
        pipe.producer_commit();
        
        // Process previous data while copy is happening
        if (i > 0) {
            // Process data from previous iteration
            process_data(shared_buffer);
        }
        
        // Wait for current copy to complete before next iteration
        pipe.consumer_wait();
        // Use the newly copied data
        process_data(shared_buffer);
        pipe.consumer_release();
    }
}
```

### Advanced cp.async Example
```cuda
// Two-buffer async copy pattern
__global__ void double_buffer_async(float* input, float* output, int n) {
    __shared__ float buffer_a[256];
    __shared__ float buffer_b[256];
    
    cuda::pipeline<cuda::thread_scope_block> pipe;
    
    float* current_buffer = buffer_a;
    float* next_buffer = buffer_b;
    
    // Prefetch first chunk
    pipe.producer_acquire();
    cuda::memcpy_async(current_buffer, input, 256 * sizeof(float), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();
    
    for (int i = 256; i < n; i += 256) {
        // Swap buffers
        float* temp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
        
        // Start copy of next chunk
        pipe.producer_acquire();
        cuda::memcpy_async(next_buffer, input + i, 256 * sizeof(float), pipe);
        pipe.producer_commit();
        
        // Process current chunk while next copy happens
        process_chunk(current_buffer, output + i - 256);
        
        // Wait for current copy to complete
        pipe.consumer_wait();
        pipe.consumer_release();
    }
    
    // Process final chunk
    process_chunk(current_buffer, output + ((n-1)/256)*256);
}
```

## Tensor Memory Accelerator (TMA)

### Introduction to TMA
TMA is a dedicated hardware unit in newer GPUs (Ampere and later) that handles complex multi-dimensional memory operations with automatic address calculation.

### TMA Descriptor Setup
```cuda
#include <cuda.h>
#include <cuda_runtime.h>

// TMA requires descriptor setup for memory layouts
__global__ void tma_example(half* input, half* output, int m, int n) {
    // TMA descriptors are typically set up on host and passed to kernel
    // This is a simplified example - real usage involves more setup
    
    // Example of TMA tile operations (conceptual)
    // Actual TMA usage requires descriptor creation on host
    /*
    // Host code would create descriptors like:
    CUtensorMap hTensorMap;
    cuTensorMapEncodeTiled(
        &hTensorMap,                    // descriptor
        CU_TENSOR_MAP_DATA_TYPE_HALF,   // data type
        2,                             // number of dimensions
        bounds,                        // size of each dimension
        base_ptr,                      // base address
        strides,                       // stride for each dimension
        element_strides,               // element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    */
    
    // In kernel: use TMA to load tiles
    // This is pseudocode as TMA requires specific setup
    /*
    __syncthreads(); // Ensure TMA descriptors are visible
    
    // Load tile using TMA
    tma_load_tile(shared_tile, tensor_map, coords);
    
    // Process tile
    process_tile(shared_tile);
    
    // Store result using TMA
    tma_store_tile(output_tensor, shared_tile, coords);
    */
}
```

### TMA Benefits
- Automatic address calculation for complex layouts
- Hardware-managed memory transactions
- Support for various data types and layouts
- Built-in support for padding and striding
- Better memory bandwidth utilization

## Practical Async Copy Implementation

### Simple Async Copy with Manual Pipeline
```cuda
// Manual implementation using traditional async copy techniques
__global__ void manual_async_copy(float* input, float* output, int n) {
    __shared__ float sdata[512];  // Two buffers of 256 each
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    
    // Phase 1: Load first chunk
    if (tid < 256) {
        sdata[tid] = input[bid * 512 + tid];
    }
    if (tid >= 256 && tid < 512) {
        sdata[tid] = input[bid * 512 + 256 + (tid - 256)];
    }
    __syncthreads();
    
    // Process first chunk while loading second
    float result = sdata[tid] * 2.0f;  // Example computation
    
    // Phase 2: Load second chunk while processing first
    __syncthreads();
    if (tid < 256) {
        sdata[tid + 256] = input[bid * 512 + 256 + tid];
    }
    if (tid >= 256) {
        sdata[tid - 256] = input[bid * 512 + 512 + (tid - 256)];  // Next block
    }
    
    // Finish processing first chunk
    output[bid * 512 + tid] = result;
    __syncthreads();
}
```

### Async Copy with Cooperative Groups
```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void coop_async_copy(float* input, float* output, int n) {
    extern __shared__ float shared_mem[];
    
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);  // Warp-level operations
    
    // Async copy using cooperative groups
    for (int i = 0; i < n; i += blockDim.x) {
        // Initiate async copy
        cg::memcpy_async(block, shared_mem, input + i, 
                         sizeof(float) * min(blockDim.x, n - i));
        
        // Wait for copy to complete
        cg::wait(block);
        
        // Process data
        int idx = i + threadIdx.x;
        if (idx < n) {
            shared_mem[threadIdx.x] *= 2.0f;
            output[idx] = shared_mem[threadIdx.x];
        }
    }
}
```

## Performance Benefits

### Latency Hiding
- Memory transfer latency hidden behind computation
- Effective memory bandwidth utilization
- Reduced idle time for SMs

### Throughput Improvement
- Overlapping memory and compute operations
- Better resource utilization
- Higher overall kernel throughput

## Best Practices

### 1. Proper Synchronization
Always ensure data is ready before using it:
```cuda
// Correct: wait before using data
pipe.consumer_wait();
use_copied_data();
pipe.consumer_release();

// Incorrect: using data before copy completes
use_copied_data(); // RACE CONDITION!
pipe.consumer_wait();
```

### 2. Adequate Buffering
Provide enough buffering to hide memory latency:
```cuda
// Good: 2+ buffers allow overlap
// Copy buffer N+2 while processing buffer N

// Insufficient: 1 buffer doesn't allow overlap
```

### 3. Memory Access Patterns
Combine with coalesced access patterns:
```cuda
// Ensure async copies also have coalesced access
// This maximizes the benefit of async operations
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Implement asynchronous data movement to overlap memory transfers with computation
- Use modern CUDA async copy instructions effectively
- Understand and utilize Tensor Memory Accelerator for complex memory operations
- Design multi-stage pipelines that hide memory latency with computation

## Hands-on Tutorial

See the `async_copy_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.