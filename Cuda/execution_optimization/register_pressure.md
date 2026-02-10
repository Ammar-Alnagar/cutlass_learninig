# Register Pressure and GPU Memory Model

## Concept Overview

Register pressure refers to the constraint on the number of registers each thread can use, which directly affects occupancy and performance. The GPU memory model defines the rules for how memory operations behave in a parallel execution environment, including visibility and ordering guarantees. Understanding both concepts is crucial for optimizing CUDA kernels.

## Register Pressure

### What are Registers?
- Fastest memory on GPU (single-cycle access)
- Thread-local storage
- Limited quantity per SM
- Each thread reserves registers for its entire lifetime

### Register Allocation
- Compiler assigns variables to registers
- Spill to local memory if register limit exceeded
- Register usage affects occupancy calculations

### Effects of Register Pressure
- **High register usage** → Fewer threads per block → Lower occupancy
- **Register spilling** → Slow local memory access → Performance degradation
- **Resource competition** → Limits parallelism

## Managing Register Pressure

### 1. Compiler Directives
```cuda
// Limit register usage to optimize occupancy
__global__ 
__launch_bounds__(256, 4)  // At most 256 threads/block, at least 4 blocks/SM
void optimized_kernel(float* data, int n) {
    // Kernel code here
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] *= 2.0f;
    }
}
```

### 2. Code Restructuring
```cuda
// High register usage
__global__ void high_register_usage(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float a = data[tid];
        float b = a * 2.0f;
        float c = b + 1.0f;
        float d = c * a;
        float e = d - b;
        float f = e * 0.5f;
        float g = f + a;
        // Many variables in use simultaneously
        data[tid] = a + b + c + d + e + f + g;
    }
}

// Lower register usage - reuse variables
__global__ void low_register_usage(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];
        val = val * 2.0f;
        val = val + 1.0f;
        val = val * data[tid];
        val = val - (data[tid] * 2.0f);
        data[tid] = val;
    }
}
```

### 3. Using Local Arrays Sparingly
```cuda
// High register pressure - array allocated in registers
__global__ void high_reg_array(float* input, float* output, int n) {
    float local_array[16];  // May consume many registers
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Use local_array...
        output[tid] = local_array[0];
    }
}

// Better - use shared memory for larger local data
__global__ void shared_memory_approach(float* input, float* output, int n) {
    extern __shared__ float sdata[];  // Use shared memory instead
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    if (tid < n) {
        sdata[local_tid] = input[tid];
        // Process in shared memory...
        output[tid] = sdata[local_tid];
    }
}
```

## Monitoring Register Usage

### Compile-Time Information
```bash
# Get register usage information
nvcc -Xptxas -v your_kernel.cu
# Output includes register usage per thread
```

### Runtime Information
```cuda
// Get register usage programmatically
struct cudaFuncAttributes attr;
cudaFuncGetAttributes(&attr, your_kernel);
printf("Registers per thread: %d\n", attr.numRegs);
printf("Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
```

## GPU Memory Model

### Relaxed Consistency Model
- Unlike CPUs, GPUs implement a relaxed memory consistency model
- Threads may observe memory operations in different orders
- Hardware optimizations reorder operations for performance
- Explicit synchronization required for ordering guarantees

### Memory Ordering Levels
- **Within a thread**: Sequential consistency maintained
- **Within a warp**: Generally consistent but not guaranteed
- **Within a block**: Requires `__syncthreads()` for ordering
- **Across blocks**: No ordering guarantees without global synchronization

### Memory Visibility
```cuda
// Example of memory visibility issues
__global__ void visibility_example(int* data, int* flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        data[0] = 42;      // Write data
        __threadfence();   // Ensure visibility
        flag[0] = 1;       // Signal other threads
    }
    
    if (tid == 1) {
        while (flag[0] == 0) { /* wait */ }
        // Without __threadfence(), not guaranteed to see data[0] = 42
        int val = data[0];   // May see old value without proper fencing
    }
}
```

### Memory Fence Types
```cuda
// Block-level fence
__threadfence_block();  // All memory ops in block visible to other threads in block

// Grid-level fence  
__threadfence();        // All memory ops visible to all threads in grid

// System-level fence
__threadfence_system(); // All memory ops visible to CPU and other devices
```

## Memory Model Implications

### 1. Race Conditions
```cuda
// Unsafe code without proper synchronization
__global__ void unsafe_race(int* data) {
    int tid = threadIdx.x;
    
    // Multiple threads accessing same location without synchronization
    data[0] = tid;  // Race condition!
}

// Safe code with atomics
__global__ void safe_atomic(int* data) {
    int tid = threadIdx.x;
    
    // Atomic operation ensures safety
    atomicAdd(&data[0], tid);
}
```

### 2. False Sharing Prevention
```cuda
// Problem: False sharing due to adjacent memory access
struct BadExample {
    int counter[32];  // Adjacent elements may share cache line
};

__global__ void bad_false_sharing(BadExample* data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < 32) {
        data->counter[tid]++;  // Different threads accessing same cache line
    }
}

// Solution: Padding to prevent false sharing
struct GoodExample {
    int counter[32];
    char padding[32];  // Prevent adjacent cache line sharing
};
```

### 3. Coherence Between CPU and GPU
```cuda
// Ensuring coherence when CPU and GPU access same memory
void cpu_gpu_coherence_example() {
    float *host_data, *device_data;
    
    // Allocate unified memory
    cudaMallocManaged(&host_data, N * sizeof(float));
    cudaMalloc(&device_data, N * sizeof(float));
    
    // Use data on GPU
    gpu_kernel<<<blocks, threads>>>(host_data, N);
    cudaDeviceSynchronize();
    
    // Ensure GPU writes are visible to CPU
    cudaDeviceSynchronize();  // Implicit memory fence
    
    // Now safe to use host_data on CPU
    cpu_function(host_data, N);
}
```

## Optimization Strategies

### 1. Balancing Register Usage
- Profile register usage with `nvprof` or Nsight Compute
- Use `__launch_bounds__` to guide compiler
- Restructure code to reduce live variable count
- Consider algorithm trade-offs between registers and shared memory

### 2. Memory Model Compliance
- Always use appropriate synchronization primitives
- Apply memory fences when ordering matters
- Use atomics for race condition prevention
- Understand visibility guarantees for your use case

## Performance Impact

### Register Pressure Effects
- **Low pressure**: Higher occupancy, better latency hiding
- **High pressure**: Lower occupancy, potential register spilling
- **Sweet spot**: Balance between register usage and occupancy

### Memory Model Effects
- **Improper synchronization**: Incorrect results, race conditions
- **Over-synchronization**: Reduced parallelism, performance loss
- **Right balance**: Correct results with optimal performance

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Monitor and optimize register usage to prevent spilling and occupancy loss
- Understand the GPU's relaxed memory consistency model and its implications
- Apply appropriate synchronization mechanisms to ensure correctness
- Balance register usage and memory access patterns for optimal performance

## Hands-on Tutorial

See the `register_pressure_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.