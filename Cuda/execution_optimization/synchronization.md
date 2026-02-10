# Memory Fences and Synchronization

## Concept Overview

GPU memory systems implement a relaxed consistency model where threads may observe different orders of memory operations. Memory fences and synchronization primitives ensure that memory operations complete before subsequent operations begin, providing ordering guarantees across thread groups. Understanding these mechanisms is crucial for writing correct concurrent GPU code.

## GPU Memory Model Fundamentals

### Relaxed Consistency Model
- GPU threads may not observe memory operations in program order
- Different threads may see memory updates at different times
- Hardware optimizations can reorder operations for performance
- Explicit synchronization is required for ordering guarantees

### Memory Ordering Scopes
- **Thread level**: Operations within a single thread
- **Block level**: Operations across threads in a block
- **Grid level**: Operations across all blocks in a kernel
- **System level**: Operations visible to CPU and other kernels

## Types of Synchronization Primitives

### 1. Block-Level Synchronization
```cuda
// Synchronize all threads in a block
__syncthreads();

// Synchronize with return value (post-Voltaic)
__syncwarp(unsigned mask);
```

### 2. Memory Fence Operations
```cuda
// Block-level fence: ensures all memory ops complete before proceeding
__threadfence_block();

// Grid-level fence: ensures all blocks in grid see updates
__threadfence();

// System-level fence: ensures all operations visible to CPU
__threadfence_system();
```

### 3. Atomic Operations
```cuda
// Basic atomic operations
int atomicAdd(int* address, int val);
float atomicAdd(float* address, float val);  // Compute capability 2.0+
double atomicAdd(double* address, double val);  // Compute capability 6.0+

// Compare-and-swap
int atomicCAS(int* address, int compare, int val);

// Exchange operations
int atomicExch(int* address, int val);

// Min/max operations
int atomicMin(int* address, int val);
int atomicMax(int* address, int val);

// Bitwise operations
int atomicAnd(int* address, int val);
int atomicOr(int* address, int val);
int atomicXor(int* address, int val);
```

## Memory Ordering Guarantees

### Sequential Consistency
- Operations appear to execute in program order
- All threads observe the same order of operations
- Highest consistency, lowest performance

### Acquire/Release Semantics
- **Acquire**: Ensures subsequent operations don't move before acquire
- **Release**: Ensures prior operations don't move after release
- Better performance than sequential consistency

### Memory Fence Examples
```cuda
// Example: Producer-Consumer pattern with proper synchronization
__global__ void producer_consumer_example(int* buffer, int* flag, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Producer: fill buffer
        for (int i = 0; i < n; i++) {
            buffer[i] = i * 2;
        }
        
        // Ensure all writes to buffer complete before setting flag
        __threadfence();  // Release fence
        
        // Signal consumer that data is ready
        *flag = 1;
    }
    else if (tid == 1) {
        // Consumer: wait for data to be ready
        while (*flag == 0) {
            // Busy wait
        }
        
        // Acquire fence: ensure we see all producer's writes
        __threadfence();
        
        // Now safe to read buffer contents
        for (int i = 0; i < n; i++) {
            buffer[i] *= 3;  // Process the data
        }
    }
}
```

## Synchronization Patterns

### 1. Reduction with Proper Synchronization
```cuda
__global__ void synchronized_reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Synchronize after each step
    }
    
    // Write result to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### 2. Grid-Wide Synchronization Pattern
```cuda
__global__ void grid_sync_example(int* counters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each block increments its counter
    atomicAdd(&counters[blockIdx.x], 1);
    
    // Block-level sync to ensure all threads in block complete increment
    __syncthreads();
    
    // Wait for all blocks to reach this point
    // Note: This is a simplified example - real grid sync requires more complex approaches
    __threadfence();  // Ensure all atomic operations are visible
    
    // Now all blocks have incremented their counters
    if (tid == 0) {
        // Process global state
        int total = 0;
        for (int i = 0; i < gridDim.x; i++) {
            total += counters[i];
        }
        counters[gridDim.x] = total;  // Store total
    }
}
```

### 3. Memory Consistency in Shared Memory
```cuda
__global__ void shared_memory_consistency(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data cooperatively
    if (gid < n) {
        sdata[tid] = input[gid];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();  // Ensure all threads have loaded data
    
    // Process data - threads depend on each other's data
    float result = sdata[tid];
    if (tid > 0) {
        result += sdata[tid - 1];  // Use neighbor's data
    }
    
    __syncthreads();  // Ensure all computations complete before storing
    
    if (gid < n) {
        output[gid] = result;
    }
}
```

## Memory Fence Scenarios

### Block-Level Fence (`__threadfence_block`)
- Ensures all memory operations within block are visible to other threads in the same block
- Lighter weight than `__syncthreads()` but provides memory ordering without execution synchronization
- Useful when you need memory ordering but threads don't need to reach the same execution point

### Grid-Level Fence (`__threadfence`)
- Ensures memory operations are visible to all threads in the same grid
- Also visible to subsequent kernel launches on the same device
- Heavier weight than block-level fence

### System-Level Fence (`__threadfence_system`)
- Ensures memory operations are visible to CPU and other devices
- Used for CPU-GPU synchronization
- Heaviest weight fence operation

## Common Synchronization Mistakes

### 1. Missing Synchronization
```cuda
// WRONG: Race condition
__global__ void wrong_sync_example(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid > 0 && tid < n) {
        // Reading data[tid-1] without ensuring it's written
        data[tid] += data[tid-1];  // Race condition!
    }
}

// CORRECT: Proper synchronization
__global__ void correct_sync_example(float* data, float* temp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        temp[tid] = data[tid];  // Copy to temporary location
    }
    __syncthreads();
    
    if (tid > 0 && tid < n) {
        data[tid] += temp[tid-1];  // Safe to read from temp
    }
}
```

### 2. Incorrect Fence Placement
```cuda
// WRONG: Fence placed incorrectly
__global__ void wrong_fence_example(int* data, int* flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        *data = 42;
        *flag = 1;  // Setting flag before data write is visible
        __threadfence();  // Too late!
    }
}

// CORRECT: Proper fence placement
__global__ void correct_fence_example(int* data, int* flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        *data = 42;
        __threadfence();  // Ensure data write is visible
        *flag = 1;  // Now safe to set flag
    }
}
```

## Performance Considerations

### Synchronization Overhead
- Synchronization primitives introduce latency
- Overuse can serialize parallel execution
- Balance correctness with performance

### Fence Performance Impact
- `__syncthreads()`: High overhead, full block synchronization
- `__threadfence_block()`: Lower overhead, memory ordering only
- `__threadfence()`: Higher overhead, grid visibility
- `__threadfence_system()`: Highest overhead, system visibility

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Use appropriate synchronization primitives to ensure memory ordering guarantees
- Understand the differences between various fence operations and their performance implications
- Design correct concurrent GPU algorithms that properly synchronize memory operations
- Recognize and fix common synchronization-related race conditions

## Hands-on Tutorial

See the `synchronization_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.