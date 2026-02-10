# CUDA Memory Hierarchy

## Concept Overview

GPUs have a complex memory hierarchy with multiple memory types, each offering different performance characteristics. Understanding these memory types and their trade-offs is crucial for writing high-performance CUDA applications.

## Memory Types and Characteristics

### 1. Global Memory
- **Location**: On-chip DRAM (or external memory)
- **Scope**: Accessible by all threads in all blocks
- **Size**: Large (GBs)
- **Speed**: Slowest (hundreds of cycles of latency)
- **Features**: Cached (L1/L2), supports atomic operations
- **Use Case**: Storing large datasets, input/output arrays

### 2. Shared Memory
- **Location**: On-chip SRAM
- **Scope**: Visible only to threads within the same block
- **Size**: Small (typically 48KB-164KB per block)
- **Speed**: Very fast (single cycle access)
- **Features**: Banked structure, can be configured as L1 cache
- **Use Case**: Data sharing and cooperation between threads in a block

### 3. Registers
- **Location**: On-chip storage
- **Scope**: Thread-local
- **Size**: Limited (~255 per thread)
- **Speed**: Fastest (single cycle access)
- **Features**: Volatile, temporary storage
- **Use Case**: Storing frequently accessed variables

### 4. Constant Memory
- **Location**: On-chip DRAM
- **Scope**: Read-only, accessible by all threads
- **Size**: Small (64KB)
- **Speed**: Cached, fast for broadcast access
- **Features**: Optimized for uniform access patterns
- **Use Case**: Storing read-only data that's accessed by many threads simultaneously

### 5. Texture Memory
- **Location**: On-chip DRAM
- **Scope**: Read-only, cached
- **Size**: Large
- **Speed**: Cached, optimized for spatial locality
- **Features**: Hardware interpolation, boundary handling
- **Use Case**: Image processing, irregular access patterns

## Memory Hierarchy Visualization

```
                    CPU Memory Space
                           |
                    Application Memory
                           |
                    CUDA Runtime
                           |
         ┌─────────────────┼─────────────────┐
         |                 |                 |
    Global Memory    Constant Memory   Texture Memory
         |                 |                 |
         └─────────────────┼─────────────────┘
                           |
                    L2 Cache (Unified)
                           |
         ┌─────────────────┼─────────────────┐
         |                                   |
    L1 Cache                          Shared Memory
         |                              (Per Block)
         |
    Registers
 (Per Thread)
```

## Performance Characteristics

| Memory Type | Latency | Bandwidth | Scope | Size | Persistence |
|-------------|---------|-----------|-------|------|-------------|
| Registers   | ~1 cycle| Peak      | Thread| ~255 | Thread      |
| Shared      | ~1 cycle| High      | Block | ~48KB| Block       |
| L1 Cache    | ~20-40 cycles| High | SM    | ~16-128KB| SM        |
| L2 Cache    | ~100-200 cycles| Medium-High | GPU | ~400KB-4MB| GPU       |
| Global      | ~200-300 cycles| Limited by bandwidth | All   | GBs   | Kernel/Program |

## Memory Access Patterns

### Global Memory Access
- **Coalesced**: Consecutive threads access consecutive memory addresses
- **Uncoalesced**: Random or strided access patterns
- **Performance Impact**: Can vary by 10x or more between coalesced and uncoalesced

### Shared Memory Access
- **Bank Conflicts**: Multiple threads access the same memory bank
- **Broadcast**: Multiple threads access the same address (efficient)
- **Performance Impact**: Bank conflicts serialize access, reducing effective bandwidth

## Practical Examples

### Using Global Memory
```cuda
// Simple vector addition
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];  // Global memory access
    }
}
```

### Using Shared Memory
```cuda
// Shared memory for cooperative loading
__global__ void sharedMemExample(float* input, float* output, int N) {
    __shared__ float sData[256];  // Shared memory array
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperative loading into shared memory
    if (i < N) {
        sData[tid] = input[i];
    }
    __syncthreads();  // Synchronize threads in block
    
    // Process data in shared memory
    if (tid > 0) {
        sData[tid] += sData[tid - 1];  // Shared memory access
    }
    __syncthreads();
    
    // Write back to global memory
    if (i < N) {
        output[i] = sData[tid];
    }
}
```

### Using Constant Memory
```cuda
__constant__ float coefficients[256];

__global__ void constantMemExample(float* input, float* output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // Coefficients are cached and broadcast efficiently
        output[i] = input[i] * coefficients[i % 256];
    }
}
```

## Memory Optimization Strategies

### 1. Data Locality
- Keep frequently accessed data in faster memory (registers → shared → global)
- Reuse data once loaded into faster memory

### 2. Memory Coalescing
- Ensure consecutive threads access consecutive memory addresses
- Use appropriate data structures and access patterns

### 3. Shared Memory Banking
- Organize data to avoid bank conflicts
- Use padding when necessary to align data properly

### 4. Memory Prefetching
- Load data early to hide memory latency
- Use asynchronous memory operations when possible

## Memory Management Best Practices

1. **Minimize Global Memory Accesses**: Reduce the number of trips to slow global memory
2. **Maximize Data Reuse**: Keep data in faster memory as long as possible
3. **Optimize Access Patterns**: Ensure coalesced access for global memory
4. **Size Shared Memory Appropriately**: Balance between occupancy and shared memory usage
5. **Monitor Memory Bandwidth**: Use profiling tools to identify bottlenecks

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Choose appropriate memory types for different data access patterns
- Understand the performance trade-offs between different memory types
- Design memory access patterns that maximize performance
- Optimize memory usage based on the specific requirements of your algorithm

## Hands-on Tutorial

See the `memory_hierarchy_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.