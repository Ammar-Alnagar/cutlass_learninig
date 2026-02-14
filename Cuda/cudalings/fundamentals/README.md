# CUDA Fundamentals

This directory contains basic CUDA concepts and programming patterns that form the foundation for more advanced optimizations.

## Key Concepts Covered

### 1. Kernel Launch Configuration
- Grid and block dimensions
- Thread indexing
- Memory access patterns

### 2. Memory Types
- Global memory
- Shared memory
- Constant memory
- Texture memory

### 3. Synchronization
- `__syncthreads()`
- Warp-level primitives
- Cooperative groups

### 4. Occupancy and Performance Metrics
- Occupancy calculation
- Memory bandwidth
- Arithmetic intensity

### 5. Common Optimization Techniques
- Coalesced memory access
- Shared memory usage
- Loop unrolling
- Register blocking

## Basic CUDA Kernel Template

```cuda
__global__ void kernel_name(float* input, float* output, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (idx < n) {
        // Perform computation
        output[idx] = input[idx] * 2.0f;
    }
}
```

## Compilation and Execution
```bash
nvcc -o program program.cu
./program
```

## Profiling Commands
```bash
# Basic profiling
nvprof ./program

# Detailed memory access info
nvprof --metrics gld_transactions,gst_transactions ./program

# Occupancy analysis
nvprof --metrics achieved_occupancy ./program
```