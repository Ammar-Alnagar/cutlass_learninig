# Shared Memory Swizzling

## Concept Overview

Shared memory swizzling is an advanced technique that transforms memory address patterns to redistribute data across different memory banks. This technique is particularly useful in tiled algorithms like GEMM (General Matrix Multiplication) where regular access patterns would otherwise cause systematic bank conflicts.

## Understanding Swizzling

### What is Swizzling?

Swizzling refers to applying mathematical transformations (such as XOR operations or permutations) to memory addresses to change how data is distributed across memory banks. The goal is to eliminate systematic bank conflicts that occur with regular access patterns.

### Why Swizzling is Needed

In many algorithms, especially those involving matrix operations, threads follow predictable access patterns that can systematically conflict with the bank structure. Swizzling breaks these patterns by changing the mapping between logical addresses and physical memory locations.

## Mathematical Foundation

### Basic XOR Swizzling

The most common form of swizzling uses XOR operations:
```
swizzled_address = original_address XOR (original_address >> k)
```

Where `k` determines the swizzling pattern.

### Bank Mapping

For a 32-bank system:
- Address `addr` maps to bank `(addr / sizeof(element)) % 32`
- Swizzling changes this mapping to distribute accesses more evenly

## Practical Examples

### Example 1: Basic Swizzling Pattern

```cuda
// Define swizzling function
__device__ __forceinline__ unsigned int swizzle_row(unsigned int addr, unsigned int width) {
    // Simple swizzling: XOR with right-shifted version
    return addr ^ (addr >> 5);  // 5 = log2(32 banks)
}

// Using swizzled addresses
__global__ void swizzled_access_example(float* input, float* output, int N) {
    __shared__ float sdata[1024];  // Raw shared memory array
    int tid = threadIdx.x;
    
    // Calculate swizzled address
    int swizzled_addr = swizzle_row(tid, 32);
    
    // Store with swizzling
    sdata[swizzled_addr] = input[tid];
    __syncthreads();
    
    // Retrieve with swizzling
    output[tid] = sdata[swizzled_addr];
}
```

### Example 2: Matrix Tiling with Swizzling

```cuda
#define TILE_SIZE 32
#define SWIZZLE_MASK 31  // For 32 banks

// Swizzle function for 2D access
__device__ __forceinline__ int swizzle_2d(int row, int col, int width) {
    return row * width + (col ^ (row & SWIZZLE_MASK));
}

__global__ void tiled_matrix_operation(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding approach
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];  // +1 padding approach
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Load data cooperatively
    for (int k = 0; k < N; k += TILE_SIZE) {
        // Load tile of A with swizzling consideration
        As[ty][tx] = (row < N && k + tx < N) ? A[row * N + (k + tx)] : 0.0f;
        
        // Load tile of B with swizzling consideration
        Bs[ty][tx] = (k + ty < N && col < N) ? B[(k + ty) * N + col] : 0.0f;
        
        __syncthreads();
        
        // Perform computation
        for (int i = 0; i < TILE_SIZE; ++i) {
            C[row * N + col] += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
}
```

### Example 3: Advanced Swizzling for GEMM

```cuda
#define WARP_SIZE 32
#define TILE_WIDTH 8

// More sophisticated swizzling for GEMM
__device__ __forceinline__ int swizzle_gemm(int row, int col, int width) {
    // Custom swizzling pattern for GEMM
    int bank_offset = (row / 2) % 32;  // Distribute based on row
    int swizzle_factor = (col ^ bank_offset) & 31;  // XOR with bank offset
    return row * width + (col ^ swizzle_factor);
}

// Alternative swizzling using permutation
__device__ __forceinline__ int permute_swizzle(int addr) {
    // Permutation: swap certain bit positions
    int bank = (addr >> 2) & 31;  // Extract bank bits
    int offset = addr & 3;        // Extract offset within bank
    return (addr & ~127) | ((bank << 2) | offset);  // Reconstruct
}
```

## Types of Swizzling Patterns

### 1. XOR-Based Swizzling
- Simple and effective for many cases
- Formula: `addr ^ (addr >> k)`
- Good for linear access patterns

### 2. Permutation-Based Swizzling
- More complex but can handle specific patterns
- Rearranges bit positions in address
- Better for 2D access patterns

### 3. Hash-Based Swizzling
- Uses hash functions to distribute addresses
- More random distribution
- Higher overhead but better distribution

## Application in Tiled Algorithms

### Matrix Multiplication (GEMM)
In GEMM, threads often access matrices in patterns that create systematic conflicts. Swizzling helps by:

1. Redistributing data across banks
2. Breaking regular access patterns
3. Maintaining coalesced access properties

### Convolution Operations
Similar benefits apply to convolution where filters access overlapping regions.

## Trade-offs and Considerations

### Benefits
- Eliminates systematic bank conflicts
- Improves shared memory bandwidth utilization
- Can significantly improve performance in memory-bound kernels

### Costs
- Adds computational overhead for address calculation
- Makes code more complex
- May not be beneficial if conflicts are already minimal

### When to Use
- Regular access patterns that cause systematic conflicts
- Tiled algorithms with predictable access patterns
- When profiling shows significant bank conflict overhead

## Performance Measurement

### Metrics to Watch
- `shared_efficiency`: Measures shared memory efficiency
- `shared_trans_util`: Shared memory transaction utilization
- `bank_conflict_cycles`: Cycles spent resolving bank conflicts

### Profiling Commands
```bash
# Using Nsight Compute
ncu --metrics smsp__shared_utilization ./your_program

# Using nvprof (legacy)
nvprof --metrics shared_efficiency ./your_program
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Apply swizzling patterns to eliminate bank conflicts in tiled algorithms
- Design custom swizzling functions for specific access patterns
- Evaluate when swizzling provides performance benefits
- Implement swizzling in complex algorithms like GEMM

## Hands-on Tutorial

See the `swizzling_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.