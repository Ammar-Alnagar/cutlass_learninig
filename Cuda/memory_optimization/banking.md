# Shared Memory Banking

## Concept Overview

Shared memory in CUDA GPUs is organized into banks that can be accessed in parallel. When multiple threads access different banks simultaneously, the access is efficient. However, when multiple threads access the same bank (except for broadcast), bank conflicts occur, causing serialization and reducing effective bandwidth.

## Understanding Shared Memory Banks

### Bank Structure
- Modern GPUs typically have 32 shared memory banks
- Each bank is 4 bytes (32 bits) wide
- Successive 32-bit words are assigned to successive banks
- Bank numbers wrap around after the last bank (bank 0, 1, 2, ..., 31, 0, 1, ...)

### Bank Access Pattern
```
Address: 0  4  8  12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 128 132 ...
Bank:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 0  1  ...
```

## Types of Bank Conflicts

### 1. No Conflicts (Ideal)
```cuda
// Each thread accesses a different bank
__shared__ float sdata[32];  // 32 elements, each in different bank
int tid = threadIdx.x;

// Each thread accesses a different bank - no conflicts
float value = sdata[tid];  // Thread 0->Bank 0, Thread 1->Bank 1, etc.
```

### 2. Broadcast (Efficient)
```cuda
// Multiple threads accessing the same address (broadcast)
__shared__ float sdata[32];
int tid = threadIdx.x;

// All threads read the same element - efficient broadcast
float value = sdata[0];  // All threads read from bank 0, but same address
```

### 3. Bank Conflicts (Inefficient)
```cuda
// Multiple threads accessing the same bank with different addresses
__shared__ float sdata[128];  // 128 elements
int tid = threadIdx.x;

// If threads access sdata[0], sdata[32], sdata[64], sdata[96] - all map to bank 0
// This creates a 4-way bank conflict
float value = sdata[tid * 32];  // 4-way bank conflict if 4 threads access this
```

## Practical Examples

### Example 1: Matrix Transpose with Bank Conflicts
```cuda
#define TILE_SIZE 32
__global__ void transposeWithConflicts(float* input, float* output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];  // 32x32 = 1024 floats
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read in coalesced pattern (good)
    tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    __syncthreads();
    
    // Write out with transpose - causes bank conflicts!
    // When writing tile[threadIdx.x][threadIdx.y], threads with same threadIdx.x
    // access the same column, causing bank conflicts
    output[x * N + y] = tile[threadIdx.x][threadIdx.y];  // BANK CONFLICTS!
}
```

### Example 2: Matrix Transpose without Bank Conflicts
```cuda
#define TILE_SIZE 32
__global__ void transposeWithoutConflicts(float* input, float* output, int N) {
    // Add padding to avoid bank conflicts: TILE_SIZE + 1 instead of TILE_SIZE
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // Note the +1 padding
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read in coalesced pattern
    tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    __syncthreads();
    
    // Write out - no bank conflicts due to padding
    output[x * N + y] = tile[threadIdx.x][threadIdx.y];  // NO BANK CONFLICTS!
}
```

## Common Bank Conflict Scenarios

### 1. Diagonal Access
```cuda
__shared__ float sdata[32][32];
int tid = threadIdx.x;

// Diagonal access causes bank conflicts
float val = sdata[tid][tid];  // Multiple threads accessing same diagonal
```

### 2. Stride Access
```cuda
__shared__ float sdata[128];
int tid = threadIdx.x;

// Stride-32 access causes bank conflicts
float val = sdata[tid * 32];  // All threads access same bank
```

### 3. Column Access in Row-Major Storage
```cuda
__shared__ float matrix[32][32];
int tid = threadIdx.x;

// Accessing columns causes bank conflicts in row-major storage
float val = matrix[0][tid];   // OK - different banks
float val2 = matrix[tid][0];  // OK - different banks
float val3 = matrix[1][tid];  // OK - different banks  
float val4 = matrix[tid][1];  // OK - different banks
// But accessing multiple columns simultaneously can cause conflicts
```

## Solutions to Bank Conflicts

### 1. Padding
```cuda
// Add padding to shift elements to different banks
__shared__ float sdata[32][33];  // 33 instead of 32 adds padding
```

### 2. Reorganizing Data Access
```cuda
// Change access pattern to avoid conflicts
__shared__ float sdata[64];
int tid = threadIdx.x;

// Instead of accessing sdata[tid * 2] which might cause conflicts,
// access sdata[tid] and reorganize algorithm
```

### 3. Using Different Data Types
```cuda
// Use wider data types to change bank mapping
__shared__ float4 sdata[8];  // 8 float4s = 32 floats, but different access pattern
```

## Advanced Techniques

### 1. Shared Memory Swizzling
Swizzling applies transformations (like XOR) to addresses to distribute accesses across banks.

### 2. Bank Conflict-Free Layouts
Design data layouts that naturally avoid bank conflicts for specific access patterns.

## Performance Impact

- **No conflicts**: Maximum shared memory bandwidth
- **2-way conflicts**: ~2x slower than no conflicts
- **4-way conflicts**: ~4x slower than no conflicts
- **32-way conflicts**: ~32x slower than no conflicts (effectively serialized)

## Debugging Bank Conflicts

### Profiler Tools
- Use `nvprof` or Nsight Compute to detect bank conflicts
- Look for "shared_replay_overhead" metrics
- Check "gst_efficiency" and "gld_efficiency" metrics

### Manual Detection
- Analyze access patterns mathematically
- Consider which threads access which addresses simultaneously
- Calculate bank numbers for each access

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Design shared memory layouts that avoid bank conflicts
- Identify potential bank conflicts in existing code
- Apply padding and other techniques to eliminate conflicts
- Understand how different access patterns affect bank conflicts

## Hands-on Tutorial

See the `banking_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.