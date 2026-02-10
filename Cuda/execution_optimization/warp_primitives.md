# Warp-Level Primitives

## Concept Overview

Warp-level primitives are specialized operations that enable efficient communication and coordination within a warp (group of 32 threads) without requiring shared memory or synchronization. These operations leverage the SIMT (Single Instruction, Multiple Thread) nature of GPU execution to perform collective operations efficiently.

## Understanding Warps

### Warp Execution Model
- A warp consists of 32 consecutive threads that execute in lockstep
- All threads in a warp execute the same instruction at the same time
- When threads diverge (take different branches), the warp executes each branch serially
- Warp-level operations exploit this synchronized execution for efficient communication

### SIMD vs SIMT
- Traditional SIMD: Single Instruction, Multiple Data
- GPU SIMT: Single Instruction, Multiple Thread (with individual data)
- Warp-level operations bridge the gap between individual thread execution and collective operations

## Types of Warp-Level Primitives

### 1. Shuffle Operations
Enable threads to exchange data directly with other threads in the same warp.

#### Warp Shuffle Functions
```cuda
// Shuffle data between threads in a warp
__device__ T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
__device__ T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=warpSize);
__device__ T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=warpSize);
__device__ T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```

#### Shuffle Examples
```cuda
// Basic shuffle: thread i gets value from thread j
__global__ void basic_shuffle(float* input, float* output) {
    int laneId = threadIdx.x % 32;  // Thread ID within warp
    
    float value = input[threadIdx.x];
    
    // Thread 0 gets value from thread 15, thread 1 gets from thread 16, etc.
    float shuffled = __shfl_sync(0xFFFFFFFF, value, 15);  // All threads participate
    
    output[threadIdx.x] = shuffled;
}

// Shuffle up: each thread gets value from higher-numbered thread
__global__ void shuffle_up_example(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Each thread gets value from (laneId + delta) thread
    float shifted = __shfl_up_sync(0xFFFFFFFF, value, 4);
    
    output[threadIdx.x] = shifted;
}

// Shuffle down: each thread gets value from lower-numbered thread
__global__ void shuffle_down_example(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Each thread gets value from (laneId - delta) thread
    float shifted = __shfl_down_sync(0xFFFFFFFF, value, 4);
    
    output[threadIdx.x] = shifted;
}

// Shuffle XOR: useful for butterfly operations
__global__ void shuffle_xor_example(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Each thread exchanges with thread at (laneId ^ mask)
    float exchanged = __shfl_xor_sync(0xFFFFFFFF, value, 16);  // Exchange with thread+/-16
    
    output[threadIdx.x] = exchanged;
}
```

### 2. Warp Vote Operations
Perform boolean operations across all threads in a warp.

#### Vote Functions
```cuda
__device__ int __all_sync(unsigned mask);           // All threads return true?
__device__ int __any_sync(unsigned mask);           // Any thread returns true?
__device__ int __ballot_sync(unsigned mask, int predicate);  // Bitmask of threads with true
```

#### Vote Examples
```cuda
// Check if all threads meet a condition
__global__ void vote_all_example(int* input, int* result) {
    int laneId = threadIdx.x % 32;
    bool condition = (input[threadIdx.x] > 0);
    
    // Returns true only if ALL threads in the warp have input > 0
    int all_positive = __all_sync(0xFFFFFFFF, condition);
    
    if (laneId == 0) {  // Only one thread writes result
        result[blockIdx.x] = all_positive;
    }
}

// Check if any thread meets a condition
__global__ void vote_any_example(int* input, int* result) {
    int laneId = threadIdx.x % 32;
    bool condition = (input[threadIdx.x] > 100);
    
    // Returns true if ANY thread in the warp has input > 100
    int any_large = __any_sync(0xFFFFFFFF, condition);
    
    if (laneId == 0) {
        result[blockIdx.x] = any_large;
    }
}

// Get bitmask of threads meeting condition
__global__ void vote_ballot_example(int* input, unsigned int* result) {
    int laneId = threadIdx.x % 32;
    bool condition = (input[threadIdx.x] % 2 == 0);  // Even numbers
    
    // Returns 32-bit mask where bit i is set if thread i meets condition
    unsigned int even_mask = __ballot_sync(0xFFFFFFFF, condition);
    
    if (laneId == 0) {
        result[blockIdx.x] = even_mask;
    }
}
```

### 3. Warp Match Operations
Identify threads with matching values.

#### Match Function
```cuda
__device__ unsigned int __match_any_sync(unsigned mask, T value);
__device__ unsigned int __match_all_sync(unsigned mask, T value, int *pred);
```

#### Match Examples
```cuda
// Find threads with matching values
__global__ void match_example(int* input, unsigned int* result) {
    int laneId = threadIdx.x % 32;
    int value = input[threadIdx.x];
    
    // Returns mask of threads with same value as current thread
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, value);
    
    result[threadIdx.x] = match_mask;
}
```

## Practical Applications

### 1. Warp-Level Reduction
```cuda
__device__ float warpReduce(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void warp_sum_reduction(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    float sum = (tid < n) ? input[tid] : 0.0f;
    
    // Perform warp-level reduction
    sum = warpReduce(sum);
    
    // First thread in each warp writes partial result to global memory
    if (laneId == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warpId] = sum;
    }
}
```

### 2. Broadcast Within Warp
```cuda
__global__ void warp_broadcast(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float value = input[threadIdx.x];
    
    // Broadcast value from thread 0 to all threads in warp
    float broadcast_val = __shfl_sync(0xFFFFFFFF, value, 0);
    
    output[threadIdx.x] = broadcast_val;
}
```

### 3. Prefix Sum Within Warp
```cuda
__device__ float warpPrefixSum(float val) {
    float result = val;
    for (int offset = 1; offset < warpSize; offset *= 2) {
        float temp = __shfl_up_sync(0xFFFFFFFF, result, offset);
        if ((threadIdx.x % 32) >= offset) {
            result += temp;
        }
    }
    return result;
}

__global__ void warp_prefix_sum(float* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    float val = input[threadIdx.x];
    float prefix_sum = warpPrefixSum(val);
    
    output[threadIdx.x] = prefix_sum;
}
```

## Warp Divergence and Predication

### Understanding Divergence
```cuda
// Problematic: causes warp divergence
__global__ void divergent_example(int* input, float* output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (input[tid] > 0) {
        // Only some threads execute this
        output[tid] = sqrt(input[tid]);
    } else {
        // Other threads execute this
        output[tid] = 0.0f;
    }
    // Both paths must complete before continuing
}

// Better: use warp primitives to handle divergence
__global__ void warp_divergence_example(int* input, float* output) {
    int laneId = threadIdx.x % 32;
    
    int val = input[threadIdx.x];
    bool positive = (val > 0);
    
    // Use ballot to identify which threads have positive values
    unsigned int pos_mask = __ballot_sync(0xFFFFFFFF, positive);
    
    if (positive) {
        output[threadIdx.x] = sqrtf((float)val);
    } else {
        output[threadIdx.x] = 0.0f;
    }
}
```

## Performance Considerations

### Advantages of Warp Primitives
- Extremely fast (single instruction)
- No shared memory required
- No synchronization overhead
- Efficient for intra-warp communication

### Limitations
- Only work within a single warp (32 threads)
- Require all participating threads to execute the same instruction
- Mask parameter required for newer architectures (compute capability 7.0+)

### When to Use
- Communication between threads in the same warp
- Collective operations like reductions
- Broadcasting values within a warp
- Implementing efficient parallel algorithms

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Use warp-level operations for efficient intra-warp communication
- Implement reductions, scans, and other collective operations using warp primitives
- Understand and mitigate warp divergence in your kernels
- Design algorithms that take advantage of warp-level parallelism

## Hands-on Tutorial

See the `warp_primitives_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.