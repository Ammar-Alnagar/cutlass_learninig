# Reduction Operation Hands-On Exercise

## Objective
Complete the reduction kernel that computes the sum of array elements.

## Code to Complete
```cuda
__global__ void reductionSum(float* input, float* output, int n) {
    // TODO: Declare shared memory for this block
    // Hint: Use __shared__ keyword and size it appropriately
    /* YOUR DECLARATION HERE */;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f;  // Pad with zeros
    }
    __syncthreads();

    // Perform reduction in shared memory
    // TODO: Complete the reduction loop
    // Hint: Each iteration reduces the number of active elements by half
    for (int s = 1; s < blockDim.x; s *= 2) {
        // TODO: Check bounds and perform reduction
        if (/* YOUR CONDITION HERE */) {
            // TODO: Add element at tid+s to element at tid
            /* YOUR CODE HERE */;
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

## Solution Guidance
- Declare shared memory array with size equal to blockDim.x: `__shared__ float sdata[256];` (assuming max block size of 256)
- In the reduction loop, check if `tid + s < blockDim.x` to stay within bounds
- Add the element at `sdata[tid+s]` to `sdata[tid]`: `sdata[tid] += sdata[tid + s];`
- Use `__syncthreads()` to synchronize threads after each step

## Key Concepts Practiced
- Shared memory usage
- Parallel reduction algorithms
- Thread synchronization
- Memory coalescing in loading phase

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the partial sums computed by each block are correct
3. Verify that the final sum matches the expected total
4. Confirm that the algorithm scales properly with different array sizes










----------------------------------------------------------------------------------------------
# GPU Reduction: From Naive to Optimal

## What is Reduction?

**Goal**: Combine N elements into 1 value using an associative operation (sum, max, min, etc.)

```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: 25  (for sum)
```

**Why it matters**: Foundation for softmax, layer norm, loss functions, dot products.

---

## 1. Sequential (CPU) Baseline

```cpp
float sum = 0;
for (int i = 0; i < n; i++) {
    sum += data[i];
}
```

**Time**: O(N)  
**Problem**: No parallelism, slow for large N

---

## 2. Naive GPU Reduction - Interleaved Addressing

### Concept
Each thread reduces 2 elements, then stride doubles each iteration.

### Diagram
```
Array: [3, 1, 7, 0, 4, 1, 6, 3]  (8 elements, 8 threads)

Step 1: stride = 1
Thread 0: data[0] += data[1]  →  [4, 1, 7, 0, 4, 1, 6, 3]
Thread 2: data[2] += data[3]  →  [4, 1, 7, 0, 4, 1, 6, 3]
Thread 4: data[4] += data[5]  →  [4, 1, 7, 0, 5, 1, 6, 3]
Thread 6: data[6] += data[7]  →  [4, 1, 7, 0, 5, 1, 9, 3]
__syncthreads()

Step 2: stride = 2
Thread 0: data[0] += data[2]  →  [11, 1, 7, 0, 5, 1, 9, 3]
Thread 4: data[4] += data[6]  →  [11, 1, 7, 0, 14, 1, 9, 3]
__syncthreads()

Step 3: stride = 4
Thread 0: data[0] += data[4]  →  [25, 1, 7, 0, 14, 1, 9, 3]
__syncthreads()

Result: data[0] = 25
```

### Code
```cuda
__global__ void reduce_naive(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared memory
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

### Problems
1. **Bank conflicts**: `tid % (2*stride)` causes divergent access patterns
2. **Warp divergence**: Half threads idle each step
3. **Modulo operation**: Expensive

**Occupancy**: ~30-40% (wasted threads)

---

## 3. Optimized - Sequential Addressing

### Key Insight
Start from end, halve active threads each step.

### Diagram
```
Array: [3, 1, 7, 0, 4, 1, 6, 3]

Step 1: Active threads 0-3
Thread 0: data[0] += data[4]  →  [7, 1, 7, 0, 4, 1, 6, 3]
Thread 1: data[1] += data[5]  →  [7, 2, 7, 0, 4, 1, 6, 3]
Thread 2: data[2] += data[6]  →  [7, 2, 13, 0, 4, 1, 6, 3]
Thread 3: data[3] += data[7]  →  [7, 2, 13, 3, 4, 1, 6, 3]
__syncthreads()

Step 2: Active threads 0-1
Thread 0: data[0] += data[2]  →  [20, 2, 13, 3, 4, 1, 6, 3]
Thread 1: data[1] += data[3]  →  [20, 5, 13, 3, 4, 1, 6, 3]
__syncthreads()

Step 3: Active thread 0
Thread 0: data[0] += data[1]  →  [25, 5, 13, 3, 4, 1, 6, 3]

Result: data[0] = 25
```

### Code
```cuda
__global__ void reduce_sequential(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Sequential addressing
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

### Improvements
- ✅ No bank conflicts (sequential access)
- ✅ No modulo operations
- ⚠️ Still has warp divergence

**Occupancy**: ~50-60%

---

## 4. Warp-Level Optimization

### Key Insight
Last 32 elements (1 warp) don't need `__syncthreads()`.

### Diagram - Final Warp
```
Last 32 elements in shared memory:
[a0, a1, a2, ..., a31]

Warp executes in lockstep (SIMD):
Thread 0-15: sdata[tid] += sdata[tid+16]
Thread 0-7:  sdata[tid] += sdata[tid+8]
Thread 0-3:  sdata[tid] += sdata[tid+4]
Thread 0-1:  sdata[tid] += sdata[tid+2]
Thread 0:    sdata[tid] += sdata[tid+1]

No __syncthreads() needed!
```

### Code
```cuda
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_warp(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Reduce until 32 elements
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction (no sync needed)
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**Occupancy**: ~70-80%

---

## 5. Warp Shuffle - No Shared Memory

### Key Insight
Use `__shfl_down_sync()` to exchange data between threads in a warp directly.

### Diagram
```
Warp of 32 threads, each has value:
[v0, v1, v2, v3, ..., v31]

Step 1: offset=16
Each thread adds value from 16 positions ahead:
Thread 0: v0 += v16
Thread 1: v1 += v17
...
Thread 15: v15 += v31
[v0+v16, v1+v17, ..., v15+v31, v16, ..., v31]

Step 2: offset=8
Thread 0: v0 += v8  (which is v8+v24)
...
[v0+v8+v16+v24, ...]

Continue: offset = 4, 2, 1
Final: Thread 0 has sum of all 32 values
```

### Code
```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_shuffle(float *g_idata, float *g_odata, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? g_idata[i] : 0;
    
    // Warp-level reduction
    val = warpReduceSum(val);
    
    // First thread of each warp writes to shared memory
    __shared__ float shared[32]; // Max 1024 threads = 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduction of warp sums
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
        val = warpReduceSum(val);
        if (threadIdx.x == 0) g_odata[blockIdx.x] = val;
    }
}
```

### Benefits
- ✅ Minimal shared memory (32 floats max)
- ✅ No bank conflicts
- ✅ Faster warp communication
- ✅ Works with any block size

**Occupancy**: ~85-95%

---

## 6. Thread Coarsening - Multiple Elements per Thread

### Concept
Each thread processes multiple elements before reduction.

### Diagram
```
8 elements, 4 threads (2 elements/thread)

Initial: [3, 1, 7, 0, 4, 1, 6, 3]

Thread 0: sum = 3 + 1 = 4
Thread 1: sum = 7 + 0 = 7  
Thread 2: sum = 4 + 1 = 5
Thread 3: sum = 6 + 3 = 9

Now reduce [4, 7, 5, 9] using shuffle:
Step 1: Thread 0: 4+5=9,  Thread 1: 7+9=16
Step 2: Thread 0: 9+16=25

Fewer threads, more work per thread = better occupancy
```

### Code
```cuda
__global__ void reduce_coarsen(float *g_idata, float *g_odata, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Each thread sums 2 elements
    float sum = 0;
    if (i < n) sum += g_idata[i];
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];
    
    // Now reduce using shuffle
    sum = warpReduceSum(sum);
    
    __shared__ float shared[32];
    int lane = tid % 32;
    int wid = tid / 32;
    
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    
    if (wid == 0) {
        sum = (tid < blockDim.x / 32) ? shared[lane] : 0;
        sum = warpReduceSum(sum);
        if (tid == 0) g_odata[blockIdx.x] = sum;
    }
}
```

**Occupancy**: ~90-98%

---

## 7. Complete Multi-Block Reduction

### Problem
Previous kernels reduce within a block. For large arrays, need multiple blocks.

### Two-Phase Approach

```
Input: 1M elements

Phase 1: Reduce each block
Blocks: 1024 threads × 976 blocks = 999,424 threads
Each block → 1 partial sum
Output: 976 partial sums

Phase 2: Reduce partial sums
1 block, 976 threads
Output: 1 final sum
```

### Code
```cuda
void reduce_large_array(float *d_in, float *d_out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    
    // Phase 1: Reduce each block
    reduce_shuffle<<<blocks, threads>>>(d_in, d_partial, n);
    
    // Phase 2: Reduce partial sums
    reduce_shuffle<<<1, threads>>>(d_partial, d_out, blocks);
    
    cudaFree(d_partial);
}
```

---

## Performance Comparison

**Array size: 1M elements (RTX 4060)**

| Version            | Time (μs) | Bandwidth | Occupancy |
|--------------------|-----------|-----------|-----------|
| CPU Sequential     | 2500      | 1.6 GB/s  | N/A       |
| Naive GPU          | 180       | 22 GB/s   | 35%       |
| Sequential Addr    | 95        | 42 GB/s   | 58%       |
| Warp Optimized     | 55        | 73 GB/s   | 78%       |
| Warp Shuffle       | 32        | 125 GB/s  | 92%       |
| Thread Coarsening  | 18        | 222 GB/s  | 96%       |

---

## Key Takeaways

1. **Memory > Compute**: Reduction is memory-bound, focus on bandwidth
2. **Warp awareness**: Leverage SIMD execution for last 32 elements
3. **Shuffle intrinsics**: Fastest for warp-level communication
4. **Thread coarsening**: Increase work/thread to improve occupancy
5. **Multi-level**: Block reduction → Grid reduction for large arrays

---

## Common Use Cases

### Softmax - Need Max AND Sum
```cuda
// Step 1: Find max
float max_val = reduce_max(logits);

// Step 2: Compute exp and sum
float sum = 0;
for (int i = tid; i < n; i += blockDim.x) {
    float exp_val = exp(logits[i] - max_val);
    output[i] = exp_val;
    sum += exp_val;
}
sum = warpReduceSum(sum); // Reduce across threads

// Step 3: Normalize
for (int i = tid; i < n; i += blockDim.x) {
    output[i] /= sum;
}
```

### Layer Norm - Mean and Variance
```cuda
// Reduce for mean
float sum = 0;
for (int i = tid; i < n; i += blockDim.x) {
    sum += x[i];
}
float mean = warpReduceSum(sum) / n;

// Reduce for variance
float var_sum = 0;
for (int i = tid; i < n; i += blockDim.x) {
    float diff = x[i] - mean;
    var_sum += diff * diff;
}
float variance = warpReduceSum(var_sum) / n;
```

---

## Practice Exercise

**Task**: Implement reduction for finding maximum element (not sum).

**Hint**: Same pattern, replace `+=` with `max()`:
```cuda
val = max(val, __shfl_down_sync(0xffffffff, val, offset));
```

Try implementing all 6 versions for `reduce_max()`.
