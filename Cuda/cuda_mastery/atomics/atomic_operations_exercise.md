# Atomic Operations Hands-On Exercise

## Objective
Complete the atomic operations kernel to handle race conditions properly.

## Code to Complete
```cuda
__global__ void atomicHistogram(unsigned char* input, unsigned int* histogram, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        unsigned char value = input[tid];
        
        // TODO: Use atomic operation to increment histogram bin safely
        // Without atomics, multiple threads might update the same bin simultaneously
        // Hint: Use atomicAdd function
        /* YOUR CODE HERE */;
    }
}
```

## Solution Guidance
- Use `atomicAdd(&histogram[value], 1)` to safely increment the histogram bin
- This ensures that even if multiple threads try to update the same bin simultaneously, the operation will be performed atomically
- The atomic operation prevents race conditions that would corrupt the histogram

## Key Concepts Practiced
- Race conditions in parallel programming
- Atomic operations
- Histogram computation
- Thread safety in GPU kernels

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the histogram counts match the expected distribution
3. Compare results with the non-atomic version to see the difference
4. Understand why atomic operations are necessary in this case