# Memory Coalescing Hands-On Exercise

## Objective
Fix the memory access pattern to ensure coalesced access for optimal performance.

## Code to Complete
```cuda
__global__ void coalescedCopy(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Implement coalesced memory access
    // Current implementation has poor coalescing - fix it
    if (tid < n) {
        // Implement proper coalesced access where consecutive threads
        // access consecutive memory locations
        /* YOUR CODE HERE */;
    }
}
```

## Solution Guidance
- Ensure that consecutive threads access consecutive memory addresses
- The simplest coalesced access is `output[tid] = input[tid]`
- This ensures that threads 0, 1, 2, ... access memory locations 0, 1, 2, ..., which maximizes memory bandwidth utilization

## Key Concepts Practiced
- Memory access patterns
- Coalesced vs. uncoalesced memory access
- Performance implications of memory access patterns
- GPU memory subsystem optimization

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the output matches the input (since it's a copy operation)
3. Compare performance with the uncoalesced version provided
4. Understand why coalesced access is more efficient