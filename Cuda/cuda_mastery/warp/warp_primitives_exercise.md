# Warp-Level Primitives Hands-On Exercise

## Objective
Complete the kernel using warp-level primitives like shuffle operations.

## Code to Complete
```cuda
__global__ void warpShuffleExample(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;  // Thread's position within its warp
    
    if (tid < n) {
        float value = input[tid];
        
        // TODO: Use warp shuffle to get value from next thread in warp
        // Hint: Use __shfl_down_sync or __shfl_sync
        // Example: float nextValue = __shfl_down_sync(0xFFFFFFFF, value, 1, 32);
        float neighborValue = /* YOUR SHUFFLE CODE */;
        
        // Store the neighbor's value
        output[tid] = neighborValue;
    }
}

__global__ void warpVoteExample(int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Check if the value is greater than threshold
        bool isGreater = (input[tid] > 50);
        
        // TODO: Use warp vote to check if ANY thread in the warp has isGreater = true
        // Hint: Use __any_sync
        int anyGreater = /* YOUR VOTE CODE */;
        
        // TODO: Use warp vote to check if ALL threads in the warp have isGreater = true
        // Hint: Use __all_sync
        int allGreater = /* YOUR VOTE CODE */;
        
        // Store results
        output[tid] = anyGreater * 100 + allGreater * 10;
    }
}
```

## Solution Guidance
- For shuffle: `float neighborValue = __shfl_down_sync(0xFFFFFFFF, value, 1, 32);`
- For vote any: `int anyGreater = __any_sync(0xFFFFFFFF, isGreater);`
- For vote all: `int allGreater = __all_sync(0xFFFFFFFF, isGreater);`

## Key Concepts Practiced
- Warp shuffle operations
- Warp vote operations (__any_sync, __all_sync)
- Cooperative operations within a warp
- Efficient data exchange between threads in a warp

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that shuffle operations correctly exchange data between threads
3. Verify that vote operations correctly aggregate boolean values across the warp
4. Understand the performance benefits of warp-level primitives