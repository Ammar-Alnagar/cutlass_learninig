# Shared Memory Banking Hands-On Exercise

## Objective
Fix bank conflicts in the shared memory transpose operation.

## Code to Complete
```cuda
__global__ void sharedMemoryTranspose(float* input, float* output, int width) {
    // TODO: Modify shared memory declaration to avoid bank conflicts
    // Hint: Add padding to avoid bank conflicts during transposed access
    __shared__ float tile[32][32];  // Standard tile - causes bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load data into shared memory (coalesced read)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Corrected transposed write that avoids bank conflicts
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        // TODO: Fix the transpose operation to avoid bank conflicts
        // Current implementation causes bank conflicts
        // tile[threadIdx.x][threadIdx.y] = tile[threadIdx.x][threadIdx.y];  // Wrong approach
        
        // Implement correct transposed write that avoids bank conflicts
        // Hint: You may need to modify the shared memory declaration above
        output[y * width + x] = /* YOUR CORRECTED ACCESS HERE */;
    }
}
```

## Solution Guidance
- Add padding to the shared memory declaration (e.g., `[32][33]` instead of `[32][32]`)
- This prevents multiple threads from accessing the same memory bank simultaneously
- The padding breaks the alignment that causes bank conflicts during transposed access
- The corrected access should be `tile[threadIdx.x][threadIdx.y]` after fixing the declaration

## Key Concepts Practiced
- Shared memory banking
- Bank conflict avoidance
- Memory layout optimization
- Transpose operations on GPUs

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the transpose operation produces correct results
3. Understand how the padding eliminates bank conflicts
4. Compare performance with and without padding