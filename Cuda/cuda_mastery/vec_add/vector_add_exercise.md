# Vector Addition Hands-On Exercise

## Objective
Complete the vector addition kernel that computes `C = A + B`.

## Code to Complete
```cuda
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    // TODO: Calculate the global thread index
    // Hint: Use blockIdx.x, blockDim.x, and threadIdx.x
    int i = /* YOUR CODE HERE */;

    // TODO: Add bounds checking to prevent out-of-bounds access
    if (/* YOUR CONDITION HERE */) {
        // TODO: Perform the vector addition: C[i] = A[i] + B[i]
        /* YOUR CODE HERE */;
    }
}
```

## Solution Guidance
- Calculate the global thread index using the formula: `blockIdx.x * blockDim.x + threadIdx.x`
- Check that the calculated index is less than N to avoid out-of-bounds access
- Perform the element-wise addition operation: `C[i] = A[i] + B[i]`

## Key Concepts Practiced
- Thread indexing in CUDA
- Bounds checking
- Basic memory access patterns
- Grid-block-thread hierarchy

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the output matches expected results (each element of C should be the sum of corresponding elements in A and B)
3. Verify that no out-of-bounds memory accesses occur