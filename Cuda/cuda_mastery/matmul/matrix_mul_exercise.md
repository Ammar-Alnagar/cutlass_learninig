# Matrix Multiplication Hands-On Exercise

## Objective
Complete the matrix multiplication kernel that computes `C = A × B`.

## Code to Complete
```cuda
__global__ void matrixMul(float* A, float* B, float* C, int width) {
    // TODO: Calculate row and column indices for this thread
    int row = /* YOUR CODE HERE */;
    int col = /* YOUR CODE HERE */;

    // Only compute if within matrix bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // TODO: Compute dot product of row from A and column from B
        // Hint: Loop from 0 to width and accumulate the products
        for (int k = 0; k < width; k++) {
            // YOUR CODE HERE: multiply A[row][k] by B[k][col] and add to sum
            /* YOUR CODE HERE */;
        }
        
        // Store result in C[row][col]
        C[row * width + col] = sum;
    }
}
```

## Solution Guidance
- Calculate row index using block and thread Y coordinates: `blockIdx.y * blockDim.y + threadIdx.y`
- Calculate column index using block and thread X coordinates: `blockIdx.x * blockDim.x + threadIdx.x`
- Implement the dot product calculation by multiplying corresponding elements (`A[row][k] * B[k][col]`) and accumulating the result

## Key Concepts Practiced
- 2D thread indexing
- Nested loops in kernels
- Matrix addressing patterns
- Bounds checking for 2D operations

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the output matches expected matrix multiplication results
3. Verify that no out-of-bounds memory accesses occur
4. Confirm that the algorithm correctly implements the mathematical definition of matrix multiplication