# Tensor Cores (MMA Instructions)

## Concept Overview

Tensor cores are specialized hardware units in modern NVIDIA GPUs (Volta and later) that perform high-throughput matrix multiply-accumulate (MMA) operations on small tiles of data. They provide massive acceleration for mixed-precision computations, particularly in deep learning applications.

## Understanding Tensor Cores

### What Are Tensor Cores?
- Dedicated hardware units for accelerating matrix operations
- Available in Volta, Turing, Ampere, and Hopper architectures
- Perform 4x4x4 or larger matrix operations in a single instruction
- Support mixed precision (FP16 inputs, FP32 accumulation)

### Architecture Evolution
- **Volta**: First introduction (4x4x4 operations)
- **Turing**: Added INT8 and INT4 integer operations
- **Ampere**: Enhanced precision, introduced BF16, sparse operations
- **Hopper**: Introduced FP8, larger tile sizes (16x8x16, 16x16x16)

## Tensor Core Specifications

### Volta Tensor Cores
- Operation: 4×4×4 matrix multiply-accumulate
- Input: FP16 (half precision)
- Accumulation: FP32 (single precision)
- Result: FP32
- Throughput: Up to 128 TFLOPS on V100

### Ampere Tensor Cores
- Operation: 16×8×16 matrix multiply-accumulate
- Inputs: Various types (FP64, FP32, TF32, FP16, BF16, INT8, INT4)
- Throughput: Up to 312 TFLOPS for sparse operations on A100

### Hopper Tensor Cores
- Operation: 16×16×16 matrix multiply-accumulate
- New data types: FP8
- Improved sparsity: 2:4 structured sparsity
- Throughput: Up to 1000 TFLOPS for FP8 on H100

## Programming Tensor Cores

### Using WMMA API (Warp Matrix Multiply Accumulate)

```cuda
#include <mma.h>
using namespace nvcuda;

// Example: 16x16x16 matrix multiplication using Tensor Cores
__global__ void wmma_example(half* a, half* b, float* c, float* d, int m, int n, int k) {
    // Tile dimensions
    const int M = 16;
    const int N = 16;
    const int K = 16;
    
    // Leading dimensions for memory layout
    const int lda = M;  // leading dimension of matrix A
    const int ldb = K;  // leading dimension of matrix B
    const int ldc = M;  // leading dimension of matrix C
    const int ldd = M;  // leading dimension of matrix D (result)
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_c;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_d;
    
    // Calculate thread's position in the matrix
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Bounds checking
    if (warpM * M >= m || warpN * N >= n) return;
    
    // Load inputs
    wmma::load_matrix_sync(frag_a, a + warpM * M * lda, lda);
    wmma::load_matrix_sync(frag_b, b + warpN * ldb + warpM * K, ldb);
    wmma::load_matrix_sync(frag_c, c + warpM * M * ldc + warpN * N, ldc);
    
    // Perform matrix multiplication: d = a*b + c
    wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);
    
    // Store result
    wmma::store_matrix_sync(d + warpM * M * ldd + warpN * N, frag_d, ldd, wmma::mem_row_major);
}
```

### More Complex WMMA Example

```cuda
// Larger GEMM using Tensor Cores with tiling
__global__ void gemm_tensor_core(half* A, half* B, float* C, float* D, 
                                 int M, int N, int K) {
    // Tile sizes for tensor cores
    const int BLOCK_M = 128;  // 128x128 output tile
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;   // K dimension of each tile
    
    // Tensor core fragment sizes
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate which block this thread block is processing
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    
    // Calculate which warp this thread block is processing
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Shared memory for tiling
    __shared__ half smem_a[BLOCK_M][BLOCK_K];
    __shared__ half smem_b[BLOCK_K][BLOCK_N];
    
    // Fragments for tensor core operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c, frag_acc;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(frag_acc, 0.0f);
    
    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += BLOCK_K) {
        // Cooperative loading of tiles into shared memory
        // Each warp loads a portion of the tile
        
        // Load A tile (cooperative among threads)
        int a_row = block_row * BLOCK_M + warp_id / (BLOCK_N/WMMA_N) * WMMA_M;
        int a_col = k;
        
        // Load B tile (cooperative among threads) 
        int b_row = k;
        int b_col = block_col * BLOCK_N + (warp_id % (BLOCK_N/WMMA_N)) * WMMA_N;
        
        // This is a simplified example - in practice, you'd have more complex
        // cooperative loading patterns
        
        // Loop over the K dimension in tensor core chunks
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            // Load fragments for tensor core operation
            wmma::load_matrix_sync(frag_a, 
                &smem_a[warp_id / (BLOCK_N/WMMA_N) * WMMA_M][(kk)],
                BLOCK_K);
                
            wmma::load_matrix_sync(frag_b, 
                &smem_b[kk][(warp_id % (BLOCK_N/WMMA_N)) * WMMA_N],
                BLOCK_K);
            
            // Perform tensor core operation: frag_acc = frag_a * frag_b + frag_acc
            wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }
    }
    
    // Store result to global memory
    int c_row = block_row * BLOCK_M + warp_id / (BLOCK_N/WMMA_N) * WMMA_M;
    int c_col = block_col * BLOCK_N + (warp_id % (BLOCK_N/WMMA_N)) * WMMA_N;
    
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, frag_acc, N, wmma::mem_row_major);
    }
}
```

## Data Layout Requirements

### Matrix Layout for Tensor Cores
Tensor cores require specific data layouts:

```cuda
// Proper layout for efficient tensor core usage
__global__ void layout_example() {
    // For optimal performance, matrices should be laid out in memory
    // to match tensor core access patterns
    
    // Matrix A: row-major (for matrix_a fragment)
    // Matrix B: column-major (for matrix_b fragment) 
    // Matrix C/D: row-major (for accumulator fragment)
    
    // Example of proper memory access:
    // A[m][k] stored in row-major: A[m * lda + k]
    // B[k][n] stored in column-major: B[n * ldb + k] 
    // C[m][n] stored in row-major: C[m * ldc + n]
}
```

### Fragment Types and Layouts
```cuda
// Different fragment types for different matrix layouts
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;  // Note: col_major
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
```

## Performance Considerations

### Memory Bandwidth Requirements
- Tensor cores require high memory bandwidth to feed data
- Need to sustain ~1.5 TB/s for FP16 on A100
- Proper memory access patterns are critical

### Occupancy Considerations
- Tensor core operations are compute-intensive
- May reduce occupancy due to register pressure
- Balance between occupancy and computational efficiency

### Precision Trade-offs
- Mixed precision: FP16 inputs, FP32 accumulation
- Maintains accuracy while gaining speed
- Consider numerical stability for your application

## Practical Implementation Tips

### 1. Proper Initialization
```cuda
// Always initialize accumulator fragments
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
wmma::fill_fragment(acc_frag, 0.0f);  // Initialize to zero
```

### 2. Bounds Checking
```cuda
// Handle cases where matrix dimensions aren't multiples of tile size
if (row < matrix_height && col < matrix_width) {
    wmma::store_matrix_sync(output_ptr, result_frag, ld, wmma::mem_row_major);
}
```

### 3. Memory Coalescing
```cuda
// Ensure global memory accesses are coalesced
// Even with tensor cores, memory bandwidth matters
```

## Advanced Features

### Sparse Tensor Cores (Ampere)
```cuda
// Ampere introduces structural sparsity support
// 2:4 sparsity pattern can double effective throughput
// Specialized APIs for sparse operations
```

### Multi-Stage Pipelines with Tensor Cores
```cuda
// Combine async memory operations with tensor cores
// Load next tile while computing current tile
// Achieve maximum overlap of memory and compute
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Utilize tensor cores for accelerated matrix operations in deep learning kernels
- Understand the specific data layout and fragment requirements
- Implement efficient GEMM operations using WMMA API
- Balance memory bandwidth and computational efficiency when using tensor cores

## Hands-on Tutorial

See the `tensor_cores_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.