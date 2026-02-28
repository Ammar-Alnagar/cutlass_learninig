# Module 02: CuTe Tensors - Comprehensive Reading Materials

## Table of Contents
1. [Introduction to CuTe Tensors](#introduction-to-cute-tensors)
2. [Tensor Creation and Memory Wrapping](#tensor-creation-and-memory-wrapping)
3. [Tensor Access Patterns](#tensor-access-patterns)
4. [Tensor Slicing and Views](#tensor-slicing-and-views)
5. [Tensor Transformations](#tensor-transformations)
6. [Multi-Dimensional Tensors](#multi-dimensional-tensors)
7. [Memory Space Management](#memory-space-management)
8. [Tensor Broadcasting](#tensor-broadcasting)
9. [Advanced Tensor Operations](#advanced-tensor-operations)
10. [Performance Considerations](#performance-considerations)

---

## Introduction to CuTe Tensors

### What is a CuTe Tensor?

A **CuTe tensor** is a high-level abstraction that combines:
1. A **pointer** to raw memory (data)
2. A **layout** that maps logical coordinates to memory offsets

```
Tensor = (Pointer, Layout)
```

This combination provides:
- **Type safety**: Compile-time type checking
- **Indexing**: Natural coordinate-based access
- **Views**: Zero-copy sub-tensor extraction
- **Composability**: Build complex tensors from simple ones

### Relationship to Layouts

Layouts (Module 01) define the **mapping function**:
```cpp
auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
int offset = layout(row, col);  // Just calculates offset
```

Tensors add **data access** on top:
```cpp
float* data = ...;
auto tensor = make_tensor(data, layout);
float value = tensor(row, col);  // Accesses data[offset]
```

### Why Tensors Matter

Traditional CUDA code:
```cpp
// Manual pointer arithmetic
float* matrix = ...;
int pitch = width;
float value = matrix[row * pitch + col];
```

CuTe tensor code:
```cpp
// Abstracted, safe access
auto tensor = make_tensor(ptr, layout);
float value = tensor(row, col);  // Layout handles the math
```

**Benefits:**
- Less error-prone (no manual offset calculation)
- More readable (intent is clear)
- Easier to refactor (change layout, keep access code)
- Compiler-friendly (enables optimizations)

---

## Tensor Creation and Memory Wrapping

### Basic Tensor Creation

```cpp
#include <cute/tensor.hpp>

using namespace cute;

// From raw pointer
float* raw_ptr = ...;
auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
auto tensor = make_tensor(raw_ptr, layout);

// Access elements
tensor(0, 0) = 1.0f;
float val = tensor(1, 2);
```

### Memory Space Wrappers

CuTe provides wrappers for different CUDA memory spaces:

```cpp
// Global memory (device DRAM)
float* gmem_ptr = ...;
auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_ptr), layout);

// Shared memory (on-chip, block-scoped)
extern __shared__ float smem[];
auto smem_tensor = make_tensor(make_smem_ptr(smem), layout);

// Register memory (thread-local)
float rmem[16];
auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), layout);

// Constant memory (read-only, cached)
__constant__ float cmem[256];
auto cmem_tensor = make_tensor(make_cmem_ptr(cmem), layout);
```

### Tensor Properties

Query tensor information:

```cpp
auto tensor = make_tensor(ptr, layout);

// Get layout
auto layout = tensor.layout();

// Get shape
auto shape = tensor.shape();
int rows = get<0>(shape);
int cols = get<1>(shape);

// Get stride
auto stride = tensor.stride();

// Get data pointer
auto ptr = tensor.data();

// Get rank and size
int rank = rank(tensor);
int size = size(tensor);
```

---

## Tensor Access Patterns

### Coordinate-Based Access

Tensors support natural coordinate indexing:

```cpp
// 2D tensor
auto tensor_2d = make_tensor(ptr, make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{}));

// Access with coordinates
float val = tensor_2d(3, 7);  // Row 3, Column 7

// Assign values
tensor_2d(0, 0) = 1.0f;
```

### Linear Access

For 1D tensors or flattened access:

```cpp
// 1D tensor
auto tensor_1d = make_tensor(ptr, make_layout(Int<256>{}));

// Linear access
tensor_1d(42) = value;
float val = tensor_1d(100);
```

### Multi-Dimensional Access

```cpp
// 3D tensor (volume)
auto tensor_3d = make_tensor(ptr, make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}), GenRowMajor{}));

// Access with 3 coordinates
float val = tensor_3d(x, y, z);

// 4D tensor (NCHW)
auto tensor_4d = make_tensor(ptr, make_layout(make_shape(Int<4>{}, Int<3>{}, Int<32>{}, Int<32>{}), GenRowMajor{}));

// Access with 4 coordinates
float val = tensor_4d(n, c, h, w);
```

### Access Pattern Analysis

Understanding coalescing:

```cpp
// Row-major 32×32 matrix
auto rm_tensor = make_tensor(ptr, make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{}));

// Coalesced access (consecutive threads access consecutive columns)
__global__ void coalesced_kernel(float* ptr) {
    auto tensor = make_tensor(ptr, layout);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Consecutive threadIdx.x -> consecutive col -> consecutive memory
    float val = tensor(row, col);  // COALESCED
}

// Uncoalesced access (consecutive threads access consecutive rows)
__global__ void uncoalesced_kernel(float* ptr) {
    auto tensor = make_tensor(ptr, layout);
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Consecutive threadIdx.x -> consecutive row -> stride of 32
    float val = tensor(row, col);  // UNCOALESCED
}
```

---

## Tensor Slicing and Views

### What are Tensor Views?

A **view** is a tensor that references the same underlying data but with a different layout. Views are **zero-copy** operations.

### Row/Column Extraction

```cpp
// Full matrix
auto matrix = make_tensor(ptr, make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{}));

// Extract row 3 (view into row)
auto row_layout = make_layout(Int<8>{});  // 1D layout for 8 elements
auto row_view = make_tensor(ptr + matrix.layout()(3, 0), row_layout);

// Extract column 5
auto col_layout = make_layout(Int<8>{});
auto col_ptr = ptr + matrix.layout()(0, 5);
auto col_view = make_tensor(col_ptr, col_layout);
```

### Sub-Matrix Extraction

```cpp
// Extract a 4×4 sub-matrix from an 8×8 matrix
auto full = make_tensor(ptr, make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{}));

// Sub-matrix starting at (2, 2)
auto sub_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<8>{}, Int<1>{}));
auto sub_ptr = ptr + full.layout()(2, 2);
auto sub_view = make_tensor(sub_ptr, sub_layout);

// Note: sub_view uses same stride as parent (8) for correct access
```

### Strided Slices

```cpp
// Every-other-row slice
auto original = make_tensor(ptr, make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}));

// Strided layout: skip every other row
auto strided_layout = make_layout(
    make_shape(Int<8>{}, Int<16>{}),
    make_stride(Int<32>{}, Int<1>{})  // Stride of 32 = 2 rows
);

auto strided_view = make_tensor(ptr, strided_layout);
```

### Reshape Views

```cpp
// 2D tensor
auto tensor_2d = make_tensor(ptr, make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{}));

// View as 1D (64 elements)
auto flat_layout = make_layout(Int<64>{});
auto flat_view = make_tensor(ptr, flat_layout);

// Both views access the same data
tensor_2d(1, 0) == flat_view(8);  // Same memory location
```

---

## Tensor Transformations

### Transpose

Create a transposed view without copying data:

```cpp
auto original = make_tensor(ptr, make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{}));

// Transposed layout
auto transposed_layout = make_layout(
    get<1>(original.layout()),  // Swap shape modes
    get<0>(original.layout())   // Swap stride modes
);

auto transposed = make_tensor(ptr, transposed_layout);

// Verification
original(2, 5) == transposed(5, 2);  // Same memory location
```

### Permute Dimensions

For higher-dimensional tensors:

```cpp
// NCHW tensor
auto nchw = make_tensor(ptr, make_layout(make_shape(Int<4>{}, Int<3>{}, Int<32>{}, Int<32>{}), GenRowMajor{}));

// View as NHWC (requires layout transformation)
auto nhwc_layout = make_layout(
    make_shape(Int<4>{}, Int<32>{}, Int<32>{}, Int<3>{}),
    make_stride(Int<32*32*3>{}, Int<3>{}, Int<32*3>{}, Int<1>{})
);

auto nhwc_view = make_tensor(ptr, nhwc_layout);
```

### Flatten and Unflatten

```cpp
// 3D tensor
auto tensor_3d = make_tensor(ptr, make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}), GenRowMajor{}));

// Flatten to 1D
auto flat = make_tensor(ptr, make_layout(Int<512>{}));

// Unflatten back to 3D
auto unflattened = make_tensor(ptr, make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}), GenRowMajor{}));
```

---

## Multi-Dimensional Tensors

### 3D Tensors (Volumes)

```cpp
// 3D volume: depth × height × width
auto volume = make_tensor(
    ptr,
    make_layout(make_shape(Int<16>{}, Int<32>{}, Int<32>{}), GenRowMajor{})
);

// Access
float val = volume(d, h, w);

// Stride analysis
// depth stride: 32 × 32 = 1024
// height stride: 32
// width stride: 1 (contiguous)
```

### 4D Tensors (Batched Images)

**NCHW Format** (PyTorch default):
```cpp
auto nchw = make_tensor(
    ptr,
    make_layout(make_shape(Int<N>{}, Int<C>{}, Int<H>{}, Int<W>{}), GenRowMajor{})
);
// Stride: (C×H×W, H×W, W, 1)
// Contiguous dimension: W (width)
```

**NHWC Format** (TensorFlow default):
```cpp
auto nhwc = make_tensor(
    ptr,
    make_layout(make_shape(Int<N>{}, Int<H>{}, Int<W>{}, Int<C>{}), GenRowMajor{})
);
// Stride: (H×W×C, W×C, C, 1)
// Contiguous dimension: C (channel)
```

### 5D Tensors (Video, Batched 3D)

```cpp
// Video: batch × frames × channels × height × width
auto video = make_tensor(
    ptr,
    make_layout(make_shape(Int<B>{}, Int<F>{}, Int<C>{}, Int<H>{}, Int<W>{}), GenRowMajor{})
);

// Access
float val = video(b, f, c, h, w);
```

---

## Memory Space Management

### Global Memory (gmem)

**Characteristics:**
- Largest capacity (GBs)
- Highest latency (400-800 cycles)
- Persistent across kernel lifetime
- Accessible by all threads

```cpp
__global__ void global_memory_kernel(float* input, float* output, int size) {
    auto in_tensor = make_tensor(make_gmem_ptr(input), make_layout(Int<size>{}));
    auto out_tensor = make_tensor(make_gmem_ptr(output), make_layout(Int<size>{}));
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out_tensor(idx) = in_tensor(idx) * 2.0f;
    }
}
```

### Shared Memory (smem)

**Characteristics:**
- Limited capacity (KBs per SM)
- Low latency (~20 cycles)
- Block-scoped (shared by threads in block)
- Must be explicitly managed

```cpp
__global__ void shared_memory_kernel(float* input, float* output, int size) {
    extern __shared__ float smem[];
    
    auto smem_tensor = make_tensor(make_smem_ptr(smem), make_layout(Int<256>{}));
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared
    smem_tensor(threadIdx.x) = input[idx];
    __syncthreads();
    
    // Process from shared memory
    float val = smem_tensor(threadIdx.x);
    
    // Store result
    output[idx] = val * 2.0f;
}
```

### Register Memory (rmem)

**Characteristics:**
- Smallest capacity (KBs per thread)
- Lowest latency (1 cycle)
- Thread-private
- Automatic allocation

```cpp
__global__ void register_memory_kernel(float* input, float* output) {
    float rmem[4];  // Thread-local registers
    
    auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), make_layout(Int<4>{}));
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to registers
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rmem_tensor(i) = input[idx * 4 + i];
    }
    
    // Process in registers
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rmem_tensor(i) *= 2.0f;
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        output[idx * 4 + i] = rmem_tensor(i);
    }
}
```

### Memory Space Transfers

```cpp
__global__ void memory_transfer_kernel(float* gmem_in, float* gmem_out) {
    // Shared memory buffer
    extern __shared__ float smem[];
    auto smem_tensor = make_tensor(make_smem_ptr(smem), make_layout(Int<256>{}));
    
    // Register buffer
    float rmem[4];
    auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), make_layout(Int<4>{}));
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Global -> Shared
    smem_tensor(threadIdx.x) = gmem_in[idx];
    __syncthreads();
    
    // Shared -> Register (vectorized)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rmem_tensor(i) = smem_tensor(threadIdx.x * 4 + i);
    }
    
    // Process in registers
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rmem_tensor(i) *= 2.0f;
    }
    
    // Register -> Shared
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        smem_tensor(threadIdx.x * 4 + i) = rmem_tensor(i);
    }
    __syncthreads();
    
    // Shared -> Global
    gmem_out[idx] = smem_tensor(threadIdx.x);
}
```

---

## Tensor Broadcasting

### What is Broadcasting?

**Broadcasting** allows tensors with different shapes to be used in operations by virtually expanding dimensions with stride 0.

### Scalar Broadcasting

```cpp
// Scalar (broadcast to any shape)
float scalar = 2.0f;
auto scalar_tensor = make_tensor(&scalar, make_layout(Int<1>{}));

// Can be "broadcast" to any shape by using stride 0
auto broadcast_layout = make_layout(
    make_shape(Int<8>{}, Int<8>{}),
    make_stride(Int<0>{}, Int<0>{})  // Both strides are 0
);

auto broadcast_tensor = make_tensor(&scalar, broadcast_layout);

// All accesses return the same scalar
broadcast_tensor(0, 0) == 2.0f;
broadcast_tensor(7, 7) == 2.0f;  // Same memory location
```

### Vector Broadcasting (Bias Addition)

```cpp
// Bias vector (1D)
float bias[64];
auto bias_tensor = make_tensor(bias, make_layout(Int<64>{}));

// Broadcast to matrix (rows share same bias)
auto broadcast_layout = make_layout(
    make_shape(Int<32>{}, Int<64>{}),  // 32 rows, 64 columns
    make_stride(Int<0>{}, Int<1>{})    // Row stride = 0 (broadcast)
);

auto bias_matrix = make_tensor(bias, broadcast_layout);

// All rows access the same bias vector
bias_matrix(0, j) == bias_matrix(31, j);  // Same memory location
```

### Matrix Broadcasting

```cpp
// Small matrix to be broadcast
float small[16];  // 4×4
auto small_tensor = make_tensor(small, make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{}));

// Broadcast to larger grid (tiled pattern)
// This requires careful layout design
```

### Broadcasting in Operations

```cpp
// Matrix + Vector (broadcast vector across rows)
__global__ void add_bias_kernel(float* matrix, float* bias, float* output, int rows, int cols) {
    auto matrix_tensor = make_tensor(matrix, make_layout(make_shape(rows, cols), GenRowMajor{}));
    auto bias_tensor = make_tensor(bias, make_layout(Int<cols>{}));
    auto output_tensor = make_tensor(output, make_layout(make_shape(rows, cols), GenRowMajor{}));
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // bias_tensor is effectively broadcast across rows
        output_tensor(row, col) = matrix_tensor(row, col) + bias_tensor(col);
    }
}
```

---

## Advanced Tensor Operations

### Tensor Composition

Combine multiple tensors hierarchically:

```cpp
// Tile tensor (2×2 tiles)
auto tile_tensor = make_tensor(tile_ptrs, make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{}));

// Element tensor within each tile (4×4 elements)
auto element_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});

// Access: first select tile, then element within tile
float val = tile_tensor(tile_row, tile_col)[elem_row, elem_col];
```

### Tensor Arithmetic

CuTe supports tensor operations:

```cpp
// Element-wise operations (conceptual)
auto result = tensor_a + tensor_b;  // Requires matching layouts
auto scaled = scalar * tensor;
```

### Tensor Reductions

```cpp
// Sum reduction (conceptual)
float sum = reduce(tensor, plus{});

// Max reduction
float max_val = reduce(tensor, max{});
```

---

## Performance Considerations

### Memory Coalescing

**Best Practice:** Match tensor layout to access pattern.

```cpp
// GOOD: Row-major tensor, row-wise access
auto rm_tensor = make_tensor(ptr, make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{}));
__global__ void good_kernel(...) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Consecutive threads (col) -> consecutive memory
    float val = rm_tensor(row, col);
}

// BAD: Row-major tensor, column-wise access
__global__ void bad_kernel(...) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // Consecutive threads (row) -> stride of 32
    float val = rm_tensor(row, col);
}
```

### Register Pressure

**Warning:** Large register tensors can reduce occupancy.

```cpp
// Too many registers per thread
float rmem[64];  // May cause register spilling!

// Better: Use shared memory for larger buffers
extern __shared__ float smem[];
```

### Alignment Requirements

**Important:** Ensure proper alignment for vectorized access.

```cpp
// 128-bit access requires 16-byte alignment
float4* aligned_ptr = reinterpret_cast<float4*>(ptr);  // ptr must be 16-byte aligned

// CuTe tensors respect alignment automatically
auto tensor = make_tensor(ptr, layout);  // Layout encodes alignment
```

### Shared Memory Bank Conflicts

**Solution:** Use padded layouts.

```cpp
// BAD: May cause bank conflicts
auto smem_tensor = make_tensor(make_smem_ptr(smem), make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{}));

// GOOD: Padded to avoid conflicts
auto padded_layout = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
auto smem_tensor = make_tensor(make_smem_ptr(smem), padded_layout);
```

---

## Further Reading

### Official Documentation
- [CuTe Tensor Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [CUTLASS 3.x Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)

### Related Modules
- Module 01: Layout Algebra (foundation)
- Module 03: Tiled Copy (data movement with tensors)
- Module 04: MMA Atoms (tensor core operations)

### Advanced Topics
- Tensor Core Programming
- Async Memory Operations
- Multi-Stage Pipelines

---

## Quick Reference Card

### Tensor Creation
```cpp
make_tensor(ptr, layout)           // Basic tensor
make_tensor(make_gmem_ptr(p), l)   // Global memory
make_tensor(make_smem_ptr(p), l)   // Shared memory
make_tensor(make_rmem_ptr(p), l)   // Register memory
```

### Tensor Access
```cpp
tensor(coord)                      // Access element
tensor.layout()                    // Get layout
tensor.shape()                     // Get shape
tensor.stride()                    // Get stride
tensor.data()                      // Get pointer
```

### Tensor Views
```cpp
make_tensor(ptr + offset, layout)  // Sub-tensor view
make_tensor(ptr, transposed_layout) // Transposed view
make_tensor(ptr, broadcast_layout) // Broadcast view
```

### Common Patterns
```cpp
// Row-major matrix
auto t = make_tensor(ptr, make_layout(make_shape(M, N), GenRowMajor{}));

// Column-major matrix
auto t = make_tensor(ptr, make_layout(make_shape(M, N), GenColMajor{}));

// Padded shared memory
auto t = make_tensor(smem, make_layout(make_shape(M, N), make_stride(P, 1)));

// Broadcast vector to matrix
auto t = make_tensor(vec, make_layout(make_shape(R, C), make_stride(0, 1)));
```
