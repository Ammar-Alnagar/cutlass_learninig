# Module 01: Layout Algebra - Comprehensive Reading Materials

## Table of Contents
1. [Introduction to Memory Layouts](#introduction-to-memory-layouts)
2. [The Mathematics of Layouts](#the-mathematics-of-layouts)
3. [Layout Construction Patterns](#layout-construction-patterns)
4. [Hierarchical Layouts and Composition](#hierarchical-layouts-and-composition)
5. [Layout Transformations](#layout-transformations)
6. [Practical Applications in GPU Programming](#practical-applications-in-gpu-programming)
7. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)
8. [Advanced Topics](#advanced-topics)

---

## Introduction to Memory Layouts

### What is a Memory Layout?

A **memory layout** is a mathematical function that maps logical coordinates (indices) to physical memory offsets. In GPU programming, understanding this mapping is crucial for:

- **Performance**: Coalesced memory access patterns maximize bandwidth
- **Correctness**: Ensuring threads access the intended data
- **Portability**: Writing code that works across different architectures

### Why CuTe Layouts Matter

Traditional CUDA programming uses manual index arithmetic:
```cpp
// Manual indexing (PMPP - Programming Massively Parallel Processors)
int offset = row * width + col;
float value = data[offset];
```

CuTe layouts abstract this into composable, type-safe objects:
```cpp
// CuTe layout abstraction
auto layout = make_layout(make_shape(M, N), GenRowMajor{});
int offset = layout(row, col);
```

**Benefits:**
- **Compile-time verification**: Errors caught at compile time
- **Composability**: Build complex layouts from simple ones
- **Readability**: Intent is clear from the code structure
- **Optimization**: Compiler can optimize layout transformations

---

## The Mathematics of Layouts

### Formal Definition

A layout `L` is defined as a pair `(S, D)` where:
- `S = (s₀, s₁, ..., sₙ₋₁)` is the **shape** (dimensions)
- `D = (d₀, d₁, ..., dₙ₋₁)` is the **stride** (step sizes)

The layout function maps coordinates to offsets:
```
L(c₀, c₁, ..., cₙ₋₁) = c₀·d₀ + c₁·d₁ + ... + cₙ₋₁·dₙ₋₁
```

### Row-Major Layouts

For a 2D row-major layout with shape `(R, C)`:
- **Shape**: `(R, C)`
- **Stride**: `(C, 1)`
- **Formula**: `offset(row, col) = row × C + col × 1`

**Example**: 4×4 row-major layout
```
Shape: (4, 4)
Stride: (4, 1)

Memory Grid (offsets):
 0  1  2  3    ← Row 0: consecutive elements
 4  5  6  7    ← Row 1
 8  9 10 11    ← Row 2
12 13 14 15    ← Row 3
```

**Key Property**: Elements in the same row are contiguous in memory.

### Column-Major Layouts

For a 2D column-major layout with shape `(R, C)`:
- **Shape**: `(R, C)`
- **Stride**: `(1, R)`
- **Formula**: `offset(row, col) = row × 1 + col × R`

**Example**: 4×4 column-major layout
```
Shape: (4, 4)
Stride: (1, 4)

Memory Grid (offsets):
 0  4  8 12    ← Row 0
 1  5  9 13    ← Row 1
 2  6 10 14    ← Row 2
 3  7 11 15    ← Row 3
     ↑
Column 0: consecutive elements
```

**Key Property**: Elements in the same column are contiguous in memory.

### Stride Analysis

The stride determines the **cost** of moving in each dimension:

| Layout Type | Stride Pattern | Coalesced Access |
|-------------|----------------|------------------|
| Row-Major | `(C, 1)` | Vary column index |
| Column-Major | `(1, R)` | Vary row index |
| Padded Row | `(P, 1)` where P > C | Vary column index |

**Rule of Thumb**: The dimension with stride 1 is the **fastest-varying** dimension.

---

## Layout Construction Patterns

### Basic Layout Creation

```cpp
// 1D layout with 64 elements
auto layout_1d = make_layout(Int<64>{});
// Shape: (64), Stride: (1)

// 2D row-major layout
auto layout_2d = make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{});
// Shape: (8, 16), Stride: (16, 1)

// 2D column-major layout
auto layout_2d_cm = make_layout(make_shape(Int<8>{}, Int<16>{}), GenColMajor{});
// Shape: (8, 16), Stride: (1, 8)

// 3D layout (NCHW format)
auto layout_3d = make_layout(
    make_shape(Int<4>{}, Int<3>{}, Int<32>{}, Int<32>{}),
    GenRowMajor{}
);
// Shape: (4, 3, 32, 32), Stride: (3072, 1024, 32, 1)
```

### Custom Strides for Padding

Padding is essential for avoiding bank conflicts in shared memory:

```cpp
// Standard 32×32 layout (may cause bank conflicts)
auto standard = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
// Stride: (32, 1)

// Padded 32×32 layout (avoids bank conflicts)
auto padded = make_layout(
    make_shape(Int<32>{}, Int<32>{}),
    make_stride(Int<33>{}, Int<1>{})
);
// Stride: (33, 1) - each row has 1 extra element padding
```

**Memory Overhead Calculation**:
```
Original: 32 × 32 = 1024 elements
Padded:   32 × 33 = 1056 elements
Overhead: (1056 - 1024) / 1024 = 3.125%
```

### Layout Generators

CuTe provides generators for common patterns:

| Generator | Description | Use Case |
|-----------|-------------|----------|
| `GenRowMajor{}` | Row-major strides | Standard matrix storage |
| `GenColMajor{}` | Column-major strides | Fortran compatibility |
| `GenUniform<Stride>{}` | Uniform stride | Custom patterns |

---

## Hierarchical Layouts and Composition

### Why Hierarchical Layouts?

GPU kernels have a natural hierarchy:
```
Grid → Thread Blocks → Warps → Threads → Elements
```

Hierarchical layouts model this structure explicitly.

### Two-Level Layout Example

Consider an 8×8 matrix divided into 2×2 tiles of 4×4 elements:

```cpp
// Tile layout: 2×2 grid of tiles
auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}));

// Element layout: 4×4 elements per tile
auto element_layout = make_layout(make_shape(Int<4>{}, Int<4>{}));

// Composed layout: hierarchical 8×8 matrix
auto composed = make_layout(
    make_shape(
        make_shape(Int<4>{}, Int<2>{}),  // Row: 4 elements × 2 tiles
        make_shape(Int<4>{}, Int<2>{})   // Col: 4 elements × 2 tiles
    ),
    make_stride(
        make_stride(Int<1>{}, Int<16>{}),  // Row strides
        make_stride(Int<4>{}, Int<32>{})   // Col strides
    )
);
```

**Understanding the Stride**:
- `1`: Move 1 element within a tile (row direction)
- `16`: Jump to next tile row (4 elements × 4 columns per tile row)
- `4`: Move 1 element within a tile (column direction)
- `32`: Jump to next tile column (16 elements × 2 tile rows)

### Thread Block Hierarchy

For a 128-thread block organized as 4 warps of 32 threads:

```cpp
// Warp-major layout
auto warp_major = make_layout(
    make_shape(Int<4>{}, Int<32>{}),  // 4 warps × 32 lanes
    make_stride(Int<32>{}, Int<1>{})  // Row-major
);

// Coordinate mapping
// warp_major(warp_id, lane_id) = warp_id × 32 + lane_id = thread_id
```

This layout ensures **consecutive thread IDs access consecutive memory**, enabling coalesced loads.

---

## Layout Transformations

### Transpose

Transpose swaps the modes (dimensions) of a layout:

```cpp
auto original = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
// Shape: (4, 8), Stride: (8, 1)

auto transposed = make_layout(
    get<1>(original),  // Swap shape modes
    get<0>(original)   // Swap stride modes
);
// Shape: (8, 4), Stride: (1, 8)

// Verification: original(r, c) == transposed(c, r)
```

### Coalesce (Flatten)

Coalesce merges compatible modes into a single dimension:

```cpp
auto layout_2d = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
// Shape: (4, 8), Stride: (8, 1)

auto flat = coalesce(layout_2d);
// Shape: (32), Stride: (1)
```

**When Coalesce Works**: Adjacent modes must have **contiguous strides**.

### Partition (logical_divide)

Partition divides a layout into tiles:

```cpp
auto matrix = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});

// Divide into 8×8 tiles
auto partitioned = logical_divide(matrix, make_shape(Int<8>{}, Int<8>{}));
// Result: 2×2 grid of 8×8 tiles
```

---

## Practical Applications in GPU Programming

### Coalesced Global Memory Access

For coalesced loads in a row-major matrix:

```cpp
__global__ void load_kernel(float* data, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Layout for row-major matrix
    auto layout = make_layout(make_shape(M, N), GenRowMajor{});
    
    // Coalesced access: consecutive threads access consecutive columns
    float value = data[layout(row, col)];
}
```

### Shared Memory Bank Conflict Avoidance

```cpp
__global__ void smem_kernel() {
    // Padded layout avoids 32-way bank conflict
    auto padded_layout = make_layout(
        make_shape(Int<32>{}, Int<32>{}),
        make_stride(Int<33>{}, Int<1>{})
    );
    
    extern __shared__ float smem[];
    auto smem_tensor = make_tensor(make_smem_ptr(smem), padded_layout);
    
    // Column access now has no bank conflicts
    smem_tensor(threadIdx.x, threadIdx.y) = value;
}
```

### Matrix Multiplication Tiling

```cpp
// 16×16 output tile, 32 threads per warp
auto output_tile = make_layout(
    make_shape(Int<16>{}, Int<16>{}),  // 16×16 elements
    make_stride(Int<16>{}, Int<1>{})   // Row-major
);

// Thread layout: 4×8 threads per warp
auto thread_layout = make_layout(
    make_shape(Int<4>{}, Int<8>{}),
    make_stride(Int<8>{}, Int<1>{})
);

// Each thread computeszes 4×2 = 8 elements
```

---

## Common Pitfalls and Best Practices

### Pitfall 1: Wrong Stride for Access Pattern

**Problem**: Using row-major layout but accessing column-wise.

```cpp
// WRONG: Row-major layout, column access (uncoalesced)
auto layout = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
float val = data[layout(threadIdx.x, 0)];  // Stride of 32!
```

**Solution**: Match layout to access pattern.

```cpp
// CORRECT: Column-major layout for column access
auto layout = make_layout(make_shape(Int<32>{}, Int<32>{}), GenColMajor{});
float val = data[layout(threadIdx.x, 0)];  // Stride of 1!
```

### Pitfall 2: Forgetting Padding in Shared Memory

**Problem**: 32-way bank conflict on column access.

```cpp
// WRONG: No padding
__shared__ float smem[32][32];
smem[threadIdx.x][threadIdx.y] = value;  // All 32 threads hit same bank!
```

**Solution**: Add padding or use swizzling.

```cpp
// CORRECT: Padded
__shared__ float smem[32][33];
smem[threadIdx.x][threadIdx.y] = value;  // Each thread hits different bank
```

### Pitfall 3: Confusing Shape with Stride

**Remember**:
- **Shape** = dimensions of the logical space
- **Stride** = step sizes for traversing dimensions

```cpp
// Same shape, different strides
auto row_major = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
// Shape: (4, 8), Stride: (8, 1)

auto col_major = make_layout(make_shape(Int<4>{}, Int<8>{}), GenColMajor{});
// Shape: (4, 8), Stride: (1, 4)
```

### Best Practices

1. **Always verify with `print()`**: Visualize layouts during development
2. **Draw diagrams**: Sketch the memory layout for complex cases
3. **Test with small examples**: Verify offset calculations manually
4. **Document stride choices**: Explain why a particular stride was chosen
5. **Use type-safe integers**: `Int<N>{}` enables compile-time checks

---

## Advanced Topics

### Layout Arithmetic

Layouts support arithmetic operations:

```cpp
// Layout addition (concatenation)
auto l1 = make_layout(Int<16>{});
auto l2 = make_layout(Int<16>{});
auto combined = layout_concat(l1, l2);  // Shape: (32)

// Layout multiplication (tensor product)
auto l3 = make_layout(Int<4>{});
auto l4 = make_layout(Int<8>{});
auto product = layout_product(l3, l4);  // Shape: (4, 8)
```

### Dynamic Layouts

While CuTe favors compile-time layouts, dynamic layouts are supported:

```cpp
// Runtime dimensions
int M = get_m();
int N = get_n();
auto dynamic_layout = make_layout(make_shape(M, N), GenRowMajor{});

// Mixed static/dynamic
auto mixed = make_layout(make_shape(Int<16>{}, N), GenRowMajor{});
```

### Layout Introspection

Query layout properties at runtime:

```cpp
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});

// Get shape
auto shape = layout.shape();
int rows = get<0>(shape);
int cols = get<1>(shape);

// Get stride
auto stride = layout.stride();
int row_stride = get<0>(stride);
int col_stride = get<1>(stride);

// Get rank (number of dimensions)
int rank = rank(layout);  // Returns 2

// Get total size
int size = size(layout);  // Returns 32
```

---

## Further Reading

### Official Documentation
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [CUTLASS 3.x Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)

### Academic Resources
- "Programming Massively Parallel Processors" - Hwu & Kirk
- "CUDA by Example" - Sanders & Kandrot
- NVIDIA CUDA Programming Guide

### Related Topics
- Module 02: CuTe Tensors (building on layouts)
- Module 03: Tiled Copy (using layouts for data movement)
- Module 05: Shared Memory Swizzling (advanced layout techniques)

---

## Quick Reference Card

### Layout Creation
```cpp
make_layout(shape, stride)           // General layout
make_layout(shape, GenRowMajor{})    // Row-major
make_layout(shape, GenColMajor{})    // Column-major
make_layout(shape, make_stride(...)) // Custom strides
```

### Common Patterns
```cpp
// Row-major M×N
make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{})

// Column-major M×N
make_layout(make_shape(Int<M>{}, Int<N>{}), GenColMajor{})

// Padded M×N with stride P
make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<P>{}, Int<1>{}))

// 3D layout (N, C, H, W)
make_layout(make_shape(Int<N>{}, Int<C>{}, Int<H>{}, Int<W>{}), GenRowMajor{})
```

### Debugging
```cpp
print(layout);           // Detailed layout info
print_layout(layout);    // Grid visualization
layout.shape();          // Get shape tuple
layout.stride();         // Get stride tuple
layout(coord);           // Calculate offset
```
