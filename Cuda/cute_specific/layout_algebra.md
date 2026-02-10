# Layout Algebra

## Concept Overview

CuTe (CUDA Templates) represents multi-dimensional data layouts as algebraic structures that map logical coordinates to memory addresses. This abstraction allows for automatic handling of complex tiling, padding, and transposition patterns without manual index arithmetic, making GPU programming more expressive and less error-prone.

## Understanding Layout Algebra

### What is a Layout?

In CuTe, a layout is a mathematical mapping that defines how logical coordinates map to physical memory addresses:

```
layout(logical_coord) → memory_address
```

This allows expressing complex memory access patterns algebraically rather than through manual index calculations.

### Basic Layout Types

#### Identity Layout
```cpp
#include "cutlass/cute/layout.hpp"
using namespace cute;

// 1D identity layout: layout(i) = i
auto identity_1d = make_layout(make_shape(Int<10>{}));  // Maps 0-9 to addresses 0-9

// 2D identity layout: layout(i,j) = i*cols + j
auto identity_2d = make_layout(make_shape(Int<4>{}, Int<5>{}));  // 4x5 matrix
```

#### Stride Layout
```cpp
// Custom stride layout
auto custom_layout = make_layout(make_shape(Int<4>{}, Int<5>{}),
                                make_stride(Int<1>{}, Int<8>{}));  // Row-major with stride 8 for columns
```

## Layout Composition

### Product Layouts (Tiling)
Product layouts combine multiple sub-layouts to create hierarchical structures:

```cpp
// Create a 2D layout as a product of 1D layouts
auto shape_2d = make_shape(Int<4>{}, Int<6>{});  // 4x6 matrix
auto layout_2d = make_layout(shape_2d);          // Row-major layout

// Tiled layout: divide 4x6 into 2x2 tiles
auto tile_shape = make_shape(Int<2>{}, Int<3>{});  // Each tile is 2x3
auto tiled_layout = tile(layout_2d, tile_shape);

// This creates a layout where:
// - Outer dimensions represent tile indices
// - Inner dimensions represent positions within tiles
```

### Layout Transformations
```cpp
// Transpose transformation
auto original_layout = make_layout(make_shape(Int<4>{}, Int<5>{}));
auto transposed_layout = composition(original_layout, make_shape(_5{}, _4{}));  // 5x4 transposed

// Swizzle transformation (for avoiding bank conflicts)
auto base_layout = make_layout(make_shape(Int<32>{}, Int<32>{}));
auto swizzled_layout = transform_layout(base_layout, make_offset_transform(swizzle_fn));
```

## Practical Layout Examples

### Matrix Layouts
```cpp
#include "cutlass/cute/layout.hpp"
using namespace cute;

// Standard row-major matrix layout
template<int M, int N>
auto make_rowmajor_matrix_layout() {
    return make_layout(make_shape(Int<M>{}, Int<N>{}),
                       make_stride(Int<1>{}, Int<M>{}));  // stride_col = M, stride_row = 1
}

// Standard column-major matrix layout
template<int M, int N>
auto make_colmajor_matrix_layout() {
    return make_layout(make_shape(Int<M>{}, Int<N>{}),
                       make_stride(Int<N>{}, Int<1>{}));  // stride_row = N, stride_col = 1
}

// Example usage
auto A_layout = make_rowmajor_matrix_layout<128, 256>();
auto B_layout = make_colmajor_matrix_layout<256, 512>();
```

### Tiled Matrix Layouts
```cpp
// Create a tiled layout for matrix multiplication
template<int M, int N, int K, int TM, int TN, int TK>
auto make_tiled_gemm_layouts() {
    // Original matrix dimensions
    auto A_shape = make_shape(Int<M>{}, Int<K>{});
    auto B_shape = make_shape(Int<K>{}, Int<N>{});
    auto C_shape = make_shape(Int<M>{}, Int<N>{});
    
    // Tile dimensions
    auto tile_A = make_shape(Int<TM>{}, Int<TK>{});
    auto tile_B = make_shape(Int<TK>{}, Int<TN>{});
    auto tile_C = make_shape(Int<TM>{}, Int<TN>{});
    
    // Create layouts
    auto A_layout = make_layout(A_shape, make_stride(_1{}, _M{}));  // Row-major A
    auto B_layout = make_layout(B_shape, make_stride(_N{}, _1{}));  // Col-major B
    auto C_layout = make_layout(C_shape, make_stride(_1{}, _M{}));  // Row-major C
    
    // Create tiled versions
    auto A_tiled = tile(A_layout, tile_A);
    auto B_tiled = tile(B_layout, tile_B);
    auto C_tiled = tile(C_layout, tile_C);
    
    return make_tuple(A_tiled, B_tiled, C_tiled);
}
```

## Layout Operations

### Layout Composition
```cpp
// Compose layouts to create complex mappings
auto outer_shape = make_shape(Int<4>{}, Int<4>{});   // 4x4 tiles
auto inner_shape = make_shape(Int<8>{}, Int<8>{});   // 8x8 elements per tile
auto composite_shape = make_shape(outer_shape, inner_shape);  // 32x32 total

// Create the composite layout
auto composite_layout = make_layout(composite_shape);
```

### Layout Access
```cpp
// Access elements using the layout
auto layout = make_layout(make_shape(Int<4>{}, Int<5>{}));
auto coord = make_coord(2, 3);  // Access element at (2,3)

// Get the memory address for this coordinate
int address = layout(coord);  // Returns 2*5 + 3 = 13
```

## Advanced Layout Concepts

### Layout Swizzling
```cpp
// Swizzling to avoid bank conflicts
auto base_layout = make_layout(make_shape(Int<32>{}, Int<32>{}));

// Define a swizzling function
auto swizzle_fn = [](int row, int col) {
    // Simple XOR swizzling to distribute across banks
    int swizzled_col = col ^ (row & 0x7);  // XOR with lower 3 bits of row
    return make_coord(row, swizzled_col);
};

// Apply swizzling transformation
auto swizzled_layout = transform_layout(base_layout, swizzle_fn);
```

### Layout Padding
```cpp
// Add padding to avoid bank conflicts
auto unpadded_layout = make_layout(make_shape(Int<32>{}, Int<32>{}));
auto padded_layout = make_layout(make_shape(Int<32>{}, Int<33>{}));  // +1 padding to avoid conflicts
```

## Layout Algebra Operations

### Layout Concatenation
```cpp
// Concatenate layouts along an axis
auto layout_a = make_layout(make_shape(Int<4>{}, Int<5>{}));
auto layout_b = make_layout(make_shape(Int<4>{}, Int<3>{}));

// Concatenate along the second dimension: 4x(5+3) = 4x8
auto concatenated = append<1>(layout_a, layout_b);
```

### Layout Splitting
```cpp
// Split a layout into parts
auto big_layout = make_layout(make_shape(Int<8>{}, Int<12>{}));

// Split into 4x6 and 4x6 along the first dimension
auto [top, bottom] = split<0>(big_layout, Int<4>{});

// Split into 8x5 and 8x7 along the second dimension
auto [left, right] = split<1>(big_layout, Int<5>{});
```

## Integration with Memory Operations

### Using Layouts with Copy Operations
```cpp
// Layout-aware copy operations
template<class SrcLayout, class DstLayout>
__device__ void layout_copy(float const* src, float* dst, 
                           SrcLayout const& src_layout, DstLayout const& dst_layout) {
    // Iterate through the layout space
    for (int i = 0; i < size(src_layout); ++i) {
        auto coord = idx2crd(i, src_layout.shape());  // Convert linear index to coordinate
        int src_addr = src_layout(coord);
        int dst_addr = dst_layout(coord);  // Same logical coordinate, different layout
        dst[dst_addr] = src[src_addr];
    }
}
```

## Benefits of Layout Algebra

### 1. Abstraction
- Separates logical data organization from physical memory layout
- Enables algorithmic thinking without low-level indexing concerns

### 2. Composability
- Complex layouts built from simple components
- Easy to experiment with different tiling strategies

### 3. Verification
- Mathematical properties can be verified at compile time
- Less prone to indexing errors

### 4. Optimization
- Compiler can optimize layout operations
- Automatic generation of efficient address calculations

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Understand CuTe's abstraction for expressing memory layouts algebraically
- Create and manipulate complex layouts using layout algebra operations
- Apply layout transformations for optimization purposes
- Design efficient memory access patterns using algebraic representations

## Hands-on Tutorial

See the `layout_algebra_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.