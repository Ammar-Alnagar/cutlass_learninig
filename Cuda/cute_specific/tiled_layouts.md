# Tiled Layouts

## Concept Overview

Tiled layouts in CuTe organize data into hierarchical tiles that match the GPU memory hierarchy (registers, shared memory, global memory). This approach automatically computes addresses for complex access patterns without manual index arithmetic, enabling efficient data movement and computation across different memory levels.

## Understanding Tiled Layouts

### Hierarchical Data Organization

Tiled layouts create a multi-level hierarchy:
1. **Outer tiles**: Represent data blocks moved between memory levels
2. **Inner tiles**: Represent data blocks processed together
3. **Elements**: Individual data items within tiles

This hierarchy mirrors the GPU's memory system:
- Global memory ↔ Outer tiles
- Shared memory ↔ Middle tiles  
- Registers ↔ Inner tiles / individual elements

### Basic Tiled Layout Example

```cpp
#include "cutlass/cute/layout.hpp"
using namespace cute;

// Create a simple 2D layout
auto full_layout = make_layout(make_shape(Int<128>{}, Int<256>{}));  // 128x256 matrix

// Define tile dimensions
auto tile_shape = make_shape(Int<32>{}, Int<64>{});  // Each tile is 32x64

// Create tiled layout
auto tiled_layout = tile(full_layout, tile_shape);

// This creates a layout with:
// - Outer shape: (128/32, 256/64) = (4, 4) tiles
// - Inner shape: (32, 64) elements per tile
```

## Creating Tiled Layouts

### Simple Tiling

```cpp
// Tile a 1D array
auto array_layout = make_layout(make_shape(Int<1024>{}));  // 1024 elements
auto tile_1d = make_shape(Int<256>{});                    // Tile size of 256
auto tiled_1d = tile(array_layout, tile_1d);              // 4 tiles of 256 elements each

// Tile a 2D matrix
auto matrix_layout = make_layout(make_shape(Int<64>{}, Int<128>{}));  // 64x128 matrix
auto tile_2d = make_shape(Int<16>{}, Int<32>{});                     // 16x32 tiles
auto tiled_2d = tile(matrix_layout, tile_2d);                        // 4x4 tiles of 16x32 each
```

### Multi-Level Tiling

```cpp
// Create a 3-level tiling hierarchy
auto full_matrix = make_layout(make_shape(Int<128>{}, Int<256>{}));

// Level 1: Coarse tiles for global/shared memory transfer
auto coarse_tile = make_shape(Int<64>{}, Int<128>{});
auto coarse_tiled = tile(full_matrix, coarse_tile);  // 2x2 coarse tiles

// Level 2: Fine tiles for shared/register transfer
auto fine_tile = make_shape(Int<16>{}, Int<32>{});
auto fine_tiled = tile(coarse_tiled, fine_tile);     // Each coarse tile has 4x4 fine tiles

// This creates a hierarchy: (2,2) coarse tiles × (4,4) fine tiles × (16,32) elements
```

## Practical Tiled Layout Examples

### Matrix Multiplication Tiling

```cpp
// Tiled layout for GEMM (General Matrix Multiplication)
template<int M, int N, int K, int BM, int BN, int BK>
auto make_gemm_tiled_layouts() {
    // Matrix dimensions
    auto A_shape = make_shape(Int<M>{}, Int<K>{});
    auto B_shape = make_shape(Int<K>{}, Int<N>{});
    auto C_shape = make_shape(Int<M>{}, Int<N>{});
    
    // Tile dimensions
    auto A_tile = make_shape(Int<BM>{}, Int<BK>{});
    auto B_tile = make_shape(Int<BK>{}, Int<BN>{});
    auto C_tile = make_shape(Int<BM>{}, Int<BN>{});
    
    // Create base layouts (assuming row-major for simplicity)
    auto A_base = make_layout(A_shape, make_stride(_1{}, _M{}));
    auto B_base = make_layout(B_shape, make_stride(_1{}, _K{}));
    auto C_base = make_layout(C_shape, make_stride(_1{}, _M{}));
    
    // Create tiled layouts
    auto A_tiled = tile(A_base, A_tile);
    auto B_tiled = tile(B_base, B_tile);
    auto C_tiled = tile(C_base, C_tile);
    
    return make_tuple(A_tiled, B_tiled, C_tiled);
}

// Example: 128x256x512 GEMM with 32x32x16 tiles
auto [A_layout, B_layout, C_layout] = make_gemm_tiled_layouts<128, 256, 512, 32, 32, 16>();
```

### Memory Hierarchy Matching

```cpp
// Layout that matches GPU memory hierarchy
template<int REG_TILE_M, int REG_TILE_N, 
         int SHARED_TILE_M, int SHARED_TILE_N,
         int GLOBAL_TILE_M, int GLOBAL_TILE_N>
auto make_hierarchical_layout() {
    // Start with global memory tile
    auto global_shape = make_shape(Int<GLOBAL_TILE_M>{}, Int<GLOBAL_TILE_N>{});
    auto global_layout = make_layout(global_shape);
    
    // Add shared memory tiling
    auto shared_tile = make_shape(Int<SHARED_TILE_M>{}, Int<SHARED_TILE_N>{});
    auto shared_layout = tile(global_layout, shared_tile);
    
    // Add register tiling
    auto reg_tile = make_shape(Int<REG_TILE_M>{}, Int<REG_TILE_N>{});
    auto reg_layout = tile(shared_layout, reg_tile);
    
    return reg_layout;
    // This creates: (G_M/S_M, G_N/S_N) shared tiles × (S_M/R_M, S_N/R_N) reg tiles × (R_M, R_N) elements
}
```

## Working with Tiled Layouts

### Accessing Tiled Data

```cpp
// Function to iterate through a tiled layout
template<class TiledLayout>
__device__ void process_tiled_data(float* data, TiledLayout const& layout) {
    // Iterate through outer tiles
    for (int outer_i = 0; outer_i < size<0>(layout.shape()); ++outer_i) {
        for (int outer_j = 0; outer_j < size<1>(layout.shape()); ++outer_j) {
            auto outer_coord = make_coord(outer_i, outer_j);
            
            // Get the inner tile layout for this outer tile
            auto inner_layout = layout(outer_coord);
            
            // Iterate through elements in the inner tile
            for (int inner_idx = 0; inner_idx < size(inner_layout); ++inner_idx) {
                auto inner_coord = idx2crd(inner_idx, inner_layout.shape());
                auto full_coord = make_coord(get<0>(outer_coord), get<1>(outer_coord), 
                                           get<0>(inner_coord), get<1>(inner_coord));
                
                int addr = layout(make_coord(outer_coord, inner_coord));
                data[addr] *= 2.0f;  // Process the element
            }
        }
    }
}
```

### Tiled Data Movement

```cpp
// Tiled copy between different memory spaces
template<class Layout>
__device__ void tiled_copy(float const* src, float* dst, 
                          Layout const& layout) {
    // Copy each tile separately
    CUTE_UNROLL
    for (int tile_idx = 0; tile_idx < size<0>(layout.shape()); ++tile_idx) {
        auto tile_layout = layout(make_coord(tile_idx));
        
        // Copy elements within this tile
        CUTE_UNROLL
        for (int elem_idx = 0; elem_idx < size<1>(layout.shape()); ++elem_idx) {
            auto elem_coord = idx2crd(elem_idx, tile_layout.shape());
            int src_addr = tile_layout(elem_coord, _0{});
            int dst_addr = tile_layout(elem_coord, _1{});
            dst[dst_addr] = src[src_addr];
        }
    }
}
```

## Advanced Tiling Concepts

### Padding for Bank Conflict Avoidance

```cpp
// Tiled layout with padding to avoid shared memory bank conflicts
template<int TILE_M, int TILE_N>
auto make_padded_tiled_layout() {
    // Add padding to avoid bank conflicts (commonly +1 for 32-bank systems)
    auto padded_shape = make_shape(Int<TILE_M>{}, Int<TILE_N+1>{});  // +1 padding
    auto base_layout = make_layout(padded_shape);
    
    // Original tile without padding
    auto logical_shape = make_shape(Int<TILE_M>{}, Int<TILE_N>{});
    
    // Create a layout that maps logical coordinates to padded physical coordinates
    auto padded_layout = make_layout(
        logical_shape,
        make_stride(size<0>(logical_shape),  // stride for rows
                   size<0>(logical_shape)+1) // stride for cols with padding
    );
    
    return padded_layout;
}
```

### Swizzled Tiled Layouts

```cpp
// Tiled layout with swizzling to avoid bank conflicts
template<int TILE_M, int TILE_N>
auto make_swizzled_tiled_layout() {
    auto base_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
    
    // Define swizzling function
    auto swizzle_fn = [](int row, int col) {
        // XOR-based swizzling to distribute across banks
        int swizzled_col = col ^ (row & 0x7);  // Swizzle with lower 3 bits of row
        return make_coord(row, swizzled_col);
    };
    
    // Apply swizzling to the layout
    auto swizzled_layout = transform_layout(base_layout, swizzle_fn);
    
    return swizzled_layout;
}
```

## Performance Considerations

### Tile Size Selection

```cpp
// Guidelines for selecting tile sizes
struct TileSizeGuidelines {
    // For register-level tiling
    static constexpr int REG_TILE_MIN = 8;   // Minimum efficient tile size
    static constexpr int REG_TILE_MAX = 32;  // Maximum practical tile size
    
    // For shared memory tiling
    static constexpr int SHARED_TILE_MIN = 32;   // Minimum for coalescing
    static constexpr int SHARED_TILE_PREF = 128; // Preferred for occupancy
    
    // Should be multiples of warp size (32) for coalescing
    static constexpr int COALESCING_FACTOR = 32;
};
```

### Memory Hierarchy Alignment

```cpp
// Align tile sizes with memory hierarchy characteristics
template<int SM_COUNT, int MAX_THREADS_PER_SM, int WARP_SIZE = 32>
struct MemoryHierarchyAlignment {
    // Calculate optimal tile sizes based on hardware characteristics
    static constexpr int OPTIMAL_SHARED_MEM_PER_BLOCK = 48 * 1024; // 48KB typical
    
    // Ensure tiles fit in shared memory
    template<int ELEMENT_SIZE_BYTES>
    static constexpr int max_tile_elements() {
        return OPTIMAL_SHARED_MEM_PER_BLOCK / ELEMENT_SIZE_BYTES;
    }
    
    // Balance between occupancy and memory usage
    static constexpr int preferred_concurrent_tiles() {
        return (MAX_THREADS_PER_SM / WARP_SIZE) * 2; // 2x for occupancy
    }
};
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Recognize how CuTe expresses hierarchical tiling through layout composition
- Create multi-level tiled layouts that match GPU memory hierarchy
- Apply tiling strategies for different memory levels (global, shared, register)
- Design efficient tiled access patterns for computational kernels

## Hands-on Tutorial

See the `tiled_layouts_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.