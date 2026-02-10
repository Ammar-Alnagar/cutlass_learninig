# CuTe Tiled Layouts

## Concept Overview

CuTe organizes data into hierarchical tiles that match the GPU memory hierarchy (registers, shared memory, global memory). Tiled layouts represent data decomposition into blocks that can be processed efficiently at different memory levels. Layout transformations automatically compute addresses for tiled access patterns without manual index arithmetic, enabling efficient implementations of algorithms like matrix multiplication.

## Understanding Tiled Layouts

### What are Tiled Layouts?
- Hierarchical decomposition of data into rectangular blocks (tiles)
- Match GPU memory hierarchy: register tiles → shared memory tiles → global memory tiles
- Enable efficient data movement between memory levels
- Support various tiling strategies for different algorithms

### Tiled Layout Structure
```
Global Memory (Large Matrix)
    ↓
Tiled Decomposition
    ↓
Shared Memory Tiles (Block-level)
    ↓
Register Tiles (Thread-level)
```

## Tiled Layout Creation

### 1. Basic Tiled Layout
```cpp
#include <cute/layout.hpp>
using namespace cute;

// Create a 128x128 matrix decomposed into 32x32 tiles
auto matrix_shape = make_shape(Int<128>{}, Int<128>{});
auto tile_shape = make_shape(Int<32>{}, Int<32>{});
auto tiled_layout = tile_to_shape(matrix_shape, tile_shape);
// Results in a 4x4 grid of 32x32 tiles
```

### 2. Multi-Level Tiling
```cpp
// Hierarchical tiling: Global → Shared → Register
auto global_shape = make_shape(Int<128>{}, Int<128>{});
auto shared_tile = make_shape(Int<64>{}, Int<64>{});
auto register_tile = make_shape(Int<8>{}, Int<8>{});

// First level: Global to Shared
auto shared_layout = tile_to_shape(global_shape, shared_tile);

// Second level: Shared to Register
auto register_layout = tile_to_shape(shared_tile, register_tile);
```

### 3. Thread Mapping to Tiles
```cpp
// Map threads to elements within a tile
auto tile_shape = make_shape(Int<16>{}, Int<16>{});
auto thread_block = make_shape(Int<256>{});  // 256 threads per block
auto thr_layout = make_layout(thread_block);

// Distribute threads across the tile
auto tile_thr_layout = zipped_divide(make_layout(tile_shape), thr_layout);
```

## Tiled Layout Operations

### 1. Tile Extraction
```cpp
// Extract a specific tile from a tiled layout
auto tiled_matrix = make_tiled_layout(make_shape(Int<128>{}, Int<128>{}), 
                                     make_shape(Int<32>{}, Int<32>{}));
auto tile_coord = make_coord(1, 2);  // Get tile at position (1,2)
auto specific_tile = logical_divide(tiled_matrix, make_shape(Int<32>{}, Int<32>{}))[tile_coord];
```

### 2. Tile Composition
```cpp
// Combine multiple tiles into a larger layout
auto tile_A = make_layout(make_shape(Int<16>{}, Int<16>{}));
auto tile_B = make_layout(make_shape(Int<16>{}, Int<16>{}));
auto combined = make_layout(make_shape(tile_A, tile_B), 
                           make_stride(size<0>(tile_A), size<0>(tile_B)));
```

### 3. Tile Transformation
```cpp
// Transform tile layout (e.g., transpose)
auto original_tile = make_layout(make_shape(Int<8>{}, Int<4>{}));
auto transposed_tile = right_inverse(original_tile);  // 4x8 transposed
```

## Practical Tiled Layout Examples

### 1. Matrix Multiplication Tiling
```cpp
// Tiled GEMM: C = A * B
template<int BM, int BN, int BK>
auto make_gemm_tiled_layouts() {
    // Tile shapes
    auto M = Int<BM>{};
    auto N = Int<BN>{};
    auto K = Int<BK>{};
    
    // Tiled layouts for each matrix
    auto layout_A = make_layout(make_shape(M, K));
    auto layout_B = make_layout(make_shape(K, N));
    auto layout_C = make_layout(make_shape(M, N));
    
    return make_tuple(layout_A, layout_B, layout_C);
}
```

### 2. Shared Memory Tiling
```cpp
// Layout for loading tiles to shared memory
template<int TileM, int TileN, int Threads>
auto make_shared_layout() {
    auto tile_shape = make_shape(Int<TileM>{}, Int<TileN>{});
    auto thread_layout = make_layout(Int<Threads>{});
    
    // Distribute elements among threads
    auto elements_per_thread = size(tile_shape) / Threads;
    auto shared_layout = make_layout(tile_shape, thread_layout);
    
    return shared_layout;
}
```

### 3. Register Tiling
```cpp
// Layout for distributing tile elements to registers
template<int TileM, int TileN, int WarpSize = 32>
auto make_register_layout() {
    auto tile_shape = make_shape(Int<TileM>{}, Int<TileN>{});
    auto warp_layout = make_layout(Int<WarpSize>{});
    
    // Each thread gets multiple elements
    auto elements_per_thread = size(tile_shape) / WarpSize;
    auto register_layout = zipped_divide(tile_shape, warp_layout);
    
    return register_layout;
}
```

## Memory Hierarchy Matching

### 1. Global to Shared Movement
```cpp
// Define layouts for data movement
auto global_layout = make_layout(make_shape(Int<1024>{}, Int<1024>{}));
auto shared_shape = make_shape(Int<128>{}, Int<128>{});
auto shared_layout = tile_to_shape(global_layout.shape(), shared_shape);

// Each block handles one shared tile
auto block_layout = make_layout(make_shape(Int<256>{}));  // 256 threads per block
```

### 2. Shared to Register Movement
```cpp
// Layout for moving from shared to registers
auto shared_tile = make_layout(make_shape(Int<128>{}, Int<128>{}));
auto thread_layout = make_layout(Int<256>{});  // 256 threads in block

// Distribute shared tile elements to threads
auto thr_tile_layout = zipped_divide(shared_tile, thread_layout);
```

## Tiled Layout Optimization Strategies

### 1. Coalescing Optimization
```cpp
// Ensure tiled access maintains coalescing
auto tile_shape = make_shape(Int<32>{}, Int<32>{});
auto thread_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // 32x8 = 256 threads

// Arrange threads to access consecutive memory
auto coalesced_layout = make_layout(tile_shape, thread_layout);
```

### 2. Bank Conflict Avoidance
```cpp
// Add padding to avoid shared memory bank conflicts
auto base_tile = make_shape(Int<32>{}, Int<32>{});
auto padded_tile = make_shape(Int<32>{}, Int<33>{});  // +1 to avoid conflicts
auto bank_safe_layout = make_layout(padded_tile);
```

### 3. Occupancy Optimization
```cpp
// Balance tile size with occupancy
auto small_tile = make_shape(Int<16>{}, Int<16>{});   // Smaller = more blocks
auto large_tile = make_shape(Int<64>{}, Int<64>{});   // Larger = more work per block

// Choose based on shared memory and occupancy requirements
```

## Advanced Tiled Layout Concepts

### 1. Irregular Tiling
```cpp
// Non-uniform tiling for irregular data
auto irregular_shape = make_shape(Int<100>{}, Int<50>{});
auto tile_pattern = make_shape(Int<32>{}, Int<32>{});
auto tiled_layout = make_layout(irregular_shape, tile_pattern);
```

### 2. Overlapping Tiles (Stencil Operations)
```cpp
// Tiling with overlap for stencil operations
auto matrix_shape = make_shape(Int<128>{}, Int<128>{});
auto tile_shape = make_shape(Int<32>{}, Int<32>{});
auto overlap = make_shape(Int<2>{}, Int<2>{});  // 2-element overlap

// Create overlapping tiles
auto stencil_layout = make_overlapped_layout(matrix_shape, tile_shape, overlap);
```

### 3. Dynamic Tiling
```cpp
// Runtime-configurable tile sizes
template<class M, class N, class K>
auto make_dynamic_tiled_layout(M m, N n, K k) {
    auto matrix_shape = make_shape(m, n);
    auto tile_shape = make_shape(k, k);
    return tile_to_shape(matrix_shape, tile_shape);
}
```

## Performance Considerations

### 1. Tile Size Selection
- **Too small**: High overhead, poor data reuse
- **Too large**: Exceeds shared memory capacity
- **Just right**: Balances reuse and capacity

### 2. Memory Access Patterns
- Maintain coalescing within tiles
- Consider access patterns of the algorithm
- Optimize for cache line utilization

### 3. Thread Divergence
- Ensure uniform work distribution across threads
- Minimize divergent execution paths within tiles

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Recognize how CuTe expresses hierarchical tiling through layout composition
- Design tiled layouts that match the GPU memory hierarchy
- Apply tiling strategies to optimize data reuse and memory access patterns
- Understand how tiling enables efficient implementations of algorithms like GEMM

## Hands-on Tutorial

See the `tiled_layouts_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.