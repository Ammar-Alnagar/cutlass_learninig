# CuTe Copy Atoms and Engines

## Concept Overview

CuTe abstracts data movement operations through copy atoms and copy engines, hiding hardware-specific details while providing efficient, portable data movement patterns. Copy engines generate optimized code for different memory types and hardware generations, automatically handling addressing for various data movement scenarios.

## Understanding Copy Atoms and Engines

### What are Copy Atoms?
- Represent a single data movement operation
- Specify source and destination layouts
- Abstract the actual copying mechanism
- Can represent different types of memory movement (GMEM ↔ SMEM, SMEM ↔ REG, etc.)

### What are Copy Engines?
- Orchestrate multiple copy atoms
- Generate efficient code for data movement
- Handle hardware-specific optimizations
- Manage different memory types and access patterns

## Copy Atom Types

### 1. Standard Copy Atoms
```cpp
#include <cute/atom/copy_atom.hpp>
using namespace cute;

// Default copy atom - works for most scenarios
auto default_copy = make_copy_atom(DefaultCopy{}, /*src=*/nullptr, /*dst=*/nullptr);

// Direct copy atom - for simple, direct memory copies
auto direct_copy = make_copy_atom(DirectCopy{}, /*src=*/nullptr, /*dst=*/nullptr);
```

### 2. Specialized Copy Atoms
```cpp
// Vectorized copy atom - for vectorized memory access
auto vec_copy = make_copy_atom(VecCopy<4>{}, /*src=*/nullptr, /*dst=*/nullptr);  // 4-element vectors

// Async copy atom - for asynchronous memory operations
auto async_copy = make_copy_atom(ACopy<CP_ASYNC_CACHE_LEVEL_L2>{}, /*src=*/nullptr, /*dst=*/nullptr);
```

## Copy Engine Concepts

### 1. Basic Copy Engine
```cpp
// Create a copy engine for GMEM to SMEM transfer
auto gmem_layout = make_layout(make_shape(Int<128>{}, Int<128>{}));
auto smem_layout = make_layout(make_shape(Int<32>{}, Int<32>{}));

// Create copy atom for the transfer
auto copy_atom = make_copy_atom(DefaultCopy{}, gmem_layout, smem_layout);

// The engine orchestrates the copy operation
// copy_atom.execute(gmem_ptr, smem_ptr);
```

### 2. Tiled Copy Engine
```cpp
// Copy engine for tiled data movement
auto tile_shape = make_shape(Int<16>{}, Int<16>{});
auto thread_layout = make_layout(Int<256>{});  // 256 threads per block

// Create copy atom that distributes work among threads
auto tiled_copy = make_copy_atom(DefaultCopy{}, 
                                make_layout(tile_shape), 
                                make_layout(tile_shape, thread_layout));
```

## Copy Operation Patterns

### 1. Global to Shared Memory
```cpp
// Pattern for loading data from global to shared memory
template<class Engine, class Layout>
__device__ void gmem_to_smem_copy(typename Engine::SPointer gptr,
                                  typename Engine::DPointer sptr,
                                  Engine const& engine,
                                  Layout const& layout) {
    // Execute copy for each thread's portion
    engine.execute(layout, gptr, sptr);
}
```

### 2. Shared to Register Memory
```cpp
// Pattern for loading from shared to register memory
template<class Engine, class Layout>
__device__ void smem_to_reg_copy(typename Engine::SPointer sptr,
                                 typename Engine::DPointer& reg_ref,
                                 Engine const& engine,
                                 Layout const& layout) {
    engine.execute(layout, sptr, reg_ref);
}
```

### 3. Register to Shared Memory
```cpp
// Pattern for storing from register to shared memory
template<class Engine, class Layout>
__device__ void reg_to_smem_copy(typename Engine::SPointer& reg_ref,
                                 typename Engine::DPointer sptr,
                                 Engine const& engine,
                                 Layout const& layout) {
    engine.execute(layout, reg_ref, sptr);
}
```

## Hardware-Agnostic Data Movement

### 1. Architecture-Specific Optimizations
```cpp
// CuTe automatically selects appropriate copy mechanism based on architecture
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Use cp.async for Ampere and later
    auto copy_engine = make_copy_atom(ACopy<CP_ASYNC_CACHE_LEVEL_L2>{}, src, dst);
#else
    // Use traditional copy for older architectures
    auto copy_engine = make_copy_atom(DefaultCopy{}, src, dst);
#endif
```

### 2. Memory Type Adaptation
```cpp
// Copy engine adapts to different memory types
template<class SrcType, class DstType>
auto make_adaptive_copy(SrcType const& src, DstType const& dst) {
    if constexpr (is_global_memory<src_type>::value && 
                  is_shared_memory<dst_type>::value) {
        // Optimize for GMEM to SMEM transfer
        return make_copy_atom(ACopy<CP_ASYNC_CACHE_LEVEL_L2>{}, src, dst);
    } else if constexpr (is_shared_memory<src_type>::value && 
                         is_register<src_type>::value) {
        // Optimize for SMEM to REG transfer
        return make_copy_atom(DefaultCopy{}, src, dst);
    } else {
        // Generic copy
        return make_copy_atom(DefaultCopy{}, src, dst);
    }
}
```

## Practical Copy Atom Examples

### 1. Matrix Tile Loading
```cpp
// Load a tile of matrix A from global to shared memory
template<int TileM, int TileN>
__device__ void load_tile_A(float const* gA, float* sA) {
    // Define tile shape
    auto tile_shape = make_shape(Int<TileM>{}, Int<TileN>{});
    auto gA_layout = make_layout(tile_shape);
    auto sA_layout = make_layout(tile_shape);
    
    // Create copy atom
    auto copy_atom = make_copy_atom(DefaultCopy{}, gA_layout, sA_layout);
    
    // Execute copy
    copy_atom(gA, sA);
}
```

### 2. Vectorized Copy
```cpp
// Copy with vectorization for better bandwidth utilization
template<int VecSize>
__device__ void vectorized_copy(float const* src, float* dst, int count) {
    // Create vectorized layout
    auto vec_layout = make_layout(make_shape(Int<VecSize>{}, Int<count/VecSize>{}));
    
    // Use vectorized copy atom
    auto vec_copy = make_copy_atom(VecCopy<VecSize>{}, vec_layout, vec_layout);
    
    // Execute vectorized copy
    vec_copy(src, dst);
}
```

### 3. Asynchronous Copy
```cpp
// Asynchronous copy with cp.async instructions
__device__ void async_copy(float const* gmem, float* smem, int count) {
    // Create async copy atom
    auto async_atom = make_copy_atom(ACopy<CP_ASYNC_CACHE_LEVEL_L2>{}, 
                                    make_layout(Int<count>{}), 
                                    make_layout(Int<count>{}));
    
    // Initiate async copy
    async_atom(gmem, smem);
    
    // Wait for completion if needed
    cp_async_wait_all();
}
```

## Copy Engine Configuration

### 1. Thread Distribution
```cpp
// Configure how threads participate in the copy
auto thread_block = make_layout(Int<256>{});  // 256 threads
auto data_shape = make_shape(Int<1024>{});
auto copy_layout = zipped_divide(make_layout(data_shape), thread_block);

auto copy_atom = make_copy_atom(DefaultCopy{}, copy_layout, copy_layout);
```

### 2. Memory Access Patterns
```cpp
// Configure for different access patterns
auto row_major = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
auto col_major = make_layout(make_shape(Int<32>{}, Int<32>{}), GenColMajor{});

auto row_copy = make_copy_atom(DefaultCopy{}, row_major, row_major);
auto col_copy = make_copy_atom(DefaultCopy{}, col_major, col_major);
```

## Performance Optimization Strategies

### 1. Coalesced Access Patterns
```cpp
// Ensure copy operations maintain coalesced access
auto coalesced_layout = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
auto copy_atom = make_copy_atom(VecCopy<4>{}, coalesced_layout, coalesced_layout);
```

### 2. Vectorization
```cpp
// Use vectorized copies when possible
template<int VecSize>
auto make_vectorized_copy_engine(auto src_layout, auto dst_layout) {
    // Ensure data size is compatible with vectorization
    if (size(src_layout) % VecSize == 0) {
        return make_copy_atom(VecCopy<VecSize>{}, src_layout, dst_layout);
    } else {
        return make_copy_atom(DefaultCopy{}, src_layout, dst_layout);
    }
}
```

### 3. Asynchronous Operations
```cpp
// Overlap computation with memory transfers
__device__ void compute_with_overlap(float* data, int n) {
    // Start async copy
    auto async_copy = make_copy_atom(ACopy<CP_ASYNC_CACHE_LEVEL_L2>{}, 
                                    make_layout(Int<n>{}), 
                                    make_layout(Int<n>{}));
    async_copy(next_data, next_smem);
    
    // Perform computation on current data
    compute_current(current_data, n);
    
    // Wait for async copy to complete
    cp_async_wait_all();
}
```

## Integration with Tiled Layouts

### 1. Tiled Data Movement
```cpp
// Combine tiled layouts with copy engines
template<int TileM, int TileN, int Threads>
__device__ void tiled_data_movement(float const* gmem, float* smem) {
    // Create tiled layout
    auto tile_shape = make_shape(Int<TileM>{}, Int<TileN>{});
    auto thread_layout = make_layout(Int<Threads>{});
    
    // Divide tile among threads
    auto tiled_layout = zipped_divide(make_layout(tile_shape), thread_layout);
    
    // Create copy atom for tiled movement
    auto copy_atom = make_copy_atom(DefaultCopy{}, tiled_layout, tiled_layout);
    
    // Execute tiled copy
    copy_atom(gmem, smem);
}
```

### 2. Hierarchical Copy Patterns
```cpp
// Copy pattern that matches memory hierarchy
auto gmem_tile = make_layout(make_shape(Int<128>{}, Int<128>{}));
auto smem_tile = make_layout(make_shape(Int<32>{}, Int<32>{}));
auto reg_tile = make_layout(make_shape(Int<8>{}, Int<8>{}));

// GMEM → SMEM copy
auto g2s_copy = make_copy_atom(DefaultCopy{}, gmem_tile, smem_tile);

// SMEM → REG copy  
auto s2r_copy = make_copy_atom(DefaultCopy{}, smem_tile, reg_tile);
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Understand CuTe's abstraction for hardware-agnostic data movement patterns
- Create and configure copy atoms for different memory movement scenarios
- Use copy engines to efficiently move data between memory hierarchy levels
- Apply appropriate copy strategies based on memory types and access patterns

## Hands-on Tutorial

See the `copy_atoms_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.