# Copy Atoms and Engines

## Concept Overview

CuTe abstracts data movement operations through "copy atoms" and "copy engines" to hide hardware-specific details. Copy atoms represent the fundamental data movement operations, while copy engines orchestrate these operations, generating efficient asynchronous copies with appropriate addressing for different memory types and hardware generations.

## Understanding Copy Atoms and Engines

### Copy Atoms

Copy atoms are the fundamental building blocks for data movement in CuTe. They represent:
- A specific data movement operation
- The source and destination layouts
- The data type being moved
- Hardware-specific implementation details

### Copy Engines

Copy engines are higher-level orchestrators that:
- Schedule and execute copy atom operations
- Handle hardware-specific optimizations
- Manage asynchronous data movement
- Abstract away low-level implementation details

## Copy Atom Basics

### Defining Copy Atoms

```cpp
#include "cutlass/cute/atom/copy_atom.hpp"
using namespace cute;

// Basic copy atom for global to shared memory
using Global2SharedCopyAtom = Copy_Atom<DefaultCopy, float>;

// More specific copy atom with custom operation
using CustomCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, half>;
```

### Common Copy Atom Types

```cpp
// Different copy atom types for various scenarios
using DefaultCopyAtom = Copy_Atom<DefaultCopy, float>;           // Default, optimized for most cases
using VectorCopyAtom = Copy_Atom<VectorCopy<4>, float>;         // Vectorized copy (4 elements at once)
using TiledCopyAtom = Copy_Atom<SM90_U32x4_STSM_N, float>;     // Hardware-specific for Hopper
using AsyncCopyAtom = Copy_Atom<cp_async, float>;               // Asynchronous copy for newer architectures
```

## Copy Engine Fundamentals

### Creating Copy Engines

```cpp
// Create a copy engine from a copy atom and layout
template<class SrcLayout, class DstLayout>
__device__ void create_copy_engine_example(
    float const* src, float* dst,
    SrcLayout const& src_layout,
    DstLayout const& dst_layout) {
    
    // Create a copy atom
    using CopyOp = Copy_Atom<DefaultCopy, float>;
    
    // Create the copy engine with the atom and layouts
    auto copy_engine = make_copy_engine(CopyOp{}, src_layout, dst_layout);
    
    // Execute the copy
    copy_engine(src, dst);
}
```

### Tiled Copy Engines

```cpp
// More complex example with tiled layouts
template<int M, int N, int TM, int TN>
__device__ void tiled_copy_example(
    float const* A_global, float* A_shared) {
    
    // Define matrix dimensions and tile sizes
    auto A_full = make_layout(make_shape(Int<M>{}, Int<N>{}));
    auto A_tile = make_shape(Int<TM>{}, Int<TN>{});
    
    // Create tiled layout
    auto A_tiled = tile(A_full, A_tile);
    
    // Create copy atom for the operation
    using CopyAtom = Copy_Atom<DefaultCopy, float>;
    
    // Create copy engine for the tiled operation
    auto copy_op = make_tiled_copy(CopyAtom{}, A_tiled);
    
    // Execute the tiled copy
    copy_op(A_global, A_shared);
}
```

## Practical Copy Engine Examples

### Global to Shared Memory Copy

```cpp
// Copy from global to shared memory with tiling
template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N>
__device__ void gmem_to_smem_copy(
    float const* gmem_src, 
    float* smem_dst) {
    
    // Define the copy operation
    using CopyOp = Copy_Atom<DefaultCopy, float>;
    
    // Create the tiled copy operation
    // This creates a copy that moves data in tiles matching the compute pattern
    auto tiled_copy = make_tiled_copy(
        CopyOp{},
        make_layout(make_shape(Int<WARP_M>{}, Int<WARP_N>{})),  // ThrID layout
        make_layout(make_shape(_1{}, _1{})),                    // Val layout (1 value per thread)
        make_layout(make_shape(Int<BLOCK_M/WARP_M>{}, Int<BLOCK_N/WARP_N>{}))  // Blk layout
    );
    
    // Get thread and block information
    int tid = threadIdx.x;
    auto thread_idx = make_coord(tid % (BLOCK_M/WARP_M), tid / (BLOCK_M/WARP_M));
    
    // Execute the copy
    copy(tiled_copy, gmem_src, smem_dst, thread_idx);
}
```

### Asynchronous Copy Engine

```cpp
// Asynchronous copy using cp.async (Ampere and later)
template<int TILE_M, int TILE_N>
__device__ void async_copy_example(
    float const* gmem_src,
    float* smem_dst) {
    
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 800  // For Ampere and later
    using AsyncCopyAtom = Copy_Atom<cp_async, float>;
    #else
    using AsyncCopyAtom = Copy_Atom<DefaultCopy, float>;
    #endif
#else
    using AsyncCopyAtom = Copy_Atom<DefaultCopy, float>;
#endif
    
    // Create tiled layout for the async copy
    auto copy_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
    
    // Create the async copy engine
    auto async_copy = make_tiled_copy(
        AsyncCopyAtom{},
        copy_layout
    );
    
    // Initiate async copy
    copy_with_thread_slice(async_copy, gmem_src, smem_dst);
    
    // Synchronize before using the data
    cp_async_fence();
    cp_async_wait_all();
}
```

## Advanced Copy Engine Concepts

### Multi-Stage Copy Pipelines

```cpp
// Three-stage copy pipeline: global → shared A → shared B → registers
template<int STAGES>
struct MultiStageCopyPipeline {
    float* buffers[STAGES];
    
    template<int PIPELINE_STAGE>
    __device__ void advance_stage(
        float const* gmem_src,
        int stage_offset) {
        
        using CopyAtom = Copy_Atom<DefaultCopy, float>;
        auto copy_op = make_tiled_copy(CopyAtom{}, 
                                     make_layout(make_shape(Int<32>{}, Int<32>{})));
        
        // Copy from global to current stage buffer
        if constexpr (PIPELINE_STAGE < STAGES) {
            copy(copy_op, 
                 gmem_src + stage_offset, 
                 buffers[PIPELINE_STAGE]);
        }
    }
};
```

### Hardware-Specific Copy Engines

```cpp
// Hardware-optimized copy based on architecture
template<int ARCH>
struct HardwareSpecificCopy {
    template<class Layout>
    __device__ static void copy_data(float const* src, float* dst, Layout const& layout) {
        if constexpr (ARCH >= 900) {  // Hopper
            using CopyOp = Copy_Atom<SM90_U16x2_STSM_N, float>;
            auto copy_engine = make_tiled_copy(CopyOp{}, layout);
            copy(copy_engine, src, dst);
        } 
        else if constexpr (ARCH >= 800) {  // Ampere
            using CopyOp = Copy_Atom<cp_async, float>;
            auto copy_engine = make_tiled_copy(CopyOp{}, layout);
            copy_with_thread_slice(copy_engine, src, dst);
        }
        else {  // Older architectures
            using CopyOp = Copy_Atom<DefaultCopy, float>;
            auto copy_engine = make_tiled_copy(CopyOp{}, layout);
            copy(copy_engine, src, dst);
        }
    }
};
```

## Copy Engine Configuration

### Thread Mapping

```cpp
// Configure how threads participate in the copy operation
template<int COPY_THREADS, int ELEMENTS_PER_THREAD>
__device__ void configure_thread_mapping() {
    // Define how threads map to data elements
    auto thr_layout = make_layout(make_shape(Int<COPY_THREADS>{}));
    auto val_layout = make_layout(make_shape(Int<ELEMENTS_PER_THREAD>{}));
    
    // Create copy atom
    using CopyAtom = Copy_Atom<DefaultCopy, float>;
    
    // Create tiled copy with specific thread/value mapping
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        thr_layout,   // How threads are arranged
        val_layout    // How values are distributed per thread
    );
}
```

### Memory Space Considerations

```cpp
// Different copy strategies for different memory spaces
enum class MemorySpace { Global, Shared, Texture, Constant };

template<MemorySpace SPACE>
struct CopyStrategy {
    template<class Layout>
    __device__ static void copy(float const* src, float* dst, Layout const& layout) {
        if constexpr (SPACE == MemorySpace::Global) {
            // Optimize for global memory characteristics
            using CopyAtom = Copy_Atom<DefaultCopy, float>;
            auto copy_op = make_tiled_copy(CopyAtom{}, layout);
            copy(copy_op, src, dst);
        }
        else if constexpr (SPACE == MemorySpace::Shared) {
            // Optimize for shared memory characteristics
            using CopyAtom = Copy_Atom<VectorCopy<4>, float>;  // Vectorized for shared mem
            auto copy_op = make_tiled_copy(CopyAtom{}, layout);
            copy(copy_op, src, dst);
        }
        // Additional memory spaces...
    }
};
```

## Performance Optimization

### Copy Engine Tuning

```cpp
// Performance tuning parameters for copy engines
struct CopyEngineParams {
    static constexpr int PREFETCH_DISTANCE = 2;     // How many tiles ahead to prefetch
    static constexpr int UNROLL_COPY_LOOPS = true;  // Unroll copy loops for performance
    static constexpr int USE_VECTOR_LOADS = true;   // Use vectorized memory operations
    static constexpr int ALIGNMENT_REQUIREMENTS = 128; // Byte alignment for optimal performance
};

// Optimized copy with tuning parameters
template<int TILE_M, int TILE_N>
__device__ void optimized_copy(
    float const* src, 
    float* dst,
    bool use_vectorization = CopyEngineParams::USE_VECTOR_LOADS) {
    
    if (use_vectorization) {
        using CopyAtom = Copy_Atom<VectorCopy<4>, float>;  // 4-element vectorized copy
        auto copy_op = make_tiled_copy(CopyAtom{}, 
                                     make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{})));
        copy(copy_op, src, dst);
    } else {
        using CopyAtom = Copy_Atom<DefaultCopy, float>;
        auto copy_op = make_tiled_copy(CopyAtom{}, 
                                     make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{})));
        copy(copy_op, src, dst);
    }
}
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Understand CuTe's abstraction for hardware-agnostic data movement patterns
- Create and configure copy atoms and engines for different memory operations
- Implement efficient data movement using CuTe's high-level abstractions
- Select appropriate copy strategies based on hardware capabilities and memory types

## Hands-on Tutorial

See the `copy_atoms_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.