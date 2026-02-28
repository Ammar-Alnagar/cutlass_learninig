# Module 03: Tiled Copy - Comprehensive Reading Materials

## Table of Contents
1. [Introduction to Tiled Copy](#introduction-to-tiled-copy)
2. [Memory Hierarchy and Data Movement](#memory-hierarchy-and-data-movement)
3. [Copy Atoms and TiledCopy](#copy-atoms-and-tiledcopy)
4. [Vectorized Memory Operations](#vectorized-memory-operations)
5. [Async Copy with cp.async](#async-copy-with-cpasync)
6. [Coalescing Strategies](#coalescing-strategies)
7. [Thread Cooperation Patterns](#thread-cooperation-patterns)
8. [Shared Memory Transfer Patterns](#shared-memory-transfer-patterns)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Copy Patterns](#advanced-copy-patterns)

---

## Introduction to Tiled Copy

### What is Tiled Copy?

**Tiled Copy** is CuTe's abstraction for efficient, hardware-aware data movement. It generalizes simple memory copies to:
- **Vectorized loads/stores**: 128-bit (or wider) memory transactions
- **Thread cooperation**: Multiple threads working together
- **Tiled organization**: Data organized in tiles for reuse
- **Async operations**: Overlap memory transfer with computation

### Why Tiled Copy Matters

Traditional CUDA copy:
```cpp
__global__ void copy_kernel(float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];  // Scalar load/store
    }
}
```

Tiled Copy approach:
```cpp
// Vectorized, coalesced, tiled copy
TiledCopy tiled_copy;
tiled_copy.copy(src_tile, dst_tile);  // 4x or more bandwidth
```

**Benefits:**
- **4x bandwidth** with 128-bit vectorized loads
- **Better occupancy** with organized thread cooperation
- **Compiler-friendly** enables better optimization
- **Portable** works across architectures

### The Copy Hierarchy

```
Copy Operation
    ├── Copy Atom (single thread's work)
    │       └── Vectorized load/store (float4, etc.)
    │
    └── TiledCopy (thread block cooperation)
            └── Multiple atoms working together
```

---

## Memory Hierarchy and Data Movement

### GPU Memory Spaces

| Memory Space | Size | Latency | Scope | Use Case |
|--------------|------|---------|-------|----------|
| Global (gmem) | GBs | 400-800 cycles | All threads | Input/output data |
| Shared (smem) | KBs/SM | ~20 cycles | Block | Tile buffering |
| Register (rmem) | KBs/thread | 1 cycle | Thread | Active computation |
| L1/Texture | KBs/SM | ~100 cycles | All | Caching |
| L2 | MBs | ~200 cycles | All | Cross-SM sharing |

### Data Flow in GEMM

```
Global Memory (A, B)
       ↓ (coalesced load)
Shared Memory (tiles)
       ↓ (vectorized load)
Registers (MMA inputs)
       ↓ (MMA operation)
Registers (accumulators)
       ↓ (coalesced store)
Global Memory (C)
```

### Copy Patterns

**Global to Shared (gmem → smem):**
```cpp
__global__ void gmem_to_smem(float* gmem, float* smem) {
    // Coalesced load from global
    // Efficient store to shared
}
```

**Shared to Register (smem → rmem):**
```cpp
__global__ void smem_to_rmem(float* smem, float* rmem) {
    // Vectorized load from shared
    // Register storage for compute
}
```

**Register to Global (rmem → gmem):**
```cpp
__global__ void rmem_to_gmem(float* rmem, float* gmem) {
    // Register data after compute
    // Coalesced store to global
}
```

---

## Copy Atoms and TiledCopy

### What is a Copy Atom?

A **Copy Atom** is the smallest unit of copy work assigned to a single thread. It defines:
- **Data type**: float, half, int, etc.
- **Vectorization**: 1, 2, 4, or 8 elements per access
- **Memory space**: gmem, smem, rmem

```cpp
// Conceptual copy atom structure
struct CopyAtom {
    using ValueType = float;
    static constexpr int Elements = 4;  // 128-bit vector
    using DstSpace = SharedMemory;
    using SrcSpace = GlobalMemory;
};
```

### TiledCopy Abstraction

**TiledCopy** coordinates multiple threads performing copy operations:

```cpp
// TiledCopy configuration
struct TiledCopyConfig {
    // Thread layout (how threads are organized)
    using ThreadLayout = Layout<Shape<Int<16>, Int<16>>>;
    
    // Value type and vectorization
    using ValueType = float;
    static constexpr int VectorSize = 4;
    
    // Memory spaces
    using SrcSpace = GlobalMemory;
    using DstSpace = SharedMemory;
};
```

### CuTe TiledCopy Usage

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithms/copy.hpp>

using namespace cute;

// Define copy traits
struct CopyTraits {
    using ValType = float;
    using MemType = GlobalMemory;
    static constexpr int Alignment = 128;  // bits
};

// Create TiledCopy
auto tiled_copy = make_tiled_copy_C(
    Copy_Atom<CopyTraits>{},
    thread_layout,
    value_layout
);

// Execute copy
tiled_copy.copy(src_tensor, dst_tensor);
```

---

## Vectorized Memory Operations

### Vectorized Loads

Load multiple elements in a single instruction:

```cpp
// 128-bit load (4 floats)
float4 val = reinterpret_cast<float4*>(&src[idx])[0];

// Store
reinterpret_cast<float4*>(&dst[idx])[0] = val;
```

**Alignment Requirements:**
- `float4`: 16-byte alignment (idx % 4 == 0 for float*)
- `float2`: 8-byte alignment (idx % 2 == 0)
- `float1`: 4-byte alignment (any idx)

### Vectorized Copy in CuTe

```cpp
// Vectorized copy atom
struct VectorizedCopy {
    static constexpr int VectorSize = 4;  // 4 floats = 128 bits
    
    template <typename Dst, typename Src>
    static __device__ void copy(Dst& dst, Src const& src, int idx) {
        using VecType = typename VectorType<float, VectorSize>::Type;
        VecType val = reinterpret_cast<VecType*>(&src[idx])[0];
        reinterpret_cast<VecType*>(&dst[idx])[0] = val;
    }
};
```

### Bandwidth Comparison

| Operation | Elements/Cycle | Bandwidth (A100) |
|-----------|----------------|------------------|
| Scalar (32-bit) | 1x | 387 GB/s |
| float2 (64-bit) | 2x | 774 GB/s |
| float4 (128-bit) | 4x | 1548 GB/s |

**Note:** Actual bandwidth depends on many factors including cache hits and memory clock.

---

## Async Copy with cp.async

### What is cp.async?

**cp.async** is an async copy instruction (sm_80+) that:
- Initiates memory transfer without blocking
- Allows compute to overlap with memory transfer
- Uses special shared memory (shared memory cluster)

### Basic cp.async Usage

```cpp
// Async copy from global to shared
__device__ void async_copy(float* gmem_ptr, float* smem_ptr, int bytes) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;"
        : : "r"(smem_offset), "l"(gmem_ptr), "r"(bytes)
    );
}

// Wait for all async copies to complete
__device__ void async_wait() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}
```

### Async Copy Pipeline

```
Time 0: [cp.async stage 0]
Time 1: [compute stage 0] [cp.async stage 1]
Time 2: [compute stage 1] [cp.async stage 2]
Time 3: [compute stage 2] [cp.async stage 3]
Time 4: [compute stage 3]
```

### cp.async with CuTe

```cpp
#include <cute/algorithms/copy_async.hpp>

// Async copy atom
using AsyncCopyAtom = Copy_Atom<SM80_CP_ASYNC<float>>;

// Create tiled async copy
auto async_copy = make_tiled_copy_C(
    AsyncCopyAtom{},
    thread_layout,
    value_layout
);

// Issue async copy
async_copy.copy(src_tensor, dst_tensor);

// Wait for completion
cp_async_wait<0>();  // Wait for all pending copies
```

### Async Copy Best Practices

1. **Use multiple stages** (double/triple buffering)
2. **Wait at the right time** (not too early, not too late)
3. **Check for errors** (async copy can fail)
4. **Align memory** (128-byte alignment for best performance)

```cpp
// Double buffering pattern
__shared__ float smem[2][TILE_SIZE];
int write_stage = 0;
int read_stage = 0;

for (int k = 0; k < K; ++k) {
    // Issue async copy for next stage
    async_copy(src[k], smem[write_stage]);
    cp_async_fence();
    
    // Wait for previous copy
    if (k > 0) {
        cp_async_wait<0>();
        __syncthreads();
        
        // Compute with data
        compute(smem[read_stage]);
    }
    
    // Ping-pong buffers
    write_stage = 1 - write_stage;
    read_stage = 1 - read_stage;
}
```

---

## Coalescing Strategies

### What is Coalescing?

**Coalescing** is when consecutive threads access consecutive memory addresses, enabling the hardware to combine multiple accesses into fewer transactions.

### Coalesced Access Patterns

**Row-Major Matrix - Coalesced:**
```cpp
__global__ void coalesced_load(float* matrix, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Consecutive threadIdx.x -> consecutive col -> consecutive memory
    float val = matrix[row * width + col];  // COALESCED
}
```

**Row-Major Matrix - Uncoalesced:**
```cpp
__global__ void uncoalesced_load(float* matrix, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Consecutive threadIdx.x -> consecutive row -> stride of width
    float val = matrix[row * width + col];  // UNCOALESCED
}
```

### Coalescing Efficiency

| Pattern | Transactions | Efficiency |
|---------|--------------|------------|
| Perfect coalescing | 1 per warp | 100% |
| Half coalescing | 2 per warp | 50% |
| No coalescing | 16 per warp | 6.25% |

### Layout-Based Coalescing

Use CuTe layouts to ensure coalescing:

```cpp
// For row-major data, use row-major layout
auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
auto tensor = make_tensor(ptr, layout);

// Access with thread layout that matches
int col = threadIdx.x;  // Consecutive threads -> consecutive columns
float val = tensor[row, col];  // Coalesced!
```

---

## Thread Cooperation Patterns

### Thread Organization

**1D Thread Block:**
```cpp
// 128 threads in 1D
auto thread_layout = make_layout(Int<128>{});
```

**2D Thread Block:**
```cpp
// 16x16 thread block
auto thread_layout = make_layout(
    make_shape(Int<16>{}, Int<16>{}),
    make_stride(Int<16>{}, Int<1>{})
);
```

**Warp-Based Organization:**
```cpp
// 4 warps of 32 threads
auto thread_layout = make_layout(
    make_shape(Int<4>{}, Int<32>{}),
    make_stride(Int<32>{}, Int<1>{})
);
```

### Work Division

**Equal Division:**
```cpp
// Each thread copies the same amount
int elements_per_thread = total_elements / num_threads;
int start = threadIdx.x * elements_per_thread;
```

**Tiled Division:**
```cpp
// 2D tile per thread
auto tile_layout = make_layout(make_shape(Int<4>{}, Int<4>{}));
int row_start = blockIdx.y * tile_size_y + threadIdx.y * tile_h;
int col_start = blockIdx.x * tile_size_x + threadIdx.x * tile_w;
```

### Cooperative Copy Example

```cpp
__global__ void cooperative_copy(float* src, float* dst, int M, int N) {
    // Each thread copies a 4x4 tile
    constexpr int TILE_H = 4;
    constexpr int TILE_W = 4;
    
    int row_start = blockIdx.y * blockDim.y * TILE_H + threadIdx.y * TILE_H;
    int col_start = blockIdx.x * blockDim.x * TILE_W + threadIdx.x * TILE_W;
    
    for (int tile_row = 0; tile_row < TILE_H; ++tile_row) {
        for (int tile_col = 0; tile_col < TILE_W; ++tile_col) {
            int row = row_start + tile_row;
            int col = col_start + tile_col;
            
            if (row < M && col < N) {
                dst[row * N + col] = src[row * N + col];
            }
        }
    }
}
```

---

## Shared Memory Transfer Patterns

### Basic gmem → smem Copy

```cpp
__global__ void gmem_to_smem(float* gmem, float* smem, int size) {
    extern __shared__ float smem_buffer[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Coalesced load
    if (idx < size) {
        smem_buffer[threadIdx.x] = gmem[idx];
    }
    __syncthreads();
    
    // Process from shared memory
    float val = smem_buffer[threadIdx.x];
}
```

### Tiled gmem → smem with 2D Threads

```cpp
__global__ void tiled_gmem_to_smem(float* gmem, float* smem, int M, int N) {
    extern __shared__ float smem_buffer[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D coalesced load
    if (row < M && col < N) {
        smem_buffer[row * N + col] = gmem[row * N + col];
    }
    __syncthreads();
}
```

### Bank Conflict Avoidance

```cpp
// Without padding (may have bank conflicts)
__shared__ float smem[32][32];

// With padding (no bank conflicts)
__shared__ float smem[32][33];  // +1 padding

// CuTe layout with padding
auto smem_layout = make_layout(
    make_shape(Int<32>{}, Int<32>{}),
    make_stride(Int<33>{}, Int<1>{})  // Padded stride
);
```

### smem → rmem Transfer

```cpp
__global__ void smem_to_rmem(float* smem, float* output, int size) {
    extern __shared__ float smem_buffer[];
    __syncthreads();
    
    // Vectorized load from shared memory
    float rmem[4];
    int idx = threadIdx.x * 4;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rmem[i] = smem_buffer[idx + i];
    }
    
    // Process in registers
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rmem[i] *= 2.0f;
    }
    
    // Store results
    for (int i = 0; i < 4; ++i) {
        output[idx + i] = rmem[i];
    }
}
```

---

## Performance Optimization

### Occupancy Considerations

**Register Usage:**
- Fewer registers → higher occupancy
- More registers → lower occupancy but potentially faster per-thread

**Shared Memory Usage:**
- Less smem → more blocks per SM
- More smem → larger tiles but fewer blocks

### Latency Hiding

Use multiple independent operations to hide latency:

```cpp
// Without latency hiding
load_A();
compute_A();
load_B();
compute_B();
// Total time: load + compute + load + compute

// With latency hiding (pipelined)
load_A();
load_B();
compute_A();  // Hides load_B latency
compute_B();
// Total time: 2*load + 2*compute (overlapped)
```

### Profiling Copy Performance

**Key Metrics:**
- **Memory Throughput**: GB/s achieved
- **Compute Throughput**: TFLOPS achieved
- **Occupancy**: Active warps / Max warps
- **Bank Conflicts**: Shared memory conflicts

**Nsight Compute Commands:**
```bash
# Profile memory throughput
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum

# Profile shared memory
ncu --metrics sm__transactions

# Profile occupancy
ncu --metrics sm__warps_per_sm
```

---

## Advanced Copy Patterns

### Matrix Transpose Copy

```cpp
__global__ void transpose_copy(float* src, float* dst, int M, int N) {
    extern __shared__ float smem[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Coalesced load (row-major)
    if (row < M && col < N) {
        smem[threadIdx.y * blockDim.x + threadIdx.x] = src[row * N + col];
    }
    __syncthreads();
    
    // Transposed store (column-major becomes row-major)
    int dst_row = col;
    int dst_col = row;
    if (dst_row < N && dst_col < M) {
        dst[dst_row * M + dst_col] = smem[threadIdx.x * blockDim.y + threadIdx.y];
    }
}
```

### Multi-Stage Pipeline Copy

```cpp
template <int Stages>
__global__ void multi_stage_copy(float* src, float* dst, int size) {
    extern __shared__ float smem[][TILE_SIZE];  // [Stages][TILE_SIZE]
    
    int stage = 0;
    
    for (int k = 0; k < num_tiles; ++k) {
        int next_stage = (stage + 1) % Stages;
        
        // Issue copy for next stage
        if (k + 1 < num_tiles) {
            copy_tile(src[k + 1], smem[next_stage]);
        }
        
        // Wait for current stage
        __syncthreads();
        
        // Process current stage
        process_tile(smem[stage], dst[k]);
        
        stage = next_stage;
    }
}
```

### Broadcast Copy

```cpp
__global__ void broadcast_copy(float* src, float* dst, int rows, int cols) {
    // Broadcast single row to all rows
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        float val = src[col];  // Load once
        
        for (int row = 0; row < rows; ++row) {
            dst[row * cols + col] = val;  // Broadcast to all rows
        }
    }
}
```

---

## Further Reading

### Official Documentation
- [CuTe Copy Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [CUTLASS 3.x Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [cp.async Programming Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Related Modules
- Module 01: Layout Algebra (foundation)
- Module 02: CuTe Tensors (data abstraction)
- Module 04: MMA Atoms (compute after copy)
- Module 05: Shared Memory Swizzling (optimization)

### Advanced Topics
- Async Copy Pipelines
- Multi-Stage Buffering
- TMA (Tensor Memory Accelerator) on sm_90

---

## Quick Reference Card

### Copy Atom Creation
```cpp
using CopyAtom = Copy_Atom<GlobalMemory, SharedMemory, float>;
auto atom = make_copy_atom<CopyAtom>();
```

### TiledCopy Usage
```cpp
auto tiled_copy = make_tiled_copy_C(
    Copy_Atom<...>{},
    thread_layout,
    value_layout
);
tiled_copy.copy(src, dst);
```

### Async Copy Pattern
```cpp
cp_async.ca.shared.global [smem], [gmem], bytes;
cp_async_fence();
cp_async_wait<0>();
```

### Vectorized Access
```cpp
// Load
float4 val = reinterpret_cast<float4*>(&ptr[idx])[0];

// Store
reinterpret_cast<float4*>(&ptr[idx])[0] = val;
```

### Coalescing Check
```cpp
// Good: consecutive threads -> consecutive memory
tensor(row, threadIdx.x);

// Bad: consecutive threads -> strided memory
tensor(threadIdx.x, col);
```
