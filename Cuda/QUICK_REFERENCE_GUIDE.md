# Quick Reference Guide: Raw CUDA to CuTe Mapping

This guide provides a quick reference for mapping raw CUDA concepts to their CuTe abstractions, helping you understand how CuTe simplifies complex CUDA programming patterns.

## Memory Layout and Access Patterns

### Raw CUDA → CuTe Layout Algebra
```
// Raw CUDA: Manual indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;
data[idx] = value;

// CuTe: Layout algebra
auto layout = make_layout(make_shape(Int<1024>{}));
int idx = layout(make_coord(blockIdx.x * blockDim.x + threadIdx.x));
data[idx] = value;

// Raw CUDA: 2D indexing
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// CuTe: 2D layout
auto layout = make_layout(make_shape(Int<height>{}, Int<width>{}));
int idx = layout(make_coord(row, col));
```

## Shared Memory Banking and Swizzling

### Raw CUDA → CuTe Banking Solutions
```
// Raw CUDA: Manual padding to avoid bank conflicts
__shared__ float sdata[32][33];  // +1 to avoid conflicts

// CuTe: Layout transformations handle this automatically
auto layout = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajorSwizzle<4>{});
```

## Warp-Level Operations

### Raw CUDA → CuTe Equivalents
```
// Raw CUDA: Warp shuffle
float value_from_thread_5 = __shfl_sync(0xFFFFFFFF, local_value, 5);

// CuTe: Higher-level abstractions often handle this internally
// (Used in copy atoms and MMA operations)

// Raw CUDA: Warp reduction
__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// CuTe: Built into MMA and copy operations
```

## Memory Movement Patterns

### Raw CUDA → CuTe Copy Atoms
```
// Raw CUDA: Manual shared memory loading
__shared__ float tile[32][32];
int tid = threadIdx.x;
tile[threadIdx.y][threadIdx.x] = input[global_idx];

// CuTe: Copy atoms abstract this
auto copy_atom = make_copy_atom(DefaultCopy{}, gmem_layout, smem_layout);
copy_atom(gmem_ptr, smem_ptr);
```

## Tensor Core Operations

### Raw CUDA → CuTe MMA Atoms
```
// Raw CUDA: Manual tensor core programming (complex assembly-like code)
// Requires manual fragment management, layout specification, etc.

// CuTe: High-level MMA atoms
auto mma_atom = make_mma_atom(MMA_Traits_HSHS_HS<>{});
auto frag_A = make_fragment_like(mma_atom.ALayout(), A_ptr);
auto frag_B = make_fragment_like(mma_atom.BLayout(), B_ptr);
auto frag_C = make_fragment_like(mma_atom.CLayout(), C_ptr);
mma_atom(frag_A, frag_B, frag_C);  // C = A * B + C
```

## Tiled Algorithms

### Raw CUDA → CuTe Tiled Layouts
```
// Raw CUDA: Manual tiling with complex indexing
for (int mm = 0; mm < M; mm += TILE_M) {
    for (int nn = 0; nn < N; nn += TILE_N) {
        for (int kk = 0; kk < K; kk += TILE_K) {
            // Complex indexing for each tile
            int a_idx = (mm + ty) * K + (kk + tx);
            int b_idx = (kk + ty) * N + (nn + tx);
            // ... more complex indexing
        }
    }
}

// CuTe: Tiled layouts handle indexing automatically
auto A_tile_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}));
auto B_tile_layout = make_layout(make_shape(Int<TILE_K>{}, Int<TILE_N>{}));
auto C_tile_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
// Indexing handled automatically by layouts
```

## Performance Optimization Patterns

### Raw CUDA → CuTe Optimizations
```
// Raw CUDA: Manual software pipelining
// Stage 1: Load next tile
// Stage 2: Compute current tile  
// Stage 3: Store previous tile
// Complex state management required

// CuTe: Copy engines and layout algebra simplify this
auto pipeline_layout = make_layout(/* hierarchical layout */);
auto copy_engine = make_copy_atom(ACopy<CP_ASYNC_CACHE_LEVEL_L2>{}, /* layouts */);
// Pipeline stages managed by CuTe abstractions
```

## Common Performance Metrics

### Raw CUDA Profiling → CuTe Considerations
```
// Raw CUDA: Manual performance measurement
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<blocks, threads>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0; cudaEventElapsedTime(&ms, start, stop);

// CuTe: Still use same CUDA profiling tools
// But CuTe layouts and operations often provide better performance out of the box
// due to optimized memory access patterns and reduced indexing overhead
```

## Key Takeaways

1. **CuTe simplifies complex indexing**: Layout algebra replaces manual index calculations
2. **Abstraction without loss of control**: CuTe generates efficient code while hiding complexity
3. **Hardware-aware optimizations**: CuTe automatically applies optimizations for different GPU architectures
4. **Composability**: CuTe components can be combined to build complex algorithms
5. **Performance**: CuTe abstractions often result in better performance than manual implementations

## When to Use Each Approach

- **Raw CUDA**: When you need maximum control, debugging, or understanding the fundamentals
- **CuTe**: When building production code, complex algorithms, or when rapid prototyping

Remember: Understanding raw CUDA is essential for effective CuTe usage. This curriculum teaches both approaches sequentially to build strong foundations.