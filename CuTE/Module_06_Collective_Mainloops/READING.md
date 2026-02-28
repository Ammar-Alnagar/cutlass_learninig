# Module 06: Collective Mainloops - Comprehensive Reading Materials

## Table of Contents
1. [Introduction to Collective Mainloops](#introduction-to-collective-mainloops)
2. [Producer-Consumer Pipeline](#producer-consumer-pipeline)
3. [Collective Operations](#collective-operations)
4. [Multi-Stage Pipeline Design](#multi-stage-pipeline-design)
5. [Thread Block Cooperation](#thread-block-cooperation)
6. [Mainloop Scheduling](#mainloop-scheduling)
7. [Double Buffering](#double-buffering)
8. [Complete GEMM Mainloop](#complete-gemm-mainloop)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Kernel Orchestration](#advanced-kernel-orchestration)

---

## Introduction to Collective Mainloops

### What is a Mainloop?

The **mainloop** is the core computation loop in a GPU kernel that:
- Iterates over the K-dimension (reduction dimension)
- Loads data tiles from global memory
- Performs computation (MMA, etc.)
- Accumulates results
- Stores final results

### Why "Collective"?

**Collective** means threads work together in coordinated patterns:
- **Cooperative loading**: Multiple threads load data together
- **Synchronized computation**: Threads compute in lockstep
- **Coordinated storage**: Threads store results together

### Complete Kernel Structure

```
GEMM Kernel
    ├── Prologue
    │       └── Load first tiles, initialize accumulators
    │
    ├── Mainloop (K-dimension iteration)
    │       ├── Wait for data load
    │       ├── Compute (MMA)
    │       ├── Load next tiles
    │       └── Pipeline synchronization
    │
    └── Epilogue
            └── Store results, cleanup
```

---

## Producer-Consumer Pipeline

### Pipeline Concept

**Producer-Consumer** pattern separates data loading from computation:

```
Time →
    Producer: [Load 0] [Load 1] [Load 2] [Load 3]
    Consumer:         [Compute 0] [Compute 1] [Compute 2] [Compute 3]
```

### Single-Stage Pipeline

```cpp
// Sequential (no overlap)
for (int k = 0; k < K; ++k) {
    load_tile(k);      // Producer
    compute_tile(k);   // Consumer
}
// Time: N × (load + compute)
```

### Two-Stage Pipeline

```cpp
// Overlapping load and compute
load_tile(0);
for (int k = 1; k < K; ++k) {
    compute_tile(k-1);  // Consumer processes previous
    load_tile(k);       // Producer loads next
}
compute_tile(K-1);
// Time: load + N × compute (overlapped!)
```

### Pipeline Throughput

| Stages | Throughput | Complexity | Use Case |
|--------|------------|------------|----------|
| 1 | 1.0x | Low | Baseline |
| 2 | ~1.8x | Medium | Common |
| 3 | ~2.3x | High | Performance |
| 4 | ~2.6x | Very High | Maximum |

---

## Collective Operations

### Collective Copy (TiledCopy)

**TiledCopy** coordinates multiple threads for efficient data movement:

```cpp
#include <cute/algorithms/copy.hpp>

using namespace cute;

// Define TiledCopy
auto tiled_copy = make_tiled_copy_C(
    Copy_Atom<GlobalMemory, SharedMemory, float>{},
    thread_layout,    // How threads are organized
    value_layout      // How values are distributed
);

// Execute collective copy
tiled_copy.copy(src_tensor, dst_tensor);
```

### Thread Cooperation in Copy

```cpp
// Each thread copies a portion of the tile
__global__ void collective_copy(float* src, float* dst, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperative: all threads work together
    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}
```

### Collective MMA

```cpp
// 32 threads cooperate on one MMA operation
__device__ void collective_mma(
    float accum[16][16],
    float A_tile[16][16],
    float B_tile[16][16]
) {
    // All 32 threads participate
    mma_sync(accum, A_tile, B_tile);
}
```

---

## Multi-Stage Pipeline Design

### Pipeline Stage Count

**Choosing the right number of stages:**

| Factor | 2-Stage | 3-Stage | 4-Stage |
|--------|---------|---------|---------|
| Shared Memory | 2× tile | 3× tile | 4× tile |
| Registers | Lower | Medium | Higher |
| Latency Hiding | Good | Better | Best |
| Complexity | Low | Medium | High |

### 2-Stage Pipeline (Double Buffering)

```cpp
template <typename TileType>
__global__ void two_stage_pipeline(...) {
    extern __shared__ uint8_t smem[];
    
    TileType smem_buffer[2];
    int produce_stage = 0;
    int consume_stage = 0;
    
    // Prologue: Load first tile
    load_tile(data[0], smem_buffer[produce_stage]);
    produce_stage = 1 - produce_stage;
    
    // Mainloop
    for (int k = 1; k < num_tiles; ++k) {
        // Load next tile
        load_tile(data[k], smem_buffer[produce_stage]);
        
        __syncthreads();
        
        // Compute with current tile
        compute(smem_buffer[consume_stage]);
        
        __syncthreads();
        
        produce_stage = 1 - produce_stage;
        consume_stage = 1 - consume_stage;
    }
    
    // Epilogue: Final compute
    compute(smem_buffer[consume_stage]);
}
```

### 3-Stage Pipeline (Triple Buffering)

```cpp
template <typename TileType>
__global__ void three_stage_pipeline(...) {
    extern __shared__ uint8_t smem[];
    
    TileType smem_buffer[3];
    int stage = 0;
    
    // Prologue: Load first two tiles
    load_tile(data[0], smem_buffer[0]);
    load_tile(data[1], smem_buffer[1]);
    
    // Mainloop
    for (int k = 2; k < num_tiles; ++k) {
        int produce = stage;
        int compute = (stage + 2) % 3;
        
        __syncthreads();
        
        // Compute with oldest tile
        compute(smem_buffer[compute]);
        
        // Load newest tile
        load_tile(data[k], smem_buffer[produce]);
        
        stage = (stage + 1) % 3;
    }
    
    // Epilogue: Compute remaining tiles
    __syncthreads();
    compute(smem_buffer[(stage + 2) % 3]);
    compute(smem_buffer[(stage + 1) % 3]);
}
```

---

## Thread Block Cooperation

### Block Organization

**Grid of thread blocks for GEMM:**

```
Output Matrix C (M×N)
    ├── Block (0,0): computes C[0:64][0:64]
    ├── Block (0,1): computes C[0:64][64:128]
    ├── Block (1,0): computes C[64:128][0:64]
    └── Block (1,1): computes C[64:128][64:128]
```

### Work Distribution

```cpp
// Calculate block's output region
int block_row = blockIdx.y;
int block_col = blockIdx.x;

int tile_m = 64;  // Elements per block in M dimension
int tile_n = 64;  // Elements per block in N dimension

int m_start = block_row * tile_m;
int n_start = block_col * tile_n;

// Each block computes its tile independently
```

### Inter-Block Independence

**Key insight:** Blocks don't synchronize with each other!

```cpp
// Each block is independent
__global__ void gemm_kernel(...) {
    // No inter-block synchronization
    // Each block computes its output tile
    // Global synchronization happens at kernel end
}
```

---

## Mainloop Scheduling

### K-Dimension Iteration

```cpp
// Basic mainloop structure
for (int k = 0; k < K / TILE_K; ++k) {
    // Load A[k] and B[k] tiles
    load_A_tile(k);
    load_B_tile(k);
    
    __syncthreads();
    
    // MMA operation
    mma_sync(accum, A_tile, B_tile);
}
```

### Optimal Scheduling

**Considerations:**
1. **Occupancy**: More blocks = more latency hiding
2. **Register usage**: Fewer registers = higher occupancy
3. **Shared memory**: Less smem = more blocks per SM
4. **Instruction mix**: Balance load/compute ratio

### Loop Unrolling

```cpp
// Manual unrolling for performance
#pragma unroll 4
for (int k = 0; k < K / TILE_K; ++k) {
    mma_sync(accum, A_tile[k], B_tile[k]);
}

// Compiler unrolls the loop
// Reduces loop overhead
// Increases code size
```

---

## Double Buffering

### Double Buffering Concept

**Use two buffers to overlap load and compute:**

```
Buffer 0: [Load 0] [Compute 0] [Load 2] [Compute 2] ...
Buffer 1:         [Load 1] [Compute 1] [Load 3] ...

Time:     [--Load--][--Compute--]
```

### Double Buffering Implementation

```cpp
__global__ void double_buffered_gemm(...) {
    extern __shared__ float smem[];
    
    float* As[2];
    float* Bs[2];
    As[0] = smem;
    As[1] = &smem[TILE_SIZE];
    Bs[0] = &smem[2 * TILE_SIZE];
    Bs[1] = &smem[3 * TILE_SIZE];
    
    int write_stage = 0;
    
    // Prologue
    load_A(A, As[write_stage], 0);
    load_B(B, Bs[write_stage], 0);
    
    // Mainloop
    for (int k = 1; k < K / TILE_K; ++k) {
        int read_stage = 1 - write_stage;
        
        __syncthreads();
        
        // Load next
        load_A(A, As[write_stage], k);
        load_B(B, Bs[write_stage], k);
        
        // Compute current
        mma_sync(accum, As[read_stage], Bs[read_stage]);
        
        write_stage = 1 - write_stage;
    }
    
    // Epilogue
    __syncthreads();
    mma_sync(accum, As[1 - write_stage], Bs[1 - write_stage]);
    store_C(C, accum);
}
```

### Multi-Buffering Extension

```cpp
// Extend to N buffers
constexpr int NUM_BUFFERS = 4;
float* As[NUM_BUFFERS];
float* Bs[NUM_BUFFERS];

for (int k = 0; k < num_tiles; ++k) {
    int produce = k % NUM_BUFFERS;
    int consume = (k + NUM_BUFFERS - 1) % NUM_BUFFERS;
    
    load_next(As[produce], Bs[produce]);
    compute_current(As[consume], Bs[consume]);
}
```

---

## Complete GEMM Mainloop

### Full Kernel Structure

```cpp
template <
    typename MMA_Atom,
    int BLOCK_M, int BLOCK_N, int BLOCK_K
>
__global__ void complete_gemm(
    float* C, const half* A, const half* B,
    int M, int N, int K
) {
    // =========================================================================
    // Setup
    // =========================================================================
    
    // Calculate output tile position
    int m_start = blockIdx.y * BLOCK_M;
    int n_start = blockIdx.x * BLOCK_N;
    
    // Shared memory
    extern __shared__ uint8_t smem[];
    half* As = reinterpret_cast<half*>(smem);
    half* Bs = reinterpret_cast<half*>(&smem[BLOCK_M * BLOCK_K]);
    
    // Accumulator
    float accum[MMA_M][MMA_N];
    zero_fill(accum);
    
    // =========================================================================
    // Prologue
    // =========================================================================
    
    // Load first tiles
    load_A_tile(A, As, m_start, 0);
    load_B_tile(B, Bs, n_start, 0);
    cp_async_fence();
    
    // =========================================================================
    // Mainloop
    // =========================================================================
    
    for (int k = 1; k < K / BLOCK_K; ++k) {
        // Wait for previous load
        cp_async_wait<0>();
        __syncthreads();
        
        // MMA operation
        mma_sync(accum, As, Bs);
        
        // Load next tiles
        load_A_tile(A, As, m_start, k);
        load_B_tile(B, Bs, n_start, k);
        cp_async_fence();
    }
    
    // =========================================================================
    // Epilogue
    // =========================================================================
    
    // Final MMA
    cp_async_wait<0>();
    __syncthreads();
    mma_sync(accum, As, Bs);
    
    // Store results
    store_C_tile(C, accum, m_start, n_start);
}
```

### Launch Configuration

```cpp
// Problem dimensions
int M = 4096, N = 4096, K = 4096;

// Tile sizes
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;

// Grid dimensions
dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
          (M + BLOCK_M - 1) / BLOCK_M);

// Block dimensions
dim3 block(128);  // 128 threads per block

// Shared memory size
int smem_size = BLOCK_M * BLOCK_K * sizeof(half) +
                BLOCK_K * BLOCK_N * sizeof(half);

// Launch
complete_gemm<<<grid, block, smem_size>>>(C, A, B, M, N, K);
```

---

## Performance Optimization

### Roofline Model

**Understand performance bottlenecks:**

```
Performance (TFLOPS)
    ↑
    │    ┌────────────── Compute Bound
    │    │
    │    │
312│    │
    │    │
    │   ╱
    │  ╱  Memory Bound
    │ ╱
    └─┴────────────────→
      Low    High    Arithmetic Intensity (FLOPs/byte)
```

### Optimization Checklist

**Memory Access:**
- [ ] Coalesced global memory access
- [ ] Vectorized loads (128-bit)
- [ ] No bank conflicts in shared memory
- [ ] Minimal redundant loads

**Compute:**
- [ ] Tensor Core utilization >80%
- [ ] Pipeline depth 2-4 stages
- [ ] Loop unrolling
- [ ] Minimal overhead

**Resources:**
- [ ] Occupancy >50%
- [ ] Registers <64 per thread
- [ ] Shared memory <192 KB/SM
- [ ] No register spilling

### Profiling Workflow

```bash
# 1. Baseline measurement
ncu --set full ./gemm_app

# 2. Check memory throughput
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_app

# 3. Check Tensor Core utilization
ncu --metrics sm__inst_executed_pipe_tensor ./gemm_app

# 4. Check occupancy
ncu --metrics sm__warps_per_sm ./gemm_app

# 5. Identify bottlenecks
ncu --metrics speedometer ./gemm_app
```

---

## Advanced Kernel Orchestration

### Kernel Fusion

**Combine multiple operations:**

```cpp
// Fused GEMM + Bias + ReLU
__global__ void fused_gemm_bias_relu(...) {
    // GEMM
    gemm_mainloop(accum, A, B);
    
    // Add bias
    accum[i][j] += bias[j];
    
    // ReLU
    accum[i][j] = max(0.0f, accum[i][j]);
    
    // Store
    C[idx] = accum[i][j];
}
```

### Persistent Kernels

**Keep kernel running for multiple batches:**

```cpp
__global__ void persistent_gemm(...) {
    int batch = 0;
    
    while (batch < num_batches) {
        // Process one batch
        process_batch(batch);
        
        batch++;
    }
}
```

### Multi-Kernel Pipelines

```
Kernel 1: Load A, B → Shared Memory
    ↓
Kernel 2: Shared Memory → MMA → Accum
    ↓
Kernel 3: Accum → Apply activation → Store C
```

---

## Further Reading

### Official Documentation
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Related Modules
- Module 01-05: Foundation for collective mainloops
- All previous modules contribute to complete kernel

### Advanced Topics
- CUTLASS 3.x Architecture
- Hopper TMA (Tensor Memory Accelerator)
- Multi-Instance GPU (MIG) Optimization

---

## Quick Reference Card

### Pipeline Pattern
```cpp
// 2-stage pipeline
load(k);
for (k = 1; k < K; ++k) {
    compute(k-1);
    load(k);
}
compute(K-1);
```

### Double Buffering
```cpp
int stage = 0;
for (int k = 0; k < num_tiles; ++k) {
    load(buffer[1-stage], k);
    __syncthreads();
    compute(buffer[stage]);
    stage = 1 - stage;
}
```

### Kernel Launch
```cpp
dim3 grid(N / BLOCK_N, M / BLOCK_M);
dim3 block(128);
kernel<<<grid, block, smem_size>>>(...);
```

### Performance Targets
```
FP16 TFLOPS: >250 (A100)
Memory BW:   >1000 GB/s
Occupancy:   >50%
Tensor Core: >80%
```
