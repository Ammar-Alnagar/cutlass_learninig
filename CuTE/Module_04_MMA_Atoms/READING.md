# Module 04: MMA Atoms - Comprehensive Reading Materials

## Table of Contents
1. [Introduction to Matrix Multiply-Accumulate](#introduction-to-matrix-multiply-accumulate)
2. [Tensor Core Architecture](#tensor-core-architecture)
3. [MMA Atom Abstraction](#mma-atom-abstraction)
4. [Thread to Tensor Core Mapping](#thread-to-tensor-core-mapping)
5. [MMA Configurations](#mma-configurations)
6. [Accumulator Management](#accumulator-management)
7. [Mixed Precision Operations](#mixed-precision-operations)
8. [GEMM with MMA Atoms](#gemm-with-mma-atoms)
9. [Warp-Level MMA](#warp-level-mma)
10. [Performance Optimization](#performance-optimization)

---

## Introduction to Matrix Multiply-Accumulate

### What is MMA?

**Matrix Multiply-Accumulate (MMA)** is the fundamental operation:
```
D = A × B + C
```

Where:
- **A**: Left operand matrix (M×K)
- **B**: Right operand matrix (K×N)
- **C**: Accumulator matrix (M×N)
- **D**: Result matrix (M×N)

### Why MMA Matters

Traditional CUDA GEMM:
```cpp
// Scalar multiply-accumulate
for (int k = 0; k < K; ++k) {
    C[i][j] += A[i][k] * B[k][j];
}
// 1 operation per cycle
```

Tensor Core MMA:
```cpp
// Matrix multiply-accumulate
mma.sync.aligned.m16n16k16...
// 1024 operations per cycle (16×16×4)
```

**Performance Gain:**
- **32× throughput** for FP16 (vs FP32 CUDA cores)
- **64× throughput** for INT8
- **Up to 312 TFLOPS** on A100 (FP16 with sparsity)

### MMA in Deep Learning

MMA accelerates:
- **Fully Connected Layers**: Matrix multiplication
- **Convolutions**: Im2col + GEMM
- **Attention**: Q×K^T and Attention×V
- **RNNs/LSTMs**: Recurrent matrix operations

---

## Tensor Core Architecture

### Tensor Core Generations

| Generation | Architecture | FP16 | INT8 | FP64 | BF16 | TF32 |
|------------|--------------|------|------|------|------|------|
| 1st | Volta (V100) | ✓ | ✓ | ✗ | ✗ | ✗ |
| 2nd | Ampere (A100) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3rd | Hopper (H100) | ✓ | ✓ | ✓ | ✓ | ✓ |

### sm_80 Tensor Cores (A100)

**Capabilities:**
- **FP16 MMA**: 16×16×16 matrices
- **FP32 Accumulation**: Higher precision output
- **BF16 Support**: Brain floating point
- **TF32**: TensorFloat32 for ML
- **Sparsity**: 2:4 structured sparsity (2× speedup)

### Tensor Core Throughput

**A100 (sm_80) Peak Performance:**

| Precision | Operations/Clock/SM | Peak TFLOPS |
|-----------|---------------------|-------------|
| FP64 | 32 | 9.7 |
| TF32 | 512 | 156 |
| FP16 | 1024 | 312 |
| FP16 (sparse) | 2048 | 624 |
| INT8 | 2048 | 624 |
| INT4 | 4096 | 1248 |

**Calculation:**
```
Peak TFLOPS = Ops/clock × SMs × Frequency(GHz)
FP16 = 1024 × 108 × 1.4 GHz = 154 TFLOPS (dense)
FP16 (sparse) = 2048 × 108 × 1.4 GHz = 309 TFLOPS
```

---

## MMA Atom Abstraction

### What is an MMA Atom?

An **MMA Atom** is CuTe's abstraction for a single Tensor Core operation. It encapsulates:
- **Matrix dimensions**: M×N×K tile size
- **Data types**: Input and output precisions
- **Thread layout**: How 32 threads cooperate
- **Register layout**: How data is distributed

### MMA Atom Structure

```cpp
// Conceptual MMA atom definition
struct MMA_Atom {
    // Matrix dimensions
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 16;
    
    // Data types
    using A_Type = half_t;      // FP16 input A
    using B_Type = half_t;      // FP16 input B
    using C_Type = float;       // FP32 accumulator
    using D_Type = float;       // FP32 output
    
    // Thread organization
    using ThreadLayout = Shape<Int<4>, Int<8>>;  // 32 threads
};
```

### CuTe MMA Atom Usage

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithms/mma.hpp>

using namespace cute;

// Define MMA atom for sm_80
using MMA_Atom = MMA_Atom<
    SM80<16, 16, 16>,    // Architecture and tile size
    F16, F16, F32, F32   // A, B, C, D types
>;

// Create MMA operation
auto mma_atom = MMA_Atom{};

// Execute MMA
mma_atom(A_tile, B_tile, C_accumulator);
```

---

## Thread to Tensor Core Mapping

### Warp-Level Organization

**32 threads** (1 warp) cooperate to execute one MMA operation:

```
Warp (32 threads)
    ├── Thread 0-31
    │       └── Cooperate on 16×16×16 MMA
    │
    └── Each thread handles subset of elements
```

### Thread Roles in MMA

For 16×16×16 MMA with 32 threads:

**Each thread computes:**
- **A elements**: 8 values (16×16 / 32 threads × K-dimension)
- **B elements**: 8 values
- **C/D elements**: 8 values (16×16 / 32 threads)

**Register usage per thread:**
```
A fragment: 8 × FP16 = 16 bytes
B fragment: 8 × FP16 = 16 bytes
C fragment: 8 × FP32 = 32 bytes
Total: 64 bytes = 16 registers (4 bytes each)
```

### Thread Layout Example

```cpp
// 2D thread layout for 16×16 MMA
auto thread_layout = make_layout(
    make_shape(Int<4>{}, Int<8>{}),  // 4×8 = 32 threads
    make_stride(Int<8>{}, Int<1>{})
);

// Each thread's responsibility
int thread_row = threadIdx.x / 8;
int thread_col = threadIdx.x % 8;

// Each thread computes 2×4 elements of 16×16 output
int elem_row_start = thread_row * 4;
int elem_col_start = thread_col * 2;
```

---

## MMA Configurations

### sm_80 MMA Configurations

**Common Configurations:**

| Name | M×N×K | A Type | B Type | C/D Type | Use Case |
|------|-------|--------|--------|----------|----------|
| `f16_f16_f32_f32` | 16×16×16 | FP16 | FP16 | FP32 | General GEMM |
| `f16_f16_f32_f32` | 16×16×32 | FP16 | FP16 | FP32 | K-parallel |
| `bf16_bf16_f32_f32` | 16×16×8 | BF16 | BF16 | FP32 | ML Training |
| `f32_f32_f32_f32` | 8×8×4 | FP32 | FP32 | FP32 | Scientific |
| `s8_s8_s32_s32` | 16×16×32 | INT8 | INT8 | INT32 | Inference |

### Configuration Selection

**Factors to consider:**

1. **Precision requirements**:
   - FP16: Maximum throughput
   - BF16: Better range for gradients
   - FP32: Maximum accuracy

2. **Matrix dimensions**:
   - Large K: Use larger K tile (16×16×32)
   - Small K: Use smaller K tile (16×16×8)

3. **Memory constraints**:
   - More registers: Larger tiles
   - Less registers: Smaller tiles

### Configuration Example

```cpp
// FP16 GEMM (maximum throughput)
using MMA_Config = MMA_Atom<
    SM80_16x16x16_F16F16F32F32,
    ThreadLayout,
    ValueLayout
>;

// BF16 GEMM (better for training)
using MMA_Config = MMA_Atom<
    SM80_16x16x8_BF16BF16F32F32,
    ThreadLayout,
    ValueLayout
>;
```

---

## Accumulator Management

### Accumulator Registers

**Accumulators** hold intermediate and final results:

```
C_accumulator[M][N] = Σ(A[M][K] × B[K][N])
```

**For 16×16 MMA:**
- **Size**: 16×16 = 256 elements
- **Type**: FP32 (higher precision)
- **Registers**: 256 / 8 = 32 registers (8 elements per register)

### Multi-Step Accumulation

For K > tile_K, accumulate across multiple MMA operations:

```cpp
// Initialize accumulator
float accum[16][16];
zero_fill(accum);

// K-dimension loop
for (int k_tile = 0; k_tile < K / 16; ++k_tile) {
    // Load A and B tiles
    load_A_tile(A, k_tile);
    load_B_tile(B, k_tile);
    
    // MMA with accumulation
    mma_sync(accum, A_tile, B_tile);
    // accum = A_tile × B_tile + accum
}

// Store result
store_C(accum);
```

### Register Allocation

**Register budget per thread:**

| Component | Size | Registers |
|-----------|------|-----------|
| A fragment | 8×FP16 | 4 |
| B fragment | 8×FP16 | 4 |
| C fragment | 8×FP32 | 8 |
| **Total** | | **16** |

**Occupancy impact:**
- 16 registers/thread → 64 threads/SM max
- 32 registers/thread → 32 threads/SM max
- 64 registers/thread → 16 threads/SM max

---

## Mixed Precision Operations

### Why Mixed Precision?

**Benefits:**
- **Speed**: FP16 compute is 8× faster than FP32
- **Accuracy**: FP32 accumulation prevents overflow/underflow
- **Memory**: FP16 uses half the bandwidth

### Common Mixed Precision Patterns

**FP16 × FP16 → FP32:**
```cpp
// Most common for deep learning
using MMA = MMA_Atom<SM80_16x16x16, F16, F16, F32, F32>;
// A, B: FP16 | C, D: FP32
```

**BF16 × BF16 → FP32:**
```cpp
// Better for training (larger range)
using MMA = MMA_Atom<SM80_16x16x8, BF16, BF16, F32, F32>;
// A, B: BF16 | C, D: FP32
```

**INT8 × INT8 → INT32:**
```cpp
// Inference with quantization
using MMA = MMA_Atom<SM80_16x16x32, S8, S8, S32, S32>;
// A, B: INT8 | C, D: INT32
```

### Precision Conversion

```cpp
// FP32 to FP16 conversion
float fp32_val = 3.14159f;
half fp16_val = __float2half(fp32_val);

// FP16 to FP32 conversion
half fp16_val = __float2half(3.14f);
float fp32_val = __half2float(fp16_val);

// Vectorized conversion
float4 fp32_vec;
half2 fp16_vec[2];
// Convert 4 FP32 to 4 FP16
```

---

## GEMM with MMA Atoms

### Complete GEMM Structure

```
GEMM: C = A × B
    ├── Grid: Multiple thread blocks
    │       └── Each block computes C tile
    │
    ├── Block Level (128-256 threads)
    │       ├── Load A tile to shared memory
    │       ├── Load B tile to shared memory
    │       └── MMA loop over K
    │
    └── Warp Level (32 threads)
            └── Execute MMA atom
```

### Tiling Strategy

**Multi-level tiling:**

```cpp
// Problem: C[M][N] = A[M][K] × B[K][N]

// Level 1: Thread block tile
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;

// Level 2: Warp tile (MMA atom)
constexpr int WARP_M = 16;
constexpr int WARP_N = 16;
constexpr int WARP_K = 16;

// Level 3: Thread tile
constexpr int THREAD_M = 2;
constexpr int THREAD_N = 4;
```

### GEMM Kernel Structure

```cpp
template <typename MMA_Atom, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_kernel(
    float* C, const half* A, const half* B,
    int M, int N, int K
) {
    // Shared memory for tiles
    extern __shared__ uint8_t smem[];
    half* As = reinterpret_cast<half*>(smem);
    half* Bs = reinterpret_cast<half*>(&smem[BLOCK_M * BLOCK_K]);
    
    // Accumulator
    float accum[WARP_M][WARP_N];
    zero_fill(accum);
    
    // Prologue: Load first tiles
    load_tile_A(A, As, 0);
    load_tile_B(B, Bs, 0);
    cp_async_fence();
    
    // Mainloop
    for (int k = 1; k < K / BLOCK_K; ++k) {
        // Wait for previous load
        cp_async_wait<0>();
        __syncthreads();
        
        // MMA operation
        mma_sync(accum, As, Bs);
        
        // Load next tiles
        load_tile_A(A, As, k);
        load_tile_B(B, Bs, k);
        cp_async_fence();
    }
    
    // Epilogue: Final MMA and store
    cp_async_wait<0>();
    __syncthreads();
    mma_sync(accum, As, Bs);
    store_C(C, accum);
}
```

---

## Warp-Level MMA

### Warp Primitives

**Warp-level operations** require coordination:

```cpp
// Warp synchronization
__syncwarp();  // Barrier for warp

// Warp shuffle (register-to-register)
float val = __shfl_sync(0xFFFFFFFF, my_val, src_lane);

// Warp broadcast
float val = __shfl_sync(0xFFFFFFFF, my_val, 0);  // From lane 0
```

### Warp Assignment

**Multiple warps per block:**

```cpp
// 128-thread block = 4 warps
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;

// Each warp computes different output tile
int warp_row = warp_id / 2;
int warp_col = warp_id % 2;

// Warp's output region
int warp_m_start = warp_row * WARP_M;
int warp_n_start = warp_col * WARP_N;
```

### Warp Synchronization

```cpp
// Correct warp synchronization pattern
__global__ void warp_kernel(...) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Step 1: All threads load data
    load_data();
    __syncthreads();  // Block-level barrier
    
    // Step 2: Warp executes MMA
    if (lane_id < 32) {
        mma_operation();
    }
    __syncwarp();  // Warp-level barrier
    
    // Step 3: Store results
    store_data();
}
```

---

## Performance Optimization

### Occupancy Optimization

**Register tuning:**

```cpp
// Launch configuration
int registers_per_thread = 24;  // Tune for occupancy
cudaFuncSetAttribute(
    gemm_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    shared_mem_size
);

// Launch
gemm_kernel<<<grid, block, shared_mem_size>>>(...);
```

### Instruction-Level Optimization

**Unroll loops:**
```cpp
#pragma unroll 4
for (int i = 0; i < 4; ++i) {
    mma_sync(accum, A_frag[i], B_frag[i]);
}
```

**Use const memory for constants:**
```cpp
__constant__ float alpha[1];
float a = alpha[0];  // Cached access
```

### Profiling MMA Performance

**Key metrics:**

| Metric | Target | Tool |
|--------|--------|------|
| Tensor Core Util | >80% | Nsight Compute |
| Occupancy | >50% | Nsight Compute |
| Memory Throughput | >80% peak | Nsight Compute |
| SM Efficiency | >90% | Nsight Compute |

**Nsight Compute commands:**
```bash
# Profile Tensor Core utilization
ncu --metrics sm__inst_executed_pipe_tensor

# Profile occupancy
ncu --metrics sm__warps_per_sm

# Profile memory throughput
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global
```

---

## Further Reading

### Official Documentation
- [CuTe MMA Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [CUTLASS GEMM Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [Tensor Core Programming Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Related Modules
- Module 03: Tiled Copy (feeding MMA)
- Module 05: Shared Memory (optimizing data movement)
- Module 06: Collective Mainloops (complete GEMM)

### Advanced Topics
- Sparse MMA (2:4 sparsity)
- Multi-Stage Pipelines
- Hopper TMA (Tensor Memory Accelerator)

---

## Quick Reference Card

### MMA Atom Creation
```cpp
using MMA = MMA_Atom<SM80_16x16x16, F16, F16, F32, F32>;
auto mma = MMA{};
```

### MMA Operation
```cpp
// D = A × B + C
mma_sync(accum, a_frag, b_frag);
```

### Thread Mapping
```cpp
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;
```

### Common Configurations
```cpp
SM80_16x16x16_F16F16F32F32  // Standard FP16 GEMM
SM80_16x16x8_BF16BF16F32F32 // BF16 training
SM80_16x16x32_S8S8S32S32    // INT8 inference
```
