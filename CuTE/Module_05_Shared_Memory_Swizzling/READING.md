# Module 05: Shared Memory & Swizzling - Comprehensive Reading Materials

## Table of Contents
1. [Introduction to Shared Memory](#introduction-to-shared-memory)
2. [Shared Memory Bank Structure](#shared-memory-bank-structure)
3. [Bank Conflict Analysis](#bank-conflict-analysis)
4. [Padding for Conflict Avoidance](#padding-for-conflict-avoidance)
5. [Swizzling Fundamentals](#swizzling-fundamentals)
6. [XOR-Based Swizzling](#xor-based-swizzling)
7. [Shared Memory Layouts for GEMM](#shared-memory-layouts-for-gemm)
8. [Swizzle Pattern Design](#swizzle-pattern-design)
9. [Bank Conflict-Free Transpose](#bank-conflict-free-transpose)
10. [Advanced Optimization Techniques](#advanced-optimization-techniques)

---

## Introduction to Shared Memory

### What is Shared Memory?

**Shared memory** is on-chip memory shared by all threads in a thread block:

```
GPU Memory Hierarchy:
    ├── Global Memory (off-chip DRAM)
    │       └── High latency (400-800 cycles)
    │
    └── Per-SM Memory
            ├── L1 Cache / Shared Memory
            │       └── Low latency (~20 cycles)
            │
            └── Registers
                    └── Lowest latency (1 cycle)
```

### Shared Memory Characteristics

| Property | Value |
|----------|-------|
| Size per SM | 195 KB (A100) |
| Size per Block | Configurable (up to 195 KB) |
| Latency | ~20 cycles |
| Bandwidth | ~19 TB/s (on-chip) |
| Scope | Thread block |
| Persistence | Block lifetime |

### Shared Memory Use Cases

**1. Data Reuse:**
```cpp
// Load once from global, reuse multiple times
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Load from global
tile[threadIdx.y][threadIdx.x] = global[row][col];
__syncthreads();

// Reuse from shared (multiple times)
for (int k = 0; k < K; ++k) {
    result += tile[threadIdx.y][k] * other_tile[k][threadIdx.x];
}
```

**2. Communication:**
```cpp
// Thread-to-thread communication within block
__shared__ float data[256];

// Thread 0 writes
if (threadIdx.x == 0) {
    data[0] = computed_value;
}
__syncthreads();

// Other threads read
float val = data[0];
```

**3. Bank Conflict Avoidance:**
```cpp
// Use padding or swizzling to avoid conflicts
__shared__ float smem[32][33];  // Padded
```

---

## Shared Memory Bank Structure

### Bank Organization

**A100 Shared Memory:**
- **32 banks** numbered 0-31
- **4 bytes per bank** per cycle
- **Consecutive addresses** map to consecutive banks

```
Address Mapping:
    Address 0-3   → Bank 0
    Address 4-7   → Bank 1
    Address 8-11  → Bank 2
    ...
    Address 124-127 → Bank 31
    Address 128-131 → Bank 0 (wrap around)
```

### Bank Calculation Formula

```cpp
// For 32-bit words (4 bytes)
int word_address = byte_address / 4;
int bank = word_address % 32;
```

### Bank Access Patterns

**No Conflict (32 threads access different banks):**
```cpp
__shared__ float smem[32][32];

// Row access (consecutive threads → consecutive banks)
float val = smem[threadIdx.y][threadIdx.x];
// Thread 0 → Bank 0
// Thread 1 → Bank 1
// ...
// Thread 31 → Bank 31
// No conflict!
```

**32-Way Conflict (32 threads access same bank):**
```cpp
__shared__ float smem[32][32];

// Column access (all threads access Bank 0)
float val = smem[threadIdx.x][threadIdx.y];
// Thread 0 → smem[0][y] → Address 0*32+y → Bank (0*32+y)/4 % 32
// Thread 1 → smem[1][y] → Address 1*32+y → Bank (1*32+y)/4 % 32
// For y=0: All threads access Bank 0!
// 32-way conflict → 32× slower
```

---

## Bank Conflict Analysis

### Conflict Severity

| Conflict Type | Threads/Bank | Cycles | Slowdown |
|---------------|--------------|--------|----------|
| No conflict | 1 | 1 | 1× |
| 2-way | 2 | 2 | 2× |
| 4-way | 4 | 4 | 4× |
| ... | ... | ... | ... |
| 32-way | 32 | 32 | 32× |

### Analyzing Access Patterns

**Example 1: Row Access (No Conflict)**
```cpp
__shared__ float smem[32][32];

// Access pattern
for (int i = 0; i < 32; ++i) {
    int bank = (i * 32 + threadIdx.x) % 32;
    // bank = (i * 32 + threadIdx.x) % 32
    // For i=0: bank = threadIdx.x (0-31, all different)
    // No conflict!
}
```

**Example 2: Column Access (32-Way Conflict)**
```cpp
__shared__ float smem[32][32];

// Access pattern
for (int j = 0; j < 32; ++j) {
    int bank = (threadIdx.x * 32 + j) % 32;
    // bank = (threadIdx.x * 32 + j) % 32
    // For j=0: bank = (threadIdx.x * 32) % 32 = 0 for all threads!
    // 32-way conflict!
}
```

**Example 3: Diagonal Access (No Conflict)**
```cpp
__shared__ float smem[32][32];

// Diagonal access
float val = smem[threadIdx.x][(threadIdx.x + offset) % 32];
// Each thread accesses different row and column
// Banks are all different
// No conflict!
```

### Conflict Detection Tool

```cpp
__device__ void analyze_bank_conflicts(
    __shared__ float* smem,
    int rows, int cols
) {
    int banks[32] = {0};
    
    // Count accesses per bank
    for (int t = 0; t < 32; ++t) {
        int addr = /* compute address for thread t */;
        int bank = (addr / 4) % 32;
        banks[bank]++;
    }
    
    // Find max conflict
    int max_conflict = 0;
    for (int b = 0; b < 32; ++b) {
        if (banks[b] > max_conflict) {
            max_conflict = banks[b];
        }
    }
    
    printf("Max conflict: %d-way\n", max_conflict);
}
```

---

## Padding for Conflict Avoidance

### How Padding Works

**Add extra elements to change stride:**

```cpp
// Without padding (column conflict)
__shared__ float smem[32][32];
// Stride = 32 elements = 128 bytes = 32 banks
// Column access: all threads hit same bank

// With padding (no conflict)
__shared__ float smem[32][33];
// Stride = 33 elements = 132 bytes
// Column access: threads hit different banks
```

### Padding Calculation

**For 32 banks with 4-byte elements:**

```cpp
// Original stride
int original_stride = 32;  // elements

// Padded stride (avoid multiple of 32)
int padded_stride = 33;  // +1 padding

// Memory overhead
float overhead = (padded_stride - original_stride) / (float)original_stride;
// overhead = 1/32 = 3.125%
```

### Common Padding Patterns

| Matrix Size | Padded Stride | Overhead |
|-------------|---------------|----------|
| 32×32 | 33 | 3.1% |
| 64×64 | 65 | 1.6% |
| 128×128 | 129 | 0.8% |
| 256×256 | 257 | 0.4% |

### Padding in GEMM

```cpp
// Shared memory for matrix A tile
constexpr int TILE_M = 16;
constexpr int TILE_K = 16;
constexpr int PADDING = 1;

__shared__ float As[TILE_M][TILE_K + PADDING];

// Load from global
As[threadIdx.y][threadIdx.x] = A[row][col];
__syncthreads();

// Access without conflicts
float val = As[threadIdx.y][k];  // k can be 0-15
```

---

## Swizzling Fundamentals

### What is Swizzling?

**Swizzling** is address remapping using XOR operations to distribute accesses across banks without padding overhead.

### Padding vs Swizzling

| Aspect | Padding | Swizzling |
|--------|---------|-----------|
| Memory overhead | Yes (1-3%) | None |
| Complexity | Low | Medium |
| Flexibility | Fixed | Configurable |
| Hardware support | Manual | Built-in (sm_80+) |

### Basic Swizzling Concept

```
Logical Address → Swizzle Function → Physical Address
       ↓                    ↓               ↓
    Thread 0          XOR with         Bank 5
    Thread 1          address bits     Bank 12
    ...               ...              ...
```

### Swizzling Benefits

1. **No memory overhead**: Use full allocated memory
2. **Conflict-free**: Distribute accesses evenly
3. **Hardware support**: Dedicated instructions on sm_80+
4. **Flexible**: Configure for different patterns

---

## XOR-Based Swizzling

### XOR Swizzle Function

**Basic 5-bit XOR swizzle:**

```cpp
__device__ __forceinline__ int swizzle_address(int addr) {
    // XOR bit 4 with bit 0
    // This redistributes addresses across banks
    return addr ^ (addr >> 5);
}

// Inverse (for de-swizzling)
__device__ __forceinline__ int unswizzle_address(int swizzled) {
    return swizzled ^ (swizzled >> 5);
}
```

### XOR Properties

**XOR is its own inverse:**
```
swizzled = addr XOR mask
addr = swizzled XOR mask  // Same operation!
```

**Example:**
```
Address:     00101 (5)
Mask:        00001 (1)  // addr >> 5 for small addresses
Swizzled:    00100 (4)

Reverse:
Swizzled:    00100 (4)
Mask:        00001 (1)
Address:     00101 (5)  // Original recovered!
```

### Multi-Bit Swizzling

```cpp
// 2-bit swizzle (for finer distribution)
__device__ __forceinline__ int swizzle_2bit(int addr) {
    int swizzled = addr;
    swizzled ^= (addr >> 5);  // XOR bit 4 with bit 0
    swizzled ^= (addr >> 6);  // XOR bit 5 with bit 1
    return swizzled;
}

// Full 5-bit swizzle (for 32 banks)
__device__ __forceinline__ int swizzle_5bit(int addr) {
    int swizzled = addr;
    swizzled ^= (addr >> 5);
    swizzled ^= (addr >> 6);
    swizzled ^= (addr >> 7);
    swizzled ^= (addr >> 8);
    swizzled ^= (addr >> 9);
    return swizzled;
}
```

### CuTe Swizzle Layout

```cpp
#include <cute/layout.hpp>

using namespace cute;

// Create swizzled layout
auto swizzled_layout = make_layout(
    make_shape(Int<32>{}, Int<32>{}),
    make_stride(Int<33>{}, Int<1>{})  // XOR swizzle encoded in stride
);

// CuTe automatically applies swizzling for optimal bank distribution
```

---

## Shared Memory Layouts for GEMM

### GEMM Shared Memory Requirements

**For C = A × B:**
- **As**: M×K tile of A
- **Bs**: K×N tile of B
- Both need conflict-free access

### Optimal Layout Design

```cpp
// Matrix A tile (row-major access)
constexpr int TILE_M = 16;
constexpr int TILE_K = 16;

__shared__ float As[TILE_M][TILE_K + 1];  // +1 padding

// Matrix B tile (column-major access)
constexpr int TILE_N = 16;

__shared__ float Bs[TILE_K][TILE_N + 1];  // +1 padding

// Access patterns
// As: Row access (threadIdx.x varies) → coalesced, no conflict
// Bs: Column access (threadIdx.y varies) → coalesced, no conflict
```

### Swizzled Layout for GEMM

```cpp
// Without hardware swizzling
__shared__ float As[16][17];  // Padded

// With hardware swizzling (sm_80+)
__shared__ float As[16][16];  // No padding needed
// Apply swizzle in address calculation
int physical_addr = swizzle(logical_addr);
```

### Bank Distribution Analysis

```cpp
// Analyze bank distribution for GEMM shared memory
void analyze_gemm_banks() {
    cout << "Matrix A (16×16 with padding):" << endl;
    cout << "  Row access: 16 consecutive banks (no conflict)" << endl;
    cout << "  Column access: 16 different banks (no conflict)" << endl;
    
    cout << endl;
    
    cout << "Matrix B (16×16 with padding):" << endl;
    cout << "  Row access: 16 consecutive banks (no conflict)" << endl;
    cout << "  Column access: 16 different banks (no conflict)" << endl;
}
```

---

## Swizzle Pattern Design

### Designing Custom Swizzle Patterns

**Step 1: Analyze access pattern**
```cpp
// Identify which addresses are accessed together
for (int t = 0; t < 32; ++t) {
    int addr = compute_address(t);
    printf("Thread %d → Address %d\n", t, addr);
}
```

**Step 2: Choose swizzle bits**
```cpp
// For 32 banks, use 5-bit swizzle
int swizzle_bits = 5;
int mask = (1 << swizzle_bits) - 1;
```

**Step 3: Implement swizzle function**
```cpp
__device__ __forceinline__ int apply_swizzle(int addr, int shift) {
    return addr ^ (addr >> shift);
}
```

**Step 4: Verify distribution**
```cpp
int bank_counts[32] = {0};
for (int t = 0; t < 32; ++t) {
    int addr = compute_address(t);
    int swizzled = apply_swizzle(addr, 5);
    int bank = (swizzled / 4) % 32;
    bank_counts[bank]++;
}

// Check max conflict
int max_conflict = 0;
for (int b = 0; b < 32; ++b) {
    if (bank_counts[b] > max_conflict) {
        max_conflict = bank_counts[b];
    }
}
printf("Max conflict after swizzle: %d-way\n", max_conflict);
```

---

## Bank Conflict-Free Transpose

### Transpose Challenge

**Matrix transpose has inherent bank conflicts:**
```cpp
// Read row-major, write column-major
__global__ void transpose(float* src, float* dst, int M, int N) {
    __shared__ float smem[32][32];
    
    // Coalesced read (row-major)
    smem[threadIdx.y][threadIdx.x] = src[row * N + col];
    __syncthreads();
    
    // Uncoalesced write with conflicts (column-major)
    dst[col * M + row] = smem[threadIdx.x][threadIdx.y];
    // Column access → bank conflicts!
}
```

### Padded Transpose

```cpp
__global__ void padded_transpose(float* src, float* dst, int M, int N) {
    __shared__ float smem[32][33];  // +1 padding
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load (coalesced)
    smem[threadIdx.y][threadIdx.x] = src[row * N + col];
    __syncthreads();
    
    // Store (now coalesced due to padding)
    int dst_row = col;
    int dst_col = row;
    dst[dst_row * M + dst_col] = smem[threadIdx.x][threadIdx.y];
}
```

### Swizzled Transpose

```cpp
__global__ void swizzled_transpose(float* src, float* dst, int M, int N) {
    __shared__ float smem[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load with swizzled address
    int logical_addr = threadIdx.y * 32 + threadIdx.x;
    int physical_addr = swizzle(logical_addr);
    
    smem[physical_addr / 32][physical_addr % 32] = src[row * N + col];
    __syncthreads();
    
    // Store with inverse swizzle
    int transpose_addr = threadIdx.x * 32 + threadIdx.y;
    int swizzled_addr = swizzle(transpose_addr);
    
    dst[col * M + row] = smem[swizzled_addr / 32][swizzled_addr % 32];
}
```

---

## Advanced Optimization Techniques

### Combined Padding + Swizzling

```cpp
// For extreme cases, combine both techniques
__shared__ float smem[32][34];  // +2 padding

__device__ int get_physical_addr(int row, int col) {
    int logical_addr = row * 34 + col;
    int swizzled = logical_addr ^ (logical_addr >> 5);
    return swizzled;
}
```

### Dynamic Shared Memory

```cpp
// Configure shared memory at kernel launch
extern __shared__ float smem[];

// Use based on runtime parameters
void launch_kernel(int tile_size) {
    int smem_size = tile_size * tile_size * sizeof(float);
    kernel<<<grid, block, smem_size>>>(...);
}
```

### Shared Memory Prefetching

```cpp
// Prefetch next tile while computing current
__global__ void prefetch_kernel(float* A, float* B, float* C) {
    __shared__ float As[2][TILE_M][TILE_K];
    __shared__ float Bs[2][TILE_K][TILE_N];
    
    int stage = 0;
    
    for (int k = 0; k < num_tiles; ++k) {
        int next_stage = 1 - stage;
        
        // Prefetch next tile
        load_tile(A, As[next_stage], k + 1);
        load_tile(B, Bs[next_stage], k + 1);
        
        __syncthreads();
        
        // Compute with current tile
        compute(As[stage], Bs[stage], C);
        
        __syncthreads();
        stage = next_stage;
    }
}
```

---

## Further Reading

### Official Documentation
- [CUDA Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)
- [CuTe Layout Documentation](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [PTX ISA - Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Related Modules
- Module 01: Layout Algebra (layout design)
- Module 03: Tiled Copy (shared memory transfers)
- Module 06: Collective Mainloops (complete GEMM)

### Advanced Topics
- Hardware Swizzling (sm_80+)
- TMA (Tensor Memory Accelerator) on sm_90
- Multi-Stage Shared Memory Pipelines

---

## Quick Reference Card

### Bank Calculation
```cpp
int bank = (address / 4) % 32;  // For 32-bit words
```

### Padding Pattern
```cpp
__shared__ float smem[32][33];  // +1 to avoid 32 multiple
```

### XOR Swizzle
```cpp
int swizzle(int addr) {
    return addr ^ (addr >> 5);
}
```

### Conflict-Free Access
```cpp
// Good: Row access
smem[threadIdx.y][threadIdx.x];

// Bad: Column access (use padding/swizzle)
smem[threadIdx.x][threadIdx.y];  // Conflict!
```

### GEMM Shared Memory
```cpp
__shared__ float As[16][17];  // A tile with padding
__shared__ float Bs[17][16];  // B tile with padding
```
