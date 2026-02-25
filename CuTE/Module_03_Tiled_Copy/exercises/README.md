# Module 03: Tiled Copy - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe Tiled Copy concepts. Tiled copy enables efficient data movement through thread cooperation and vectorized operations.

## Building the Exercises

### Prerequisites
- CUDA Toolkit with sm_89 support (or modify CMakeLists.txt for your architecture)
- CUTLASS library with CuTe headers

### Build Instructions

```bash
cd exercises
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Exercises

### Exercise 01: Tiled Copy Basics
**File:** `ex01_tiled_copy_basics.cu`

Learn the fundamentals:
- What is tiled copy
- Thread division of work
- Tile-based organization

**Concepts:** Tiling, work distribution, parallel copy

---

### Exercise 02: Vectorized Loads
**File:** `ex02_vectorized_loads.cu`

Use vectorized memory operations:
- 128-bit loads (4 floats)
- Alignment requirements
- Bandwidth improvement

**Concepts:** Vectorization, alignment, bandwidth

---

### Exercise 03: Thread Cooperation
**File:** `ex03_thread_cooperation.cu`

Understand thread collaboration:
- Work division among threads
- Thread indexing
- Block configuration

**Concepts:** Cooperation, indexing, parallelism

---

### Exercise 04: Global to Shared Memory Copy
**File:** `ex04_gmem_to_smem.cu`

Master gmem -> smem transfers:
- Tiled loading patterns
- Coalesced access
- Shared memory usage

**Concepts:** Memory hierarchy, tiling, coalescing

---

### Exercise 05: Copy Atom and Tiled Copy
**File:** `ex05_copy_atom.cu`

Learn CuTe's copy atom abstraction:
- What is a copy atom
- Thread organization
- Instruction selection

**Concepts:** Atoms, abstraction, portability

---

### Exercise 06: Coalescing Strategies
**File:** `ex06_coalescing_strategies.cu`

Optimize memory access patterns:
- Coalesced vs uncoalesced
- Layout impact
- Best practices

**Concepts:** Coalescing, optimization, bandwidth

---

### Exercise 07: Matrix Transpose Copy
**File:** `ex07_matrix_transpose_copy.cu`

Implement efficient transpose:
- Tiled transpose algorithm
- Coalesced read/write
- Shared memory optimization

**Concepts:** Transpose, tiling, optimization

---

### Exercise 08: Async Copy with cp.async
**File:** `ex08_async_copy.cu`

Use asynchronous copy operations:
- cp.async instruction
- Pipeline patterns
- Overlap compute/transfer

**Concepts:** Async, pipelining, overlap

---

## Learning Path

1. **Exercise 01** - Tiled copy basics
2. **Exercise 02** - Vectorized loads
3. **Exercise 03** - Thread cooperation
4. **Exercise 04** - gmem to smem
5. **Exercise 05** - Copy atoms
6. **Exercise 06** - Coalescing strategies
7. **Exercise 07** - Matrix transpose
8. **Exercise 08** - Async copy

## Key Concepts Summary

### Tiled Copy Benefits
- Better memory coalescing
- Enables vectorized loads
- Thread cooperation
- Overlap with computation

### Memory Access Patterns
| Pattern | Row-Major | Column-Major |
|---------|-----------|--------------|
| Row Access | Coalesced | Uncoalesced |
| Column Access | Uncoalesced | Coalesced |

### Async Copy Pipeline
```
Time 0: [LOAD 0]
Time 1: [COMPUTE 0] [LOAD 1]
Time 2: [COMPUTE 1] [LOAD 2]
Time 3: [COMPUTE 2] [LOAD 3]
Time 4: [COMPUTE 3]
```

## Common Patterns

```cpp
// Basic tiled copy pattern
__global__ void tiled_copy(float* src, float* dst, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}

// Vectorized load pattern
float4 val = reinterpret_cast<float4*>(&src[idx])[0];
reinterpret_cast<float4*>(&dst[idx])[0] = val;

// Async copy pattern (sm_80+)
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2;"
    : : "r"(smem_off), "l"(gmem_ptr), "r"(bytes)
);
```

## Tips for Success

1. **Match layout to access pattern** for coalescing
2. **Use vectorized loads** when possible (4x bandwidth)
3. **Tile your data** for shared memory reuse
4. **Overlap copy with compute** using async operations

## Next Steps

After completing these exercises:
1. Move to Module 04: MMA Atoms
2. Learn Tensor Core operations
3. Study matrix multiplication

## Additional Resources

- Module 03 README.md - Concept overview
- `tiled_copy_basics.cu` - Reference implementation
- CuTe documentation - https://github.com/NVIDIA/cutlass
