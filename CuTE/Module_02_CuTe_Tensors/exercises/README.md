# Module 02: CuTe Tensors - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe Tensor concepts. Tensors wrap raw pointers with layouts to create safe, indexed views of memory.

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

### Building Individual Exercises

```bash
make ex01_tensor_creation
make ex02_tensor_access
make ex03_tensor_slicing
# ... and so on
```

## Exercises

### Exercise 01: Tensor Creation from Raw Pointers
**File:** `ex01_tensor_creation.cu`

Learn to create CuTe tensors:
- Wrap raw pointers with layouts
- Access elements using coordinates
- Work with different data types

**Concepts:** `make_tensor`, pointer wrappers, element access

---

### Exercise 02: Tensor Access Patterns
**File:** `ex02_tensor_access.cu`

Understand memory access patterns:
- Row-wise vs column-wise access
- Coalesced vs uncoalesced access
- Impact on memory throughput

**Concepts:** Coalescing, memory efficiency, access patterns

---

### Exercise 03: Tensor Slicing Operations
**File:** `ex03_tensor_slicing.cu`

Extract sub-tensors and views:
- Row and column slices
- Sub-matrix extraction
- Strided slices

**Concepts:** Slicing, views, no-copy operations

---

### Exercise 04: Tensor Transpose and View
**File:** `ex04_tensor_transpose.cu`

Create transposed views:
- Transpose without copying data
- Verify transpose relationships
- Double transpose

**Concepts:** Transpose, view operations, zero-copy

---

### Exercise 05: Tensor Composition with Layouts
**File:** `ex05_tensor_composition.cu`

Compose tensors hierarchically:
- Tile and element layouts
- Multi-level organization
- Tiled algorithms

**Concepts:** Composition, tiling, hierarchy

---

### Exercise 06: Multi-dimensional Tensors
**File:** `ex06_multidim_tensors.cu`

Work with 3D and 4D tensors:
- 3D tensors for volumes
- 4D tensors for batches (NCHW)
- Stride calculation

**Concepts:** Multi-dimensional, NCHW/NHWC, stride calculation

---

### Exercise 07: Tensor Memory Spaces
**File:** `ex07_tensor_memory_spaces.cu`

Understand CUDA memory spaces:
- Global memory (gmem)
- Shared memory (smem)
- Register memory (rmem)

**Concepts:** Memory hierarchy, pointer wrappers, data movement

---

### Exercise 08: Tensor Broadcasting
**File:** `ex08_tensor_broadcasting.cu`

Broadcast tensors for operations:
- Scalar to vector
- Vector to matrix
- Bias addition

**Concepts:** Broadcasting, stride 0, efficient reuse

---

## Learning Path

1. **Exercise 01** - Create basic tensors
2. **Exercise 02** - Access patterns matter
3. **Exercise 03** - Slice tensors
4. **Exercise 04** - Transpose views
5. **Exercise 05** - Compose layouts
6. **Exercise 06** - Multi-dimensional
7. **Exercise 07** - Memory spaces
8. **Exercise 08** - Broadcasting

## Common Patterns

```cpp
// Create a tensor from raw pointer
float* data = ...;
auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
auto tensor = make_tensor(make_gmem_ptr(data), layout);

// Access elements
float val = tensor(i, j);

// Create shared memory tensor
extern __shared__ float smem[];
auto smem_tensor = make_tensor(make_smem_ptr(smem), layout);

// Create register tensor
float rmem[16];
auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), layout);

// Broadcast layout (stride = 0)
auto broadcast = make_layout(shape, make_stride(Int<0>{}, Int<1>{}));
```

## Memory Space Comparison

| Property    | Global    | Shared    | Register  |
|-------------|-----------|-----------|-----------|
| Size        | GBs       | KBs/MBs   | KBs       |
| Latency     | High      | Low       | Lowest    |
| Scope       | All       | Block     | Thread    |
| Persistence | Kernel    | Block     | Thread    |

## Tips for Success

1. **Match layout to access pattern** for coalescing
2. **Use views** instead of copying when possible
3. **Understand memory hierarchy** for optimization
4. **Broadcast** to avoid data duplication

## Next Steps

After completing these exercises:
1. Move to Module 03: Tiled Copy
2. Learn cooperative thread operations
3. Study vectorized memory transfers

## Additional Resources

- Module 02 README.md - Concept overview
- `tensor_basics.cu` - Reference implementation
- CuTe documentation - https://github.com/NVIDIA/cutlass
