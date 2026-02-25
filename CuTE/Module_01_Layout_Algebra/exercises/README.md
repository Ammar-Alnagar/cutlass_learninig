# Module 01: Layout Algebra - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe Layout Algebra concepts. Each exercise focuses on a specific aspect of layout creation and manipulation.

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

You can build specific exercises:

```bash
make ex01_basic_layout
make ex02_rowmajor_colmajor
make ex03_custom_strides
# ... and so on
```

## Exercises

### Exercise 01: Basic Layout Creation
**File:** `ex01_basic_layout.cu`

Learn to create fundamental CuTe layouts:
- 1D, 2D, and 3D layouts
- Using `make_layout`, `make_shape`, `make_stride`
- Row-major and column-major generators

**Concepts:** Layout creation, shape, stride

---

### Exercise 02: Row-Major vs Column-Major
**File:** `ex02_rowmajor_colmajor.cu`

Understand the difference between row-major and column-major layouts:
- Create both layout types
- Compare offset mappings
- Identify coalesced access patterns

**Concepts:** Memory ordering, coalesced access

---

### Exercise 03: Custom Strides and Padding
**File:** `ex03_custom_strides.cu`

Learn to create layouts with custom strides:
- Padded layouts for shared memory
- Bank conflict avoidance
- Memory overhead calculation

**Concepts:** Padding, bank conflicts, custom strides

---

### Exercise 04: Layout Composition
**File:** `ex04_layout_composition.cu`

Master hierarchical layout composition:
- Tile and element layouts
- Thread-to-data mapping
- Tiled algorithm organization

**Concepts:** Composition, tiling, hierarchy

---

### Exercise 05: Offset Mapping Challenge
**File:** `ex05_offset_mapping_challenge.cu`

Practice calculating memory offsets:
- Predict offsets before running
- Verify with CuTe calculations
- Understand the offset formula

**Concepts:** Coordinate-to-offset mapping, formula verification

---

### Exercise 06: Hierarchical Layouts
**File:** `ex06_hierarchical_layouts.cu`

Organize thread blocks and warps:
- Block -> Warp -> Thread hierarchy
- 2D thread block layouts
- Matrix multiplication hierarchy

**Concepts:** Thread hierarchy, warps, scalability

---

### Exercise 07: Layout Transformation
**File:** `ex07_layout_transformation.cu`

Transform layouts using various operations:
- Transpose operations
- Reshape (2D to 1D)
- Partition layouts

**Concepts:** Transformation, transpose, reshape, partition

---

### Exercise 08: Debug Layouts with cute::print
**File:** `ex08_debug_with_print.cu`

Master debugging with `cute::print()`:
- Visualize layout structures
- Interpret print output
- Debug common mistakes

**Concepts:** Debugging, visualization, common pitfalls

---

## Learning Path

1. **Start with Exercise 01** - Basic layout creation
2. **Exercise 02** - Understand memory ordering
3. **Exercise 03** - Learn about padding
4. **Exercise 04** - Composition concepts
5. **Exercise 05** - Practice offset calculations
6. **Exercise 06** - Hierarchical organization
7. **Exercise 07** - Transform layouts
8. **Exercise 08** - Debug effectively

## Tips for Success

1. **Run each exercise** and study the output
2. **Complete the TODO sections** in the code
3. **Modify parameters** (sizes, strides) to see different behaviors
4. **Use cute::print()** liberally for debugging
5. **Draw diagrams** to visualize layouts

## Common Patterns

```cpp
// Row-major layout
auto rm = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});

// Column-major layout
auto cm = make_layout(make_shape(Int<M>{}, Int<N>{}), GenColMajor{});

// Padded layout
auto padded = make_layout(make_shape(Int<M>{}, Int<N>{}), 
                          make_stride(Int<P>{}, Int<1>{}));

// 3D layout
auto layout3d = make_layout(make_shape(Int<A>{}, Int<B>{}, Int<C>{}), 
                            GenRowMajor{});
```

## Next Steps

After completing these exercises:
1. Move to Module 02: CuTe Tensors
2. Apply layout knowledge to tensor operations
3. Build more complex kernels

## Additional Resources

- Module 01 README.md - Concept overview
- `layout_study.cu` - Reference implementation
- CuTe documentation - https://github.com/NVIDIA/cutlass
