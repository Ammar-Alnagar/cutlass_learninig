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

### Running Exercises

```bash
./ex01_basic_layout
./ex02_rowmajor_colmajor
```

---

## Exercises

### Exercise 01: Basic Layout Creation
**File:** `ex01_basic_layout.cu`

**Learning Objectives:**
- Understand what a CuTe layout is
- Create 1D, 2D, and 3D layouts
- Use `make_layout`, `make_shape`, `make_stride`
- Understand row-major and column-major generators

**Step-by-Step Guidance:**
1. **Read the code comments** - Each task has detailed hints
2. **Task 1:** Create a simple 1D layout with 16 elements using `make_shape(Int<16>{})`
3. **Task 2:** Create a 2D row-major layout - notice how `GenRowMajor{}` auto-generates strides
4. **Task 3:** Create a 2D column-major layout using `GenColMajor{}`
5. **Task 4:** Create a 3D layout - think about what strides make sense

**Key Concepts to Understand:**
- Layout = Shape + Stride
- Shape defines dimensions (e.g., 8×4)
- Stride defines how to step through memory (e.g., row-major: (4,1))
- `print(layout)` shows the structure

**Common Pitfalls:**
- Forgetting that row-major means rightmost dimension has stride 1
- Confusing shape with stride
- Not using `Int<N>{}` for compile-time constants

**Expected Output:**
You should see layout structures showing shape and stride tuples.

**Verification:**
- For 2D row-major (8×4): stride should be (4, 1)
- For 2D column-major (8×4): stride should be (1, 8)
- For 3D row-major (4×4×2): stride should be (8, 2, 1)

**Extension Challenge:**
Try creating a layout with shape (2, 3, 4) and manually specify strides.

**Concepts:** Layout creation, shape, stride

---

### Exercise 02: Row-Major vs Column-Major
**File:** `ex02_rowmajor_colmajor.cu`

**Learning Objectives:**
- Understand the difference between row-major and column-major layouts
- Map coordinates to offsets and observe differences
- Identify which access pattern is coalesced for each layout
- Visualize memory layout in grid format

**Step-by-Step Guidance:**
1. **Create both layouts** - 4×4 row-major and column-major
2. **Print the structures** - Compare shape and stride
3. **Visualize as grids** - See how offsets are arranged
4. **Compare offset mappings** - Run through the comparison table
5. **Analyze coalesced access** - Understand which pattern is efficient

**Key Concepts to Understand:**
- **Row-Major:** Elements in same row are contiguous (stride: N, 1)
- **Column-Major:** Elements in same column are contiguous (stride: 1, M)
- **Coalesced Access:** Consecutive threads access consecutive memory addresses

**Visual Example:**
```
Row-Major 4×4:        Column-Major 4×4:
 0  1  2  3            0  4  8 12
 4  5  6  7            1  5  9 13
 8  9 10 11            2  6 10 14
12 13 14 15            3  7 11 15
```

**Common Pitfalls:**
- Assuming row-major is always better (depends on access pattern)
- Not considering how data is accessed when choosing layout

**Expected Output:**
Two different grid layouts showing how the same coordinates map to different offsets.

**Verification:**
- Row-major: layout(0,1) - layout(0,0) should equal 1
- Column-major: layout(1,0) - layout(0,0) should equal 1

**Extension Challenge:**
Predict layout(2,3) for both layouts before running, then verify.

**Concepts:** Memory ordering, coalesced access

---

### Exercise 03: Custom Strides and Padding
**File:** `ex03_custom_strides.cu`

**Learning Objectives:**
- Create layouts with custom strides
- Understand padding for bank conflict avoidance
- Calculate memory overhead of padding
- Create strided layouts for submatrix access

**Step-by-Step Guidance:**
1. **Task 1:** Create standard 8×8 layout - observe the stride pattern
2. **Task 2:** Create padded layout with stride 9 instead of 8
3. **Task 3:** Calculate memory overhead percentage
4. **Task 4:** Create strided layout for accessing every other row

**Key Concepts to Understand:**
- **Padding:** Adding extra elements to improve access patterns
- **Bank Conflicts:** Multiple threads accessing same memory bank (slow)
- **Custom Strides:** Manually specifying how to step through dimensions

**Why Padding Helps:**
```
Without padding (8×8):    With padding (8×9):
Bank: 0 1 2 3 4 5 6 7     Bank: 0 1 2 3 4 5 6 7
        8 9 0 1 2 3 4 5         8 9 0 1 2 3 4 5
Column access causes      Column access is now
bank conflicts!           conflict-free!
```

**Common Pitfalls:**
- Forgetting to account for padding in calculations
- Using too much padding (wasteful) or too little (still conflicts)

**Expected Output:**
Layout structures showing different strides and memory overhead calculation.

**Verification:**
- Standard 8×8: stride = (8, 1)
- Padded 8×8: stride = (9, 1)
- Overhead should be ~12.5% for stride 9

**Extension Challenge:**
Calculate optimal padding for a 32×32 matrix with 32 banks.

**Concepts:** Padding, bank conflicts, custom strides

---

### Exercise 04: Layout Composition
**File:** `ex04_layout_composition.cu`

**Learning Objectives:**
- Master hierarchical layout composition
- Understand tile and element layouts
- Learn thread-to-data mapping
- Organize tiled algorithms

**Step-by-Step Guidance:**
1. **Understand composition** - Layouts can contain other layouts
2. **Create tile layout** - Define how tiles are organized
3. **Create element layout** - Define elements within each tile
4. **Compose them** - Combine tile and element layouts

**Key Concepts to Understand:**
- **Composition:** Building complex layouts from simpler ones
- **Tiling:** Dividing work into manageable chunks
- **Hierarchy:** Block → Warp → Thread organization

**Common Pattern:**
```cpp
auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), ...);
auto element_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), ...);
auto composed = composition(tile_layout, element_layout);
```

**Expected Output:**
Hierarchical layout showing tile and element organization.

**Verification:**
- Total size should equal tile_size × element_size
- Each tile should contain the expected number of elements

**Extension Challenge:**
Create a 3-level hierarchy: Block → Warp → Thread.

**Concepts:** Composition, tiling, hierarchy

---

### Exercise 05: Offset Mapping Challenge
**File:** `ex05_offset_mapping_challenge.cu`

**Learning Objectives:**
- Practice calculating memory offsets manually
- Predict offsets before running code
- Verify with CuTe calculations
- Understand the offset formula deeply

**Step-by-Step Guidance:**
1. **Study the layout** - Note shape and stride
2. **Predict offsets** - Calculate manually before running
3. **Run and verify** - Compare your predictions
4. **Understand the formula:** offset = Σ(coord_i × stride_i)

**Key Formula:**
```
For layout with shape (S0, S1) and stride (D0, D1):
offset(i, j) = i × D0 + j × D1
```

**Practice Problems:**
- Layout (8, 4) with stride (4, 1): offset(3, 2) = ?
- Layout (4, 8) with stride (1, 4): offset(2, 5) = ?

**Expected Output:**
Offset calculations showing coordinate-to-memory mapping.

**Verification:**
Your manual calculations should match CuTe's output.

**Extension Challenge:**
Create a 3D layout and calculate offset(2, 3, 1) manually.

**Concepts:** Coordinate-to-offset mapping, formula verification

---

### Exercise 06: Hierarchical Layouts
**File:** `ex06_hierarchical_layouts.cu`

**Learning Objectives:**
- Organize thread blocks and warps hierarchically
- Create 2D thread block layouts
- Design matrix multiplication hierarchy
- Understand scalability patterns

**Step-by-Step Guidance:**
1. **Understand GPU hierarchy** - Grid → Block → Warp → Thread
2. **Create block layout** - How blocks are organized
3. **Create warp layout** - Warps within each block
4. **Create thread layout** - Threads within each warp

**Key Concepts:**
- **Block:** Independent thread groups
- **Warp:** 32 threads executing in lockstep
- **Thread:** Individual execution unit

**Common Pattern for GEMM:**
```
Grid: (M/tile_M, N/tile_N)
Block: (warps_per_block × 32 threads)
Each warp computes a tile of the output
```

**Expected Output:**
Hierarchical organization showing block/warp/thread relationships.

**Verification:**
- Total threads = blocks × warps_per_block × 32
- Each level should properly nest within parent

**Extension Challenge:**
Design hierarchy for a 256×256 matrix multiply with 16×16 tiles.

**Concepts:** Thread hierarchy, warps, scalability

---

### Exercise 07: Layout Transformation
**File:** `ex07_layout_transformation.cu`

**Learning Objectives:**
- Transform layouts using various operations
- Perform transpose operations
- Reshape layouts (2D to 1D)
- Partition layouts into sub-layouts

**Step-by-Step Guidance:**
1. **Transpose:** Swap dimensions of a layout
2. **Reshape:** Change shape while preserving size
3. **Partition:** Extract sub-layouts from larger layout

**Key Operations:**
- **Transpose:** (M, N) → (N, M)
- **Reshape:** (8, 4) → (32,) or (16, 2)
- **Partition:** Extract rows 4-7 from 8-row layout

**Common Transformations:**
```cpp
// Transpose
auto transposed = make_layout(shape(layout), stride(layout).transpose());

// Reshape (conceptually)
auto flat = make_layout(Int<32>{});  // Same size, different shape
```

**Expected Output:**
Original and transformed layouts showing the changes.

**Verification:**
- Transpose should swap shape and stride elements
- Reshape should preserve total size
- Partition should be valid sub-region

**Extension Challenge:**
Chain multiple transformations: transpose → reshape → partition.

**Concepts:** Transformation, transpose, reshape, partition

---

## Learning Path

1. **Start with Exercise 01** - Basic layout creation
2. **Exercise 02** - Understand memory ordering
3. **Exercise 03** - Learn about padding
4. **Exercise 04** - Composition concepts
5. **Exercise 05** - Practice offset calculations
6. **Exercise 06** - Hierarchical organization
7. **Exercise 07** - Transform layouts

## Tips for Success

1. **Run each exercise** and study the output
2. **Complete the TODO sections** in the code
3. **Modify parameters** (sizes, strides) to see different behaviors
4. **Use print()** liberally for debugging
5. **Draw diagrams** to visualize layouts
6. **Calculate offsets manually** before running to verify understanding

## Common Patterns Reference

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

// Custom stride layout
auto custom = make_layout(shape, make_stride(D0, D1, D2));
```

## Offset Calculation Formula

For a layout with shape `(S0, S1, ..., Sn)` and stride `(D0, D1, ..., Dn)`:

```
offset(c0, c1, ..., cn) = c0×D0 + c1×D1 + ... + cn×Dn
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
- CUTLASS examples - https://github.com/NVIDIA/cutlass/tree/master/examples
