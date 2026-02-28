# CuTe Learning Path - Complete Reference Guide

## Overview

This guide provides a comprehensive overview of the complete CuTe learning path across all 6 modules, including learning objectives, key concepts, exercises, and additional resources.

---

## Module Summary

| Module | Topic | Key Focus | Exercises | Reading |
|--------|-------|-----------|-----------|---------|
| 01 | Layout Algebra | Memory organization | 10 exercises | READING.md |
| 02 | CuTe Tensors | Safe memory access | 9 exercises | READING.md |
| 03 | Tiled Copy | Efficient data movement | 9 exercises | READING.md |
| 04 | MMA Atoms | Tensor Core operations | 9 exercises | READING.md |
| 05 | Shared Memory | Bank conflict avoidance | 9 exercises | READING.md |
| 06 | Collective Mainloops | Complete kernel construction | 9 exercises | READING.md |

---

## Learning Path Checklist

### Module 01: Layout Algebra Fundamentals

**Learning Objectives:**
- [ ] Understand layout definition and purpose
- [ ] Create row-major and column-major layouts
- [ ] Design custom strides for padding
- [ ] Compose hierarchical layouts
- [ ] Transform layouts (transpose, reshape, partition)
- [ ] Debug layouts with `cute::print()`

**Core Exercises:**
- [ ] ex01_basic_layout.cu - Basic layout creation
- [ ] ex02_rowmajor_colmajor.cu - Memory ordering
- [ ] ex03_custom_strides.cu - Padding and custom strides
- [ ] ex04_layout_composition.cu - Hierarchical layouts
- [ ] ex05_offset_mapping_challenge.cu - Offset calculations
- [ ] ex06_hierarchical_layouts.cu - Thread hierarchy
- [ ] ex07_layout_transformation.cu - Layout transformations
- [ ] ex08_debug_with_print.cu - Debugging techniques

**Advanced Exercises:**
- [ ] ex09_layout_arithmetic.cu - Advanced operations
- [ ] ex10_real_world_layouts.cu - Real-world patterns

**Key Concepts:**
- [ ] Shape and stride relationship
- [ ] Row-major vs column-major ordering
- [ ] Padding for bank conflict avoidance
- [ ] Layout composition and hierarchy
- [ ] Offset calculation formula
- [ ] Layout transformations

**Reading Materials:**
- [ ] Module 01 README.md
- [ ] Module 01 READING.md (comprehensive guide)
- [ ] Layout Algebra - Quick Reference Card

---

### Module 02: CuTe Tensors

**Learning Objectives:**
- [ ] Create tensors from raw pointers
- [ ] Access tensor elements using coordinates
- [ ] Slice tensors without copying data
- [ ] Create transposed and broadcast views
- [ ] Work with multi-dimensional tensors
- [ ] Understand memory space management

**Core Exercises:**
- [ ] ex01_tensor_creation.cu - Tensor creation
- [ ] ex02_tensor_access.cu - Access patterns
- [ ] ex03_tensor_slicing.cu - Slicing operations
- [ ] ex04_tensor_transpose.cu - Transpose views
- [ ] ex05_tensor_composition.cu - Layout composition
- [ ] ex06_multidim_tensors.cu - Multi-dimensional tensors
- [ ] ex07_tensor_memory_spaces.cu - Memory spaces
- [ ] ex08_tensor_broadcasting.cu - Broadcasting

**Advanced Exercises:**
- [ ] ex09_advanced_tensor_manipulations.cu - Advanced manipulations

**Key Concepts:**
- [ ] Tensor = Pointer + Layout
- [ ] Zero-copy views
- [ ] Memory space wrappers (gmem, smem, rmem)
- [ ] Broadcasting with stride 0
- [ ] Tensor slicing and reshaping
- [ ] Alignment requirements

**Reading Materials:**
- [ ] Module 02 README.md
- [ ] Module 02 READING.md (comprehensive guide)
- [ ] Tensor Algebra - Quick Reference Card

---

### Module 03: Tiled Copy

**Learning Objectives:**
- [ ] Understand tiled copy fundamentals
- [ ] Use vectorized loads (128-bit)
- [ ] Implement thread cooperation patterns
- [ ] Copy from global to shared memory
- [ ] Use CuTe copy atoms
- [ ] Implement async copy with cp.async

**Core Exercises:**
- [ ] ex01_tiled_copy_basics.cu - Tiled copy basics
- [ ] ex02_vectorized_loads.cu - Vectorized operations
- [ ] ex03_thread_cooperation.cu - Thread cooperation
- [ ] ex04_gmem_to_smem.cu - Global to shared copy
- [ ] ex05_copy_atom.cu - Copy atoms
- [ ] ex06_coalescing_strategies.cu - Coalescing
- [ ] ex07_matrix_transpose_copy.cu - Transpose copy
- [ ] ex08_async_copy.cu - Async copy operations

**Advanced Exercises:**
- [ ] ex09_advanced_tiled_copy.cu - Advanced patterns

**Key Concepts:**
- [ ] Copy atoms and TiledCopy abstraction
- [ ] Vectorized loads (float4, 128-bit)
- [ ] Coalesced memory access
- [ ] Async copy with cp.async
- [ ] Thread cooperation patterns
- [ ] Memory hierarchy utilization

**Reading Materials:**
- [ ] Module 03 README.md
- [ ] Module 03 READING.md (comprehensive guide)
- [ ] Tiled Copy - Quick Reference Card

---

### Module 04: MMA Atoms

**Learning Objectives:**
- [ ] Understand MMA operation fundamentals
- [ ] Configure MMA atoms for different precisions
- [ ] Map threads to Tensor Core operations
- [ ] Manage accumulator registers
- [ ] Use mixed precision operations
- [ ] Build complete GEMM with MMA

**Core Exercises:**
- [ ] ex01_mma_atom_basics.cu - MMA basics
- [ ] ex02_tensor_core_sim.cu - Tensor Core simulation
- [ ] ex03_thread_tensor_mapping.cu - Thread mapping
- [ ] ex04_accumulator_management.cu - Accumulator management
- [ ] ex05_mixed_precision_mma.cu - Mixed precision
- [ ] ex06_gemm_with_mma.cu - GEMM implementation
- [ ] ex07_mma_configurations.cu - MMA configurations
- [ ] ex08_warp_level_mma.cu - Warp-level MMA

**Advanced Exercises:**
- [ ] ex09_advanced_mma_configurations.cu - Advanced configurations

**Key Concepts:**
- [ ] MMA operation: D = A × B + C
- [ ] Tensor Core architecture (sm_80, sm_89)
- [ ] Thread-to-Tensor-Core mapping
- [ ] Register allocation and occupancy
- [ ] Mixed precision (FP16, BF16, INT8)
- [ ] Multi-step accumulation

**Reading Materials:**
- [ ] Module 04 README.md
- [ ] Module 04 READING.md (comprehensive guide)
- [ ] MMA Atoms - Quick Reference Card

---

### Module 05: Shared Memory & Swizzling

**Learning Objectives:**
- [ ] Understand shared memory architecture
- [ ] Analyze bank conflicts
- [ ] Use padding for conflict avoidance
- [ ] Implement XOR-based swizzling
- [ ] Design GEMM shared memory layouts
- [ ] Create bank conflict-free transpose

**Core Exercises:**
- [ ] ex01_shared_memory_basics.cu - Shared memory basics
- [ ] ex02_bank_conflict_analysis.cu - Conflict analysis
- [ ] ex03_padding_conflict_avoidance.cu - Padding
- [ ] ex04_swizzling_fundamentals.cu - Swizzling basics
- [ ] ex05_xor_swizzling.cu - XOR swizzling
- [ ] ex06_smem_layouts_gemm.cu - GEMM layouts
- [ ] ex07_swizzle_pattern_design.cu - Pattern design
- [ ] ex08_conflict_free_transpose.cu - Transpose

**Advanced Exercises:**
- [ ] ex09_advanced_swizzling.cu - Advanced optimization

**Key Concepts:**
- [ ] Shared memory bank structure (32 banks)
- [ ] Bank conflict analysis and severity
- [ ] Padding vs swizzling trade-offs
- [ ] XOR-based address remapping
- [ ] GEMM shared memory layout design
- [ ] Conflict-free transpose patterns

**Reading Materials:**
- [ ] Module 05 README.md
- [ ] Module 05 READING.md (comprehensive guide)
- [ ] Shared Memory - Quick Reference Card

---

### Module 06: Collective Mainloops

**Learning Objectives:**
- [ ] Understand producer-consumer pipelines
- [ ] Implement collective copy operations
- [ ] Design multi-stage pipelines
- [ ] Coordinate thread block cooperation
- [ ] Schedule GEMM mainloops
- [ ] Implement double buffering
- [ ] Build complete GEMM kernels
- [ ] Profile and optimize performance

**Core Exercises:**
- [ ] ex01_producer_consumer.cu - Producer-consumer
- [ ] ex02_collective_copy.cu - Collective copy
- [ ] ex03_multi_stage_pipeline.cu - Multi-stage pipeline
- [ ] ex04_thread_block_cooperation.cu - Block cooperation
- [ ] ex05_mainloop_scheduling.cu - Mainloop scheduling
- [ ] ex06_double_buffering.cu - Double buffering
- [ ] ex07_complete_gemm_mainloop.cu - Complete GEMM
- [ ] ex08_performance_optimization.cu - Optimization

**Advanced Exercises:**
- [ ] ex09_complete_gemm_implementation.cu - Full implementation

**Key Concepts:**
- [ ] Producer-consumer pipeline pattern
- [ ] Collective operations and cooperation
- [ ] Multi-stage pipeline design (2, 3, 4 stages)
- [ ] Thread block organization
- [ ] Mainloop scheduling and unrolling
- [ ] Double and multi-buffering
- [ ] Performance profiling and tuning

**Reading Materials:**
- [ ] Module 06 README.md
- [ ] Module 06 READING.md (comprehensive guide)
- [ ] Collective Mainloops - Quick Reference Card

---

## Cross-Module Concepts

### Memory Hierarchy

| Level | Module | Concept |
|-------|--------|---------|
| Layout | 01 | Logical to physical mapping |
| Tensor | 02 | Pointer + Layout abstraction |
| Copy | 03 | Data movement between levels |
| MMA | 04 | Register-level computation |
| Shared | 05 | On-chip memory optimization |
| Mainloop | 06 | Complete hierarchy utilization |

### Performance Optimization Flow

```
Module 01: Efficient layout design
       ↓
Module 02: Optimal tensor access
       ↓
Module 03: Vectorized, coalesced copies
       ↓
Module 04: Tensor Core utilization
       ↓
Module 05: Bank conflict avoidance
       ↓
Module 06: Pipeline optimization
```

### Thread Hierarchy Across Modules

```
Grid (Module 06)
    └── Thread Block (Module 06)
            ├── Shared Memory (Module 05)
            │       └── Tiled Copy (Module 03)
            │
            ├── Warp (Module 04)
            │       └── MMA Atom (Module 04)
            │
            └── Thread (Module 01, 02)
                    ├── Layout (Module 01)
                    └── Tensor (Module 02)
```

---

## Quick Reference Cards

### Layout Creation (Module 01)

```cpp
// Row-major layout
auto rm = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});

// Column-major layout
auto cm = make_layout(make_shape(Int<M>{}, Int<N>{}), GenColMajor{});

// Padded layout
auto padded = make_layout(make_shape(Int<M>{}, Int<N>{}),
                          make_stride(Int<P>{}, Int<1>{}));

// Debug
print(layout);
print_layout(layout);
```

### Tensor Operations (Module 02)

```cpp
// Create tensor
auto tensor = make_tensor(ptr, layout);

// Access element
float val = tensor(row, col);

// Create view
auto view = make_tensor(ptr + offset, new_layout);

// Memory spaces
make_gmem_ptr(p);   // Global
make_smem_ptr(p);   // Shared
make_rmem_ptr(p);   // Register
```

### Tiled Copy (Module 03)

```cpp
// Vectorized load
float4 val = reinterpret_cast<float4*>(&ptr[idx])[0];

// Async copy
cp.async.ca.shared.global [smem], [gmem], bytes;
cp_async_fence();
cp_async_wait<0>();
```

### MMA Operation (Module 04)

```cpp
// MMA atom
using MMA = MMA_Atom<SM80_16x16x16, F16, F16, F32, F32>;

// Execute MMA
mma_sync(accum, a_frag, b_frag);

// Thread mapping
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;
```

### Shared Memory (Module 05)

```cpp
// Padded shared memory
__shared__ float smem[32][33];

// XOR swizzle
int swizzle(int addr) {
    return addr ^ (addr >> 5);
}

// Bank calculation
int bank = (address / 4) % 32;
```

### Pipeline Pattern (Module 06)

```cpp
// 2-stage pipeline
load(0);
for (k = 1; k < K; ++k) {
    compute(k-1);
    load(k);
}
compute(K-1);

// Kernel launch
kernel<<<grid, block, smem_size>>>(...);
```

---

## Additional Resources

### Official Documentation
- [CuTe GitHub Repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/tree/main/docs)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Tutorials and Examples
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Profiling Tools
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [CUDA Profiling Tools](https://developer.nvidia.com/cuda-profiling-tools)

### Books and Papers
- "Programming Massively Parallel Processors" - Hwu & Kirk
- "CUDA by Example" - Sanders & Kandrot
- "Efficient Matrix Multiplication on GPUs" - Various papers

---

## Completion Certificate

Congratulations on completing the CuTe Learning Path!

**You have mastered:**
1. ✅ Layout Algebra - Memory organization
2. ✅ CuTe Tensors - Safe memory access
3. ✅ Tiled Copy - Efficient data movement
4. ✅ MMA Atoms - Tensor Core operations
5. ✅ Shared Memory - Bank conflict avoidance
6. ✅ Collective Mainloops - Complete kernel construction

**You are now ready to:**
- Read and understand CUTLASS 3.x code
- Write custom CuTe-based kernels
- Optimize GEMM and related operations
- Contribute to high-performance CUDA projects

---

## Next Steps

### Continue Learning
1. Study advanced CUTLASS examples
2. Explore Hopper architecture features (sm_90)
3. Learn about TMA (Tensor Memory Accelerator)
4. Study sparse MMA operations

### Practice Projects
1. Implement optimized GEMM kernel
2. Build convolution kernel using GEMM
3. Create attention kernel for Transformers
4. Optimize existing CUDA kernels with CuTe

### Contribute
1. Report issues in CUTLASS repository
2. Contribute examples or documentation
3. Share optimizations with the community
4. Help others learn CuTe

---

## Study Tips

1. **Start with Module 01**: Layout algebra is foundational
2. **Complete all exercises**: Hands-on practice is essential
3. **Use the reading materials**: Deep dive into concepts
4. **Experiment**: Modify parameters and observe effects
5. **Profile your code**: Use Nsight Compute for insights
6. **Join the community**: Discuss with other learners
7. **Build incrementally**: Master each module before moving on
8. **Reference the quick cards**: Use for quick lookups

Good luck on your CuTe learning journey!
