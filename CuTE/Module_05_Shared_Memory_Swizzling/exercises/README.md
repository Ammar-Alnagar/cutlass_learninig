# Module 05: Shared Memory Swizzling - Exercises

## Overview
This directory contains hands-on exercises to practice Shared Memory and Swizzling concepts. Learn to avoid bank conflicts and optimize shared memory access patterns.

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

### Exercise 01: Shared Memory Basics
**File:** `ex01_shared_memory_basics.cu`

Learn shared memory fundamentals:
- Characteristics and benefits
- Bank structure (32 banks)
- Common use cases

**Concepts:** On-chip memory, banks, latency

---

### Exercise 02: Bank Conflict Analysis
**File:** `ex02_bank_conflict_analysis.cu`

Identify and analyze bank conflicts:
- Conflict causes
- Access pattern analysis
- Severity calculation

**Concepts:** Conflicts, serialization, analysis

---

### Exercise 03: Padding for Conflict Avoidance
**File:** `ex03_padding_conflict_avoidance.cu`

Use padding to avoid conflicts:
- How padding works
- Calculate requirements
- Memory overhead trade-off

**Concepts:** Padding, stride, overhead

---

### Exercise 04: Swizzling Fundamentals
**File:** `ex04_swizzling_fundamentals.cu`

Understand swizzling concepts:
- XOR-based transformation
- Address remapping
- Compare with padding

**Concepts:** Swizzling, XOR, remapping

---

### Exercise 05: XOR-Based Swizzling
**File:** `ex05_xor_swizzling.cu`

Master XOR swizzling patterns:
- XOR properties
- Common patterns
- Multi-bit swizzling

**Concepts:** XOR, bit manipulation, reversibility

---

### Exercise 06: Shared Memory Layouts for GEMM
**File:** `ex06_smem_layouts_gemm.cu`

Design GEMM shared memory layouts:
- Tile A and B requirements
- Access pattern optimization
- Padding vs swizzling

**Concepts:** GEMM, tiles, optimization

---

### Exercise 07: Swizzle Pattern Design
**File:** `ex07_swizzle_pattern_design.cu`

Design custom swizzle patterns:
- Pattern analysis
- Bit selection
- Verification

**Concepts:** Design, analysis, verification

---

### Exercise 08: Bank Conflict-Free Matrix Transpose
**File:** `ex08_conflict_free_transpose.cu`

Implement conflict-free transpose:
- Transpose access patterns
- Padded transpose
- Swizzled transpose

**Concepts:** Transpose, padding, swizzling

---

## Learning Path

1. **Exercise 01** - Shared memory basics
2. **Exercise 02** - Bank conflict analysis
3. **Exercise 03** - Padding technique
4. **Exercise 04** - Swizzling fundamentals
5. **Exercise 05** - XOR swizzling
6. **Exercise 06** - GEMM layouts
7. **Exercise 07** - Pattern design
8. **Exercise 08** - Conflict-free transpose

## Bank Conflict Summary

### Conflict Scenarios

| Access Pattern | Stride | Conflict | Solution |
|----------------|--------|----------|----------|
| Row access | 1 | None | None needed |
| Column access | 32 | 32-way | Padding/Swizzle |
| Diagonal access | 33 | None | None needed |
| Transpose write | 32 | 32-way | Padding/Swizzle |

### Padding Options

| Matrix Size | Padded Stride | Overhead |
|-------------|---------------|----------|
| 32×32 | 33 | 3.1% |
| 64×64 | 65 | 1.6% |
| 128×128 | 129 | 0.8% |

## Key Formulas

### Bank Calculation
```
bank = (address / 4) % 32  // For 32-bit words
```

### XOR Swizzle
```
swizzled_addr = addr XOR (addr >> shift)
Common: shift = 5 for 32 banks
```

### Padding Overhead
```
overhead % = (padded_elements - original_elements) / original_elements × 100
```

## Common Patterns

```cpp
// Padded shared memory
__shared__ float smem[32][33];  // +1 padding

// XOR swizzle function
__device__ __forceinline__ int swizzle(int addr) {
    return addr ^ (addr >> 5);
}

// In kernel:
int logical_addr = threadIdx.y * 32 + threadIdx.x;
int physical_addr = swizzle(logical_addr);
smem[physical_addr / 32][physical_addr % 32] = value;
```

## Tips for Success

1. **Analyze access patterns first** - Understand before optimizing
2. **Padding is simpler** - Use when memory allows
3. **Swizzling has no overhead** - Use when memory is tight
4. **Verify with analysis** - Always check bank distribution
5. **Consider both phases** - Load and compute must both be optimized

## Next Steps

After completing these exercises:
1. Move to Module 06: Collective Mainloops
2. Learn producer-consumer pipelines
3. Study complete kernel integration

## Additional Resources

- Module 05 README.md - Concept overview
- `shared_memory_layouts.cu` - Reference implementation
- CUDA Programming Guide - Shared Memory chapter
- CUTLASS documentation - https://github.com/NVIDIA/cutlass
