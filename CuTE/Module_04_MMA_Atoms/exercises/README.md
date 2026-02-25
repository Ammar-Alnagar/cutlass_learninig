# Module 04: MMA Atoms - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe MMA (Matrix Multiply-Accumulate) Atom concepts. MMA atoms are the fundamental building blocks for Tensor Core operations.

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

### Exercise 01: MMA Atom Basics
**File:** `ex01_mma_atom_basics.cu`

Learn MMA fundamentals:
- D = A * B + C operation
- Tensor Core concepts
- Small matrix multiplication

**Concepts:** MMA operation, Tensor Cores, warp-level

---

### Exercise 02: Tensor Core Operation Simulation
**File:** `ex02_tensor_core_sim.cu`

Simulate Tensor Core operations:
- Throughput comparison
- Performance benefits
- Operation counting

**Concepts:** Simulation, throughput, acceleration

---

### Exercise 03: Thread to Tensor Core Mapping
**File:** `ex03_thread_tensor_mapping.cu`

Understand thread organization:
- Warp-level organization
- Thread roles in MMA
- Operand loading

**Concepts:** Warp, thread mapping, cooperation

---

### Exercise 04: Accumulator Management
**File:** `ex04_accumulator_management.cu`

Manage accumulators:
- Register allocation
- Multi-step accumulation
- Precision handling

**Concepts:** Accumulators, registers, K-reduction

---

### Exercise 05: Mixed Precision MMA
**File:** `ex05_mixed_precision_mma.cu`

Use mixed precision:
- FP16 inputs, FP32 accumulation
- BF16, INT8 configurations
- Precision selection

**Concepts:** Mixed precision, FP16, BF16, INT8

---

### Exercise 06: GEMM with MMA Atoms
**File:** `ex06_gemm_with_mma.cu`

Build complete GEMM:
- Multi-level tiling
- K-dimension reduction
- Full GEMM structure

**Concepts:** GEMM, tiling, reduction

---

### Exercise 07: MMA Atom Configurations
**File:** `ex07_mma_configurations.cu`

Explore configurations:
- Naming conventions
- Architecture support
- Configuration selection

**Concepts:** Configurations, architectures, selection

---

### Exercise 08: Warp-Level Matrix Multiply
**File:** `ex08_warp_level_mma.cu`

Master warp-level operations:
- Warp primitives
- Warp assignment
- Warp synchronization

**Concepts:** Warp-level, primitives, assignment

---

## Learning Path

1. **Exercise 01** - MMA basics
2. **Exercise 02** - Tensor Core simulation
3. **Exercise 03** - Thread mapping
4. **Exercise 04** - Accumulator management
5. **Exercise 05** - Mixed precision
6. **Exercise 06** - Complete GEMM
7. **Exercise 07** - Configurations
8. **Exercise 08** - Warp-level MMA

## MMA Configurations Summary

### Common sm_80 Configurations

| Configuration | M×N×K | Types | Use Case |
|---------------|-------|-------|----------|
| SM80_16x8x16 | 16×8×16 | F32/F16/F16/F32 | General GEMM |
| SM80_16x8x32 | 16×8×32 | F32/F16/F16/F32 | K-parallel |
| SM80_8x8x4 | 8×8×4 | F32/FP64/FP64/FP64 | Scientific |
| SM80_16x8x32 | 16×8×32 | S32/S8/S8/S32 | INT8 Inference |
| SM80_16x8x8 | 16×8×8 | F32/BF16/BF16/F32 | ML Training |

### Architecture Support

| Arch | GPU | FP16 | INT8 | FP64 |
|------|-----|------|------|------|
| sm_70 | V100 | ✓ | ✓ | ✗ |
| sm_75 | T4 | ✓ | ✓ | ✗ |
| sm_80 | A100 | ✓ | ✓ | ✓ |
| sm_86 | A10 | ✓ | ✓ | ✓ |
| sm_89 | H100 | ✓ | ✓ | ✓ |

## Key Formulas

### GEMM Complexity
```
Total Operations = M × N × K
MMA Operations = (M/16) × (N/16) × (K/16)  [for 16×16×16 MMA]
```

### Throughput Calculation
```
Peak TFLOPS = Ops/clock × Warps/SM × SMs × Frequency
A100 FP16 = 512 × 8 × 108 × 1.4 GHz = 312 TFLOPS
```

### Register Allocation
```
Registers/thread = (Accum + Operand A + Operand B) / Threads
For 16×16×16: (256 + 128 + 128) / 32 = 16 registers/thread
```

## Common Patterns

```cpp
// Warp-level MMA instruction
asm volatile(
    "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32"
    "{%0, %1, ...}, {%2, ...}, {%3, ...}, {%4, ...};"
    : "+f"(accum[0]) : "r"(a_frag[0]), "r"(b_frag[0]), "f"(accum[0])
);

// Multi-step accumulation
for (int k_tile = 0; k_tile < K / 16; ++k_tile) {
    load_operands(frag_a, frag_b, k_tile);
    mma_sync(accum, frag_a, frag_b);
}
store_results(accum, C);
```

## Tips for Success

1. **Understand warp organization** - 32 threads cooperate
2. **Manage registers carefully** - Limited resource
3. **Use mixed precision** - FP16 inputs, FP32 accumulation
4. **Tile appropriately** - Match MMA atom size
5. **Profile configurations** - Different configs for different sizes

## Next Steps

After completing these exercises:
1. Move to Module 05: Shared Memory Swizzling
2. Learn bank conflict avoidance
3. Study swizzling techniques

## Additional Resources

- Module 04 README.md - Concept overview
- `mma_atom_basics.cu` - Reference implementation
- CUTLASS documentation - https://github.com/NVIDIA/cutlass
- Tensor Core Programming Guide - NVIDIA Developer
