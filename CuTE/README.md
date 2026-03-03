# CuTe Learning Directory — LLM Inference Kernel Engineering

## Purpose

This is a **hands-on, code-first** CuTe curriculum for building production LLM inference kernels. You will write FlashAttention-2, tiled GEMM, and quantized kernels from scratch in CuTe/CUTLASS 3.x.

**Target roles:** NVIDIA (Inference Kernel Engineer), Cerebras (LLM Inference Performance), Modular (AI Kernel Engineer).

**Outcome:** A public GitHub portfolio with 24 runnable exercises + 2 complete projects demonstrating CuTe fluency.

---

## Prerequisites

You already have:
- Expert C++ (memory, templates, performance)
- Expert CUDA (kernels, shared memory, Nsight profiling, roofline analysis)
- RTX 4060 (sm_89, Ada Lovelace) + Blackwell access
- CUDA 12.x + CUTLASS 3.x installed

You do **not** need:
- Linear algebra review
- Basic CUDA tutorials
- General GPU programming intro

---

## Directory Structure

```
CuTe/
├── README.md                          ← This file
├── Module_01_Layouts/
│   ├── README.md                      ← Layout algebra, tiling, GQA patterns
│   └── exercises/
│       ├── ex01_basic_layouts.cu      ← make_layout, shape, stride, print_layout
│       ├── ex02_tiling_with_logical_divide.cu  ← logical_divide for FlashAttention tiling
│       ├── ex03_attention_tensor_layouts.cu    ← 4D [batch, heads, seqlen, head_dim]
│       └── ex04_gqa_stride_zero.cu    ← Stride-0 broadcast for GQA
├── Module_02_Tensors/
│   ├── README.md                      ← Tensor creation, slicing, local_tile
│   └── exercises/
│       ├── ex01_tensor_creation.cu    ← make_tensor, make_gmem_ptr, make_smem_ptr
│       ├── ex02_slicing_views.cu      ← Underscore slicing for per-head access
│       ├── ex03_local_tile.cu         ← Block iteration (FlashAttention outer loop)
│       └── ex04_local_partition.cu    ← Thread/warp distribution
├── Module_03_TiledCopy/
│   ├── README.md                      ← Vectorized loads, cp.async pipeline
│   └── exercises/
│       ├── ex01_basic_copy.cu         ← make_tiled_copy, Copy atoms
│       ├── ex02_vectorized_128bit.cu  ← float4 loads, bandwidth measurement
│       ├── ex03_gmem_to_smem.cu       ← K/V tile storage, smem sizing
│       └── ex04_async_copy_pipeline.cu← cp.async, cp_async_fence, cp_async_wait
├── Module_04_TiledMMA/
│   ├── README.md                      ← Tensor Core MMA, mixed precision
│   └── exercises/
│       ├── ex01_mma_atom_setup.cu     ← MMA_Atom for sm_89
│       ├── ex02_tiled_gemm.cu         ← gemm() call, QK^T pattern
│       ├── ex03_fragment_layout.cu    ← partition_fragment_A/B/C
│       └── ex04_mixed_precision_fp16.cu ← FP16 inputs, FP32 accumulator
├── Module_05_Swizzle/
│   ├── README.md                      ← Bank-conflict-free smem
│   └── exercises/
│       ├── ex01_bank_conflicts_demo.cu← Visualize 32-way conflicts
│       ├── ex02_apply_swizzle.cu      ← Swizzle<2,3,3> parameters
│       └── ex03_verify_with_nsight.cu ← ncu profiling workflow
├── Module_06_Pipeline/
│   ├── README.md                      ← Double buffering, load/compute overlap
│   └── exercises/
│       ├── ex01_double_buffer.cu      ← 2-stage pipeline, ping-pong buffers
│       ├── ex02_pipelined_gemm.cu     ← QK^T with overlapped K/V load
│       └── ex03_async_mma_overlap.cu  ← cp_async_wait<1>, production pattern
└── Projects/
    ├── 01_tiled_gemm/
    │   ├── README.md                  ← Specification, roofline analysis
    │   └── gemm.cu                    ← Complete tiled GEMM (64x64 tiles)
    └── 02_flash_attention_prefill/
        ├── README.md                  ← Capstone specification
        └── flash_attention.cu         ← Full FlashAttention-2 prefill
```

---

## Learning Path

| Module | Topic | Exercises | Key Concepts |
|--------|-------|-----------|--------------|
| 01 | Layouts | 4 | `make_layout`, `logical_divide`, GQA stride-0 |
| 02 | Tensors | 4 | `make_tensor`, slicing, `local_tile`, `local_partition` |
| 03 | TiledCopy | 4 | Vectorized loads, gmem→smem, `cp.async` |
| 04 | TiledMMA | 4 | MMA atoms, `gemm()`, fragment layout, FP16/FP32 |
| 05 | Swizzle | 3 | Bank conflicts, `Swizzle<2,3,3>`, Nsight verification |
| 06 | Pipeline | 3 | Double buffering, prologue/mainloop/epilogue |
| **Projects** | **2** | **2** | **Tiled GEMM, FlashAttention-2** |

**Total: 32 files** (6 module READMEs + 24 exercises + 2 project READMEs + 2 project kernels)

---

## How to Use This Directory

1. **Work sequentially.** Each module builds on the previous.
2. **Run every exercise.** All files compile and produce output.
3. **Answer CHECKPOINT questions** before moving to the next exercise.
4. **Profile with Nsight.** Each exercise includes `ncu` commands.
5. **Build the projects.** The capstone is FlashAttention-2 prefill.

---

## Environment Setup

```bash
# Verify CUDA
nvcc --version

# Verify CUTLASS 3.x is available (needed for CuTe headers)
# CuTe headers are in: <cutlass/include/cute>

# Compile Module 01, Exercise 01
nvcc -std=c++17 -I/path/to/cutlass/include \
     -arch=sm_89 -O3 \
     Module_01_Layouts/exercises/ex01_basic_layouts.cu -o ex01 && ./ex01

# Compile Project 01 (Tiled GEMM)
nvcc -std=c++17 -I/path/to/cutlass/include \
     -arch=sm_89 -O3 \
     Projects/01_tiled_gemm/gemm.cu -o gemm && ./gemm

# Compile Project 02 (FlashAttention-2)
nvcc -std=c++17 -I/path/to/cutlass/include \
     -arch=sm_89 -O3 \
     Projects/02_flash_attention_prefill/flash_attention.cu -o flash_attention && ./flash_attention
```

---

## Job Mapping

Every exercise maps to a specific job requirement:

| Job | Requirement | Modules |
|-----|-------------|---------|
| NVIDIA DL Software Engineer (Inference) | CuTe kernels, FlashAttention, TiledMMA, TiledCopy | 03, 04, 06, Projects |
| NVIDIA DL Software Engineer (Model Optimization) | Kernel fusion, INT8/FP8 GEMM, TRT-LLM | 04, 05, 06 |
| Modular AI Kernel Engineer | High-performance attention/GEMM, Tensor Cores | 04, 05, 06, Projects |
| Cerebras LLM Inference Performance | FlashAttention variants, profiling-driven optimization | 03, 05, 06, Projects |
| Cerebras Inference ML Runtime | Latency/throughput optimization, vLLM/SGLang | 06, Projects |

---

## Exercise Format

Every exercise follows this structure:

```cpp
/*
 * WHAT THIS TEACHES:              ← 3-5 lines, plain English
 * WHY THIS MATTERS FOR LLM INFERENCE:  ← Production kernel mapping
 * MENTAL MODEL:                 ← Concept explanation before code
 */

// ... code with inline comments explaining WHY, not just what ...

// PREDICT BEFORE RUNNING:       ← Questions to answer before execution
// Q1: ...
// Q2: ...

// CHECKPOINT:                   ← Questions to answer before next exercise
// Q1: ...
// Q2: ...
```

---

## Exit Criteria (Full Directory)

Before considering this complete, you must be able to:

1. **Write a tiled GEMM** in CuTe with swizzled smem and double buffering
2. **Write FlashAttention-2 prefill** from scratch (no CUTLASS templates)
3. **Profile with Nsight** and identify bottlenecks (compute vs. memory bound)
4. **Explain warp-level fragment layout** for QK^T and PV GEMMs
5. **Achieve >70% of roofline bandwidth** on your tiled GEMM
6. **Achieve >50% of peak TFLOPS** on FlashAttention-2

---

## Performance Targets

| Kernel | Metric | Target |
|--------|--------|--------|
| Tiled GEMM (Project 01) | TFLOPS efficiency | >70% of peak |
| Tiled GEMM (Project 01) | Bank conflicts | 0 (verified with Nsight) |
| FlashAttention-2 (Project 02) | Bandwidth efficiency | >70% of peak |
| FlashAttention-2 (Project 02) | Numerical error | <1% vs. CPU reference |
| FlashAttention-2 (Project 02) | Speedup over naive | 2-10x |

---

## Nsight Profiling Commands

Each exercise includes profiling commands. Key metrics:

```bash
# Bank conflicts (Module 05)
ncu --metrics smem__conflict_requests.sum ./ex02_apply_swizzle

# Tensor Core usage (Module 04)
ncu --metrics smsp__inst_executed_op_tensor.sum ./ex02_tiled_gemm

# Memory throughput (Module 03)
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./ex02_vectorized_128bit

# Full analysis (Projects)
ncu --set full --section SpeedOfLight ./gemm
```

---

## Quick Start

```bash
# Start with Module 01
cd /home/ammar/work/AI-Kernel-learning/CuTE/Module_01_Layouts

# Read the module overview
cat README.md

# Compile and run first exercise
nvcc -std=c++17 -I/path/to/cutlass/include -arch=sm_89 -O3 \
     exercises/ex01_basic_layouts.cu -o ex01 && ./ex01

# Answer CHECKPOINT questions, then move to ex02
```

---

## What You'll Build

By completing this directory, you will have:

1. **24 working CuTe exercises** — each demonstrating a specific concept
2. **Tiled GEMM kernel** — production-quality, benchmarked, profiled
3. **FlashAttention-2 prefill** — complete capstone with causal masking
4. **Nsight profiling screenshots** — evidence of optimization skills
5. **GitHub portfolio** — ready to share with NVIDIA/Cerebras/Modular recruiters

---

## Next Steps

Start with **Module 01: Layouts**. It teaches the layout algebra you need to tile attention tensors and express GQA patterns.

```bash
cd Module_01_Layouts
cat README.md
nvcc -std=c++17 -I/path/to/cutlass/include -arch=sm_89 -O3 exercises/ex01_basic_layouts.cu -o ex01 && ./ex01
```
