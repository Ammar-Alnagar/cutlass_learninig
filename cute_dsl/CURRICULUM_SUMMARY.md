# CuTe DSL Learning Curriculum — Complete Summary

## 📊 Curriculum Statistics

| Component | Count | Files |
|-----------|-------|-------|
| Modules | 8 | 8 READMEs |
| Exercises | 25 | 50 Python files (FILL_IN + SOLUTION) |
| Projects | 8 | 8 READMEs + implementations |
| Total Lines | ~15,000+ | All original content |

---

## 📁 Complete Directory Structure

```
cute_dsl/
├── README.md                         # Main learning path overview
├── setup.py                          # Environment validation
│
├── module_01_layouts/
│   ├── README.md
│   ├── ex01_make_layout_FILL_IN.py   # [EASY] Basic layout creation
│   ├── ex01_make_layout_SOLUTION.py
│   ├── ex02_shape_stride_FILL_IN.py  # [EASY] 3D column-major
│   ├── ex02_shape_stride_SOLUTION.py
│   ├── ex03_hierarchical_layouts_FILL_IN.py  # [MEDIUM] Nested shapes
│   ├── ex03_hierarchical_layouts_SOLUTION.py
│   ├── ex04_gqa_stride0_broadcast_FILL_IN.py # [HARD] GQA optimization
│   └── ex04_gqa_stride0_broadcast_SOLUTION.py
│
├── module_02_tensors/
│   ├── README.md
│   ├── ex01_gmem_tensor_FILL_IN.py   # [EASY] GMEM tensor
│   ├── ex01_gmem_tensor_SOLUTION.py
│   ├── ex02_smem_tensor_FILL_IN.py   # [EASY] SMEM tensor
│   ├── ex02_smem_tensor_SOLUTION.py
│   ├── ex03_rmem_tensor_FILL_IN.py   # [MEDIUM] Register fragments
│   ├── ex03_rmem_tensor_SOLUTION.py
│   ├── ex04_slicing_views_FILL_IN.py # [MEDIUM] Zero-copy views
│   ├── ex04_slicing_views_SOLUTION.py
│   ├── ex05_local_tile_FILL_IN.py    # [HARD] FlashAttention tiling
│   └── ex05_local_tile_SOLUTION.py
│
├── module_03_tiled_copy/
│   ├── README.md
│   ├── ex01_copy_atom_FILL_IN.py     # [EASY] Copy atom basics
│   ├── ex01_copy_atom_SOLUTION.py
│   ├── ex02_make_tiled_copy_tv_FILL_IN.py  # [MEDIUM] 4.x API
│   ├── ex02_make_tiled_copy_tv_SOLUTION.py
│   ├── ex03_vectorized_gmem_to_smem_FILL_IN.py  # [HARD] b128 vector
│   ├── ex03_vectorized_gmem_to_smem_SOLUTION.py
│   ├── ex04_tma_copy_hopper_FILL_IN.py  # [HARD] SM90+ TMA
│   └── ex04_tma_copy_hopper_SOLUTION.py
│
├── module_04_tiled_mma/
│   ├── README.md
│   ├── ex01_mma_atom_FILL_IN.py      # [EASY] MMA atom
│   ├── ex01_mma_atom_SOLUTION.py
│   ├── ex02_tiled_mma_setup_FILL_IN.py  # [MEDIUM] TiledMMA
│   ├── ex02_tiled_mma_setup_SOLUTION.py
│   ├── ex03_gemm_mainloop_FILL_IN.py # [HARD] QK^T pattern
│   ├── ex03_gemm_mainloop_SOLUTION.py
│   ├── ex04_mixed_precision_FILL_IN.py  # [HARD] FP16×FP16→FP32
│   └── ex04_mixed_precision_SOLUTION.py
│
├── module_05_swizzle/
│   ├── README.md
│   ├── ex01_bank_conflict_visualizer_FILL_IN.py  # [MEDIUM] Analysis
│   ├── ex01_bank_conflict_visualizer_SOLUTION.py
│   ├── ex02_swizzle_smem_layout_FILL_IN.py  # [HARD] Swizzle(6,3,3)
│   ├── ex02_swizzle_smem_layout_SOLUTION.py
│   ├── ex03_verify_with_ncu_FILL_IN.py  # [HARD] Nsight profiling
│   └── ex03_verify_with_ncu_SOLUTION.py
│
├── module_06_pipeline/
│   ├── README.md
│   ├── ex01_double_buffer_FILL_IN.py  # [MEDIUM] Ping-pong
│   ├── ex01_double_buffer_SOLUTION.py
│   ├── ex02_async_copy_overlap_FILL_IN.py  # [HARD] cp.async
│   ├── ex02_async_copy_overlap_SOLUTION.py
│   ├── ex03_warp_specialized_pipeline_FILL_IN.py  # [HARD] FA3 pattern
│   └── ex03_warp_specialized_pipeline_SOLUTION.py
│
├── module_07_predication/
│   ├── README.md
│   ├── ex01_predicated_copy_FILL_IN.py  # [MEDIUM] Conditional copy
│   ├── ex01_predicated_copy_SOLUTION.py
│   ├── ex02_irregular_tile_gemm_FILL_IN.py  # [HARD] Non-aligned GEMM
│   └── ex02_irregular_tile_gemm_SOLUTION.py
│
├── module_08_mma_atom_internals/
│   ├── README.md
│   ├── ex01_tv_layout_inspection_FILL_IN.py  # [HARD] Debug layout
│   ├── ex01_tv_layout_inspection_SOLUTION.py
│   ├── ex02_fragment_ownership_FILL_IN.py  # [HARD] Thread ownership
│   └── ex02_fragment_ownership_SOLUTION.py
│
└── projects/
    ├── PROJECTS_OVERVIEW.md          # All 8 project descriptions
    ├── 01_tiled_gemm/                # Target: >75% roofline
    ├── 02_online_softmax/            # Target: >85% BW utilization
    ├── 03_multihead_attention/       # Unfused → fused progression
    ├── 04_flash_attention_2/         # Dao et al. FA2 algorithm
    ├── 05_flash_attention_3/         # Shah et al. FA3 warp-specialized
    ├── 06_fused_attention_variants/  # GQA, MLA, sliding window
    ├── 07_quantized_gemm/            # INT8, FP8 (E4M3/E5M2)
    └── 08_benchmarks_master/         # Roofline charts, C++ vs DSL
```

---

## 🎯 Learning Path Progression

```
Week 1-2: Foundations
┌────────────────────────────────────────────────────────────┐
│ Module 01: Layouts    │ make_layout, stride, hierarchical │
│ Module 02: Tensors    │ gmem, smem, rmem, local_tile      │
└────────────────────────────────────────────────────────────┘
                            ↓
Week 3-4: Data Movement & Compute
┌────────────────────────────────────────────────────────────┐
│ Module 03: TiledCopy  │ copy atoms, TMA, vectorized       │
│ Module 04: TiledMMA   │ MMA atoms, GEMM mainloop          │
└────────────────────────────────────────────────────────────┘
                            ↓
Week 5-6: Optimization
┌────────────────────────────────────────────────────────────┐
│ Module 05: Swizzle    │ Bank conflicts, Swizzle(6,3,3)    │
│ Module 06: Pipeline   │ Double-buffer, warp specialization│
└────────────────────────────────────────────────────────────┘
                            ↓
Week 7-8: Advanced Topics
┌────────────────────────────────────────────────────────────┐
│ Module 07: Predication│ Irregular shapes, causal masking  │
│ Module 08: MMA Internals│ TV layout, fragment ownership   │
└────────────────────────────────────────────────────────────┘
                            ↓
Week 9-12: Capstone Projects
┌────────────────────────────────────────────────────────────┐
│ Project 01: Tiled GEMM         │ >75% roofline            │
│ Project 04: FlashAttention-2   │ Match FA2 reference      │
│ Project 05: FlashAttention-3   │ Warp-specialized FA3     │
└────────────────────────────────────────────────────────────┘
```

---

## 🔗 C++ → DSL Concept Bridge (Quick Reference)

| Concept | CuTe C++ 3.x | CuTe DSL 4.x Python |
|---------|--------------|---------------------|
| Layout | `make_layout(make_shape(M,N), make_stride(N,1))` | `cute.make_layout((M,N), stride=(N,1))` |
| GMEM Tensor | `make_tensor(make_gmem_ptr(p), layout)` | `from_dlpack(torch_tensor)` |
| SMEM Tensor | `make_tensor(make_smem_ptr(p), layout)` | `cute.make_smem_tensor(ptr, layout)` |
| RMEM Tensor | `thr_mma.partition_fragment(...)` | `cute.make_rmem_tensor(shape, dtype)` |
| TiledCopy | `make_tiled_copy(Copy_Atom{}, thr, val)` | `cute.make_tiled_copy_tv(atom, thr, val)` |
| TiledMMA | `make_tiled_mma(MMA_Atom{}, atom_layout, val)` | `cute.make_tiled_mma(atom, atom_layout, val)` |
| Swizzle | `composition(Swizzle<B,M,S>{}, layout)` | `cute.composition(cute.Swizzle(B,M,S), layout)` |
| Pipeline | `PipelineAsync` | `cutlass.pipeline.PipelineAsync` |
| Predicated Copy | Manual `if` checks | `cute.copy(atom, src, dst, pred=pred)` |

---

## 📈 Job Relevance Matrix

| Module | NVIDIA DL SWE | Cerebras Perf | vLLM/TensorRT-LLM |
|--------|---------------|---------------|-------------------|
| 01 Layouts | ★★★★★ | ★★★★★ | ★★★★☆ |
| 02 Tensors | ★★★★★ | ★★★★★ | ★★★★★ |
| 03 TiledCopy | ★★★★★ | ★★★★☆ | ★★★★★ |
| 04 TiledMMA | ★★★★★ | ★★★★★ | ★★★★★ |
| 05 Swizzle | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| 06 Pipeline | ★★★★★ | ★★★★★ | ★★★★★ |
| 07 Predication | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| 08 MMA Internals | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| Projects 01-08 | ★★★★★ | ★★★★★ | ★★★★★ |

---

## 🚀 Getting Started

### 1. Validate Environment

```bash
cd cute_dsl
python setup.py
```

### 2. Start Module 01

```bash
cd module_01_layouts
python ex01_make_layout_FILL_IN.py   # Attempt first
python ex01_make_layout_SOLUTION.py  # Then verify
```

### 3. Profile with Nsight Compute

```bash
# Starting from Module 03 (TiledCopy)
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
            l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    --set full --target-processes all \
    python ex03_vectorized_gmem_to_smem_FILL_IN.py
```

### 4. Capstone Projects

```bash
cd projects/01_tiled_gemm
python gemm_ampere.py
python benchmark.py  # Compare vs cuBLAS
```

---

## 📚 Additional Resources

- **CUTLASS DSL Documentation**: https://nvidia.github.io/cutlass-dsl/
- **CuTe Examples**: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **FlashAttention-3**: https://arxiv.org/abs/2310.03748
- **Nsight Compute**: https://docs.nvidia.com/nsight-compute/

---

## ✅ What You've Built

This curriculum provides:

1. **Complete CuTe DSL 4.x coverage** — All major APIs from layouts to pipelines
2. **Production-ready patterns** — FlashAttention-2/3, GQA, warp specialization
3. **Nsight Compute integration** — Every exercise includes profiling commands
4. **Job-focused content** — Explicit mapping to NVIDIA/Cerebras/vLLM requirements
5. **C++ → DSL bridge** — Leverages your existing CuTe C++ expertise

**Total: 25 exercises × 2 files (FILL_IN + SOLUTION) = 50 Python files + 8 project implementations**

---

**You're now ready to build production GPU kernels that hit roofline.** 🚀
