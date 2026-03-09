# CUTLASS 3.x Kernel Engineering Curriculum

Production-grade GEMM and attention kernel engineering using NVIDIA's CUTLASS 3.x framework.

## Prerequisites

This curriculum assumes **mastery** of:
- CuTe C++ primitives (layouts, tensors, TiledCopy, TiledMMA, swizzle, pipelines)
- CUDA C++ expertise (shared memory, warp primitives, async copy)
- ThunderKittens experience
- Transformer math (attention, softmax, GQA, MLA, quantization)

**This is NOT a CuTe tutorial.** CuTe is the foundation — CUTLASS 3.x is the production framework built on top of it.

## CUTLASS 3.x Layer Map

```
┌─────────────────────────────────────────────────────────────┐
│  CollectiveBuilder                                          │
│  ── Auto-selects optimal kernel configuration per arch      │
├─────────────────────────────────────────────────────────────┤
│  GemmUniversalAdapter                                       │
│  ── Launch wrapper, grid/tile configuration                 │
├─────────────────────────────────────────────────────────────┤
│  CollectiveMma           │  CollectiveEpilogue             │
│  ── Mainloop: data move  │  ── Output: EVT fusion,         │
│     + compute            │     quantization, bias          │
├──────────────────────────┴─────────────────────────────────┤
│  CuTe C++ primitives (foundation — assumed mastered)       │
│  TiledMMA │ TiledCopy │ Pipeline │ Swizzle │ MMA Atom      │
└─────────────────────────────────────────────────────────────┘
```

## CuTe → CUTLASS 3.x Concept Bridge

| CuTe Primitive | CUTLASS 3.x Abstraction |
|----------------|-------------------------|
| `TiledMMA` (manual setup) | `CollectiveMma` (auto-configured) |
| `TiledCopy` (manual) | `CollectiveEpilogue` / TMA loader |
| Manual pipeline wiring | `PipelineAsync` / `PipelineTmaAsync` |
| Manual swizzle layout | `SmemLayout` (auto-selected by Builder) |
| Manual grid config | `GemmUniversalAdapter` |
| Custom epilogue math | **EVT** (Epilogue Visitor Tree) |
| Manual MMA atom | `CollectiveBuilder<SM80/SM90/SM100>` |

## Architecture Targets

| Arch | SM | Key Features |
|------|-----|--------------|
| Ampere | SM80 | Tensor Core, async copy |
| Hopper | SM90 | TMA, warp-specialization, clusters |
| Blackwell | SM100 | Persistent kernels, FP8 native |

## Curriculum Structure

### Modules

| Module | Topic | Key Skill |
|--------|-------|-----------|
| `01_collective_builder` | CollectiveBuilder anatomy | Auto-configured GEMM |
| `02_epilogue_visitor_tree` | EVT fusion | Fused activation + quant |
| `03_warp_specialized_gemm` | Producer/consumer warp split | FA3 architecture |
| `04_streamk` | StreamK decomposition | Wave quantization fix |
| `05_grouped_gemm` | Variable-batch GEMM | MoE expert routing |
| `06_mixed_precision` | TF32/FP16/BF16/FP8/INT8 | Quantized inference |
| `07_sparse_gemm` | 2:4 structured sparsity | SpGEMM |
| `08_kernel_fusion` | LayerNorm, RoPE, softmax | End-to-end fusion |

### Projects

| Project | Target | Benchmark |
|---------|--------|-----------|
| `01_production_gemm` | >90% cuBLAS on Ampere/Hopper | TFLOPS + roofline |
| `02_fused_attention` | FA2/FA3 in CUTLASS | vs FA2/FA3 reference |
| `03_moe_inference` | 2× tokens/sec vs naive | FP8 quantized MoE |
| `04_quantized_inference` | FP8 >1.5× over FP16 | Accuracy within 0.5% |
| `05_benchmarks_master` | CUTLASS vs TK vs cuBLAS | Interview artifact |

## Build Instructions

```bash
mkdir build && cd build
cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc)
```

## Profiling Commands

```bash
# CollectiveBuilder GEMM — verify Tensor Core utilization
ncu --metrics sm__inst_executed_pipe_tensor.sum,\
          l2tex__t_bytes.sum,\
          smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct \
    ./gemm_basic

# EVT fusion — verify fusion happened (fewer global stores)
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
          smsp__inst_executed.sum \
    ./epilogue_relu

# Warp-specialized — verify DMA/MMA warp overlap
ncu --metrics smsp__thread_inst_executed_per_pipe_tensor.ratio,\
          l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    ./hopper_ws_gemm

# Full profile (always target this)
ncu --set full --target-processes all [binary]
```

## Performance Targets

| Kernel | Target | Metric |
|--------|--------|--------|
| Production GEMM (Ampere) | >90% cuBLAS | TFLOPS |
| Production GEMM (Hopper) | TMA + warp-spec advantage | % theoretical peak |
| FA2 CUTLASS | Within 10% of FA2 reference | Tokens/sec |
| FA3 warp-specialized | Producer/consumer split | Overlap efficiency |
| MoE grouped GEMM | 2× tokens/sec vs naive | Expert routing |
| FP8 linear | >1.5× over FP16 on H100 | Accuracy within 0.5% |

## Interview Artifacts

This curriculum produces GitHub-ready artifacts:
- Roofline charts for every kernel
- Side-by-side: CUTLASS 3.x vs ThunderKittens vs cuBLAS
- Warp-specialization producer/consumer analysis
- EVT fusion verification via Nsight Compute metrics

**Key interview story:** *"I matched cuBLAS on H100 using CollectiveBuilder — here's what the builder auto-selected and why it was optimal. I implemented warp-specialized attention following the FA3 architecture with explicit producer/consumer warp split."*

## References

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass/tree/main/examples/)
- [CUTLASS 3.x Documentation](https://nvidia.github.io/cutlass/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
