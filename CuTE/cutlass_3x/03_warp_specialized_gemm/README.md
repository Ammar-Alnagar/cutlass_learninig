# Module 03 — Warp-Specialized GEMM

## Overview

Warp specialization is Hopper's (SM90) most important architectural feature for kernel optimization. It divides warps within a block into specialized roles:

- **Producer warps**: Handle TMA loads (memory operations)
- **Consumer warps**: Handle MMA compute (Tensor Core)

This enables true async memory/compute overlap at the warp level.

## Why Warp Specialization Matters

Traditional GEMM (all warps do everything):
```
Warp 0: Load → Compute → Store
Warp 1: Load → Compute → Store
...
```

Warp-specialized GEMM:
```
Producer Warps: Load → Load → Load → ...
Consumer Warps: Compute → Compute → Compute → ...
```

**Result:** Eliminated load/store interference, maximized Tensor Core utilization.

## FA3 Architecture

Flash Attention 3 (FA3) is built on warp specialization:

```
┌─────────────────────────────────────────────────────────────┐
│  SM90 Warp Block (128 warps typical)                        │
├─────────────────────────────────────────────────────────────┤
│  Producer Warps (16-24 warps)                               │
│    ├─ TMA Load Q tiles                                      │
│    ├─ TMA Load K tiles                                      │
│    └─ TMA Load V tiles                                      │
├─────────────────────────────────────────────────────────────┤
│  Consumer Warps (remaining warps)                           │
│    ├─ QK^T MMA                                              │
│    ├─ Softmax (in registers)                                │
│    └─ PV MMA                                                │
└─────────────────────────────────────────────────────────────┘
```

## Warp Split Configuration

Typical H100 configuration (128 warps/SM):

| Role | Warp Count | Responsibility |
|------|------------|----------------|
| Producer | 4-8 | TMA loads for A, B matrices |
| Consumer | 28-32 | MMA compute, epilogue |
| Coordination | 2-4 | Semaphore management |

The CollectiveBuilder auto-selects optimal split, but you can override:

```cpp
using WarpSpecialization = cutlass::gemm::collective::WarpSpecializedPolicy
  <cutlass::gemm::collective::ProducerWarpCount<4>,
   cutlass::gemm::collective::ConsumerWarpCount<28>>;
```

## Pipeline Stages

Warp specialization requires multi-stage pipelining:

```
Stage 0: Load tile 0 → Compute tile 0
Stage 1: Load tile 1 → Compute tile 1
...
Stage N: Load tile N → Compute tile N
```

More stages = more latency hiding, but higher shared memory usage.

H100 shared memory: 230 KB per SM
- Each stage: ~16-32 KB (depends on tile size)
- Optimal: 5-8 stages for typical GEMM

## Profiling Warp Specialization

```bash
# Verify producer/consumer overlap
ncu --metrics smsp__thread_inst_executed_per_pipe_tensor.ratio,\
          l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    ./hopper_ws_gemm

# Key metrics:
# - High tensor instruction ratio = good compute utilization
# - High load bandwidth = producer warps keeping up
```

## Production Use Cases

| Use Case | Warp Split | Speedup |
|----------|------------|---------|
| Dense GEMM (H100) | 4 prod / 28 cons | 1.5-2.0× |
| FA3 Attention | 8 prod / 24 cons | 1.8-2.5× |
| MoE Grouped GEMM | 6 prod / 26 cons | 1.6-2.2× |

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_hopper_ws_FILL_IN.cu` | Basic warp-specialized GEMM | Medium |
| `ex02_pingpong_FILL_IN.cu` | Ping-pong pipeline | Hard |
| `ex03_ws_attention_FILL_IN.cu` | Warp-spec applied to attention | Hard |
