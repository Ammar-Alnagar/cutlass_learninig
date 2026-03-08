# Module 06 — Pipeline: Double-Buffer and Warp Specialization

## Concept Overview

Pipelines overlap data movement with compute to hide memory latency. CuTe provides pipeline primitives for double-buffering, async copy overlap, and warp-specialized execution (DMA warps load, MMA warps compute).

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| `PipelineAsync` | `cutlass.pipeline.PipelineAsync` |
| `PipelineTmaAsync` (SM90+) | `cutlass.pipeline.PipelineTmaAsync` |
| `barrier.arrive_and_wait()` | `pipeline.sync()` |
| `fence_view_async_mbarrier()` | `cute.fence()` |

### Pipeline Types

- **PipelineAsync**: Software-managed async pipeline (Ampere+)
- **PipelineTmaAsync**: TMA-integrated pipeline (Hopper+)
- **PipelineSimple**: Basic double-buffer

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | Double-buffer setup | [MEDIUM] | Foundation for all pipelines |
| 02 | Async copy overlap | [HARD] | **Critical for peak performance** |
| 03 | Warp-specialized pipeline | [HARD] | FlashAttention-3 pattern |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- Pipeline design is a **senior-level interview topic**
- Warp specialization demonstrates deep GPU architecture knowledge

### FlashAttention / vLLM / TensorRT-LLM
- FA3 uses warp specialization for 1.5× speedup over FA2
- All production kernels use some form of pipelining

---

**Next:** Open `ex01_double_buffer_FILL_IN.py`
