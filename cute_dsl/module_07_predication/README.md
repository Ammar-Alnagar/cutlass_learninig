# Module 07 — Predication: Handling Irregular Shapes

## Concept Overview

Predication enables conditional execution of memory and compute operations. In CuTe DSL, predicates are passed as keyword arguments to `cute.copy()` and other operations. This is essential for handling non-power-of-2 shapes and boundary conditions.

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| Manual predicate logic | `cute.copy(atom, src, dst, pred=pred_tensor)` |
| `if (thread_idx < bound)` | Predicate tensor with boolean values |
| Boundary checks in loops | Predicated tiled operations |

### When Predication is Needed

- Sequence lengths not divisible by tile size
- Causal masking in attention
- Irregular batch sizes
- Dynamic shapes in production models

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | Predicated copy | [MEDIUM] | Foundation for irregular shapes |
| 02 | Irregular tile GEMM | [HARD] | **Production kernel requirement** |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- Handling irregular shapes is a **practical interview question**
- Shows understanding of real-world model deployment challenges

### FlashAttention / vLLM / TensorRT-LLM
- Real sequence lengths are rarely tile-aligned
- Causal masking requires predication
- Dynamic batching needs flexible predication

---

**Next:** Open `ex01_predicated_copy_FILL_IN.py`
