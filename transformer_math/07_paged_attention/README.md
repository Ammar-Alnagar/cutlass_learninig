# Module 07: PagedAttention — vLLM's Innovation

**One concept:** PagedAttention solves KV cache memory fragmentation by allocating non-contiguous blocks, like OS virtual memory. This enables 2-4x larger batch sizes.

**Job mapping:** Cerebras runtime engineer — you will implement block table management and non-contiguous gather kernels.

---

## Files in This Module

1. **01_memory_fragmentation.md** — Why naive KV cache wastes 60-80% memory.

2. **02_block_tables.md** — Logical→physical block mapping, the page table.

3. **03_kernel_implications.md** — Non-contiguous gather, ComposedLayout connection to CuTe.

4. **paged_attention_sim.py** — Simulate block table, show memory savings vs. naive.

---

## What You Must Be Able To Do After This Module

1. Compute memory savings from PagedAttention vs. naive allocation

2. Implement block table indexing for non-contiguous KV access

3. Explain the connection between PagedAttention and OS virtual memory

---

**Next:** `01_memory_fragmentation.md` — the fragmentation problem
