# Module 02: KV Cache — The Memory Wall

**One concept:** KV cache eliminates redundant computation in autoregressive decode by caching key and value vectors from previous tokens. This is essential because without it, decode would recompute all previous tokens at every step.

**Job mapping:** Cerebras/NVIDIA inference engineer — you will implement KV cache management, including memory allocation, block tables (PagedAttention), and quantization.

---

## Files in This Module

Read in order:

1. **01_why_kv_cache.md** — The redundancy in autoregressive decode, why recomputing is wasteful.

2. **02_memory_formula.md** — Exact memory formula for KV cache, worked examples for LLaMA-3 8B/70B.

3. **03_prefill_vs_decode.md** — Different compute characteristics, why decode is always bandwidth-bound.

4. **kv_cache_sim.py** — Simulate prefill + decode with KV cache, print memory usage at each step.

---

## What You Must Be Able To Do After This Module

1. Compute exact KV cache memory for any model: $2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes}$

2. Explain why decode without KV cache is $O(S^2)$ in compute, but with KV cache is $O(S)$

3. Derive the arithmetic intensity of decode with and without KV cache

4. Implement KV cache allocation and indexing correctly

---

## Before Moving To Module 03

Run `python kv_cache_sim.py`. It must print `PASS`. If it prints `FAIL`, you do not understand the memory savings yet.

**Next:** `01_why_kv_cache.md` — the redundancy in autoregressive generation
