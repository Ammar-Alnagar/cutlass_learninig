# Module 05: FlashAttention — The IO-Aware Attention Algorithm

**One concept:** FlashAttention computes attention tile-by-tile in SRAM, never materializing the $O(S^2)$ intermediate matrix in HBM. This reduces HBM traffic from $O(S^2)$ to $O(S)$ and shifts attention from memory-bound to compute-bound.

**Job mapping:** NVIDIA inference kernel engineer — this is the exact algorithm you implement in your CuTe FlashAttention kernel. The online softmax rescaling formula is the hardest math in this directory.

---

## Files in This Module

Read in order:

1. **01_the_io_problem.md** — Why naive attention is IO-bound, HBM traffic analysis, the $O(S^2)$ wall.

2. **02_tiling_insight.md** — SRAM tiling, block-serial computation, the block-parallel algorithm structure.

3. **03_online_softmax.md** — Running max $m_i$, running sum $l_i$, the rescaling formula. **Spend extra time here.**

4. **04_fa2_improvements.md** — Work partitioning, parallelism over $S$, reduced non-matmul FLOPs.

5. **flash_attention.py** — Tile-by-tile numpy implementation with intermediate prints. Maps line-for-line to your CuTe kernel.

---

## What You Must Be Able To Do After This Module

1. Derive the online softmax rescaling formula from first principles

2. Write the FlashAttention tile loop pseudocode from memory

3. Compute HBM traffic for naive vs. FlashAttention: $O(BHS^2)$ vs. $O(BHSd_h)$

4. Explain why FlashAttention is exact (no approximation) — same output as naive attention

---

## Before Moving To Module 10

Run `python flash_attention.py`. It must print `PASS`. Trace through the online softmax updates and verify the rescaling formula.

**Next:** `01_the_io_problem.md` — why naive attention is IO-bound
