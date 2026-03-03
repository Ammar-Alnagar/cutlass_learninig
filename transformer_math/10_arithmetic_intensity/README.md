# Module 10: Arithmetic Intensity — Systems Thinking for Kernel Engineers

**One concept:** Arithmetic intensity (FLOPs/byte) determines whether an operation is compute-bound or memory-bound. This is the single most important metric for kernel optimization and the most commonly tested concept in NVIDIA/Cerebras interviews.

**Job mapping:** All target jobs — NVIDIA, Modular, and Cerebras interviewers ask candidates to analyze arithmetic intensity and determine the bottleneck.

---

## Files in This Module

Read in order:

1. **01_roofline_for_attention.md** — Roofline model, compute-bound vs. memory-bound, the roofline formula.

2. **02_decode_vs_prefill.md** — Why decode is always bandwidth-bound, prefill can be compute-bound.

3. **03_batch_size_effect.md** — How batch size shifts arithmetic intensity, minimum batch for compute-bound.

4. **intensity_calculator.py** — Compute arithmetic intensity for any attention config. **Most interview-relevant file.**

---

## What You Must Be Able To Do After This Module

1. Compute arithmetic intensity for any operation: AI = FLOPs / bytes

2. Determine if an operation is compute-bound or memory-bound given hardware roofline

3. Calculate minimum batch size to become compute-bound

4. Analyze whether kernel optimizations help (tiling, fusion, etc.)

---

## Before Finishing

Run `python intensity_calculator.py`. It must print `PASS`. Use it to analyze your own kernel configurations.

**This module ties together everything:** Module 01 (attention), Module 02 (KV cache), Module 05 (FlashAttention).
