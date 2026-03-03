# Module 03: Attention Variants — MHA, MQA, GQA, MLA

**One concept:** Different attention variants trade accuracy for KV cache size and compute efficiency. GQA is the production standard for LLaMA-3.

**Job mapping:** NVIDIA inference engineer — you will implement kernels that handle MHA, MQA, and GQA with the same code path.

---

## Files in This Module

Read in order:

1. **01_mqa.md** — Multi-Query Attention: single KV head, 32x KV cache reduction.

2. **02_gqa.md** — Grouped Query Attention: middle ground, stride-0 connection to CuTe layouts.

3. **03_mla.md** — Multi-Head Latent Attention (DeepSeek): latent compression, absorb trick.

4. **attention_variants.py** — Compare MHA vs. MQA vs. GQA forward pass, KV cache sizes.

---

## What You Must Be Able To Do After This Module

1. Compute KV cache size for MHA, MQA, GQA given model config

2. Explain the CuTe stride-0 layout for GQA KV tensors

3. Analyze accuracy vs. efficiency tradeoff for each variant

---

## Before Moving To Module 04

Run `python attention_variants.py`. It must print `PASS`. Compare KV cache sizes.

**Next:** `01_mqa.md` — Multi-Query Attention
