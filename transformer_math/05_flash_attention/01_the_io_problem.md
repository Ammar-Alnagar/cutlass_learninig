# The IO Problem: Why Naive Attention Is IO-Bound

## What This Is

Naive attention materializes the full $S \times S$ attention score matrix in HBM. This requires $O(S^2)$ memory and $O(S^2)$ HBM traffic, making attention severely memory-bandwidth bound at production sequence lengths.

**The core problem:** For $S = 4096$, the $QK^T$ matrix is 1 GB (FP16). Reading and writing this from HBM dominates runtime.

## Why A Kernel Engineer Needs This

**This is the problem your FlashAttention kernel solves.** Understanding the IO bottleneck quantifies why FlashAttention matters. When you write your CuTe kernel, every optimization (tiling, online softmax, register sharing) exists to avoid this $O(S^2)$ HBM traffic.

**Interview relevance:** NVIDIA interviewers ask: "Why is FlashAttention faster? What is the IO complexity of naive vs. FlashAttention?" You must be able to derive the HBM traffic formulas.

## The Math

### Naive Attention: Three Separate Kernels

**Kernel 1: QK^T Matmul**
```cuda
// Input: Q [B,H,S,d_h], K [B,H,S,d_h]
// Output: scores [B,H,S,S]
scores = Q @ K^T / sqrt(d_h)
```

**HBM traffic:**
- Read Q: $B \cdot H \cdot S \cdot d_h \cdot 2$ bytes
- Read K: $B \cdot H \cdot S \cdot d_h \cdot 2$ bytes
- Write scores: $B \cdot H \cdot S \cdot S \cdot 2$ bytes
- **Total:** $4 B H S d_h + 2 B H S^2$ bytes

**Kernel 2: Softmax**
```cuda
// Input: scores [B,H,S,S]
// Output: P [B,H,S,S]
P = softmax(scores)
```

**HBM traffic:**
- Read scores: $B \cdot H \cdot S \cdot S \cdot 2$ bytes
- Write P: $B \cdot H \cdot S \cdot S \cdot 2$ bytes
- **Total:** $4 B H S^2$ bytes

**Kernel 3: PV Matmul**
```cuda
// Input: P [B,H,S,S], V [B,H,S,d_h]
// Output: O [B,H,S,d_h]
O = P @ V
```

**HBM traffic:**
- Read P: $B \cdot H \cdot S \cdot S \cdot 2$ bytes
- Read V: $B \cdot H \cdot S \cdot d_h \cdot 2$ bytes
- Write O: $B \cdot H \cdot S \cdot d_h \cdot 2$ bytes
- **Total:** $2 B H S^2 + 4 B H S d_h$ bytes

### Total HBM Traffic (Naive)

**Sum all three kernels:**

$$\text{Bytes}_{\text{naive}} = (4 B H S d_h + 2 B H S^2) + (4 B H S^2) + (2 B H S^2 + 4 B H S d_h)$$

$$\text{Bytes}_{\text{naive}} = 8 B H S d_h + 8 B H S^2$$

**Factored:**
$$\boxed{\text{Bytes}_{\text{naive}} = 8 B H S (d_h + S)}$$

**Worked example (LLaMA-3 8B, B=1, H=32, S=4096, d_h=128, FP16):**

$$\text{Bytes}_{\text{naive}} = 8 \cdot 1 \cdot 32 \cdot 4096 \cdot (128 + 4096)$$
$$= 1,048,576 \cdot 4224$$
$$= 4,429,185,024 \text{ bytes}$$
$$\approx 4.4 \text{ GB}$$

**Breakdown:**
- $Q, K, V, O$ (linear terms): $8 \cdot 1 \cdot 32 \cdot 4096 \cdot 128 = 134$ MB (3%)
- Scores, P (quadratic terms): $8 \cdot 1 \cdot 32 \cdot 4096 \cdot 4096 = 4.3$ GB (97%)

**The $O(S^2)$ terms dominate.**

### FLOPs for Attention

From Module 01.4:
$$\text{FLOPs} = 4 B H S^2 d_h$$

**Worked example (LLaMA-3 8B):**
$$\text{FLOPs} = 4 \cdot 1 \cdot 32 \cdot 4096^2 \cdot 128 = 274.9 \text{ GFLOPs}$$

### Arithmetic Intensity (Naive)

$$\text{AI}_{\text{naive}} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{4 B H S^2 d_h}{8 B H S (d_h + S)}$$

**Simplified:**
$$\boxed{\text{AI}_{\text{naive}} = \frac{S d_h}{2 (d_h + S)}}$$

**Worked example (LLaMA-3 8B, S=4096, d_h=128):**

$$\text{AI}_{\text{naive}} = \frac{4096 \cdot 128}{2 \cdot (128 + 4096)} = \frac{524,288}{2 \cdot 4224} = \frac{524,288}{8,448} \approx 62 \text{ FLOPs/byte}$$

**Compare to H100 roofline:**
- H100 FP16 tensor core peak: ~2000 FLOPs/byte
- Naive attention: 62 FLOPs/byte

**Conclusion:** Naive attention is **32x more memory-bound** than the hardware roofline. The GPU spends 97% of cycles waiting for memory.

### Why This Matters at Scale

**LLaMA-3 8B at different sequence lengths:**

| S | QK^T Size (FP16) | HBM Traffic | AI (FLOPs/byte) | Bound |
|---|------------------|-------------|-----------------|-------|
| 512 | 16 MB | 128 MB | 28 | Memory |
| 1024 | 64 MB | 384 MB | 42 | Memory |
| 2048 | 256 MB | 1.2 GB | 53 | Memory |
| 4096 | 1.0 GB | 4.4 GB | 62 | Memory |
| 8192 | 4.0 GB | 17.2 GB | 63 | Memory |

**Observation:** AI saturates at ~64 FLOPs/byte as $S \to \infty$:

$$\lim_{S \to \infty} \frac{S d_h}{2 (d_h + S)} = \frac{d_h}{2} = \frac{128}{2} = 64$$

**For any model with $d_h = 128$, naive attention is fundamentally limited to ~64 FLOPs/byte**, regardless of sequence length. This is 30x below H100 peak.

## Shapes and Sizes

| Tensor | Shape | Elements | Bytes (FP16) | % of HBM traffic |
|--------|-------|----------|--------------|------------------|
| Q, K, V | $[B, H, S, d_h]$ | $B H S d_h$ each | $2 B H S d_h$ each | 3% total |
| Scores | $[B, H, S, S]$ | $B H S^2$ | $2 B H S^2$ | 48% |
| P (probs) | $[B, H, S, S]$ | $B H S^2$ | $2 B H S^2$ | 48% |
| O | $[B, H, S, d_h]$ | $B H S d_h$ | $2 B H S d_h$ | 1% |

**Key insight:** The $O(S^2)$ tensors (scores, P) dominate HBM traffic but are never used after computation. They are purely intermediate.

## The Kernel Implication

### The FlashAttention Insight

**FlashAttention observes:** The $O(S^2)$ intermediates are computed and consumed in a streaming fashion. Each row of scores is:
1. Computed (one dot product at a time)
2. Softmax-normalized (requires all scores in that row)
3. Used to weight values (one value at a time)
4. Never accessed again

**Key idea:** Keep intermediates in SRAM. Only read/write $O(S)$ data from HBM.

### SRAM vs. HBM

**HBM (High Bandwidth Memory):**
- Capacity: 80 GB (H100)
- Bandwidth: 3.35 TB/s (H100)
- Latency: ~500 cycles

**SRAM (On-chip memory):**
- Capacity: 50 MB (H100 SMs, ~230 KB per SM)
- Bandwidth: ~20 PB/s (effective, via registers/shared memory)
- Latency: ~20 cycles

**SRAM is 6000x faster than HBM.** FlashAttention's goal: never write $O(S^2)$ data to HBM.

### Block-Serial Algorithm

**Naive (block-parallel):**
```
Kernel 1: Compute all QK^T → write to HBM
Kernel 2: Compute all softmax → read/write HBM
Kernel 3: Compute all PV → read HBM, write output
```

**FlashAttention (block-serial):**
```
Single Kernel:
  For each query tile Q_i:
    For each key/value tile K_j, V_j:
      Compute Q_i @ K_j^T (in SRAM)
      Update softmax running max/sum (in SRAM)
      Update output running sum (in SRAM)
    Write O_i to HBM (once)
```

**HBM traffic (FlashAttention):**
- Read Q: $B H S d_h \cdot 2$ bytes (once)
- Read K: $B H S d_h \cdot 2$ bytes (once)
- Read V: $B H S d_h \cdot 2$ bytes (once)
- Write O: $B H S d_h \cdot 2$ bytes (once)
- **Total:** $8 B H S d_h$ bytes

**Compare to naive:** $8 B H S (d_h + S)$ bytes

**Speedup in HBM traffic:**
$$\frac{8 B H S (d_h + S)}{8 B H S d_h} = \frac{d_h + S}{d_h} = 1 + \frac{S}{d_h}$$

For LLaMA-3 8B ($S = 4096, d_h = 128$):
$$\text{Speedup} = 1 + \frac{4096}{128} = 33\times$$

**FlashAttention reduces HBM traffic by 33x for LLaMA-3 8B.**

## Numbers That Matter

| Model | S | d_h | Naive HBM | FA HBM | Speedup | Naive AI | FA AI |
|-------|---|-----|-----------|--------|---------|----------|-------|
| LLaMA-3 8B | 4096 | 128 | 4.4 GB | 134 MB | 33x | 62 | 2048 |
| LLaMA-3 8B | 8192 | 128 | 17.2 GB | 268 MB | 65x | 63 | 4096 |
| LLaMA-3 70B | 4096 | 128 | 8.8 GB | 268 MB | 33x | 62 | 2048 |

**FlashAttention arithmetic intensity:**
$$\text{AI}_{\text{FA}} = \frac{4 B H S^2 d_h}{8 B H S d_h} = \frac{S}{2}$$

For $S = 4096$: $\text{AI}_{\text{FA}} = 2048$ FLOPs/byte (compute-bound on H100).

## Common Interview Questions

**Q1: Derive the HBM traffic for naive attention. Why is it $O(S^2)$?**

<details>
<summary>Answer</summary>

Naive attention uses three kernels:
1. QK^T: reads Q, K ($O(S d_h)$), writes scores ($O(S^2)$)
2. Softmax: reads scores ($O(S^2)$), writes P ($O(S^2)$)
3. PV: reads P ($O(S^2)$), V ($O(S d_h)$), writes O ($O(S d_h)$)

Total: $O(S^2)$ from scores and P tensors.

Specifically: $8 B H S (d_h + S)$ bytes.

The $O(S^2)$ terms dominate because $S \gg d_h$ (typically $S = 4096, d_h = 128$).
</details>

**Q2: What is the arithmetic intensity of naive attention for LLaMA-3 8B at S=4096?**

<details>
<summary>Answer</summary>

$\text{AI}_{\text{naive}} = \frac{S d_h}{2 (d_h + S)}$

For $S = 4096, d_h = 128$:
$\text{AI} = \frac{4096 \cdot 128}{2 \cdot (128 + 4096)} = \frac{524,288}{8,448} \approx 62$ FLOPs/byte

H100 peak: ~2000 FLOPs/byte. Naive attention achieves 3% of peak.
</details>

**Q3: How does FlashAttention reduce HBM traffic? What is the new IO complexity?**

<details>
<summary>Answer</summary>

FlashAttention computes attention tile-by-tile in SRAM:
- Load Q, K, V tiles into SRAM
- Compute QK^T, softmax, PV entirely in SRAM
- Only write output O to HBM (once)

HBM traffic: $8 B H S d_h$ bytes (linear in $S$, not quadratic)

IO complexity: $O(B H S d_h)$ instead of $O(B H S^2)$

Speedup: $(d_h + S) / d_h = 1 + S/d_h \approx 33\times$ for LLaMA-3 8B.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01.4 (FLOP and memory analysis) — you need the naive attention baseline.

**What this unlocks:**
- Module 05.2 (Tiling): The SRAM tiling strategy.
- Module 05.3 (Online Softmax): The rescaling formula that makes block-serial correct.
- Module 10 (Arithmetic Intensity): Roofline analysis for FlashAttention.

**Next:** `02_tiling_insight.md` — how SRAM tiling eliminates $O(S^2)$ HBM traffic.
