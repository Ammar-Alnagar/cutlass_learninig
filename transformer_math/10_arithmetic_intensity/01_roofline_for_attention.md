# Roofline Model for Attention

## What This Is

The roofline model relates arithmetic intensity (FLOPs/byte) to achievable performance. It tells you whether an operation is compute-bound (limited by FLOPs/s) or memory-bound (limited by bytes/s).

**The roofline formula:**
$$\text{Achievable FLOPs/s} = \min(\text{Peak FLOPs/s}, \text{Peak BW} \times \text{AI})$$

Where AI = arithmetic intensity = FLOPs / bytes.

## Why A Kernel Engineer Needs This

**This determines your optimization strategy:**
- Memory-bound: Optimize for coalesced loads, cache reuse, minimize HBM traffic
- Compute-bound: Optimize for tensor core utilization, occupancy, instruction pipelining

**Interview relevance:** NVIDIA interviewers ask: "Is attention compute-bound or memory-bound? What is the arithmetic intensity?" You must be able to compute AI and compare against the hardware roofline.

## The Math

### Arithmetic Intensity Definition

$$\text{AI} = \frac{\text{Total FLOPs}}{\text{Total HBM Bytes}}$$

**Units:** FLOPs/byte

**Interpretation:** How many computations per byte of memory traffic.

### Hardware Roofline

**H100 (FP16 tensor cores):**
- Peak FLOPs/s: 989 TFLOPs/s (dense, with sparsity: 1979 TFLOPs/s)
- Peak HBM bandwidth: 3.35 TB/s
- **Peak AI:** $989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ FLOPs/byte

**A100 (FP16 tensor cores):**
- Peak FLOPs/s: 312 TFLOPs/s (dense)
- Peak HBM bandwidth: 2.0 TB/s
- **Peak AI:** $312 / 2.0 \approx 156$ FLOPs/byte

**RTX 4060 (FP16 tensor cores):**
- Peak FLOPs/s: 184 TFLOPs/s
- Peak HBM bandwidth: 272 GB/s
- **Peak AI:** $184 / 0.272 \approx 676$ FLOPs/byte

**Interpretation:** If AI < peak AI, the operation is memory-bound. If AI > peak AI, it's compute-bound.

### Attention Arithmetic Intensity

From Module 01.4 and Module 05.1:

**Naive attention:**
$$\text{AI}_{\text{naive}} = \frac{4 B H S^2 d_h}{8 B H S (d_h + S)} = \frac{S d_h}{2 (d_h + S)}$$

**FlashAttention:**
$$\text{AI}_{\text{FA}} = \frac{4 B H S^2 d_h}{8 B H S d_h} = \frac{S}{2}$$

**Decode (with KV cache):**
$$\text{AI}_{\text{decode}} = \frac{4 B H S d_h}{4 B H S d_h \cdot \text{dtype\_bytes}} = \frac{1}{\text{dtype\_bytes}}$$

For FP16 (2 bytes):
$$\text{AI}_{\text{decode}} = 0.5 \text{ FLOPs/byte}$$

### Worked Examples

**Example 1: LLaMA-3 8B Prefill (S=4096, d_h=128)**

Naive:
$$\text{AI}_{\text{naive}} = \frac{4096 \cdot 128}{2 \cdot (128 + 4096)} = \frac{524,288}{8,448} \approx 62 \text{ FLOPs/byte}$$

FlashAttention:
$$\text{AI}_{\text{FA}} = \frac{4096}{2} = 2048 \text{ FLOPs/byte}$$

**H100 peak AI:** ~295 FLOPs/byte

**Conclusion:**
- Naive: 62 < 295 → Memory-bound (21% of peak)
- FlashAttention: 2048 > 295 → Compute-bound (can achieve 100% of peak)

**Example 2: LLaMA-3 8B Decode (S=4096, d_h=128)**

$$\text{AI}_{\text{decode}} = 0.5 \text{ FLOPs/byte}$$

**H100 peak AI:** ~295 FLOPs/byte

**Conclusion:** 0.5 << 295 → Severely memory-bound (0.17% of peak)

**This is why decode is the bottleneck in LLM serving.**

### The Roofline Plot

```
Achievable FLOPs/s
    ^
    |                          /  (compute-bound region)
    |                        /
    |                      /
    |                    /
    |                  /
    |                /
    |              /
    |------------/  (memory-bound region)
    |          /
    |        /
    |      /
    |    /
    |  /
    |/
    +------------------------> Arithmetic Intensity (FLOPs/byte)
    0
```

**Memory-bound region (left of knee):** Performance = Peak BW × AI

**Compute-bound region (right of knee):** Performance = Peak FLOPs/s

**Knee point:** AI = Peak FLOPs/s / Peak BW = Peak AI

## Shapes and Sizes

| Operation | FLOPs | Bytes | AI (FLOPs/byte) |
|-----------|-------|-------|-----------------|
| Naive attention | $4 B H S^2 d_h$ | $8 B H S (d_h + S)$ | $\frac{S d_h}{2(d_h + S)}$ |
| FlashAttention | $4 B H S^2 d_h$ | $8 B H S d_h$ | $\frac{S}{2}$ |
| Decode (KV cache) | $4 B H S d_h$ | $4 B H S d_h \cdot \text{dtype\_bytes}$ | $\frac{1}{\text{dtype\_bytes}}$ |
| GEMM (M×N×K) | $2 MNK$ | $2(MN + NK + MK) \cdot \text{dtype\_bytes}$ | $\frac{K}{2 \cdot \text{dtype\_bytes}}$ (for large M, N) |

## The Kernel Implication

### Optimization Strategy by Region

**Memory-bound (AI < peak AI):**
- Focus on reducing HBM traffic
- Use tiling to improve cache reuse
- Fuse kernels to eliminate intermediate writes
- Use quantization (INT8, FP8) to reduce bytes

**Compute-bound (AI > peak AI):**
- Focus on maximizing tensor core utilization
- Use large tiles for better occupancy
- Pipeline memory loads with compute
- Use async copy (H100) to hide latency

### FlashAttention: Moving from Memory to Compute

**Before (naive):** AI = 62 FLOPs/byte → Memory-bound
**After (FA):** AI = 2048 FLOPs/byte → Compute-bound

**What changed:** FlashAttention eliminated $O(S^2)$ HBM traffic, increasing AI by 33x.

**Result:** Can now achieve 100% of peak FLOPs/s instead of 21%.

### Decode: Stuck in Memory-Bound Region

**Decode AI = 0.5 FLOPs/byte** — this is fundamental to autoregressive generation.

**No kernel optimization can change this.** The algorithm itself is memory-bound.

**Solutions:**
1. Increase batch size (amortize across sequences)
2. Use KV cache quantization (reduce bytes)
3. Speculative decoding (generate multiple tokens, verify in parallel)

## Numbers That Matter

| Hardware | Peak FLOPs/s | Peak BW | Peak AI |
|----------|--------------|---------|---------|
| H100 (FP16) | 989 TFLOPs/s | 3.35 TB/s | 295 |
| A100 (FP16) | 312 TFLOPs/s | 2.0 TB/s | 156 |
| RTX 4060 (FP16) | 184 TFLOPs/s | 272 GB/s | 676 |

| Operation | AI (FLOPs/byte) | Bound on H100 |
|-----------|-----------------|---------------|
| Naive attention (S=4096) | 62 | Memory |
| FlashAttention (S=4096) | 2048 | Compute |
| FlashAttention (S=512) | 256 | Compute |
| Decode (FP16) | 0.5 | Memory |
| Decode (INT8) | 1.0 | Memory |
| Decode (FP8) | 1.0 | Memory |
| GEMM (4096×4096×4096) | 2048 | Compute |

## Common Interview Questions

**Q1: What is the arithmetic intensity of FlashAttention for LLaMA-3 8B at S=4096? Is it compute-bound or memory-bound on H100?**

<details>
<summary>Answer</summary>

AI_FA = S / 2 = 4096 / 2 = 2048 FLOPs/byte

H100 peak AI ≈ 295 FLOPs/byte

Since 2048 > 295, FlashAttention is compute-bound on H100.

This means the kernel can achieve 100% of peak FLOPs/s (limited by tensor core throughput, not memory bandwidth).
</details>

**Q2: Why is decode always memory-bound, regardless of sequence length?**

<details>
<summary>Answer</summary>

Decode AI = 1 / dtype_bytes = 0.5 FLOPs/byte (for FP16)

This is independent of sequence length because:
- FLOPs = 4 B H S d_h (linear in S)
- Bytes = 4 B H S d_h × dtype_bytes (also linear in S)
- AI = FLOPs / Bytes = 1 / dtype_bytes (S cancels out)

Since 0.5 << 295 (H100 peak), decode is severely memory-bound.
</details>

**Q3: At what sequence length does FlashAttention become compute-bound on A100?**

<details>
<summary>Answer</summary>

AI_FA = S / 2

Set AI = peak AI for A100 (156 FLOPs/byte):
S / 2 = 156
S = 312

For S > 312, FlashAttention is compute-bound on A100.
For S < 312, FlashAttention is memory-bound.

For LLaMA-3 8B at S=4096, FlashAttention is well into the compute-bound region.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01.4 (FLOP/memory analysis), Module 05.1 (IO problem)

**What this unlocks:**
- Module 10.2 (Decode vs. Prefill): Different characteristics of each phase
- Module 10.3 (Batch Size Effect): How batching affects AI

**Next:** `02_decode_vs_prefill.md` — why decode and prefill have different bottlenecks.
