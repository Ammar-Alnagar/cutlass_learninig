# Decode vs. Prefill: Different Bottlenecks

## What This Is

Prefill and decode have fundamentally different arithmetic intensities:
- **Prefill:** Can be compute-bound at moderate sequence lengths (AI ≈ S/2)
- **Decode:** Always memory-bound (AI ≈ 0.5 FLOPs/byte for FP16)

This difference dictates kernel design and system architecture.

## Why A Kernel Engineer Needs This

**You optimize for different bottlenecks:**
- Prefill kernels: Maximize tensor core utilization
- Decode kernels: Minimize memory latency, maximize batch size

**Interview relevance:** Cerebras interviewers ask: "Why is prefill faster than decode per token? What is the bottleneck in each phase?"

## The Math

### Prefill Arithmetic Intensity

From Module 10.1:
$$\text{AI}_{\text{prefill}} = \frac{S}{2} \text{ (FlashAttention)}$$

**Worked example (LLaMA-3 8B):**
- S = 4096: AI = 2048 FLOPs/byte → Compute-bound
- S = 512: AI = 256 FLOPs/byte → Compute-bound (barely, on H100)
- S = 128: AI = 64 FLOPs/byte → Memory-bound (on H100)

**Transition point (H100, peak AI = 295):**
$$S / 2 = 295 \Rightarrow S = 590$$

For S > 590, prefill is compute-bound on H100.

### Decode Arithmetic Intensity

**Derivation:**

At decode step $t$ (generating token $t+1$ given $t$ previous tokens):

**FLOPs (attention only, per layer):**
- $QK^T$: $2 \cdot B \cdot H \cdot 1 \cdot t \cdot d_h$ (one query, $t$ keys)
- $PV$: $2 \cdot B \cdot H \cdot 1 \cdot t \cdot d_h$ (one output, $t$ values)
- **Total:** $4 B H t d_h$

**HBM Traffic (with KV cache, per layer):**
- Read $Q$: $B \cdot H \cdot 1 \cdot d_h \cdot 2$ bytes
- Read $K_{\text{cache}}$: $B \cdot H \cdot t \cdot d_h \cdot 2$ bytes
- Read $V_{\text{cache}}$: $B \cdot H \cdot t \cdot d_h \cdot 2$ bytes
- Write $O$: $B \cdot H \cdot 1 \cdot d_h \cdot 2$ bytes
- Write new $K, V$: $2 \cdot B \cdot H \cdot 1 \cdot d_h \cdot 2$ bytes

For $t \gg 1$:
$$\text{Bytes} \approx 4 B H t d_h \cdot 2 = 8 B H t d_h \text{ bytes (FP16)}$$

**Arithmetic intensity:**
$$\text{AI}_{\text{decode}} = \frac{4 B H t d_h}{8 B H t d_h} = 0.5 \text{ FLOPs/byte}$$

**Key insight:** $t$ cancels out. AI is constant regardless of sequence length.

### Comparison Table

| Metric | Prefill (S=4096) | Decode (any S) |
|--------|------------------|----------------|
| FLOPs/layer | $4 B H S^2 d_h$ | $4 B H S d_h$ |
| Bytes/layer (FA) | $8 B H S d_h$ | $8 B H S d_h$ |
| AI (FP16) | $S/2 = 2048$ | $0.5$ |
| Bound (H100) | Compute | Memory |
| GPU utilization | ~100% | ~0.17% |

**Note:** Decode FLOPs and bytes are both linear in $S$, so AI is constant.

### Why Decode Is Fundamentally Memory-Bound

**The decode operation:**
1. Load $S$ cached keys from HBM
2. Load $S$ cached values from HBM
3. Compute $S$ dot products (one per key)
4. Produce 1 output token

**Ratio:** $2S$ loads, $S$ computes → 2 bytes per FLOP (for FP16).

**This is inherent to autoregressive generation.** No algorithmic improvement can change this ratio.

**The only ways to improve decode throughput:**
1. Increase batch size (amortize across sequences)
2. Reduce memory (quantization)
3. Increase bandwidth (HBM3, multi-GPU)
4. Speculative decoding (generate multiple, verify in parallel)

## Shapes and Sizes

| Phase | Q shape | K shape | V shape | FLOPs | Bytes | AI |
|-------|---------|---------|---------|-------|-------|-----|
| Prefill | $[B,H,S,d_h]$ | $[B,H,S,d_h]$ | $[B,H,S,d_h]$ | $4 B H S^2 d_h$ | $8 B H S d_h$ | $S/2$ |
| Decode | $[B,H,1,d_h]$ | $[B,H,S,d_h]$ | $[B,H,S,d_h]$ | $4 B H S d_h$ | $8 B H S d_h$ | $0.5$ |

## The Kernel Implication

### Prefill Kernel Optimization

**Goal:** Maximize tensor core utilization.

**Strategies:**
- Use large thread blocks (256-1024 threads)
- Maximize occupancy (hide latency)
- Use tensor cores (wmma/mma instructions)
- Pipeline memory loads with compute
- Use FlashAttention tiling

**CuTe example:**
```cpp
// Prefill: large tiles for compute efficiency
using TileShape = Shape<_128, _128>;  // Large tiles
using Atom = MMA_Atom<SM80_16x8x16_F16>;  // Tensor cores
```

### Decode Kernel Optimization

**Goal:** Minimize memory latency.

**Strategies:**
- Use small thread blocks (32-128 threads, matching warp size)
- Coalesce memory loads (sequential access patterns)
- Cache KV in L2 (reuse across warps)
- Use vectorized loads (ld.global.v4)
- Minimize register pressure (more active warps)

**CuTe example:**
```cpp
// Decode: small tiles for memory efficiency
using TileShape = Shape<_1, _128>;  // One query, 128 keys
using Atom = MMA_Atom<SM80_16x8x16_F16>;
```

### Unified Kernel (FlashAttention)

FlashAttention handles both with the same kernel but different launch parameters:

```cuda
// Prefill: many blocks, large tiles
flash_attention<<<num_query_tiles, num_heads, block_size>>>(...);

// Decode: fewer blocks, small tiles
flash_attention<<<1, num_heads, small_block_size>>>(...);
```

## Numbers That Matter

| Scenario | Model | B | S | Phase | AI | Bound | Time/layer |
|----------|-------|---|---|-------|-----|-------|------------|
| Prefill | LLaMA-3 8B | 1 | 4096 | Prefill | 2048 | Compute | ~0.5 ms |
| Decode | LLaMA-3 8B | 1 | 4096 | Decode | 0.5 | Memory | ~0.1 ms/token |
| Decode | LLaMA-3 8B | 128 | 4096 | Decode | 0.5 | Memory | ~0.1 ms/token |
| Prefill | LLaMA-3 8B | 32 | 4096 | Prefill | 2048 | Compute | ~2 ms |

**Note:** Decode time per token is constant regardless of batch size (memory-bound). Prefill time scales with batch size (compute-bound).

## Common Interview Questions

**Q1: Why is decode slower than prefill per token?**

<details>
<summary>Answer</summary>

Decode is memory-bound (AI = 0.5 FLOPs/byte), while prefill is compute-bound (AI = S/2 = 2048 for S=4096).

Memory-bound operations achieve a small fraction of peak FLOPs/s:
- Decode: 0.5 / 295 = 0.17% of H100 peak
- Prefill: 100% of H100 peak

Even though decode does fewer FLOPs per token, it's limited by memory bandwidth, not compute.
</details>

**Q2: Does increasing batch size help decode throughput? Why?**

<details>
<summary>Answer</summary>

Yes, but not because it changes AI. Decode AI is constant (0.5 FLOPs/byte) regardless of batch size.

Increasing batch size amortizes the memory cost across multiple sequences:
- B=1: 1 token per memory access
- B=128: 128 tokens per memory access

Throughput scales linearly with batch size, but latency per token remains the same.
</details>

**Q3: At what sequence length does LLaMA-3 8B prefill transition from memory-bound to compute-bound on H100?**

<details>
<summary>Answer</summary>

AI_prefill = S / 2

Set AI = H100 peak AI (295):
S / 2 = 295
S = 590

For S > 590, prefill is compute-bound.
For S < 590, prefill is memory-bound.

For LLaMA-3 8B at S=4096, prefill is well into the compute-bound region.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 10.1 (roofline model)

**What this unlocks:**
- Module 10.3 (Batch Size Effect): How batching affects AI
- Module 08 (Speculative Decoding): Addresses decode underutilization

**Next:** `03_batch_size_effect.md` — how batch size shifts arithmetic intensity.
