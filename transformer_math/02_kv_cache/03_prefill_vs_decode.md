# Prefill vs. Decode

## What This Is

Prefill and decode are the two phases of transformer inference, with fundamentally different compute characteristics:

**Prefill:** Process all prompt tokens in parallel. Compute-bound at moderate batch sizes.

**Decode:** Generate one token at a time. Always memory-bandwidth bound.

Understanding this distinction is critical for kernel design — you optimize for different bottlenecks in each phase.

## Why A Kernel Engineer Needs This

**You will write separate kernels (or kernel modes) for prefill and decode.** The same FlashAttention kernel can handle both, but the performance characteristics differ:

- Prefill: Optimize for tensor core utilization, occupancy
- Decode: Optimize for memory coalescing, cache reuse, minimizing HBM traffic

**Interview relevance:** Modular and Cerebras interviewers ask: "Why is decode slower than prefill per token? What is the bottleneck in each phase?"

## The Math

### Prefill Phase

**Input:** Prompt tokens $X_{0:P}$ (all $P$ tokens known)

**Compute:**
1. Compute $Q, K, V$ for all $P$ tokens: $[B, P, d]$ each
2. Compute attention scores: $QK^T \to [B, H, P, P]$
3. Softmax and weighted sum: $O = \text{softmax}(\cdot)V \to [B, P, d]$

**FLOPs (attention only, per layer):**
$$\text{FLOPs}_{\text{prefill}} = 4 \cdot B \cdot H \cdot P^2 \cdot d_h = 4 \cdot B \cdot P^2 \cdot d$$

Since $H \cdot d_h = d$.

**Memory traffic (per layer):**
- Read $Q, K, V$: $3 \cdot B \cdot P \cdot d \cdot \text{bytes}$
- Read weights (if not fused): $4 \cdot d^2 \cdot \text{bytes}$ (QKV + output projections)
- Write $O$: $B \cdot P \cdot d \cdot \text{bytes}$
- Write intermediates (naive): $2 \cdot B \cdot H \cdot P^2 \cdot \text{bytes}$ (scores + probs)

**Arithmetic intensity (prefill, attention only):**
$$\text{AI}_{\text{prefill}} = \frac{4 B P^2 d}{3 B P d \cdot \text{bytes} + 2 B H P^2 \cdot \text{bytes}}$$

For FP16 (2 bytes) and large $P$:
$$\text{AI}_{\text{prefill}} \approx \frac{4 P^2 d}{2 H P^2 \cdot 2} = \frac{4 P^2 d}{4 H P^2} = \frac{d}{H} = d_h$$

For LLaMA-3 8B ($d_h = 128$):
$$\text{AI}_{\text{prefill}} \approx 128 \text{ FLOPs/byte}$$

**With FlashAttention (no $O(P^2)$ intermediates):**
$$\text{AI}_{\text{prefill, FA}} \approx \frac{4 B P^2 d}{3 B P d \cdot 2 + B P d \cdot 2} = \frac{4 P^2 d}{8 P d} = \frac{P}{2}$$

For $P = 4096$:
$$\text{AI}_{\text{prefill, FA}} \approx 2048 \text{ FLOPs/byte}$$

**This is compute-bound on H100** (peak ~2000 FLOPs/byte).

### Decode Phase

**Input:** Generated token $X_t$ (one new token), cached $K_{0:t-1}, V_{0:t-1}$

**Compute:**
1. Compute $Q_t, K_t, V_t$ for token $t$: $[B, 1, d]$ each
2. Load cached $K_{0:t-1}, V_{0:t-1}$: $[B, t-1, d]$ each
3. Compute attention: $Q_t [K_{0:t-1}, K_t]^T \to [B, H, 1, t]$
4. Softmax and weighted sum: $O_t \to [B, 1, d]$

**FLOPs (attention only, per layer, per token):**
$$\text{FLOPs}_{\text{decode}} = 4 \cdot B \cdot H \cdot t \cdot d_h = 4 \cdot B \cdot t \cdot d$$

**Memory traffic (per layer, per token):**
- Read $Q_t$: $B \cdot d \cdot \text{bytes}$
- Read $K_{\text{cache}}$: $B \cdot t \cdot d \cdot \text{bytes}$
- Read $V_{\text{cache}}$: $B \cdot t \cdot d \cdot \text{bytes}$
- Write $O_t$: $B \cdot d \cdot \text{bytes}$
- Write new $K_t, V_t$ to cache: $2 \cdot B \cdot d \cdot \text{bytes}$

**Total:**
$$\text{Bytes}_{\text{decode}} = B \cdot d \cdot \text{bytes} + 2 \cdot B \cdot t \cdot d \cdot \text{bytes} + 3 \cdot B \cdot d \cdot \text{bytes}$$
$$= B \cdot d \cdot \text{bytes} \cdot (4 + 2t)$$

For FP16 (2 bytes) and $t \gg 2$:
$$\text{Bytes}_{\text{decode}} \approx 4 \cdot B \cdot t \cdot d \text{ bytes}$$

**Arithmetic intensity (decode):**
$$\text{AI}_{\text{decode}} = \frac{4 B t d}{4 B t d \cdot 2} = \frac{4 B t d}{8 B t d} = 0.5 \text{ FLOPs/byte}$$

Wait, let me recalculate with the correct FLOPs formula:

$$\text{FLOPs}_{\text{decode}} = 4 \cdot B \cdot H \cdot t \cdot d_h$$
$$\text{Bytes}_{\text{decode}} \approx 4 \cdot B \cdot t \cdot d \cdot \text{dtype\_bytes}$$

Since $H \cdot d_h = d$:
$$\text{AI}_{\text{decode}} = \frac{4 B t d}{4 B t d \cdot \text{dtype\_bytes}} = \frac{1}{\text{dtype\_bytes}}$$

For FP16 (2 bytes):
$$\text{AI}_{\text{decode}} = \frac{1}{2} = 0.5 \text{ FLOPs/byte}$$

**This is extremely memory-bound.** H100 peak is ~2000 FLOPs/byte. Decode achieves 0.5 FLOPs/byte — 4000x below peak.

### Comparison Table

| Metric | Prefill (P=4096) | Decode (t=4096) |
|--------|------------------|-----------------|
| FLOPs/layer | $4 B P^2 d$ | $4 B t d$ |
| Bytes/layer | $8 B P d$ (FA) | $4 B t d \cdot 2$ |
| AI (FP16) | $P/2 = 2048$ | $0.5$ |
| Bottleneck | Compute | Memory |
| H100 utilization | ~80% | ~0.02% |

**Key insight:** Decode is 4000x more memory-bound than prefill (for LLaMA-3 8B).

### Why Decode Is Always Bandwidth-Bound

At decode step $t$:
- Load $t$ cached keys: $B \cdot t \cdot d$ bytes
- Load $t$ cached values: $B \cdot t \cdot d$ bytes
- Compute $t$ dot products: $B \cdot t \cdot d$ FLOPs (per head, but $H$ heads)

**Per-token, per-layer:**
- Memory: $2 \cdot B \cdot t \cdot d \cdot \text{bytes}$
- Compute: $2 \cdot B \cdot t \cdot d$ FLOPs (QK^T + PV, ignoring constants)

$$\text{AI} = \frac{2 B t d}{2 B t d \cdot \text{bytes}} = \frac{1}{\text{bytes}} = 0.5 \text{ FLOPs/byte (FP16)}$$

**This is fundamental to autoregressive generation.** No kernel optimization can change this — the algorithm itself is memory-bound.

**Solutions:**
1. Increase batch size (amortize memory across sequences)
2. Use KV cache quantization (reduce bytes per element)
3. Use larger on-chip caches (HBM is the bottleneck)

## Shapes and Sizes

| Phase | Q shape | K shape | V shape | QK^T shape | Output shape |
|-------|---------|---------|---------|------------|--------------|
| Prefill | $[B, H, P, d_h]$ | $[B, H, P, d_h]$ | $[B, H, P, d_h]$ | $[B, H, P, P]$ | $[B, H, P, d_h]$ |
| Decode | $[B, H, 1, d_h]$ | $[B, H, t, d_h]$ (cached) | $[B, H, t, d_h]$ (cached) | $[B, H, 1, t]$ | $[B, H, 1, d_h]$ |

## The Kernel Implication

### Prefill Kernel Optimization

**Goal:** Maximize tensor core utilization.

**Strategies:**
- Use large thread blocks (256-1024 threads)
- Maximize occupancy (hide latency)
- Use tensor cores (wmma/mma instructions)
- Fuse QKV projections (single GEMM)
- Use FlashAttention tiling (avoid $O(P^2)$ HBM writes)

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

FlashAttention handles both prefill and decode with the same kernel:

```cuda
flash_attention(Q, K, V, O, causal=true) {
    // Q: [B, H, S_q, d_h]
    // K, V: [B, H, S_k, d_h]
    
    // For prefill: S_q = S_k = P (causal mask applied)
    // For decode: S_q = 1, S_k = t (no mask needed)
    
    for (int tile_j = 0; tile_j < ceil(S_k / BLOCK_SIZE); ++tile_j) {
        // Load K, V tile
        // Compute QK^T tile
        // Update running softmax
        // Update output
    }
}
```

**The same kernel, different launch parameters:**
- Prefill: large blocks, many threads
- Decode: small blocks, fewer threads

## Numbers That Matter

| Scenario | Model | B | S | Phase | AI (FLOPs/byte) | Bound | Time/layer (H100) |
|----------|-------|---|---|-------|-----------------|-------|-------------------|
| Prefill | LLaMA-3 8B | 1 | 4096 | Prefill | 2048 | Compute | ~0.5 ms |
| Prefill | LLaMA-3 8B | 32 | 4096 | Prefill | 65536 | Compute | ~2 ms |
| Decode | LLaMA-3 8B | 1 | 4096 | Decode | 0.5 | Memory | ~0.1 ms/token |
| Decode | LLaMA-3 8B | 128 | 4096 | Decode | 64 | Memory | ~0.1 ms/token |
| Decode | LLaMA-3 70B | 1 | 4096 | Decode | 0.5 | Memory | ~0.2 ms/token |

**Note:** Decode time per token is roughly constant regardless of batch size (memory-bound). Prefill time scales with batch size (compute-bound).

## Common Interview Questions

**Q1: Why is decode always memory-bandwidth bound, regardless of batch size?**

<details>
<summary>Answer</summary>

At decode step $t$, for each new token:
- Load $t$ cached keys and $t$ cached values from HBM
- Compute $t$ dot products (one per cached key)

The ratio is: $2t$ loads, $t$ computes → 2 bytes loaded per FLOP (for FP16).

Arithmetic intensity = 0.5 FLOPs/byte, independent of batch size.

H100 peak = ~2000 FLOPs/byte. Since $0.5 \ll 2000$, decode is always memory-bound.

Batch size increases total throughput but doesn't change the bottleneck.
</details>

**Q2: At what sequence length does LLaMA-3 8B prefill become compute-bound on H100?**

<details>
<summary>Answer</summary>

With FlashAttention: $\text{AI} \approx P/2$

Set AI = 2000 (H100 roofline):
$P/2 = 2000$
$P = 4000$

For $P > 4000$, prefill is compute-bound.
For $P < 4000$, prefill is memory-bound.

For LLaMA-3 8B at $P = 4096$, prefill is right at the boundary — approximately compute-bound.
</details>

**Q3: Why can't you optimize decode to be compute-bound?**

<details>
<summary>Answer</summary>

The algorithm itself is memory-bound. Each new token attends to $t$ cached tokens, requiring $O(t)$ memory loads for $O(t)$ computes. The ratio is fixed at ~0.5 FLOPs/byte.

No kernel optimization (tiling, fusion, etc.) can change this fundamental ratio. The only ways to improve decode throughput are:
1. Increase batch size (more sequences in parallel)
2. Reduce memory (KV cache quantization)
3. Increase memory bandwidth (HBM3, multi-GPU)
4. Speculative decoding (generate multiple tokens, verify in parallel)
</details>

## Connection To Other Concepts

**Prerequisites:** Module 02.2 (memory formula) — you need the KV cache size.

**What this unlocks:**
- Module 05 (FlashAttention): The tile loop handles both prefill and decode.
- Module 08 (Speculative Decoding): Addresses decode underutilization.
- Module 10 (Arithmetic Intensity): Roofline analysis for both phases.

**Next:** Run `kv_cache_sim.py` to see prefill and decode in action.
