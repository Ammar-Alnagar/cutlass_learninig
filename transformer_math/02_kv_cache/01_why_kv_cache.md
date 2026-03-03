# Why KV Cache Exists

## What This Is

KV cache eliminates redundant computation in autoregressive decode. At each decode step, the transformer generates one new token by attending to all previous tokens. Without caching, you would recompute the key and value vectors for all previous tokens at every step — an $O(S^2)$ waste.

**The key insight:** Keys and values for tokens $0, 1, \ldots, t-1$ do not change when generating token $t$. Cache them and reuse.

## Why A Kernel Engineer Needs This

**You will implement the kernel that manages KV cache: allocation, indexing, and potentially quantization.** In vLLM and similar systems, you also implement PagedAttention — the page table that maps logical token positions to physical cache blocks. Understanding why the cache exists is prerequisite to implementing these optimizations.

**Interview relevance:** Cerebras and NVIDIA interviewers ask: "Why is KV cache necessary? What happens if you don't use it?" You must be able to quantify the waste.

## The Math

### Autoregressive Decode Without KV Cache

At decode step $t$ (generating token $t$):

**Input:** tokens $0, 1, \ldots, t$ (where token $t$ is the newly generated token)

**Compute:**
1. Compute $Q_t, K_t, V_t$ for all tokens $0, \ldots, t$
2. Compute attention: $Q_{0:t} K_{0:t}^T$ → scores $[B, H, 1, t+1]$
3. Softmax and weighted sum: $O_t = \text{softmax}(\cdot) V_{0:t}$

**Problem:** At step $t+1$, you repeat the exact same computation for tokens $0, \ldots, t$:
- Recompute $K_0, \ldots, K_t$ (same values as before!)
- Recompute $V_0, \ldots, V_t$ (same values as before!)
- Recompute $Q_{0:t} K_{0:t}^T$ for the old tokens (wasted work)

**Compute complexity without cache:**

At step $t$, attention computes:
- $Q_t$ (new query for token $t$)
- $K_{0:t}$ (keys for all tokens, but $K_{0:t-1}$ are redundant)
- $V_{0:t}$ (values for all tokens, but $V_{0:t-1}$ are redundant)
- $Q_t K_{0:t}^T$: $t+1$ dot products

**Total FLOPs to generate $S$ tokens:**

$$\sum_{t=1}^{S} \underbrace{4 H d_h (t+1)}_{\text{attention at step } t} = 4 H d_h \sum_{t=1}^{S} (t+1) = 4 H d_h \cdot \frac{S(S+3)}{2} = O(H d_h S^2)$$

Since $H d_h = d$:
$$\text{FLOPs}_{\text{no-cache}} = O(d S^2)$$

**This is quadratic in sequence length.** For $S = 4096$, this is $16.7$ million operations per layer, per sequence.

### Autoregressive Decode With KV Cache

At decode step $t$:

**Prefill (step 0):** Process all prompt tokens $0, \ldots, P-1$ in parallel.
- Compute $K_{0:P-1}, V_{0:P-1}$ once
- Store in KV cache
- Memory: $2 \cdot P \cdot H \cdot d_h$ elements per layer

**Decode step $t$ (generating token $P+t$):**
1. Compute $Q_{P+t}, K_{P+t}, V_{P+t}$ for the new token only
2. Append $K_{P+t}, V_{P+t}$ to KV cache
3. Compute attention: $Q_{P+t} \cdot [K_{0:P+t-1}, K_{P+t}]^T$
4. Softmax and weighted sum using cached values

**Key difference:** $K_{0:P+t-1}$ and $V_{0:P+t-1}$ are loaded from cache, not recomputed.

**Compute complexity with cache:**

At step $t$, attention computes:
- $Q_{P+t}, K_{P+t}, V_{P+t}$: $3 \cdot d$ FLOPs (projections for 1 token)
- $Q_{P+t} K_{0:P+t}^T$: $H \cdot (P+t+1) \cdot d_h$ FLOPs (dot products)
- Softmax and $PV$: $O(P+t)$ FLOPs

**Total FLOPs to generate $S$ tokens (after prefill of $P$ tokens):**

$$\sum_{t=1}^{S} \underbrace{4 H d_h (P+t)}_{\text{attention at step } t} = 4 H d_h \sum_{t=1}^{S} (P+t) = 4 H d_h \left(P S + \frac{S(S+1)}{2}\right)$$

For $S \ll P$ (typical: generate 100-1000 tokens after 4096 prompt):
$$\text{FLOPs}_{\text{with-cache}} \approx 4 H d_h P S = 4 d P S$$

**This is linear in $S$ (number of generated tokens), not quadratic.**

### Comparison: With vs. Without Cache

**Example: LLaMA-3 8B, prompt $P=4096$, generate $S=128$ tokens**

Without cache:
$$\text{FLOPs} = O(d (P+S)^2) = O(4096 \cdot 4224^2) \approx 73 \text{ billion FLOPs per layer}$$

With cache:
$$\text{FLOPs} = O(d P S) = O(4096 \cdot 4096 \cdot 128) \approx 2.1 \text{ billion FLOPs per layer}$$

**Speedup: ~35x**

This is why KV cache is non-optional. Without it, autoregressive decode is infeasible.

## Shapes and Sizes

| Operation | Without cache | With cache |
|-----------|---------------|------------|
| $Q$ at step $t$ | $[B, H, t+1, d_h]$ | $[B, H, 1, d_h]$ (new token only) |
| $K$ at step $t$ | $[B, H, t+1, d_h]$ (recomputed) | $[B, H, 1, d_h]$ (new) + cache $[B, H, t, d_h]$ |
| $V$ at step $t$ | $[B, H, t+1, d_h]$ (recomputed) | $[B, H, 1, d_h]$ (new) + cache $[B, H, t, d_h]$ |
| $QK^T$ scores | $[B, H, t+1, t+1]$ | $[B, H, 1, t+1]$ |
| FLOPs at step $t$ | $O(d \cdot t)$ | $O(d)$ (loading cache is memory, not compute) |

## The Kernel Implication

### KV Cache Data Structure

**Basic layout (per layer):**
```cpp
// KV cache: [max_seq_len, batch_size, num_heads, head_dim]
// or: [batch_size, num_heads, max_seq_len, head_dim]
float* k_cache;  // Size: L * S_max * B * H * d_h * sizeof(float)
float* v_cache;
```

**Indexing at decode step $t$:**
```cpp
// Load cached K, V for positions 0..t-1
// New K, V for position t
int cache_offset = t * B * H * d_h;

// Compute new K, V
compute_kv(X[t], K_new, V_new);  // [B, H, 1, d_h]

// Append to cache
memcpy(k_cache + cache_offset, K_new, B * H * d_h * sizeof(float));
memcpy(v_cache + cache_offset, V_new, B * H * d_h * sizeof(float));

// Attention uses full cache [0..t]
attention(Q_new, k_cache[0..t], v_cache[0..t], O);
```

### Memory Bandwidth Implications

**Without cache:** Compute-bound (recomputing K, V is FLOPs)

**With cache:** Memory-bound (loading K, V from cache is bandwidth)

**This is the irony:** KV cache makes decode faster but more memory-bound. The kernel shifts from compute optimization to memory optimization.

**What this means for your kernel:**
- Optimize for coalesced loads from cache
- Minimize cache miss rate (keep hot tokens in L2)
- Consider quantization (INT8 KV cache = 2x memory savings)

## Numbers That Matter

| Model | L | d | H | d_h | KV cache (S=4096, FP16) | KV cache (S=8192, FP16) |
|-------|---|---|---|-----|------------------------|------------------------|
| LLaMA-3 8B | 32 | 4096 | 32 | 128 | 2.0 GB | 4.0 GB |
| LLaMA-3 70B | 80 | 8192 | 64 | 128 | 10.0 GB | 20.0 GB |
| LLaMA-3 405B | 126 | 16384 | 128 | 128 | 64.0 GB | 128.0 GB |

**Formula:** KV cache memory = $2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes}$

**Note:** This is per sequence. For batch size $B$, multiply by $B$.

## Common Interview Questions

**Q1: Why is KV cache necessary for autoregressive decode? What is the complexity without it?**

<details>
<summary>Answer</summary>

Without KV cache, at each decode step $t$, you recompute $K$ and $V$ for all previous tokens $0, \ldots, t-1$. This is wasteful because these values are identical to the previous step.

Complexity without cache: $O(d S^2)$ per layer for $S$ tokens (quadratic)
Complexity with cache: $O(d S)$ per layer for $S$ tokens (linear)

For LLaMA-3 8B with $S=4096$, the speedup is ~35x. Without KV cache, autoregressive decode is infeasible.
</details>

**Q2: What is the memory cost of KV cache for LLaMA-3 8B with sequence length 4096?**

<details>
<summary>Answer</summary>

Per layer: $2 \cdot S \cdot d \cdot \text{bytes} = 2 \cdot 4096 \cdot 4096 \cdot 2 = 64$ MB (FP16)

Total (32 layers): $32 \cdot 64 \text{ MB} = 2$ GB

This is per sequence. For batch size 128, you need 256 GB of KV cache — which is why PagedAttention and memory optimization are critical.
</details>

**Q3: Does KV cache make decode compute-bound or memory-bound? Why?**

<details>
<summary>Answer</summary>

Memory-bound. With KV cache, you load $S$ cached keys and values from HBM for each new token, but only compute $O(1)$ FLOPs per loaded byte.

Arithmetic intensity of decode with cache: ~1 FLOP/byte (see Module 01.4)

H100 peak: ~2000 FLOPs/byte. Since $1 \ll 2000$, decode is severely memory-bound.

This is why increasing batch size is critical: it amortizes the memory cost across multiple sequences.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01 (attention) — you need to understand the attention formula.

**What this unlocks:**
- Module 03 (Attention Variants): GQA reduces KV cache size by sharing KV heads.
- Module 06 (Quantization): INT8/FP8 KV cache reduces memory by 2-4x.
- Module 07 (PagedAttention): Solves memory fragmentation in KV cache allocation.

**Next:** `02_memory_formula.md` — exact memory formula and worked examples.
