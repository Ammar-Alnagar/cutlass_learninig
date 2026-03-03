# Multi-Query Attention (MQA)

## What This Is

Multi-Query Attention uses a single key-value head for all query heads. Instead of H independent KV heads (MHA), MQA shares one KV head across all H query heads.

**The tradeoff:** 32x KV cache reduction (for H=32) with minimal accuracy loss.

## Why A Kernel Engineer Needs This

**MQA changes the memory access pattern for KV cache.** Instead of H independent KV heads, you have 1 KV head broadcast to H query heads. This affects:
- KV cache allocation (H_kv = 1 instead of H)
- Memory layout (broadcast pattern)
- Kernel indexing (stride-0 for KV heads)

**Interview relevance:** NVIDIA interviewers ask: "What is MQA? How does it reduce memory?"

## The Math

### MHA vs. MQA Head Structure

**MHA (Multi-Head Attention):**
- Query heads: H
- KV heads: H
- KV cache per layer: $2 \cdot S \cdot H \cdot d_h$ elements

**MQA (Multi-Query Attention):**
- Query heads: H
- KV heads: 1
- KV cache per layer: $2 \cdot S \cdot 1 \cdot d_h$ elements

**KV cache reduction:**
$$\frac{\text{MQA KV}}{\text{MHA KV}} = \frac{1}{H}$$

For H=32: 32x reduction.

### Attention Formula with MQA

**MHA:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

where $Q \in \mathbb{R}^{B \times H \times S \times d_h}$, $K \in \mathbb{R}^{B \times H \times S \times d_h}$, $V \in \mathbb{R}^{B \times H \times S \times d_h}$.

**MQA:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

where $Q \in \mathbb{R}^{B \times H \times S \times d_h}$, $K \in \mathbb{R}^{B \times 1 \times S \times d_h}$, $V \in \mathbb{R}^{B \times 1 \times S \times d_h}$.

**Key difference:** K and V have only 1 head, broadcast to H query heads.

### Shapes and Broadcasting

**MQA tensor shapes:**
$$Q: [B, H, S, d_h]$$
$$K: [B, 1, S, d_h]$$
$$V: [B, 1, S, d_h]$$

**QK^T computation:**
- Q: $[B, H, S, d_h]$
- K^T: $[B, 1, d_h, S]$
- QK^T: $[B, H, S, S]$ (K is broadcast from 1 to H)

**Broadcasting rule:** The KV head dimension (1) is broadcast to match the query head dimension (H).

## Shapes and Sizes

| Model | H | d_h | MHA KV Cache | MQA KV Cache | Reduction |
|-------|---|-----|--------------|--------------|-----------|
| LLaMA-3 8B | 32 | 128 | 2.0 GB | 0.06 GB | 32x |
| LLaMA-3 70B | 64 | 128 | 10.0 GB | 0.16 GB | 64x |

**Note:** These are per-sequence at S=4096, FP16.

## The Kernel Implication

### CuTe Layout for MQA

**KV cache layout (MQA):**
```cpp
// MQA: KV has shape [B, 1, S, d_h]
// But we need to broadcast to H query heads

// Option 1: Explicit broadcast (wasteful)
auto K_broadcast = broadcast(K, H);  // Don't do this!

// Option 2: Stride-0 layout (efficient)
auto K_layout = make_layout(make_shape(B, H, S, d_h),
                            make_stride(K_stride_b, 0, K_stride_s, K_stride_d));
// Note: stride for head dimension is 0 (broadcast)
```

**Stride-0 means:** All H query heads read the same KV data (no duplication).

### Kernel Indexing

**MHA kernel:**
```cuda
// Each query head h reads its own KV head h
float* k_ptr = k_cache + h * kv_head_stride;
```

**MQA kernel:**
```cuda
// All query heads read the same KV head (index 0)
float* k_ptr = k_cache;  // Same for all h
```

**Unified kernel (handles both MHA and MQA):**
```cuda
// kv_head_stride = 0 for MQA, non-zero for MHA
int kv_head_idx = use_mqa ? 0 : h;
float* k_ptr = k_cache + kv_head_idx * kv_head_stride;
```

## Numbers That Matter

| Model | H | MHA KV (S=4096) | MQA KV (S=4096) | Speedup |
|-------|---|-----------------|-----------------|---------|
| LLaMA-3 8B | 32 | 2.0 GB | 64 MB | 32x |
| LLaMA-3 70B | 64 | 10.0 GB | 156 MB | 64x |

**Maximum batch size increase (A100 80GB, LLaMA-3 8B):**
- MHA: B_max ≈ 30
- MQA: B_max ≈ 30 × 32 = 960

## Common Interview Questions

**Q1: What is the KV cache reduction factor for MQA vs. MHA?**

<details>
<summary>Answer</summary>

MQA uses 1 KV head instead of H heads.

KV cache reduction = H / 1 = H

For LLaMA-3 8B (H=32): 32x reduction.
For LLaMA-3 70B (H=64): 64x reduction.
</details>

**Q2: How does MQA affect the attention computation?**

<details>
<summary>Answer</summary>

MQA broadcasts the single KV head to all H query heads during QK^T computation.

The attention formula is unchanged:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_h)) V

Only the shapes change: K and V have shape [B, 1, S, d_h] instead of [B, H, S, d_h].

The broadcasting is handled by the memory layout (stride-0), not by explicit data duplication.
</details>

**Q3: What is the accuracy tradeoff of MQA?**

<details>
<summary>Answer</summary>

MQA typically has small accuracy degradation (1-2% on standard benchmarks) compared to MHA.

The loss comes from reduced expressivity: all query heads share the same key-value representations.

For production models, GQA (Module 03.2) is preferred: it offers most of MQA's efficiency with MHA's accuracy.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01 (attention), Module 02 (KV cache)

**What this unlocks:**
- Module 03.2 (GQA): Generalization of MQA with multiple KV heads
- Module 03.3 (MLA): Further compression with latent representations

**Next:** `02_gqa.md` — Grouped Query Attention
