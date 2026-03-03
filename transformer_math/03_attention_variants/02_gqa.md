# Grouped Query Attention (GQA)

## What This Is

Grouped Query Attention is a middle ground between MHA and MQA. Query heads are divided into groups, and each group shares one KV head.

**LLaMA-3 8B configuration:**
- Query heads: H_q = 32
- KV heads: H_kv = 8
- Groups: 8 (each group has 32/8 = 4 query heads)

**The tradeoff:** 4x KV cache reduction with near-MHA accuracy.

## Why A Kernel Engineer Needs This

**GQA is the production standard for LLaMA-3.** You will implement GQA kernels that handle the grouped query-to-KV mapping.

**Critical insight:** GQA uses stride-0 memory layout for KV heads. This is the exact same pattern you'll use in CuTe.

**Interview relevance:** NVIDIA/Cerebras interviewers ask: "What is GQA? How does it differ from MQA? What is the stride-0 layout?"

## The Math

### GQA Head Structure

**GQA configuration:**
- Query heads: H_q
- KV heads: H_kv (where H_kv < H_q and H_q is divisible by H_kv)
- Group size: G = H_q / H_kv

**KV cache per layer:**
$$\text{KV}_{\text{GQA}} = 2 \cdot S \cdot H_{kv} \cdot d_h$$

**KV cache reduction vs. MHA:**
$$\frac{\text{KV}_{\text{GQA}}}{\text{KV}_{\text{MHA}}} = \frac{H_{kv}}{H_q} = \frac{1}{G}$$

For LLaMA-3 8B (H_q=32, H_kv=8, G=4): 4x reduction.

### Query-to-KV Head Mapping

**Group assignment:**
Query head $h_q$ attends to KV head $h_{kv}$ where:
$$h_{kv} = \lfloor h_q / G \rfloor = \lfloor h_q \cdot H_{kv} / H_q \rfloor$$

**Worked example (LLaMA-3 8B, G=4):**
- Query heads 0-3 → KV head 0
- Query heads 4-7 → KV head 1
- Query heads 8-11 → KV head 2
- ...
- Query heads 28-31 → KV head 7

### Attention Formula with GQA

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

where:
- $Q \in \mathbb{R}^{B \times H_q \times S \times d_h}$
- $K \in \mathbb{R}^{B \times H_{kv} \times S \times d_h}$
- $V \in \mathbb{R}^{B \times H_{kv} \times S \times d_h}$

**QK^T computation:**
For query head $h_q$, compute attention with KV head $h_{kv} = \lfloor h_q / G \rfloor$:
$$\text{scores}_{h_q} = \frac{Q_{h_q} K_{h_{kv}}^T}{\sqrt{d_h}}$$

## Shapes and Sizes

| Model | H_q | H_kv | G | MHA KV | GQA KV | Reduction |
|-------|-----|------|---|--------|--------|-----------|
| LLaMA-3 8B | 32 | 8 | 4 | 2.0 GB | 0.5 GB | 4x |
| LLaMA-3 70B | 64 | 8 | 8 | 10.0 GB | 1.25 GB | 8x |

**Note:** Per-sequence at S=4096, FP16.

## The Kernel Implication

### Stride-0 Layout for GQA

**This is the most important kernel concept for GQA.**

In CuTe, GQA is implemented using a stride-0 layout for the KV head dimension:

```cpp
// GQA KV cache layout
// Physical shape: [B, H_kv, S, d_h]
// Logical shape: [B, H_q, S, d_h]  (H_q > H_kv)

auto kv_layout = make_layout(
    make_shape(B, H_q, S, d_h),           // Logical shape
    make_stride(kv_stride_b, 0,           // stride-0 for head!
                kv_stride_s, kv_stride_d)
);
```

**Stride-0 means:** Query heads in the same group read the same KV data without duplication.

**Memory access pattern:**
```
Query head 0 → KV head 0 (offset 0)
Query head 1 → KV head 0 (offset 0, same data!)
Query head 2 → KV head 0 (offset 0, same data!)
Query head 3 → KV head 0 (offset 0, same data!)
Query head 4 → KV head 1 (offset kv_stride_b)
...
```

### CuTe Example

```cpp
// GQA configuration
constexpr int H_q = 32;
constexpr int H_kv = 8;
constexpr int G = H_q / H_kv;  // Group size = 4

// KV cache: physical layout [B, H_kv, S, d_h]
Tensor K_cache = make_tensor<K>(shape(B, H_kv, S, d_h));

// Logical layout for GQA: broadcast H_kv to H_q
auto K_logical = local_tile(K_cache, Tile<H_q>, make_coord(_, 0, _, _));

// In the kernel, query head h_q accesses KV head h_q / G
int kv_head_idx = h_q / G;  // or h_q >> log2(G) for power-of-2 G
```

### Kernel Indexing

**GQA kernel:**
```cuda
__global__ void gqa_attention(Q, K_cache, V_cache, O, H_q, H_kv) {
    int h_q = blockIdx.y;  // Query head index
    int h_kv = h_q / (H_q / H_kv);  // Map to KV head
    
    // Load KV for this group
    float* k_ptr = K_cache + h_kv * kv_head_stride;
    float* v_ptr = V_cache + h_kv * kv_head_stride;
    
    // Compute attention for query head h_q
    ...
}
```

**Power-of-2 optimization:**
```cuda
// If G is power of 2, use bit shift
int h_kv = h_q >> log2(G);  // Faster than division
```

## Numbers That Matter

| Model | H_q | H_kv | G | KV Cache (S=4096) | Max Batch (A100 80GB) |
|-------|-----|------|---|-------------------|----------------------|
| LLaMA-3 8B (MHA) | 32 | 32 | 1 | 2.0 GB | 30 |
| LLaMA-3 8B (GQA) | 32 | 8 | 4 | 0.5 GB | 120 |
| LLaMA-3 8B (MQA) | 32 | 1 | 32 | 0.06 GB | 960 |
| LLaMA-3 70B (GQA) | 64 | 8 | 8 | 1.25 GB | 48 |

**LLaMA-3 8B uses GQA (not MHA) for 4x KV cache reduction with minimal accuracy loss.**

## Common Interview Questions

**Q1: What is the stride-0 layout for GQA? Why is it important?**

<details>
<summary>Answer</summary>

Stride-0 layout means the KV head dimension has stride 0 in the memory layout. All query heads in a group read the same KV data without duplication.

Example (LLaMA-3 8B, G=4):
- Query heads 0,1,2,3 all read KV head 0 (same memory location)
- Query heads 4,5,6,7 all read KV head 1 (same memory location)

This is efficient because:
1. No data duplication (KV is stored once per group)
2. No explicit broadcast (hardware handles it)
3. Coalesced memory access (all threads in a warp read adjacent locations)

In CuTe, this is expressed as:
```cpp
make_stride(..., 0, ...)  // stride-0 for the head dimension
```
</details>

**Q2: How does GQA compare to MQA and MHA?**

<details>
<summary>Answer</summary>

| Aspect | MHA | GQA | MQA |
|--------|-----|-----|-----|
| KV heads | H | H_kv (1 < H_kv < H) | 1 |
| KV cache | 1.0x | H_kv/H | 1/H |
| Accuracy | Best | Near-MHA | Slight loss |
| Throughput | Baseline | 2-4x | 4-8x |

GQA is the production standard (LLaMA-3, Mistral, etc.) because it offers the best accuracy-efficiency tradeoff.
</details>

**Q3: For LLaMA-3 8B, which query heads attend to KV head 2?**

<details>
<summary>Answer</summary>

LLaMA-3 8B: H_q = 32, H_kv = 8, G = 4.

Query head h_q attends to KV head h_kv = h_q / 4.

KV head 2 is attended by query heads:
- h_q / 4 = 2
- h_q = 8, 9, 10, 11

Query heads 8, 9, 10, 11 attend to KV head 2.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 03.1 (MQA)

**What this unlocks:**
- Module 05 (FlashAttention): GQA is handled naturally in the tile loop
- Module 07 (PagedAttention): Block tables work with GQA KV heads

**Next:** `03_mla.md` — Multi-Head Latent Attention (DeepSeek's innovation).
