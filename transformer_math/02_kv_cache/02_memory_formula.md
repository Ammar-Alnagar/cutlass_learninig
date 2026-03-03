# KV Cache Memory Formula

## What This Is

Exact formula for computing KV cache memory requirements. This tells you how much GPU memory is needed to store key and value vectors for a given model, sequence length, batch size, and data type.

**The formula:**
$$\text{KV Cache Memory} = 2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes} \cdot B$$

Where the factor of 2 accounts for both keys and values.

## Why A Kernel Engineer Needs This

**You will allocate KV cache buffers and manage their memory.** At production scale, you need to know:
- How much memory to reserve per sequence
- Maximum batch size given available memory
- Whether to use FP16, INT8, or FP8 for the cache

**Interview relevance:** NVIDIA interviewers ask: "How much memory does LLaMA-3 70B need for KV cache at batch=128, S=4096?" You must be able to compute this instantly.

## The Math

### Derivation

**Per-layer, per-sequence KV cache:**

Each layer stores:
- Keys: $S \cdot H \cdot d_h$ elements
- Values: $S \cdot H \cdot d_h$ elements
- Total: $2 \cdot S \cdot H \cdot d_h$ elements

Since $H \cdot d_h = d$ (model dimension):
$$\text{Elements per layer} = 2 \cdot S \cdot d$$

**Memory per layer (in bytes):**
$$\text{Bytes per layer} = 2 \cdot S \cdot d \cdot \text{dtype\_bytes}$$

**All layers:**
$$\text{Total bytes} = L \cdot 2 \cdot S \cdot d \cdot \text{dtype\_bytes}$$

**All sequences (batch size $B$):**
$$\boxed{\text{KV Cache Memory} = 2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes} \cdot B}$$

### Data Type Sizes

| Data Type | Bytes per Element | Notes |
|-----------|-------------------|-------|
| FP32 | 4 | Rarely used for inference |
| FP16 | 2 | Standard for most models |
| BF16 | 2 | Alternative to FP16 |
| FP8 (E4M3/E5M2) | 1 | H100 native support |
| INT8 | 1 | Requires quantization |
| INT4 | 0.5 | Aggressive quantization |

### Worked Example 1: LLaMA-3 8B

**Configuration:**
- $L = 32$ layers
- $d = 4096$ (hidden size)
- $S = 4096$ (sequence length)
- $B = 1$ (batch size)
- FP16 (2 bytes)

**Calculation:**
$$\text{KV Cache} = 2 \cdot 32 \cdot 4096 \cdot 4096 \cdot 2 \cdot 1$$
$$= 2 \cdot 32 \cdot 4096 \cdot 4096 \cdot 2$$
$$= 2,147,483,648 \text{ bytes}$$
$$= 2.0 \text{ GB}$$

**Breakdown:**
- Per layer: $2 \cdot 4096 \cdot 4096 \cdot 2 = 64$ MB
- 32 layers: $32 \cdot 64 = 2048$ MB = 2 GB

**With FP8 (1 byte):**
$$\text{KV Cache} = 2 \cdot 32 \cdot 4096 \cdot 4096 \cdot 1 \cdot 1 = 1.0 \text{ GB}$$

### Worked Example 2: LLaMA-3 70B

**Configuration:**
- $L = 80$ layers
- $d = 8192$ (hidden size)
- $S = 4096$ (sequence length)
- $B = 1$ (batch size)
- FP16 (2 bytes)

**Calculation:**
$$\text{KV Cache} = 2 \cdot 80 \cdot 4096 \cdot 8192 \cdot 2 \cdot 1$$
$$= 10,737,418,240 \text{ bytes}$$
$$\approx 10.0 \text{ GB}$$

**Per layer:** $2 \cdot 4096 \cdot 8192 \cdot 2 = 128$ MB

**With batch size 128:**
$$\text{KV Cache} = 10.0 \text{ GB} \cdot 128 = 1.28 \text{ TB}$$

**This is why PagedAttention is necessary.** You cannot allocate 1.28 TB contiguously.

### Worked Example 3: LLaMA-3 8B with GQA

LLaMA-3 8B uses Grouped Query Attention (GQA):
- Query heads: $H_q = 32$
- KV heads: $H_{kv} = 8$
- Group size: $H_q / H_{kv} = 4$

**GQA reduces KV cache by factor of $H_{kv} / H_q$:**

Standard MHA would have $H_{kv} = H_q = 32$ heads.
GQA has $H_{kv} = 8$ heads.

**KV cache with GQA:**
$$\text{KV Cache}_{\text{GQA}} = 2 \cdot L \cdot S \cdot (H_{kv} \cdot d_h) \cdot \text{dtype\_bytes} \cdot B$$

Since $d = H_q \cdot d_h$ but KV cache only stores $H_{kv}$ heads:
$$\text{KV Cache}_{\text{GQA}} = 2 \cdot L \cdot S \cdot \left(\frac{H_{kv}}{H_q} \cdot d\right) \cdot \text{dtype\_bytes} \cdot B$$

For LLaMA-3 8B:
$$\text{KV Cache}_{\text{GQA}} = 2.0 \text{ GB} \cdot \frac{8}{32} = 2.0 \text{ GB} \cdot 0.25 = 0.5 \text{ GB}$$

**GQA reduces KV cache by 4x for LLaMA-3 8B.**

## Shapes and Sizes

| Tensor | Shape | Elements per layer | Bytes (FP16) |
|--------|-------|-------------------|--------------|
| K cache | $[B, H_{kv}, S, d_h]$ | $B \cdot H_{kv} \cdot S \cdot d_h$ | $2 \cdot B \cdot H_{kv} \cdot S \cdot d_h$ |
| V cache | $[B, H_{kv}, S, d_h]$ | $B \cdot H_{kv} \cdot S \cdot d_h$ | $2 \cdot B \cdot H_{kv} \cdot S \cdot d_h$ |
| Total | | $2 \cdot B \cdot H_{kv} \cdot S \cdot d_h$ | $4 \cdot B \cdot H_{kv} \cdot S \cdot d_h$ |

**Note:** For MHA, $H_{kv} = H$. For GQA, $H_{kv} < H$. For MQA, $H_{kv} = 1$.

## The Kernel Implication

### Memory Layout

**Contiguous layout (simple but inflexible):**
```cpp
// [batch, layer, head, seq, head_dim]
float* k_cache = malloc(B * L * H_kv * S * d_h * sizeof(float));
```

**Problem:** Requires contiguous allocation. Wastes memory if sequences have different lengths.

**Paged layout (vLLM):**
```cpp
// Divide KV cache into fixed-size blocks
// Each block holds KV for BLOCK_SIZE tokens
// Block table maps (batch, layer, seq_block) -> physical_block_id

int block_table[B][L][ceil(S / BLOCK_SIZE)];
float* k_cache_blocks[num_blocks][H_kv][BLOCK_SIZE][d_h];
```

**Benefit:** Non-contiguous allocation. Only allocate blocks for actual tokens.

### Allocation Strategy

**Prefill time allocation:**
```cpp
// Allocate max KV cache upfront
size_t kv_cache_bytes = 2 * L * S_max * d * dtype_bytes * B_max;
cudaMalloc(&k_cache, kv_cache_bytes);
cudaMalloc(&v_cache, kv_cache_bytes);
```

**Dynamic allocation (PagedAttention):**
```cpp
// Allocate blocks on-demand
for each new token:
    if no free block:
        allocate new block from pool
    assign block to token
```

## Numbers That Matter

| Model | L | d | H_kv | KV Cache (S=4096, B=1, FP16) | KV Cache (S=4096, B=128, FP16) |
|-------|---|-----|------|------------------------------|--------------------------------|
| LLaMA-3 8B (MHA) | 32 | 4096 | 32 | 2.0 GB | 256 GB |
| LLaMA-3 8B (GQA) | 32 | 4096 | 8 | 0.5 GB | 64 GB |
| LLaMA-3 70B | 80 | 8192 | 64 | 10.0 GB | 1.28 TB |
| LLaMA-3 405B | 126 | 16384 | 128 | 64.0 GB | 8.2 TB |

**Note:** LLaMA-3 8B uses GQA (8 KV heads), not MHA. The MHA numbers are for comparison.

## Common Interview Questions

**Q1: Derive the KV cache memory formula. What does each factor represent?**

<details>
<summary>Answer</summary>

KV Cache = $2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes} \cdot B$

- 2: Keys and values (two tensors)
- L: Number of layers (each layer has its own KV cache)
- S: Sequence length (tokens per sequence)
- d: Model dimension (total hidden size, equals $H \cdot d_h$)
- dtype_bytes: Size of each element (2 for FP16, 1 for FP8/INT8)
- B: Batch size (number of sequences)

For LLaMA-3 8B (L=32, d=4096, S=4096, B=1, FP16):
$2 \cdot 32 \cdot 4096 \cdot 4096 \cdot 2 \cdot 1 = 2.0$ GB
</details>

**Q2: How does GQA reduce KV cache memory? Derive the formula.**

<details>
<summary>Answer</summary>

GQA uses fewer KV heads than query heads. If $H_q$ is query heads and $H_{kv}$ is KV heads:

Standard MHA: $H_{kv} = H_q$, so KV cache = $2 \cdot L \cdot S \cdot (H_q \cdot d_h) \cdot \text{bytes}$

GQA: $H_{kv} < H_q$, so KV cache = $2 \cdot L \cdot S \cdot (H_{kv} \cdot d_h) \cdot \text{bytes}$

Reduction factor: $H_{kv} / H_q$

For LLaMA-3 8B: $H_q = 32$, $H_{kv} = 8$, reduction = 8/32 = 0.25 (4x smaller)
</details>

**Q3: What is the KV cache memory for LLaMA-3 70B at batch=64, S=4096, FP16?**

<details>
<summary>Answer</summary>

L=80, d=8192, S=4096, B=64, dtype_bytes=2

KV Cache = $2 \cdot 80 \cdot 4096 \cdot 8192 \cdot 2 \cdot 64$
= $10.0 \text{ GB} \cdot 64$
= $640 \text{ GB}$

This requires multiple GPUs or PagedAttention with memory offloading.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 02.1 (why KV cache) — you need to understand why caching is necessary.

**What this unlocks:**
- Module 03 (Attention Variants): MQA and GQA trade accuracy for KV cache size.
- Module 06 (Quantization): INT8/FP8 cache reduces memory by 2-4x.
- Module 07 (PagedAttention): Solves fragmentation for large batch sizes.

**Next:** `03_prefill_vs_decode.md` — different compute characteristics of prefill and decode.
