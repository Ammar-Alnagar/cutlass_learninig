# Multi-Head Latent Attention (MLA)

## What This Is

Multi-Head Latent Attention (MLA) is DeepSeek's innovation that compresses KV cache using low-rank latent representations. Instead of storing full KV vectors, MLA stores compressed latents and reconstructs KV on-the-fly.

**The tradeoff:** 5-10x KV cache reduction with comparable accuracy to GQA.

## Why A Kernel Engineer Needs This

**MLA adds a decompression step to the attention kernel.** You'll implement the "absorb trick" that fuses KV reconstruction with attention computation.

**Interview relevance:** Cerebras interviewers ask about novel attention optimizations. MLA shows you understand state-of-the-art techniques.

## The Math

### MLA Compression

**Standard GQA:**
- Store: $K, V \in \mathbb{R}^{B \times H_{kv} \times S \times d_h}$

**MLA:**
- Store: $c \in \mathbb{R}^{B \times d_c \times S}$ (compressed latent)
- Reconstruct: $K, V = W_K c, W_V c$ where $W_K, W_V \in \mathbb{R}^{d_h \times d_c}$

**Compression ratio:**
$$\frac{\text{MLA KV}}{\text{GQA KV}} = \frac{d_c}{H_{kv} \cdot d_h}$$

For DeepSeek-V2 ($d_c = 512$, $H_{kv} \cdot d_h = 8 \times 128 = 1024$): 2x compression from latent alone.

### The Absorb Trick

**Naive MLA attention:**
1. Reconstruct K, V from latent: $K = W_K c$, $V = W_V c$
2. Compute attention: $\text{softmax}(Q K^T / \sqrt{d_h}) V$

**Problem:** Reconstruction adds FLOPs and memory traffic.

**Absorb trick:** Fuse reconstruction with attention computation.

**Key insight:**
$$Q K^T = Q (W_K c)^T = Q c^T W_K^T = (Q W_K) c^T$$

Define $\tilde{Q} = Q W_K$. Then:
$$Q K^T = \tilde{Q} c^T$$

**Similarly for V:**
$$\text{Output} = \text{softmax}(\tilde{Q} c^T / \sqrt{d_h}) (W_V c)$$

Define $\tilde{V} = W_V c$. Then:
$$\text{Output} = \text{softmax}(\tilde{Q} c^T / \sqrt{d_h}) \tilde{V}$$

**Benefit:** Precompute $\tilde{Q} = Q W_K$ and $\tilde{V} = W_V c$ outside the attention loop.

### MLA Attention Formula

**Prefill:**
1. Compute latent: $c = \text{Encoder}(X)$
2. Store $c$ in KV cache (compressed)
3. Compute $\tilde{Q} = Q W_K^{\text{down}}$, $\tilde{V} = W_V^{\text{up}} c$
4. Attention: $\text{softmax}(\tilde{Q} c^T / \sqrt{d_h}) \tilde{V}$

**Decode:**
1. Compute new latent: $c_t = \text{Encoder}(X_t)$
2. Append to cached latents: $c_{0:t}$
3. Compute $\tilde{Q}_t = Q_t W_K^{\text{down}}$, $\tilde{V}_t = W_V^{\text{up}} c_t$
4. Attention: $\text{softmax}(\tilde{Q}_t c_{0:t}^T / \sqrt{d_h}) \tilde{V}_{0:t}$

## Shapes and Sizes

| Component | GQA Shape | MLA Shape | Reduction |
|-----------|-----------|-----------|-----------|
| KV Cache | $[B, H_{kv}, S, d_h]$ | $[B, d_c, S]$ | $H_{kv} \cdot d_h / d_c$ |
| DeepSeek-V2 | $[B, 8, S, 128]$ | $[B, 512, S]$ | 2x |
| With multi-token | $[B, 8, S, 128]$ | $[B, 64, S]$ | 16x |

**Note:** MLA can combine with multi-token prediction for additional compression.

## The Kernel Implication

### Fused MLA Kernel

```cuda
__global__ void mla_attention(Q, c_cache, W_K, W_V, O) {
    // Q: [B, H_q, 1, d_h]
    // c_cache: [B, d_c, S] (compressed KV)
    // W_K: [d_h, d_c], W_V: [d_h, d_c]
    
    // Step 1: Compute Q_tilde = Q @ W_K (absorb trick)
    float Q_tilde[d_h];
    for (int k = 0; k < d_h; ++k) {
        Q_tilde[k] = 0;
        for (int j = 0; j < d_c; ++j)
            Q_tilde[k] += Q[h_q, k] * W_K[k, j];
    }
    
    // Step 2: Attention with compressed KV
    for (int t = 0; t < S; ++t) {
        // Load compressed latent
        float c_t[d_c];
        load_latent(c_cache, t, c_t);
        
        // Compute score: Q_tilde @ c_t
        float score = 0;
        for (int k = 0; k < d_c; ++k)
            score += Q_tilde[k] * c_t[k];
        
        // ... online softmax update ...
    }
}
```

### Memory Layout

**MLA KV cache layout:**
```cpp
// MLA: [B, d_c, S] instead of [B, H_kv, S, d_h]
// Contiguous in d_c dimension for coalesced loads

auto c_layout = make_layout(make_shape(B, d_c, S),
                            make_stride(d_c * S, S, 1));
```

## Numbers That Matter

| Model | Method | KV Cache (S=4096) | Reduction |
|-------|--------|-------------------|-----------|
| LLaMA-3 8B | GQA | 0.5 GB | 1x |
| DeepSeek-V2 | MLA | 0.25 GB | 2x |
| DeepSeek-V2 | MLA + multi-token | 0.03 GB | 16x |

## Common Interview Questions

**Q1: What is the absorb trick in MLA?**

<details>
<summary>Answer</summary>

The absorb trick fuses KV reconstruction with attention computation.

Instead of:
1. K = W_K @ c (reconstruct)
2. scores = Q @ K^T (attention)

Compute:
1. Q_tilde = Q @ W_K (absorb W_K into Q)
2. scores = Q_tilde @ c^T (attention with latent)

This eliminates the explicit reconstruction step and reduces memory traffic.
</details>

**Q2: How does MLA compare to GQA?**

<details>
<summary>Answer</summary>

| Aspect | GQA | MLA |
|--------|-----|-----|
| KV cache | H_kv × d_h per token | d_c per token |
| Compression | H_kv/H_q | d_c / (H_kv × d_h) |
| Additional FLOPs | None | W_K, W_V projection |
| Accuracy | Near-MHA | Comparable to GQA |

MLA trades compute for memory: additional projection FLOPs for reduced KV cache.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 03.2 (GQA)

**What this unlocks:**
- Module 06 (Quantization): Alternative approach to KV cache compression

**Next:** Run `attention_variants.py` to compare MHA, MQA, and GQA.
