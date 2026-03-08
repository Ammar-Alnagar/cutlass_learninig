# Project 06 — Fused Attention Variants

## Files
- `gqa_attention_FILL_IN.py` — Grouped Query Attention
- `mla_attention_FILL_IN.py` — Multi-head Latent Attention
- `sliding_window_attn_FILL_IN.py` — Sliding window attention
- `benchmark.py`

## GQA (Grouped Query Attention)

Used in Llama-2-70B, Llama-3. Multiple query heads share KV heads.

```
# Llama-2-70B: 8 query heads, 2 KV heads
# Stride-0 broadcast eliminates redundant KV loads

for q_head in range(8):
    kv_head = q_head // 4  # Share KV heads
    Q = query_heads[q_head]
    K = kv_heads[kv_head]  # Broadcast load
    V = kv_heads[kv_head]
    ...
```

## MLA (Multi-head Latent Attention)

Compressed KV cache with latent dimension for memory efficiency.

## Sliding Window Attention

Local attention with fixed window size (Mistral, Llama-3.1).
