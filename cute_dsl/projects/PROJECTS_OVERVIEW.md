# Project 02 — Online Softmax

## Target Performance

| Metric | Target |
|--------|--------|
| Memory BW Utilization | >85% |
| Numerical Stability | Match PyTorch within 1e-5 |

## Algorithm: Numerically Stable One-Pass Softmax

```python
# Online softmax from "Online Normalizer Calculation for Softmax"
# Computes softmax in one pass without overflow

max_val = -inf
sum_exp = 0

for x in input:
    new_max = max(max_val, x)
    sum_exp = sum_exp * exp(max_val - new_max) + exp(x - new_max)
    max_val = new_max

softmax = exp(input - max_val) / sum_exp
```

## Files

- `softmax_naive_FILL_IN.py` — Baseline (unstable, two-pass)
- `softmax_online_FILL_IN.py` — Numerically stable one-pass
- `softmax_online_SOLUTION.py`
- `benchmark.py` — BW utilization vs PyTorch

## Job Relevance

- **FlashAttention**: Online softmax is core to attention computation
- **vLLM**: Softmax in every attention layer
- **NVIDIA Interview**: Common kernel optimization question

---

# Project 03 — Multi-Head Attention

## Target Performance

| Kernel | Target TFLOPS | vs cuBLAS |
|--------|---------------|-----------|
| Unfused MHA | 50% roofline | Baseline |
| Fused MHA | 70% roofline | 1.5× |

## Algorithm

```
# Unfused (baseline)
Q = X @ Wq  # GMEM
K = X @ Wk  # GMEM
V = X @ Wv  # GMEM
S = Q @ K.T / sqrt(d)  # GMEM
P = softmax(S)  # GMEM
O = P @ V  # GMEM

# Fused (target)
# Keep Q, K, V in SMEM
# Fuse QK^T + softmax + PV
```

## Files

- `mha_unfused_FILL_IN.py` — Baseline
- `mha_fused_FILL_IN.py` — Fused QK^T + softmax
- `mha_fused_SOLUTION.py`
- `benchmark.py`

---

# Project 04 — FlashAttention-2

## Paper Reference

[FlashAttention-2: Better Attention Modeling using Tiling and Recomputation](https://arxiv.org/abs/2307.08691)

## Algorithm

```
Algorithm 1: FlashAttention-2 (simplified)

for bj in range(ceil(d / Br)):
    for bk in range(ceil(d / Bc)):
        # Load Q, K, V tiles to SMEM
        Q_tile = Q[bj*Br:(bj+1)*Br, :]
        K_tile = K[bk*Bc:(bk+1)*Bc, :]
        V_tile = V[bk*Bc:(bk+1)*Bc, :]
        
        # QK^T
        S_tile = Q_tile @ K_tile.T
        
        # Online softmax
        P_tile = softmax(S_tile)
        
        # PV
        O_tile = P_tile @ V_tile
        
        # Accumulate
        O[bj*Br:(bj+1)*Br, bk*Bc:(bk+1)*Bc] += O_tile
```

## Target Performance

| GPU | Baseline FA2 | Target |
|-----|--------------|--------|
| A100 | 180 TFLOPS | 150+ TFLOPS |
| H100 | 550 TFLOPS | 450+ TFLOPS |

## Files

- `fa2_prefill_FILL_IN.py` — Tiled, causal, online softmax
- `fa2_prefill_SOLUTION.py`
- `fa2_decode_FILL_IN.py` — Single query, KV cache
- `fa2_decode_SOLUTION.py`
- `benchmark.py` — vs FlashAttention C++ baseline

---

# Project 05 — FlashAttention-3 (Warp-Specialized)

## Paper Reference

[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision](https://arxiv.org/abs/2310.03748)

## Key Optimizations

1. **Warp Specialization**: DMA warps load, MMA warps compute
2. **Asynchronous TMA**: Hopper's Tensor Memory Accelerator
3. **FP8 Support**: Optional low-precision mode

## Warp Layout

```
Total: 128 threads (4 warps)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  DMA Warp 0 │  MMA Warp 0 │  MMA Warp 1 │  MMA Warp 2 │
│  (load QKV) │  (QK^T)     │  (softmax)  │  (PV)       │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## Target Performance

| GPU | FA2 Baseline | FA3 Target |
|-----|--------------|------------|
| H100 | 550 TFLOPS | 700+ TFLOPS |

## Files

- `fa3_warp_specialized_FILL_IN.py`
- `fa3_warp_specialized_SOLUTION.py`
- `fa3_pingpong_pipeline_FILL_IN.py`
- `fa3_pingpong_pipeline_SOLUTION.py`
- `benchmark.py` — vs FA2 baseline

---

# Project 06 — Fused Attention Variants

## GQA (Grouped Query Attention)

```
# Llama-2-70B: 8 query heads, 2 KV heads
# Stride-0 broadcast eliminates redundant KV loads

for q_head in range(8):
    kv_head = q_head // 4  # Share KV heads
    Q = query_heads[q_head]
    K = kv_heads[kv_head]  # Broadcast load
    V = kv_heads[kv_head]  # Broadcast load
    ...
```

## MLA (Multi-head Latent Attention)

Compressed KV cache with latent dimension.

## Files

- `gqa_attention_FILL_IN.py`
- `mla_attention_FILL_IN.py`
- `sliding_window_attn_FILL_IN.py`
- `benchmark.py`

---

# Project 07 — Quantized GEMM

## Target Performance

| Precision | Target TFLOPS | vs FP16 |
|-----------|---------------|---------|
| INT8 | 2× | 2.0× |
| FP8 (E4M3) | 1.5× | 1.5× |

## Files

- `int8_gemm_FILL_IN.py`
- `fp8_gemm_FILL_IN.py`
- `fp8_gemm_SOLUTION.py`
- `benchmark.py`

---

# Project 08 — Benchmarks Master

## Files

- `roofline.py` — Auto roofline chart generator
- `compare_cutlass_cpp.py` — CuTe DSL vs C++ perf table
- `results/` — CSV + PNG benchmark outputs

## Roofline Chart Example

```
┌────────────────────────────────────────────────────┐
│                    Roofline Chart                  │
│                                                    │
│  1000 │                    ● H100 Peak            │
│       │                  ╱                        │
│   500 │                ╱  ● FA3 (700 TFLOPS)      │
│       │              ╱    ● FA2 (550 TFLOPS)      │
│   100 │            ╱      ● GEMM (234 TFLOPS)     │
│       │──────────╱────────────────────────────    │
│    10 │        ╱                                  │
│       └────────┴─────────┴─────────┴───────────── │
│            0.01    0.1      1       10    100     │
│                  Arithmetic Intensity (FLOP/byte) │
└────────────────────────────────────────────────────┘
```
