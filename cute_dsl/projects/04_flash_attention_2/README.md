# Project 04 — FlashAttention-2

## Paper Reference
[FlashAttention-2: Better Attention Modeling using Tiling and Recomputation](https://arxiv.org/abs/2307.08691)

## Files
- `fa2_prefill_FILL_IN.py` — Tiled prefill with causal masking
- `fa2_prefill_SOLUTION.py`
- `fa2_decode_FILL_IN.py` — Single query decode
- `fa2_decode_SOLUTION.py`
- `benchmark.py`

## Target Performance
| GPU | Baseline FA2 | Target |
|-----|--------------|--------|
| A100 | 180 TFLOPS | 150+ TFLOPS |
| H100 | 550 TFLOPS | 450+ TFLOPS |

## Algorithm (Prefill)
```
for bj in range(ceil(seq_q / Br)):
    for bk in range(ceil(seq_k / Bc)):
        Q_tile = Q[bj*Br:(bj+1)*Br, :]
        K_tile = K[bk*Bc:(bk+1)*Bc, :]
        V_tile = V[bk*Bc:(bk+1)*Bc, :]
        
        S_tile = Q_tile @ K_tile.T / sqrt(d)
        
        # Causal mask
        if causal:
            S_tile[mask] = -inf
        
        P_tile = softmax(S_tile)
        O_tile = P_tile @ V_tile
        O[bj*Br:(bj+1)*Br, bk*Bc:(bk+1)*Bc] += O_tile
```
