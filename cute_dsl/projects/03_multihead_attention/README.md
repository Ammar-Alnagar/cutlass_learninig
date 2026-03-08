# Project 03 — Multi-Head Attention

## Files

- `mha_unfused_FILL_IN.py` — Baseline unfused MHA
- `mha_fused_FILL_IN.py` — Fused QK^T + softmax + PV
- `mha_fused_SOLUTION.py`
- `benchmark.py`

## Target Performance

| Kernel | Target TFLOPS | vs Unfused |
|--------|---------------|------------|
| Unfused MHA | 50% roofline | Baseline |
| Fused MHA | 70% roofline | 1.5× |

## Algorithm

```
# Unfused (baseline)
Q = X @ Wq
K = X @ Wk  
V = X @ Wv
S = Q @ K.T / sqrt(d)
P = softmax(S)
O = P @ V

# Fused (target)
# Keep intermediate results in registers/SMEM
# Fuse QK^T + softmax + PV into single kernel
```
