# Project 05 — FlashAttention-3 (Warp-Specialized)

## Paper Reference
[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision](https://arxiv.org/abs/2310.03748)

## Files
- `fa3_warp_specialized_FILL_IN.py` — Warp-specialized FA3
- `fa3_warp_specialized_SOLUTION.py`
- `fa3_pingpong_pipeline_FILL_IN.py` — Ping-pong pipeline
- `fa3_pingpong_pipeline_SOLUTION.py`
- `benchmark.py`

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
