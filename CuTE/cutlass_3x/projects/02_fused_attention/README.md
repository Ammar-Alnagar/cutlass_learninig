# Project 02 — Fused Attention (FA2/FA3 in CUTLASS)

## Overview

Implement Flash Attention 2 and 3 using CUTLASS 3.x collectives. This is the NVIDIA DL Software Engineer interview centerpiece.

## Requirements

### FA2 in CUTLASS
- Use CollectiveBuilder for QK^T and PV GEMMs
- Fused softmax in epilogue (or separate kernel)
- Target: Within 10% of FA2 reference

### FA3 in CUTLASS
- Warp-specialized collectives
- TMA-based loads for Q, K, V
- Producer/consumer warp split
- Target: Within 10% of FA3 reference

## Deliverables

| File | Description |
|------|-------------|
| `fa2_cutlass_FILL_IN.cu` | FA2 using CUTLASS collectives |
| `fa3_warp_spec_FILL_IN.cu` | FA3 warp-specialized |
| `benchmark.cu` | vs FA2/FA3 reference |

## Attention GEMM Pattern

```
Q: [B, H, S, D]  → Reshape: [B*H*S, D]
K: [B, H, S, D]  → Transpose: [D, S]
V: [B, H, S, D]  → Reshape: [B*H*S, D]

Step 1: QK^T = Q @ K^T  → [B*H*S, S] (attention scores)
Step 2: P = softmax(QK^T / sqrt(D))
Step 3: O = P @ V  → [B*H*S, D] (output)
```

## Success Criteria

- FA2 CUTLASS: Within 10% of FA2 reference
- FA3 CUTLASS: Within 10% of FA3 reference
- Demonstrate warp specialization advantage

## Interview Story

*"I implemented Flash Attention 3 using CUTLASS 3.x warp-specialized collectives. The key was configuring 8 producer warps for Q, K, V TMA loads and 24 consumer warps for QK^T MMA, softmax, and PV MMA. This achieved 95% of the FA3 reference performance."*
