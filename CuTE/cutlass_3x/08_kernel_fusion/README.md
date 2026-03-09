# Module 08 — Kernel Fusion

## Overview

Kernel fusion combines multiple operations into a single kernel, eliminating intermediate global memory traffic. CUTLASS 3.x enables fusion through:
- EVT (Epilogue Visitor Tree)
- Custom collective mainloops
- Fused attention patterns

## Fusion Opportunities

| Pattern | Operations Fused | Speedup |
|---------|------------------|---------|
| Linear + Activation | GEMM + ReLU/GELU | 1.3-1.5× |
| Linear + Bias | GEMM + Add | 1.5-2× |
| Linear + LayerNorm | GEMM + Normalize | 2-3× |
| QK Attention | GEMM + Softmax | 1.5-2× |
| RoPE + QK | Rotate + GEMM | 1.2-1.5× |

## LayerNorm Fusion

LayerNorm is memory-bound. Fusing with GEMM:
```
Unfused: GEMM → Store → Load → LayerNorm → Store
Fused:   GEMM → LayerNorm (registers) → Store
```

## RoPE Fusion

Rotary Position Embeddings (RoPE) for LLM:
```
Unfused: RoPE(Q) → GEMM(QK^T) → Softmax
Fused:   Fused RoPE+QK → Softmax
```

## Online Softmax

Numerically stable softmax in epilogue:
```cpp
// Online softmax: max + sum in single pass
max_val = max(max_val, x)
sum += exp(x - max_val)
output = exp(x - max_val) / sum
```

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_layernorm_gemm_FILL_IN.cu` | Fused LayerNorm + GEMM | Hard |
| `ex02_rope_attention_FILL_IN.cu` | RoPE fused into QK | Hard |
| `ex03_gemm_softmax_FILL_IN.cu` | Online softmax epilogue | Medium |
