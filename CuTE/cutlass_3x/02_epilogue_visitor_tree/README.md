# Module 02 — Epilogue Visitor Tree (EVT)

## Overview

The Epilogue Visitor Tree (EVT) is CUTLASS 3.x's most powerful fusion mechanism. It allows you to compose multiple elementwise operations into a single epilogue pass, eliminating intermediate global memory traffic.

## Why EVT Matters

Traditional GEMM epilogue:
```
GEMM → Global Store (accum) → Load (accum) → ReLU → Global Store
```

EVT-fused epilogue:
```
GEMM → ReLU (in registers) → Global Store
```

**Result:** 2× fewer global memory operations for fused activation.

## EVT Node Composition

Each EVT node is a composable operation:

```cpp
// Each EVT node is a composable operation:
using EpilogueOp = Sm80EVT
  EvtOpMultiply,              // D = alpha * accum
  EvtOpAdd,                   // D = D + beta * C  
  EvtOpActivation<ReLU>       // D = ReLU(D)
>;
// Result: fused alpha*accum + beta*C + ReLU in single epilogue pass
```

## Built-in EVT Operations

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `EvtOpMultiply` | Scale accumulator | Alpha scaling |
| `EvtOpAdd` | Add bias | Residual connection |
| `EvtOpActivation<ReLU>` | ReLU activation | Transformer MLP |
| `EvtOpActivation<GELU>` | GELU activation | LLM MLP |
| `EvtOpQuantize` | FP8/INT8 quantization | Quantized inference |
| `EvtOpDequantize` | Dequantize input | Weight quantization |
| `EvtOpClamp` | Clamp to range | Activation bounding |
| `EvtOpSigmoid` | Sigmoid | Gating |

## Custom EVT Node

```cpp
struct MyEVTNode {
  using Arguments = struct { float scale; };
  
  template<class Thr> 
  CUTLASS_DEVICE auto get(Thr const& thr, Arguments const& args) {
    return [&](auto accum) { return accum * args.scale; };
  }
};
```

## Profiling EVT Fusion

Verify fusion happened using Nsight Compute:

```bash
# Fewer global stores = fusion worked
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
          smsp__inst_executed.sum \
    ./epilogue_relu
```

Compare against unfused baseline — fused should show:
- Lower `l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum`
- Same `smsp__inst_executed.sum` (or slightly higher due to fused ops)

## Production Use Cases

| Use Case | EVT Composition | Framework |
|----------|-----------------|-----------|
| Linear + ReLU | Multiply + ReLU | TRT-LLM |
| QK Attention | Multiply + Softmax | FA2/FA3 |
| Quantized Linear | Dequant + Multiply + Quantize | DeepSpeed |
| LayerNorm + GEMM | Normalize + Multiply | Megatron |

## Exercises

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| `ex01_evt_basics_FILL_IN.cu` | Scale + bias fusion | Easy |
| `ex02_evt_relu_gelu_FILL_IN.cu` | Activation fusion | Medium |
| `ex03_evt_quantize_FILL_IN.cu` | FP8 output quantization | Medium |
| `ex04_evt_custom_FILL_IN.cu` | Write your own EVT node | Hard |
