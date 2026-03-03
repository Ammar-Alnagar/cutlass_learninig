# INT8 Weight Quantization

## What This Is

INT8 quantization converts FP16 weights to 8-bit integers. Two schemes:
- **Per-tensor:** One scale for entire tensor
- **Per-channel:** One scale per output channel (better accuracy)

## The Math

### Symmetric Quantization

**Find scale:**
$$s = \frac{\max(|x|)}{127}$$

**Quantize:**
$$x_{\text{int8}} = \text{round}(x / s)$$

**Dequantize:**
$$\hat{x} = s \cdot x_{\text{int8}}$$

### Per-Channel Quantization

For weight matrix $W \in \mathbb{R}^{d \times d}$:

**Per-tensor:** One scale $s$ for all elements.

**Per-channel:** One scale $s_j$ per column $j$:
$$s_j = \frac{\max_i(|W_{ij}|)}{127}$$

**GEMM with quantized weights:**
$$Y = X \cdot W \approx X \cdot (S \cdot W_{\text{int8}}) = (X \cdot S) \cdot W_{\text{int8}}$$

where $S = \text{diag}(s_0, s_1, \ldots, s_{d-1})$.

## The Kernel Implication

**INT8 GEMM:**
```cuda
// Load INT8 weights, dequantize on-the-fly
int8_t w_int8 = load_int8(weight_ptr);
float w_fp16 = scale * (float)w_int8;

// FP16 GEMM
output += input * w_fp16;
```

**H100 tensor cores:** Native INT8 GEMM with accumulation in FP32 or FP16.
