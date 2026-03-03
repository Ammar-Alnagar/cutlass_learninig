# FP8 Formats: E4M3 vs. E5M2

## What This Is

FP8 has two formats (IEEE 754):
- **E4M3:** 4 exponent bits, 3 mantissa bits (better precision, smaller range)
- **E5M2:** 5 exponent bits, 2 mantissa bits (larger range, less precision)

## The Math

### FP8 Representation

**E4M3:**
- Sign: 1 bit
- Exponent: 4 bits (bias = 7)
- Mantissa: 3 bits
- Range: ~[6e-5, 448]
- Special: NaN encoding for max exponent

**E5M2:**
- Sign: 1 bit
- Exponent: 5 bits (bias = 15)
- Mantissa: 2 bits
- Range: ~[6e-5, 57344]
- Special: Supports ±Inf, NaN

### When to Use Each

**E4M3 (better precision):**
- Weights (bounded range)
- Activations (known range)
- KV cache (normalized values)

**E5M2 (larger range):**
- Gradients (large dynamic range)
- Attention scores (variable range)
- General-purpose FP8

## H100 Native Support

**H100 FP8 tensor cores:**
- FP8 GEMM: 4x FP16 throughput
- Accumulation: FP32 or FP16
- Conversion: FP16 ↔ FP8 in hardware

**Warp-level instructions:**
```cuda
// FP8 tensor core GEMM
wmma::mma_sync(d, a_fp8, b_fp8, c_fp32);
```
