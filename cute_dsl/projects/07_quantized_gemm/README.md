# Project 07 — Quantized GEMM

## Files
- `int8_gemm_FILL_IN.py` — INT8 GEMM with accumulation
- `fp8_gemm_FILL_IN.py` — FP8 (E4M3/E5M2) GEMM
- `fp8_gemm_SOLUTION.py`
- `benchmark.py`

## Target Performance
| Precision | Target TFLOPS | vs FP16 |
|-----------|---------------|---------|
| INT8 | 2× FP16 | 2.0× |
| FP8 (E4M3) | 1.5× FP16 | 1.5× |

## FP8 Formats
- **E4M3**: 4 exponent, 3 mantissa (±448 range, ~2 bits precision)
- **E5M2**: 5 exponent, 2 mantissa (±57344 range, ~1 bit precision)

## Quantization Pipeline
```
# Pre-processing
A_fp8 = quantize(A_fp16, format='e4m3')
B_fp8 = quantize(B_fp16, format='e4m3')

# GEMM
C_fp32 = A_fp8 @ B_fp8  # Accumulate in FP32

# Optional: dequantize output
C_fp16 = dequantize(C_fp32)
```
