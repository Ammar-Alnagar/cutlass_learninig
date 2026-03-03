# Why Quantize: The Memory Bandwidth Wall

## What This Is

Quantization reduces numerical precision (FP32 → FP16 → INT8 → FP8) to reduce memory bandwidth and increase arithmetic intensity. For memory-bound operations like decode, this directly improves throughput.

## Why A Kernel Engineer Needs This

**You will implement quantized kernels that load INT8/FP8 data and dequantize on-the-fly.** This reduces HBM traffic and increases effective bandwidth.

## The Math

### Arithmetic Intensity Shift

**FP16 decode:**
- AI = 0.5 FLOPs/byte
- H100 peak: 295 FLOPs/byte
- Utilization: 0.17%

**INT8 decode:**
- AI = 1.0 FLOPs/byte (2x improvement)
- Utilization: 0.34% (still memory-bound, but 2x better)

**FP8 decode:**
- AI = 1.0 FLOPs/byte
- H100 native FP8 tensor cores: 4x FP16 throughput

### Quantization Formula

**INT8 quantization:**
$$x_{\text{int8}} = \text{round}\left(\frac{x_{\text{fp16}}}{s}\right) + z$$

where:
- $s$ = scale factor
- $z$ = zero-point (typically 0 for symmetric quantization)

**Dequantization:**
$$x_{\text{fp16}} \approx s \cdot x_{\text{int8}}$$

## Numbers That Matter

| Data Type | Bytes | Dynamic Range | H100 Tensor Core |
|-----------|-------|---------------|------------------|
| FP32 | 4 | 24 bits | 31 TFLOPs/s |
| FP16 | 2 | 11 bits | 989 TFLOPs/s |
| INT8 | 1 | 8 bits | 1979 TFLOPs/s |
| FP8 (E4M3) | 1 | ~5 bits | 1979 TFLOPs/s |
| FP8 (E5M2) | 1 | ~4 bits | 1979 TFLOPs/s |

## Common Interview Questions

**Q: Why does INT8 quantization improve decode throughput?**

<details>
<summary>Answer</summary>

Decode is memory-bound (AI = 0.5 FLOPs/byte for FP16).

INT8 reduces KV cache size by 2x:
- FP16: 2 bytes per element
- INT8: 1 byte per element

This doubles arithmetic intensity (AI = 1.0 FLOPs/byte) and effectively doubles memory bandwidth (same bytes, more elements).

Throughput improvement: 2x (memory-bound regime).
</details>
