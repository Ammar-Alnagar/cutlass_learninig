# KV Cache Quantization

## What This Is

KV cache quantization stores K, V in INT8 or FP8 instead of FP16. This reduces KV cache memory by 2-4x, enabling larger batch sizes.

## The Math

### INT8 KV Cache

**Quantize at write:**
$$K_{\text{int8}} = \text{round}(K_{\text{fp16}} / s_K), \quad V_{\text{int8}} = \text{round}(V_{\text{fp16}} / s_V)$$

**Dequantize at read:**
$$K_{\text{fp16}} \approx s_K \cdot K_{\text{int8}}, \quad V_{\text{fp16}} \approx s_V \cdot V_{\text{int8}}$$

**Scale per token or per block:**
- Per-token: One scale per token (better accuracy)
- Per-block: One scale per block of tokens (less overhead)

### Memory Savings

**LLaMA-3 8B, S=4096, FP16:**
- KV cache: 0.54 GB (GQA)

**INT8 KV cache:**
- KV cache: 0.27 GB (2x reduction)

**FP8 KV cache:**
- KV cache: 0.27 GB (2x reduction)
- H100 native FP8: No dequantization overhead

## The Kernel Implication

**Quantized KV cache kernel:**
```cuda
// Write to cache (quantize)
float scale = compute_scale(K_fp16);
int8_t K_int8 = round(K_fp16 / scale);
store_int8(k_cache_ptr, K_int8);
store_float(scale_ptr, scale);

// Read from cache (dequantize)
int8_t K_int8 = load_int8(k_cache_ptr);
float scale = load_float(scale_ptr);
float K_fp16 = scale * (float)K_int8;
```
