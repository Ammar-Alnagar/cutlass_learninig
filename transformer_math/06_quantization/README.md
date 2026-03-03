# Module 06: Quantization — INT8, FP8, KV Cache Compression

**One concept:** Quantization reduces memory bandwidth by using lower-precision data types. INT8/FP8 KV cache cuts memory by 2-4x with minimal accuracy loss.

**Job mapping:** NVIDIA inference & model optimization — you will implement INT8/FP8 GEMM and quantized KV cache kernels.

---

## Files in This Module

1. **01_why_quantize.md** — Memory bandwidth wall, arithmetic intensity shift.

2. **02_int8_weight_quant.md** — Per-tensor vs. per-channel, scale/zero-point math.

3. **03_kv_cache_quantization.md** — INT8 KV cache, accuracy implications.

4. **04_fp8_formats.md** — E4M3 vs. E5M2, H100 native support.

5. **quantization.py** — INT8 quantize/dequantize, simulate accuracy vs. compression.

---

## What You Must Be Able To Do After This Module

1. Compute scale and zero-point for INT8 quantization

2. Explain per-tensor vs. per-channel quantization tradeoffs

3. Choose FP8 format (E4M3 vs. E5M2) for a given use case

---

**Next:** `01_why_quantize.md` — the memory bandwidth wall
