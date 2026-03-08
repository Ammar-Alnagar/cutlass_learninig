"""
Module 06 — Attention Kernels
Exercise 03 — Fused MHA with Mixed Precision Inputs (INT8 KV Cache)

LEVEL: 2 (CuTe DSL custom kernel)

WHAT YOU'RE BUILDING:
  Mixed precision FMHA with INT8 KV cache — production pattern used in 
  vLLM, TensorRT-LLM for memory-efficient long-context inference. 
  Q/K/V computed in FP16, cached in INT8, attention in FP16.

OBJECTIVE:
  - Implement mixed precision attention (FP16 compute, INT8 cache)
  - Fuse INT8 dequantization into attention
  - Understand quantization error propagation
  - Compare mixed precision vs full precision accuracy

NOTE: Requires SM90+ (Hopper) for optimal INT8 Tensor Core performance
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
import math
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What accuracy loss do you expect from INT8 KV cache vs FP16?
#     Is it acceptable for LLM inference?

# Q2: Should Q be quantized too, or stay in FP16?
#     What's the trade-off?

# Q3: How does per-channel vs per-tensor quantization affect attention?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class MixedPrecisionConfig:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    compute_dtype: torch.dtype
    cache_dtype: torch.dtype


config = MixedPrecisionConfig(
    batch_size=32,
    num_heads=32,
    seq_len=512,
    head_dim=128,
    compute_dtype=torch.float16,
    cache_dtype=torch.int8,
)

device = torch.device("cuda")

print("=" * 60)
print("Fused MHA with Mixed Precision (INT8 KV Cache)")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Batch size:    {config.batch_size}")
print(f"  Num heads:     {config.num_heads}")
print(f"  Seq len:       {config.seq_len}")
print(f"  Head dim:      {config.head_dim}")
print(f"  Compute dtype: {config.compute_dtype}")
print(f"  Cache dtype:   {config.cache_dtype}")

B, H, S, D = config.batch_size, config.num_heads, config.seq_len, config.head_dim


# ==============================================================================
# QUANTIZATION FUNCTIONS
# ==============================================================================

def quantize_per_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Per-tensor INT8 quantization."""
    scale = tensor.abs().max() / 127.0
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.item()


def quantize_per_channel(tensor: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 quantization."""
    # Move quantization dim to last
    if dim != -1 and dim != tensor.dim() - 1:
        tensor = tensor.transpose(dim, -1)
    
    shape = tensor.shape
    flat = tensor.reshape(-1, shape[-1])
    scales = flat.abs().max(dim=0).values / 127.0
    scales = scales.clamp(min=1e-8)
    
    quantized = (tensor / scales).round().clamp(-128, 127).to(torch.int8)
    return quantized, scales


def dequantize_per_tensor(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize per-tensor INT8."""
    return quantized.to(torch.float32) * scale


def dequantize_per_channel(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize per-channel INT8."""
    # scales shape: [channels]
    return quantized.to(torch.float32) * scales


# ==============================================================================
# CREATE TENSORS
# ==============================================================================

# Create Q, K, V in FP16
Q_fp16 = torch.randn(B, H, S, D, dtype=config.compute_dtype, device=device)
K_fp16 = torch.randn(B, H, S, D, dtype=config.compute_dtype, device=device)
V_fp16 = torch.randn(B, H, S, D, dtype=config.compute_dtype, device=device)

# Quantize K and V to INT8 (different methods)
K_int8_pt, K_scale_pt = quantize_per_tensor(K_fp16)
V_int8_pt, V_scale_pt = quantize_per_tensor(V_fp16)

K_int8_pc, K_scales_pc = quantize_per_channel(K_fp16, dim=-1)  # Per-head-dim
V_int8_pc, V_scales_pc = quantize_per_channel(V_fp16, dim=-1)

print(f"\nQuantization:")
print(f"  Per-tensor scale:  K={K_scale_pt:.6f}, V={V_scale_pt:.6f}")
print(f"  Per-channel scales: K=[{K_scales_pc.min().item():.4f}, {K_scales_pc.max().item():.4f}]")
print(f"  Per-channel scales: V=[{V_scales_pc.min().item():.4f}, {V_scales_pc.max().item():.4f}]")

# Memory comparison
memory_fp16 = B * H * S * D * 2 * 2 / 1e6  # K + V in FP16
memory_int8 = B * H * S * D * 2 * 1 / 1e6  # K + V in INT8

print(f"\nKV Cache Memory:")
print(f"  FP16: {memory_fp16:.1f} MB")
print(f"  INT8: {memory_int8:.1f} MB ({100 * (1 - memory_int8/memory_fp16):.0f}% savings)")


# ==============================================================================
# REFERENCE ATTENTION
# ==============================================================================

def attention_ref(Q, K, V, scale: float):
    """Reference attention."""
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output


scale = 1.0 / math.sqrt(D)

# Full precision reference
output_fp16_ref = attention_ref(Q_fp16, K_fp16, V_fp16, scale)

# Mixed precision (INT8 KV, dequantize for compute)
K_dq_pt = dequantize_per_tensor(K_int8_pt, K_scale_pt)
V_dq_pt = dequantize_per_tensor(V_int8_pt, V_scale_pt)
output_mixed_pt = attention_ref(Q_fp16, K_dq_pt, V_dq_pt, scale)

K_dq_pc = dequantize_per_channel(K_int8_pc, K_scales_pc)
V_dq_pc = dequantize_per_channel(V_int8_pc, V_scales_pc)
output_mixed_pc = attention_ref(Q_fp16, K_dq_pc, V_dq_pc, scale)


# ==============================================================================
# FILL IN: Level 2 — Mixed Precision FMHA Kernel
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: CuTe DSL Mixed Precision FMHA Kernel")
print("=" * 60)

# TODO [HARD]: Implement mixed precision FMHA with fused dequantization
# HINT:
#   - Load INT8 K/V cache
#   - Dequantize on-the-fly (fuse into load)
#   - Compute attention in FP16
#   - Use cute.make_tiled_mma for FP16 × INT8 → FP32 MMA
# REF: cutlass/examples/python/CuTeDSL/fmha_mixed_input.py

# TODO: Define mixed precision FMHA kernel
# @cutlass.jit
# def mixed_precision_fmha(Q: cute.Tensor, K_int8: cute.Tensor, 
#                          V_int8: cute.Tensor, scales: cute.Tensor,
#                          O: cute.Tensor, scale: float):
#     """
#     Mixed precision FMHA with INT8 KV cache.
#     
#     - Q: FP16
#     - K_int8, V_int8: INT8 (quantized)
#     - scales: FP32 (per-channel or per-tensor)
#     - Compute: FP16
#     """
#     # Load INT8 K/V and dequantize
#     # K_fp16 = dequantize(K_int8, scales)
#     # V_fp16 = dequantize(V_int8, scales)
#     
#     # Standard attention in FP16
#     # scores = Q @ K_fp16.T * scale
#     # weights = softmax(scores)
#     # output = weights @ V_fp16
#     ...

# Placeholder kernel
@cutlass.jit
def mixed_precision_fmha(Q: cute.Tensor, K_int8: cute.Tensor,
                         V_int8: cute.Tensor, scales: cute.Tensor,
                         O: cute.Tensor, scale: float):
    """Placeholder mixed precision FMHA kernel."""
    pass

print(f"\nMixed precision FMHA kernel defined (placeholder)")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification: Accuracy Comparison")
print("=" * 60)

# Compare mixed precision to full precision
error_pt = (output_mixed_pt - output_fp16_ref).abs().div(output_fp16_ref.abs() + 1e-8).max().item()
error_pc = (output_mixed_pc - output_fp16_ref).abs().div(output_fp16_ref.abs() + 1e-8).max().item()

print(f"\nMax Relative Error vs FP16:")
print(f"  Per-tensor INT8:  {error_pt*100:.2f}%")
print(f"  Per-channel INT8: {error_pc*100:.2f}%")

print(f"\nAccuracy assessment:")
if error_pc < 0.01:
    print(f"  ✓ Per-channel: Excellent (< 1% error)")
elif error_pc < 0.05:
    print(f"  ✓ Per-channel: Good (< 5% error, acceptable for inference)")
else:
    print(f"  ⚠ Per-channel: May need calibration")

if error_pt < 0.01:
    print(f"  ✓ Per-tensor: Excellent (< 1% error)")
elif error_pt < 0.05:
    print(f"  ✓ Per-tensor: Good (< 5% error)")
else:
    print(f"  ⚠ Per-tensor: Consider per-channel")

print(f"\nRecommendation: {'Per-channel' if error_pc < error_pt else 'Per-tensor'} quantization")


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark_mixed_precision_fmha(Q, K_int8, V_int8, scales, scale,
                                   num_warmup=10, num_iters=100) -> float:
    """Benchmark mixed precision FMHA."""
    O = torch.zeros_like(Q)
    
    # Dequantize for simulation
    if scales.dim() == 0:
        K_dq = K_int8.to(torch.float32) * scales
        V_dq = V_int8.to(torch.float32) * scales
    else:
        K_dq = K_int8.to(torch.float32) * scales
        V_dq = V_int8.to(torch.float32) * scales
    
    # Warmup
    for _ in range(num_warmup):
        _ = attention_ref(Q, K_dq, V_dq, scale)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = attention_ref(Q, K_dq, V_dq, scale)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_full_precision_fmha(Q, K, V, scale,
                                  num_warmup=10, num_iters=100) -> float:
    """Benchmark full precision FMHA."""
    O = torch.zeros_like(Q)
    
    # Warmup
    for _ in range(num_warmup):
        _ = attention_ref(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = attention_ref(Q, K, V, scale)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: Mixed vs Full Precision")
print("=" * 60)

# Per-tensor quantized
mixed_pt_latency = benchmark_mixed_precision_fmha(
    Q_fp16, K_int8_pt, V_int8_pt, 
    torch.tensor([K_scale_pt, V_scale_pt], device=device),
    scale
)

# Full precision
full_latency = benchmark_full_precision_fmha(Q_fp16, K_fp16, V_fp16, scale)

print(f"\nResults:")
print(f"  Full precision (FP16): {full_latency:.3f} ms")
print(f"  Mixed precision (INT8): {mixed_pt_latency:.3f} ms")

if mixed_pt_latency > 0 and full_latency > 0:
    # Note: In simulation, mixed may be slower due to dequantization
    # On real hardware with fused dequant, mixed should be faster
    speedup = full_latency / mixed_pt_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    print(f"\n  Note: Real speedup requires fused dequantization kernel.")
    print(f"        Memory bandwidth savings: ~2×")


# ==============================================================================
# QUANTIZATION METHOD COMPARISON
# ==============================================================================

print("\n" + "=" * 60)
print("Quantization Method Comparison")
print("=" * 60)

# Test different quantization granularities
methods = {
    "Per-tensor": (K_int8_pt, V_int8_pt, error_pt),
    "Per-channel (head)": (K_int8_pc, V_int8_pc, error_pc),
}

print(f"\n{'Method':<25} {'Max Rel Error':<15} {'Recommendation'}")
print(f"{'-'*55}")

for method, (K_q, V_q, error) in methods.items():
    if error < 0.01:
        rec = "✓ Excellent"
    elif error < 0.05:
        rec = "✓ Good"
    else:
        rec = "⚠ Consider finer granularity"
    print(f"{method:<25} {error*100:<15.2f}% {rec}")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: INT8 KV cache accuracy loss?")
print(f"        Per-tensor:  {error_pt*100:.2f}%")
print(f"        Per-channel: {error_pc*100:.2f}%")
print("        Typical: 1-5% for per-channel, acceptable for inference")

print("\n    Q2: Should Q be quantized?")
print("        Answer: Usually no. Q is single token (decode) or")
print("                small batch (prefill). Memory savings minimal.")
print("                Keep Q in FP16 for accuracy.")

print("\n    Q3: Per-channel vs per-tensor for attention?")
print("        Per-channel: Better accuracy (handles scale variation)")
print("        Per-tensor:  Simpler, less overhead")
print("        Recommendation: Per-channel for K/V, per-tensor OK for short context")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics dram__throughput.sum,\\")
print(f"                smsp__inst_executed_pipe_tensor.sum \\")
print(f"        python ex03_fmha_mixed_input_FILL_IN.py")
print("\n    Look for:")
print("      - Reduced memory traffic (INT8 cache)")
print("      - INT8 Tensor Core utilization")
print("      - Dequantization overhead (should be minimal)")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How does vLLM use INT8 KV cache?")
print("    A: vLLM approach:")
print("       1. Quantize K/V to INT8 after computation")
print("       2. Store INT8 + per-channel scales in cache")
print("       3. Dequantize on-the-fly during attention")
print("       4. 2× memory savings, < 1% accuracy loss")
print("       5. Enables 2× longer contexts or 2× batch size")

print("\n    Q: What's the trade-off of KV cache quantization?")
print("    A: Pros:")
print("       - 2× memory savings")
print("       - 2× memory bandwidth efficiency")
print("       - Enables longer contexts")
print("       Cons:")
print("       - Quantization overhead (dequantization)")
print("       - Small accuracy loss (1-5%)")
print("       - Implementation complexity")

# C4: Production guidance
print("\nC4: Production Mixed Precision Tips")
print("    Recommended approach:")
print("      1. Q: FP16 (no quantization needed)")
print("      2. K/V: INT8 per-channel quantization")
print("      3. Scales: FP16 (stored with cache)")
print("      4. Dequantization: Fuse into attention kernel")
print("      5. Calibrate scales on representative data")
print("\n    When to avoid:")
print("      - Accuracy-critical applications")
print("      - Very short context (< 64 tokens)")
print("      - Models with extreme value ranges")

print("\n" + "=" * 60)
print("Exercise 03 Complete!")
print("=" * 60)
