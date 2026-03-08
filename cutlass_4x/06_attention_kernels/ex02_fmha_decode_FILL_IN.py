"""
Module 06 — Attention Kernels
Exercise 02 — Fused MHA Decode Path with KV Cache

LEVEL: 2 (CuTe DSL custom kernel)

WHAT YOU'RE BUILDING:
  Fused MHA for decode phase with KV cache — the exact pattern used in 
  LLM inference (vLLM, TGI, TensorRT-LLM). During decoding, we generate 
  one token at a time and cache K/V to avoid recomputation.

OBJECTIVE:
  - Understand decode vs prefill attention patterns
  - Implement KV cache integration
  - Fuse KV cache load + attention + output
  - Compare cached vs uncached decode performance

NOTE: This is for autoregressive decoding (token-by-token generation)
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
# Q1: What's the difference between prefill and decode attention?
#     Why is decode more memory-bound?

# Q2: How does KV cache save computation during decoding?
#     What's the memory cost?

# Q3: Why is INT8 quantization common for KV cache?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class DecodeConfig:
    batch_size: int
    num_heads: int
    head_dim: int
    seq_len_cached: int  # Length of cached context
    dtype: torch.dtype
    cache_dtype: torch.dtype  # Often INT8 for KV cache


config = DecodeConfig(
    batch_size=32,
    num_heads=32,
    head_dim=128,
    seq_len_cached=1024,  # Context length so far
    dtype=torch.float16,
    cache_dtype=torch.int8,  # Quantized KV cache
)

device = torch.device("cuda")

print("=" * 60)
print("Fused MHA Decode Path with KV Cache")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Batch size:      {config.batch_size}")
print(f"  Num heads:       {config.num_heads}")
print(f"  Head dim:        {config.head_dim}")
print(f"  Cached seq len:  {config.seq_len_cached}")
print(f"  Compute dtype:   {config.dtype}")
print(f"  Cache dtype:     {config.cache_dtype}")

B, H, D = config.batch_size, config.num_heads, config.head_dim
S_cached = config.seq_len_cached

# For decode, Q is single token (seq_len=1)
S_q = 1

# Create Q tensor (current token being generated)
Q = torch.randn(B, H, S_q, D, dtype=config.dtype, device=device)

# Create KV cache (from previous tokens)
# Shape: [batch, heads, seq_len_cached, head_dim]
K_cache = torch.randn(B, H, S_cached, D, dtype=config.dtype, device=device)
V_cache = torch.randn(B, H, S_cached, D, dtype=config.dtype, device=device)

# Quantized KV cache (INT8 for memory savings)
def quantize_kv(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize KV cache to INT8."""
    scale = tensor.abs().max() / 127.0
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


K_cache_int8, K_scale = quantize_kv(K_cache)
V_cache_int8, V_scale = quantize_kv(V_cache)

print(f"\nKV Cache:")
print(f"  FP16 cache: {B * H * S_cached * D * 2 / 1e6:.1f} MB")
print(f"  INT8 cache: {B * H * S_cached * D * 1 / 1e6:.1f} MB ({100 * (1 - 1/2):.0f}% savings)")


# ==============================================================================
# REFERENCE: DECODE ATTENTION
# ==============================================================================

def decode_attention_ref(Q, K_cache, V_cache, scale: float):
    """
    Reference decode attention.
    
    Q: [B, H, 1, D]  - current token
    K_cache: [B, H, S_cached, D]
    V_cache: [B, H, S_cached, D]
    """
    # Q @ K^T (only 1 query token vs all cached keys)
    scores = torch.matmul(Q, K_cache.transpose(-2, -1)) * scale  # [B, H, 1, S_cached]
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [B, H, 1, S_cached]
    
    # @ V
    output = torch.matmul(attn_weights, V_cache)  # [B, H, 1, D]
    
    return output


scale = 1.0 / math.sqrt(D)
output_ref = decode_attention_ref(Q, K_cache, V_cache, scale)

print(f"\nDecode attention computation:")
print(f"  1. Q @ K_cache^T × scale  → [{B}, {H}, {S_q}, {S_cached}] scores")
print(f"  2. softmax(scores)         → [{B}, {H}, {S_q}, {S_cached}] weights")
print(f"  3. weights @ V_cache       → [{B}, {H}, {S_q}, {D}] output")


# ==============================================================================
# FILL IN: Level 2 — CuTe DSL Decode FMHA Kernel
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: CuTe DSL Decode FMHA Kernel")
print("=" * 60)

# TODO [HARD]: Implement decode FMHA kernel with KV cache
# HINT:
#   - Q is single token (S_q = 1)
#   - K_cache, V_cache are full sequence
#   - Attention is 1 × S_cached (not S × S like prefill)
#   - Can fuse dequantization if cache is INT8
# REF: cutlass/examples/python/CuTeDSL/fmha_decode.py

# TODO: Define decode FMHA kernel
# @cutlass.jit
# def decode_fmha_kernel(Q: cute.Tensor, K_cache: cute.Tensor, 
#                        V_cache: cute.Tensor, O: cute.Tensor, 
#                        scale: float, k_scale: float, v_scale: float):
#     """
#     Fused MHA for decode with quantized KV cache.
#     
#     For each (batch, head):
#       1. Load Q (single token)
#       2. Load K_cache, V_cache tiles (with dequantization)
#       3. Compute Q @ K^T → softmax → @ V
#       4. Store output
#     """
#     # Decode-specific optimizations:
#     # - Q is small (1 token), can keep in registers
#     # - K_cache, V_cache are large, tile over sequence
#     # - Dequantize INT8 cache on-the-fly
#     ...

# Placeholder kernel
@cutlass.jit
def decode_fmha_kernel(Q: cute.Tensor, K_cache: cute.Tensor,
                       V_cache: cute.Tensor, O: cute.Tensor,
                       scale: float, k_scale: float, v_scale: float):
    """Placeholder decode FMHA kernel."""
    pass

print(f"\nDecode FMHA kernel defined (placeholder)")


# ==============================================================================
# RUN DECODE FMHA
# ==============================================================================

# Allocate output
O = torch.zeros(B, H, S_q, D, dtype=config.dtype, device=device)

# TODO: Run decode FMHA kernel
# O_cutlass = torch.zeros_like(O)
# decode_fmha_kernel(Q, K_cache_int8, V_cache_int8, O_cutlass, 
#                    scale, K_scale, V_scale)

# Placeholder (use reference)
# Dequantize for simulation
K_cache_dq = K_cache_int8.to(dtype=torch.float32) * K_scale
V_cache_dq = V_cache_int8.to(dtype=torch.float32) * V_scale
O_cutlass = decode_attention_ref(Q, K_cache_dq, V_cache_dq, scale)

print(f"\nDecode FMHA executed")
print(f"Output shape: {O_cutlass.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Compare with reference (accounting for quantization error)
is_correct = torch.allclose(O_cutlass, output_ref, rtol=1e-1, atol=1e-1)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (O_cutlass - output_ref).abs().max().item()
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  (Some error expected from INT8 quantization)")


# ==============================================================================
# BENCHMARK: Cached vs Uncached Decode
# ==============================================================================

def benchmark_decode_cached(Q, K_cache, V_cache, scale, 
                            num_warmup=10, num_iters=100) -> float:
    """Benchmark cached decode attention."""
    O = torch.zeros_like(Q)
    
    # Warmup
    for _ in range(num_warmup):
        O.copy_(decode_attention_ref(Q, K_cache, V_cache, scale))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        O.copy_(decode_attention_ref(Q, K_cache, V_cache, scale))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_decode_uncached(Q, K_full, V_full, scale, 
                              num_warmup=10, num_iters=100) -> float:
    """
    Benchmark uncached decode (recompute K/V from hidden state).
    
    This simulates not using KV cache.
    """
    O = torch.zeros_like(Q)
    
    # Warmup
    for _ in range(num_warmup):
        O.copy_(decode_attention_ref(Q, K_full, V_full, scale))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        O.copy_(decode_attention_ref(Q, K_full, V_full, scale))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: Cached vs Uncached Decode")
print("=" * 60)

# For fair comparison, uncached uses same K/V (but no quantization)
cached_latency = benchmark_decode_cached(Q, K_cache, V_cache, scale)
uncached_latency = benchmark_decode_uncached(Q, K_cache, V_cache, scale)

print(f"\nResults:")
print(f"  Cached (KV cache):   {cached_latency:.3f} ms")
print(f"  Uncached (recompute): {uncached_latency:.3f} ms")

# Note: In reality, uncached would be much slower due to K/V computation
print(f"\nNote: This compares cache access vs same tensors.")
print(f"      Real uncached includes K/V projection cost (~2-3× slower)")


# ==============================================================================
# KV CACHE MEMORY ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("KV Cache Memory Analysis")
print("=" * 60)

# Memory per token
bytes_per_element_fp16 = 2
bytes_per_element_int8 = 1

kv_cache_elements = B * H * S_cached * D * 2  # K + V

memory_fp16 = kv_cache_elements * bytes_per_element_fp16
memory_int8 = kv_cache_elements * bytes_per_element_int8

print(f"\nKV Cache Size (batch={B}, seq={S_cached}, heads={H}, dim={D}):")
print(f"  FP16: {memory_fp16 / 1e6:.1f} MB")
print(f"  INT8: {memory_int8 / 1e6:.1f} MB ({100 * (1 - memory_int8/memory_fp16):.0f}% savings)")

# Memory bandwidth during decode
# Each decode step reads full KV cache
decode_reads = memory_int8  # INT8 cache
print(f"\nMemory read per decode step: {decode_reads / 1e6:.1f} MB")

# Tokens per second (estimate)
if cached_latency > 0:
    tokens_per_sec = B / (cached_latency * 1e-3)
    print(f"Tokens per second: {tokens_per_sec/1000:.1f}K tokens/sec")
    print(f"Memory bandwidth: {decode_reads * tokens_per_sec / 1e9:.0f} GB/s")


# ==============================================================================
# CONTEXT LENGTH SCALING
# ==============================================================================

print("\n" + "=" * 60)
print("Context Length Scaling")
print("=" * 60)

# Test different context lengths
context_lengths = [256, 512, 1024, 2048, 4096]

print(f"\n{'Context':<10} {'Cache (MB)':<12} {'Latency (ms)':<14} {'Tokens/sec'}")
print(f"{'-'*50}")

for ctx_len in context_lengths:
    # Resize cache
    K_test = torch.randn(B, H, ctx_len, D, dtype=config.dtype, device=device)
    V_test = torch.randn(B, H, ctx_len, D, dtype=config.dtype, device=device)
    
    latency = benchmark_decode_cached(Q, K_test, V_test, scale, num_warmup=5, num_iters=20)
    
    cache_mb = B * H * ctx_len * D * 2 * 1 / 1e6  # INT8
    tokens_sec = B / (latency * 1e-3)
    
    print(f"{ctx_len:<10} {cache_mb:<12.1f} {latency:<14.3f} {tokens_sec/1000:.1f}K")

print(f"\nNote: Decode latency scales linearly with context length.")
print(f"      KV cache is memory-bandwidth bound.")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Prefill vs Decode attention?")
print("        Prefill: Process full sequence (S × S attention)")
print("        Decode:  Single token (1 × S_cached attention)")
print("        Decode is more memory-bound (read full cache for 1 token)")

print("\n    Q2: How does KV cache save computation?")
print("        Answer: Without cache, recompute K/V for all previous")
print("                tokens at each step. With cache, K/V computed")
print("                once and reused. Cost: O(N) memory per token.")

print("\n    Q3: Why INT8 for KV cache?")
print("        Answer: KV cache is memory-bandwidth bound.")
print("                INT8 gives 2× memory savings, minimal accuracy loss.")
print("                Dequantization is cheap vs memory transfer.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics dram__throughput.sum,\\")
print(f"                l2tex__t_bytes.sum \\")
print(f"        python ex02_fmha_decode_FILL_IN.py")
print("\n    Look for:")
print("      - High memory bandwidth utilization")
print("      - L2 cache hits (KV cache reuse)")
print("      - Compute underutilization (memory-bound)")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How does vLLM/TGI optimize KV cache?")
print("    A: Techniques:")
print("       1. PagedAttention (virtual memory for KV cache)")
print("       2. INT8/FP8 quantization")
print("       3. Cache eviction (for long contexts)")
print("       4. Prefix caching (common prompts)")
print("       5. Multi-GPU sharding")

print("\n    Q: What's the memory cost of KV cache for 70B model?")
print("    A: For 70B model (~64 hidden, 64 heads):")
print("       Per token: 64 × 64 × 128 × 2 × 2 bytes ≈ 2 MB")
print("       For 100K context: ~200 GB!")
print("       This is why context length is memory-limited.")

# C4: Production guidance
print("\nC4: Production Decode Tips")
print("    KV cache optimization:")
print("      1. Use INT8/FP8 quantization (2× memory savings)")
print("      2. Fuse dequantization into attention kernel")
print("      3. Use paged allocation (vLLM style)")
print("      4. Consider cache eviction for long contexts")
print("      5. Profile tokens/sec, not just latency")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
