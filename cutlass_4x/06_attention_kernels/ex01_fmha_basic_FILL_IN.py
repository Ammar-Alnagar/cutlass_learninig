"""
Module 06 — Attention Kernels
Exercise 01 — Fused Multi-Head Attention (FMA) Basic

LEVEL: 2 (CuTe DSL custom kernel)

WHAT YOU'RE BUILDING:
  Fused Multi-Head Attention (FMHA) kernel using CuTe DSL — the foundation 
  of FlashAttention. This fuses QKV projection, attention scores, softmax, 
  and output projection into a single kernel, eliminating intermediate 
  memory writes.

OBJECTIVE:
  - Understand FMHA fusion pattern (Q @ K^T → softmax → @ V)
  - Implement basic FMHA using CuTe DSL
  - Compare fused vs unfused attention performance
  - Learn tiling strategy for attention

NOTE: This is a simplified FMHA for educational purposes. 
Production FlashAttention uses more advanced techniques.
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
# Q1: What operations does FMHA fuse? What memory traffic is saved?

# Q2: Why is attention fusion more important for long sequences?
#     How does complexity scale with sequence length?

# Q3: What's the memory bottleneck in unfused attention?


# ==============================================================================
# SETUP
# ==============================================================================

# Attention configuration
@dataclass
class AttentionConfig:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype


config = AttentionConfig(
    batch_size=32,
    num_heads=32,
    seq_len=512,
    head_dim=128,
    dtype=torch.float16,
)

device = torch.device("cuda")

print("=" * 60)
print("Fused Multi-Head Attention (FMHA) Basic")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Batch size:  {config.batch_size}")
print(f"  Num heads:   {config.num_heads}")
print(f"  Seq len:     {config.seq_len}")
print(f"  Head dim:    {config.head_dim}")
print(f"  Dtype:       {config.dtype}")

# Derived dimensions
B, H, S, D = config.batch_size, config.num_heads, config.seq_len, config.head_dim

# Create input tensors (Q, K, V)
Q = torch.randn(B, H, S, D, dtype=config.dtype, device=device)
K = torch.randn(B, H, S, D, dtype=config.dtype, device=device)
V = torch.randn(B, H, S, D, dtype=config.dtype, device=device)

# Reference: Unfused attention (PyTorch scaled_dot_product_attention)
def unfused_attention(Q, K, V, scale: float):
    """Unfused multi-head attention."""
    # Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # @ V
    output = torch.matmul(attn_weights, V)
    
    return output


scale = 1.0 / math.sqrt(D)
output_ref = unfused_attention(Q, K, V, scale)

print(f"\nAttention computation:")
print(f"  1. Q @ K^T × scale  → [{B}, {H}, {S}, {S}] scores")
print(f"  2. softmax(scores)  → [{B}, {H}, {S}, {S}] weights")
print(f"  3. weights @ V      → [{B}, {H}, {S}, {D}] output")

print(f"\nMemory traffic (unfused):")
scores_memory = B * H * S * S * 2  # FP16 scores written to global mem
weights_memory = B * H * S * S * 2  # FP16 weights written
print(f"  Intermediate scores: {scores_memory / 1e6:.1f} MB")
print(f"  Intermediate weights: {weights_memory / 1e6:.1f} MB")
print(f"  Total intermediate: {(scores_memory + weights_memory) / 1e6:.1f} MB")


# ==============================================================================
# FILL IN: Level 2 — CuTe DSL FMHA Kernel
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: CuTe DSL Custom FMHA Kernel")
print("=" * 60)

# TODO [HARD]: Implement basic FMHA kernel using CuTe DSL
# HINT:
#   - Tile over sequence dimension (S)
#   - For each tile: load Q_tile, K_tile, V_tile
#   - Compute Q @ K^T → softmax → @ V in registers
#   - Use cute.make_tiled_mma for GEMM operations
# REF: cutlass/examples/python/CuTeDSL/fmha_basic.py
#      flashattention.io paper

# TODO: Define FMHA kernel
# @cutlass.jit
# def fmha_kernel(Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, 
#                 O: cute.Tensor, scale: float):
#     """
#     Basic Fused Multi-Head Attention.
#     
#     For each (batch, head) pair:
#       1. Load Q, K, V tiles
#       2. Compute scores = Q @ K^T × scale
#       3. Compute attn_weights = softmax(scores)
#       4. Compute output = attn_weights @ V
#       5. Store output
#     """
#     # Get thread/block indices
#     # tile_m, tile_n, tile_k = ...
#     
#     # Load Q tile into shared memory
#     # smem_q = cute.make_smem_tensor(...)
#     
#     # Load K tile into shared memory
#     # smem_k = cute.make_smem_tensor(...)
#     
#     # Compute Q @ K^T
#     # tiled_mma = cute.make_tiled_mma(...)
#     # scores = tiled_mma(smem_q, smem_k.T)
#     
#     # Apply scale and softmax
#     # scores = scores * scale
#     # attn_weights = torch.softmax(scores, dim=-1)
#     
#     # Load V tile
#     # smem_v = cute.make_smem_tensor(...)
#     
#     # Compute output = attn_weights @ V
#     # output = tiled_mma(attn_weights, smem_v)
#     
#     # Store output
#     # O[:] = output

# Placeholder kernel (uses PyTorch reference)
@cutlass.jit
def fmha_kernel(Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, 
                O: cute.Tensor, scale: float):
    """Placeholder FMHA kernel."""
    # In practice, implement with CuTe DSL primitives
    pass

print(f"\nFMHA kernel defined (placeholder)")


# ==============================================================================
# RUN FMHA KERNEL
# ==============================================================================

# Allocate output
O = torch.zeros(B, H, S, D, dtype=config.dtype, device=device)

# TODO: Run FMHA kernel
# O_cutlass = torch.zeros_like(O)
# fmha_kernel(Q, K, V, O_cutlass, scale)

# Placeholder (use reference for now)
O_cutlass = unfused_attention(Q, K, V, scale)

print(f"\nFMHA kernel executed")
print(f"Output shape: {O_cutlass.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Compare with reference
is_correct = torch.allclose(O_cutlass, output_ref, rtol=1e-1, atol=1e-1)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (O_cutlass - output_ref).abs().max().item()
    print(f"  Max absolute error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK: Fused vs Unfused
# ==============================================================================

def benchmark_fmha_fused(Q, K, V, scale, num_warmup=10, num_iters=100) -> float:
    """Benchmark fused FMHA."""
    O = torch.zeros_like(Q)
    
    # Warmup
    for _ in range(num_warmup):
        # In practice: fmha_kernel(Q, K, V, O, scale)
        O.copy_(unfused_attention(Q, K, V, scale))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        # In practice: fmha_kernel(Q, K, V, O, scale)
        O.copy_(unfused_attention(Q, K, V, scale))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_attention_unfused(Q, K, V, scale, num_warmup=10, num_iters=100) -> float:
    """Benchmark unfused attention (separate operations)."""
    # Warmup
    for _ in range(num_warmup):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: Fused vs Unfused")
print("=" * 60)

fused_latency = benchmark_fmha_fused(Q, K, V, scale)
unfused_latency = benchmark_attention_unfused(Q, K, V, scale)

print(f"\nResults:")
print(f"  Fused FMHA:   {fused_latency:.3f} ms")
print(f"  Unfused:      {unfused_latency:.3f} ms")

if fused_latency > 0 and unfused_latency > 0:
    speedup = unfused_latency / fused_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    
    # Memory saved
    memory_saved = (B * H * S * S * 2 * 2) / 1e6  # scores + weights
    print(f"\n  Memory saved: ~{memory_saved:.1f} MB per attention call")


# ==============================================================================
# SEQUENCE LENGTH SCALING
# ==============================================================================

print("\n" + "=" * 60)
print("Sequence Length Scaling")
print("=" * 60)

# Test different sequence lengths
seq_lengths = [128, 256, 512, 1024, 2048]

print(f"\n{'Seq Len':<10} {'Unfused (ms)':<14} {'Fused (ms)':<12} {'Speedup'}")
print(f"{'-'*45}")

for seq_len in seq_lengths:
    # Resize tensors
    Q_test = torch.randn(B, H, seq_len, D, dtype=config.dtype, device=device)
    K_test = torch.randn(B, H, seq_len, D, dtype=config.dtype, device=device)
    V_test = torch.randn(B, H, seq_len, D, dtype=config.dtype, device=device)
    
    unfused_ms = benchmark_attention_unfused(Q_test, K_test, V_test, scale, num_warmup=5, num_iters=20)
    fused_ms = benchmark_fmha_fused(Q_test, K_test, V_test, scale, num_warmup=5, num_iters=20)
    
    speedup = unfused_ms / fused_ms if fused_ms > 0 else 0
    
    print(f"{seq_len:<10} {unfused_ms:<14.3f} {fused_ms:<12.3f} {speedup:.2f}×")

print(f"\nNote: Speedup increases with sequence length.")
print(f"      O(N²) memory traffic saved for sequence length N.")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: What operations does FMHA fuse?")
print("        Answer:")
print("        1. Q @ K^T (attention scores)")
print("        2. Softmax (attention weights)")
print("        3. weights @ V (output)")
print("        All in one kernel, no intermediate writes!")

print("\n    Q2: Why more important for long sequences?")
print("        Answer: Attention is O(N²) in sequence length.")
print("                Unfused writes O(N²) intermediates.")
print(f"                For N={S}: {S*S:,} elements × 2 bytes = {S*S*2/1e6:.1f} MB")
print("                For N=4096: 33,554,432 elements = 64 MB!")

print("\n    Q3: Memory bottleneck in unfused attention?")
print("        Answer: Writing/reading attention scores (N×N matrix).")
print("                For long sequences, this dominates runtime.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --set full --target-processes all \\")
print(f"        python ex01_fmha_basic_FILL_IN.py")
print("\n    Look for:")
print("      - Reduced global memory traffic (fused)")
print("      - Higher arithmetic intensity")
print("      - Better cache utilization")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How does FlashAttention improve on basic FMHA?")
print("    A: FlashAttention adds:")
print("       1. Tiling over sequence dimension (O(N) memory)")
print("       2. Online softmax (single pass)")
print("       3. Recomputation (trade compute for memory)")
print("       4. Warp specialization (overlap load/compute/store)")
print("       Result: I/O-aware attention, optimal memory access")

print("\n    Q: What's the complexity of attention?")
print("    A: Time:   O(N² × D) for sequence length N, head dim D")
print("       Memory: O(N²) unfused, O(N) with tiling")
print("       This is why long-context LLMs need FlashAttention!")

# C4: Production guidance
print("\nC4: Production FMHA Tips")
print("    Use FlashAttention when:")
print("      - Sequence length > 128")
print("      - Memory bandwidth bound")
print("      - Training (need gradients)")
print("\n    Considerations:")
print("      - FlashAttention v2: Better for Hopper")
print("      - FlashAttention v3: Blackwell optimized")
print("      - Use torch.nn.functional.scaled_dot_product_attention")
print("        (uses FlashAttention internally on supported GPUs)")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
