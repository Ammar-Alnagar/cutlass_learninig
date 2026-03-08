"""
Module 03 — Epilogue Fusion
Exercise 03 — Custom Epilogue Fusion Config (EFC) with Python Functor

LEVEL: 1 → 2 (High-level op → CuTe DSL custom functor)

WHAT YOU'RE BUILDING:
  Custom Python epilogue functor using CUTLASS 4.3+ Epilogue Fusion Config 
  (EFC) API. This enables arbitrary fusion patterns like GELU, SiLU/SwiGLU,
  or custom activations without writing CUDA kernels.

OBJECTIVE:
  - Define custom epilogue functor in Python
  - Use @cutlass.jit to compile custom epilogue
  - Fuse complex activation (GELU/SiLU) into GEMM
  - Compare with separate kernel implementation
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
import math
from dataclasses import dataclass
from typing import Tuple, Callable


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: GELU is more complex than ReLU (involves erf approximation). 
#     Will fusion still provide speedup?

# Q2: What's the maximum complexity epilogue you can fuse before hitting
#     register pressure limits?

# Q3: Can you fuse multiple operations (e.g., bias + GELU + scaling)?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions (typical transformer MLP)
M, K, N = 512, 8192, 8192
dtype = torch.float16
device = torch.device("cuda")

# Allocate tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
bias = torch.randn(N, dtype=dtype, device=device)

# Reference: GEMM + bias + GELU (separate)
def gemm_bias_gelu_separate(A, B, bias):
    """Reference: GEMM → Bias → GELU (3 separate operations)."""
    x = torch.mm(A, B) + bias
    return torch.nn.functional.gelu(x)


# Reference: GEMM + bias + SiLU (separate)
def gemm_bias_silu_separate(A, B, bias):
    """Reference: GEMM → Bias → SiLU (3 separate operations)."""
    x = torch.mm(A, B) + bias
    return torch.nn.functional.silu(x)


C_ref_gelu = gemm_bias_gelu_separate(A, B, bias)
C_ref_silu = gemm_bias_silu_separate(A, B, bias)

print("=" * 60)
print("Custom Epilogue Fusion Config (EFC)")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"\nTesting custom epilogue functors:")
print("  1. GELU (Gaussian Error Linear Unit)")
print("  2. SiLU (Sigmoid Linear Unit / Swish)")


# ==============================================================================
# FILL IN: Level 1/2 — Custom Epilogue Functor
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1/2: Custom Epilogue Functor (Python + @cutlass.jit)")
print("=" * 60)

# TODO [HARD]: Define custom epilogue functor for GELU
# HINT:
#   - Define Python function that takes (accum, bias) and returns result
#   - Use @cutlass.jit to compile
#   - Pass to cutlass.op.Gemm via epilogue_functor parameter
# REF: cutlass/examples/python/CuTeDSL/custom_epilogue.py

# TODO: Define GELU epilogue functor
# @cutlass.jit
# def gelu_epilogue(accum, bias):
#     """Fused: (accum + bias) → GELU"""
#     x = accum + bias
#     # GELU approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
#     return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# TODO: Define SiLU epilogue functor  
# @cutlass.jit
# def silu_epilogue(accum, bias):
#     """Fused: (accum + bias) → SiLU"""
#     x = accum + bias
#     # SiLU: x * sigmoid(x)
#     return x * torch.sigmoid(x)

# Placeholder functors (replace with implementation)
def gelu_epilogue(accum, bias):
    return torch.nn.functional.gelu(accum + bias)

def silu_epilogue(accum, bias):
    return torch.nn.functional.silu(accum + bias)


# TODO: Create GEMM plans with custom epilogue
# plan_gelu = cutlass.op.Gemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor,
#     epilogue_functor=gelu_epilogue,
# )

# plan_silu = cutlass.op.Gemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor,
#     epilogue_functor=silu_epilogue,
# )

# TODO: Run GEMM with custom epilogue
# C_gelu_fused = torch.zeros(M, N, dtype=dtype, device=device)
# plan_gelu.run(A, B, C_gelu_fused, bias=bias)

# C_silu_fused = torch.zeros(M, N, dtype=dtype, device=device)
# plan_silu.run(A, B, C_silu_fused, bias=bias)

# Placeholder (replace with implementation)
plan_gelu = None
plan_silu = None
C_gelu_fused = torch.zeros(M, N, dtype=dtype, device=device)
C_silu_fused = torch.zeros(M, N, dtype=dtype, device=device)

print(f"\nCustom epilogue functors defined")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Verify GELU
is_gelu_correct = torch.allclose(C_gelu_fused, C_ref_gelu, rtol=1e-1, atol=1e-1)
print(f"GELU correctness: {'✓ PASS' if is_gelu_correct else '✗ FAIL'}")
if not is_gelu_correct:
    max_error = (C_gelu_fused - C_ref_gelu).abs().max().item()
    print(f"  Max error: {max_error:.6f}")

# Verify SiLU
is_silu_correct = torch.allclose(C_silu_fused, C_ref_silu, rtol=1e-1, atol=1e-1)
print(f"SiLU correctness: {'✓ PASS' if is_silu_correct else '✗ FAIL'}")
if not is_silu_correct:
    max_error = (C_silu_fused - C_ref_silu).abs().max().item()
    print(f"  Max error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK: Fused vs Unfused
# ==============================================================================

def benchmark_custom_epilogue(plan, A, B, C, bias,
                              num_warmup=10, num_iters=50) -> float:
    """Benchmark GEMM with custom epilogue."""
    if plan is None:
        return 0.0
    
    # Warmup
    for _ in range(num_warmup):
        plan.run(A, B, C, bias=bias)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(A, B, C, bias=bias)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: GELU Epilogue")
print("=" * 60)

# GELU benchmark
gelu_fused_latency = benchmark_custom_epilogue(plan_gelu, A, B, C_gelu_fused, bias)

# Separate GELU benchmark
def benchmark_separate_gelu(A, B, bias, num_warmup=10, num_iters=50):
    C_temp = torch.zeros(M, N, dtype=dtype, device=device)
    for _ in range(num_warmup):
        C_temp.copy_(torch.nn.functional.gelu(torch.mm(A, B) + bias))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C_temp.copy_(torch.nn.functional.gelu(torch.mm(A, B) + bias))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000

gelu_separate_latency = benchmark_separate_gelu(A, B, bias)

print(f"\nResults:")
print(f"  Fused (GEMM+Bias+GELU):    {gelu_fused_latency:.3f} ms")
print(f"  Separate (GEMM→Bias→GELU): {gelu_separate_latency:.3f} ms")

if gelu_fused_latency > 0 and gelu_separate_latency > 0:
    gelu_speedup = gelu_separate_latency / gelu_fused_latency
    print(f"\n  Speedup: {gelu_speedup:.2f}×")


print("\n" + "=" * 60)
print("Performance: SiLU Epilogue")
print("=" * 60)

# SiLU benchmark
silu_fused_latency = benchmark_custom_epilogue(plan_silu, A, B, C_silu_fused, bias)

# Separate SiLU benchmark
def benchmark_separate_silu(A, B, bias, num_warmup=10, num_iters=50):
    C_temp = torch.zeros(M, N, dtype=dtype, device=device)
    for _ in range(num_warmup):
        C_temp.copy_(torch.nn.functional.silu(torch.mm(A, B) + bias))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C_temp.copy_(torch.nn.functional.silu(torch.mm(A, B) + bias))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000

silu_separate_latency = benchmark_separate_silu(A, B, bias)

print(f"\nResults:")
print(f"  Fused (GEMM+Bias+SiLU):    {silu_fused_latency:.3f} ms")
print(f"  Separate (GEMM→Bias→SiLU): {silu_separate_latency:.3f} ms")

if silu_fused_latency > 0 and silu_separate_latency > 0:
    silu_speedup = silu_separate_latency / silu_fused_latency
    print(f"\n  Speedup: {silu_speedup:.2f}×")


# ==============================================================================
# ADVANCED: Chained Epilogue Operations
# ==============================================================================

print("\n" + "=" * 60)
print("Advanced: Chained Epilogue Operations")
print("=" * 60)

# TODO [HARD]: Define epilogue with multiple fused operations
# Example: GEMM + bias + scaling + activation
# HINT: Chain operations in the epilogue functor

# TODO: Define scaled GELU epilogue
# scale_factor = 2.0
# @cutlass.jit
# def scaled_gelu_epilogue(accum, bias, scale):
#     """Fused: GELU((accum + bias) * scale)"""
#     x = (accum + bias) * scale
#     return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

print("\nChained epilogue pattern:")
print("  GEMM → bias → scale → activation (all fused)")
print("  (Implementation requires CUTLASS 4.3+ EFC)")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Does GELU fusion still provide speedup?")
if gelu_fused_latency > 0 and gelu_separate_latency > 0:
    print(f"        Actual GELU speedup: {gelu_speedup:.2f}×")
print("        Expected: Yes, but less than ReLU (GELU has more compute)")

print("\n    Q2: Maximum epilogue complexity?")
print("        Answer: Limited by register count.")
print("                GELU/SiLU fit comfortably.")
print("                Very complex functions may need separate kernel.")

print("\n    Q3: Can you fuse multiple operations?")
print("        Answer: Yes! CUTLASS 4.3+ EFC supports chaining.")
print("                Example: GEMM + bias + GELU + scaling")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics smsp__inst_executed.sum,\\")
print(f"                l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \\")
print(f"        python ex03_efc_custom_epilogue_FILL_IN.py")
print("\n    Look for:")
print("      - Fewer instructions for fused version")
print("      - Reduced global memory traffic")
print("      - Similar tensor core utilization")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: Why is SiLU/SwiGLU popular in modern LLMs?")
print("    A: SiLU (used in LLaMA, PaLM) provides:")
print("       - Smooth gradients (unlike ReLU)")
print("       - Better expressiveness (non-monotonic)")
print("       - Gating mechanism in SwiGLU variant")
print("       Fusing into GEMM is critical for performance.")

print("\n    Q: How does EFC compare to writing custom CUDA kernels?")
print("    A: EFC advantages:")
print("       - Python API (faster iteration)")
print("       - Auto-tuning support")
print("       - Portable across GPU architectures")
print("       Custom CUDA advantages:")
print("       - Full control over optimization")
print("       - Can exceed EFC performance limits")

# C4: Production guidance
print("\nC4: Production Custom Epilogue Tips")
print("    Use EFC for:")
print("      - Standard activations (GELU, SiLU, ReLU)")
print("      - Custom but simple element-wise ops")
print("      - Rapid prototyping")
print("\n    Use custom CUDA kernel for:")
print("      - Complex reductions in epilogue")
print("      - Non-element-wise operations")
print("      - Maximum performance (hand-tuned)")

print("\n" + "=" * 60)
print("Exercise 03 Complete!")
print("=" * 60)
