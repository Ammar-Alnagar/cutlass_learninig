"""
Module 03 — Epilogue Fusion
Exercise 02 — Bias Add Epilogue Fusion

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  Fused bias addition in GEMM epilogue — the most common fusion pattern in 
  deep learning. Every linear layer with bias benefits from this. This is 
  the foundation for more complex epilogue functors.

OBJECTIVE:
  - Configure cutlass.op.Gemm with bias epilogue
  - Understand vectorized bias loading
  - Compare fused vs separate bias add
  - Learn bias broadcasting patterns
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: Bias add is a simple element-wise addition. Why fuse it into GEMM?
#     Hint: Think about memory bandwidth and kernel launch overhead

# Q2: What's the shape of the bias vector for GEMM [M, K] @ [K, N] → [M, N]?
#     How is bias broadcast across the batch dimension?

# Q3: Does fused bias add change the arithmetic intensity of GEMM?
#     Why or why not?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions (typical transformer linear layer)
M, K, N = 256, 2048, 8192  # e.g., attention output projection
dtype = torch.float16
device = torch.device("cuda")

# Allocate tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)

# Bias vector (one per output column)
bias = torch.randn(N, dtype=dtype, device=device)

# Reference: GEMM + separate bias add
def gemm_then_bias(A, B, bias):
    """Reference: GEMM followed by separate bias addition."""
    return torch.mm(A, B) + bias  # bias broadcasts across M dimension


C_ref = gemm_then_bias(A, B, bias)

print("=" * 60)
print("Bias Add Epilogue Fusion")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"Bias shape: {bias.shape} (broadcasts to [{M}, {N}])")
print(f"\nReference: GEMM → Bias Add (2 separate operations)")


# ==============================================================================
# FILL IN: Level 1 — High-Level Op with Bias Epilogue
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.Gemm with bias)")
print("=" * 60)

# TODO [EASY]: Configure GEMM with fused bias epilogue
# HINT:
#   - cutlass.op.Gemm accepts bias parameter in run() method
#   - Bias is automatically fused into epilogue
#   - No special epilogue_functor needed for simple bias
# REF: cutlass/examples/python/CuTeDSL/gemm_bias.py

# TODO: Create GEMM plan
# plan_bias = cutlass.op.Gemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor,
# )

# TODO: Run GEMM with bias (bias fused into epilogue)
# C_fused = torch.zeros(M, N, dtype=dtype, device=device)
# plan_bias.run(A, B, C_fused, bias=bias)

# Placeholder (replace with implementation)
plan_bias = None
C_fused = torch.zeros(M, N, dtype=dtype, device=device)

print(f"\nCUTLASS GEMM + Bias (fused) completed")
print(f"Output shape: {C_fused.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

# TODO [EASY]: Verify correctness against reference
# HINT: Use torch.allclose with fp16 tolerances
# is_correct = torch.allclose(C_fused, C_ref, rtol=..., atol=...)

is_correct = torch.allclose(C_fused, C_ref, rtol=1e-2, atol=1e-2)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (C_fused - C_ref).abs().max().item()
    print(f"Max absolute error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK: Fused vs Unfused
# ==============================================================================

def benchmark_gemm_bias_fused(plan, A, B, C, bias,
                              num_warmup=10, num_iters=100) -> float:
    """Benchmark fused GEMM + Bias."""
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


def benchmark_gemm_then_bias_separate(A, B, bias, 
                                      num_warmup=10, num_iters=100) -> float:
    """Benchmark separate GEMM + Bias."""
    C_temp = torch.zeros(M, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        C_temp.copy_(torch.mm(A, B) + bias)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        C_temp.copy_(torch.mm(A, B) + bias)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


# TODO [EASY]: Benchmark both approaches
# fused_latency = benchmark_gemm_bias_fused(plan_bias, A, B, C_fused, bias)
# separate_latency = benchmark_gemm_then_bias_separate(A, B, bias)

fused_latency = 0.0
separate_latency = 0.0

print("\n" + "=" * 60)
print("Performance Comparison")
print("=" * 60)
print(f"\nResults:")
print(f"  Fused (GEMM+Bias):     {fused_latency:.3f} ms")
print(f"  Separate (GEMM→Bias):  {separate_latency:.3f} ms")

if fused_latency > 0 and separate_latency > 0:
    speedup = separate_latency / fused_latency
    improvement = (1 - fused_latency / separate_latency) * 100
    print(f"\n  Speedup:    {speedup:.2f}×")
    print(f"  Improvement: {improvement:.1f}%")


# ==============================================================================
# MEMORY TRAFFIC ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("Memory Traffic Analysis")
print("=" * 60)

bytes_per_element = 2  # FP16
output_elements = M * N

# Separate approach: GEMM writes, bias add reads GEMM output + writes result
separate_memory_bytes = output_elements * bytes_per_element * 2  # Read + write

# Fused approach: bias loaded during epilogue, no extra memory traffic for output
# Bias is loaded once (N elements) vs output read+write (2*M*N elements)
fused_bias_bytes = N * bytes_per_element  # Only bias loaded

print(f"\nMemory Traffic:")
print(f"  Separate approach: {separate_memory_bytes / 1e6:.1f} MB (output read+write)")
print(f"  Fused approach:    {fused_bias_bytes / 1e6:.2f} MB (bias load only)")
print(f"  Memory saved:      {(separate_memory_bytes - fused_bias_bytes) / 1e6:.1f} MB")
print(f"  Reduction:         {(1 - fused_bias_bytes / separate_memory_bytes) * 100:.1f}%")


# ==============================================================================
# BIAS BROADCASTING PATTERNS
# ==============================================================================

print("\n" + "=" * 60)
print("Bias Broadcasting Patterns")
print("=" * 60)

# Different bias shapes for different use cases
bias_patterns = {
    "per-column": (N,),           # Standard: one bias per output column
    "per-row": (M,),              # Rare: one bias per batch row
    "scalar": (1,),               # Single bias value (broadcast to all)
    "per-element": (M, N),        # Full bias matrix (no broadcasting)
}

print("\nBias broadcasting options:")
for name, shape in bias_patterns.items():
    try:
        bias_test = torch.randn(shape, dtype=dtype, device=device)
        # Test if broadcasting works
        result = torch.mm(A, B) + bias_test
        print(f"  {name:15} {shape}: ✓ Works")
    except RuntimeError as e:
        print(f"  {name:15} {shape}: ✗ {str(e)[:50]}")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Why fuse bias add?")
print("        Answer: Saves output read+write (2*M*N elements).")
print("                Bias loaded once during epilogue instead.")
print(f"                For M={M}, N={N}: saves ~{2*M*N*2/1e6:.1f} MB traffic")

print("\n    Q2: Bias shape for [M,K]@[K,N]→[M,N]?")
print("        Answer: (N,) — one bias per output column")
print("                Broadcasts across M (batch) dimension")

print("\n    Q3: Does fusion change arithmetic intensity?")
print("        Answer: No — same FLOPs for GEMM.")
print("                But reduces memory traffic, improving")
print("                effective arithmetic intensity.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\")
print(f"                l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \\")
print(f"        python ex02_bias_add_epilogue_FILL_IN.py")
print("\n    Look for:")
print("      - Reduced global loads for fused version")
print("      - Reduced global stores for fused version")
print("      - Similar tensor core utilization")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: When would you NOT fuse bias into GEMM?")
print("    A: Rare cases:")
print("       - Bias is computed dynamically from GEMM output")
print("       - Need GEMM output for skip connection before bias")
print("       - Debugging (verify GEMM output separately)")
print("       - Bias is zero (no-op, but fusion overhead negligible)")

print("\n    Q: How does bias fusion interact with activation fusion?")
print("    A: They compose! Epilogue can be:")
print("       GEMM → bias → activation (all fused)")
print("       CUTLASS supports chained epilogue functors.")

# C4: Production guidance
print("\nC4: Production Bias Fusion Tips")
print("    Always fuse bias when:")
print("      - Bias is available at GEMM call time")
print("      - No intermediate operation needs GEMM output")
print("\n    Bias initialization:")
print("      - Initialize to zeros for stable training start")
print("      - Use small variance (e.g., 0.01) for fine-tuning")
print("      - Consider learned bias vs fixed (e.g., positional)")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
