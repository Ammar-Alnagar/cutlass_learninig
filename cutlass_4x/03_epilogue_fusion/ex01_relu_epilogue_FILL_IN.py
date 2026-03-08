"""
Module 03 — Epilogue Fusion
Exercise 01 — ReLU Epilogue Fusion

LEVEL: 1 → 2 (High-level op → CuTe DSL)

WHAT YOU'RE BUILDING:
  Fused ReLU activation in GEMM epilogue — the exact pattern used in 
  TensorRT-LLM's quantized linear layers and MLP blocks. Fusing activation
  into the GEMM epilogue eliminates a separate kernel launch and global 
  memory round-trip.

OBJECTIVE:
  - Configure cutlass.op.Gemm with ReLU epilogue
  - Understand epilogue functor pattern
  - Measure performance gain from fusion
  - Verify numerical correctness
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: How much speedup do you expect from fusing ReLU into GEMM epilogue?
#     Consider: kernel launch overhead + memory bandwidth saved

# Q2: What's the arithmetic intensity of a standalone ReLU kernel?
#     Why is fusion especially important for element-wise ops?

# Q3: Does fusing ReLU change the numerical output vs separate ReLU?
#     Why or why not?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions (typical MLP hidden projection)
M, K, N = 512, 4096, 4096
dtype = torch.float16
device = torch.device("cuda")

# Allocate tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
C = torch.zeros(M, N, dtype=dtype, device=device)

# Bias vector (common in MLP layers)
bias = torch.randn(N, dtype=dtype, device=device)

# Reference: GEMM + separate ReLU
def gemm_then_relu(A, B, bias):
    """Reference: GEMM followed by separate ReLU."""
    output = torch.mm(A, B) + bias
    return torch.relu(output)


C_ref = gemm_then_relu(A, B, bias)

print("=" * 60)
print("ReLU Epilogue Fusion")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"\nReference: GEMM → Bias Add → ReLU (3 separate operations)")


# ==============================================================================
# FILL IN: Level 1 — High-Level Op with ReLU Epilogue
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.Gemm with epilogue)")
print("=" * 60)

# TODO [MEDIUM]: Configure GEMM with fused ReLU epilogue
# HINT:
#   - Use cutlass.op.Gemm with epilogue_functor parameter
#   - CUTLASS provides built-in functors: cutlass.epilogue.ReLU
#   - Or define custom functor (see Exercise 03)
# REF: cutlass/examples/python/CuTeDSL/epilogue_relu.py

# TODO: Create GEMM plan with ReLU epilogue
# plan_relu = cutlass.op.Gemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor,
#     epilogue_functor=cutlass.epilogue.ReLU,  # or custom functor
# )

# TODO: Run GEMM with bias (bias is fused into epilogue)
# plan_relu.run(A, B, C, bias=bias)

# Placeholder (replace with implementation)
plan_relu = None
C_fused = torch.zeros(M, N, dtype=dtype, device=device)

print(f"\nCUTLASS GEMM + ReLU (fused) completed")
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

def benchmark_gemm_relu_fused(plan, A, B, C, bias,
                              num_warmup=10, num_iters=100) -> float:
    """Benchmark fused GEMM + ReLU."""
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


def benchmark_gemm_then_relu_separate(A, B, bias, 
                                      num_warmup=10, num_iters=100) -> float:
    """Benchmark separate GEMM + ReLU."""
    C_temp = torch.zeros(M, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        C_temp.copy_(torch.relu(torch.mm(A, B) + bias))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        C_temp.copy_(torch.relu(torch.mm(A, B) + bias))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


# TODO [MEDIUM]: Benchmark both approaches
# fused_latency = benchmark_gemm_relu_fused(plan_relu, A, B, C_fused, bias)
# separate_latency = benchmark_gemm_then_relu_separate(A, B, bias)

fused_latency = 0.0
separate_latency = 0.0

print("\n" + "=" * 60)
print("Performance Comparison")
print("=" * 60)
print(f"\nResults:")
print(f"  Fused (GEMM+ReLU):     {fused_latency:.3f} ms")
print(f"  Separate (GEMM→ReLU):  {separate_latency:.3f} ms")

if fused_latency > 0 and separate_latency > 0:
    speedup = separate_latency / fused_latency
    improvement = (1 - fused_latency / separate_latency) * 100
    print(f"\n  Speedup:    {speedup:.2f}×")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Compute TFLOPS (for GEMM portion only)
    flops = 2 * M * N * K
    fused_tflops = flops / (fused_latency * 1e-3) / 1e12
    separate_tflops = flops / (separate_latency * 1e-3) / 1e12
    
    print(f"\n  Fused TFLOPS:     {fused_tflops:.1f}")
    print(f"  Separate TFLOPS:  {separate_tflops:.1f}")


# ==============================================================================
# ANALYSIS: Why Fusion Matters
# ==============================================================================

print("\n" + "=" * 60)
print("Analysis: Why Epilogue Fusion Matters")
print("=" * 60)

# Compute memory traffic for separate approach
bytes_per_element = 2  # FP16
output_elements = M * N

# Separate approach: GEMM writes to global mem, ReLU reads + writes
separate_memory_bytes = output_elements * bytes_per_element * 2  # Read + write for ReLU

# Fused approach: ReLU applied in registers, no extra memory traffic
fused_memory_bytes = 0  # ReLU happens in epilogue, no extra memory

print(f"\nMemory Traffic Analysis:")
print(f"  Separate approach: {separate_memory_bytes / 1e6:.1f} MB extra (ReLU read+write)")
print(f"  Fused approach:    {fused_memory_bytes} MB extra (in-register)")
print(f"  Memory saved:      {separate_memory_bytes / 1e6:.1f} MB per GEMM")

# Estimate bandwidth
if separate_latency > 0 and fused_latency > 0:
    delta_latency = (separate_latency - fused_latency) * 1e-3  # seconds
    if delta_latency > 0:
        effective_bandwidth = separate_memory_bytes / delta_latency / 1e9  # GB/s
        print(f"\n  Effective bandwidth saved: ~{effective_bandwidth:.0f} GB/s")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Expected speedup from ReLU fusion?")
if fused_latency > 0 and separate_latency > 0:
    print(f"        Actual speedup: {separate_latency/fused_latency:.2f}×")
print("        Typical: 1.05-1.2× for large matrices")
print("               Higher for smaller matrices (launch overhead)")

print("\n    Q2: Arithmetic intensity of standalone ReLU?")
print("        Answer: 1 op / 2 bytes (read) + 2 bytes (write) = 0.25 ops/byte")
print("                Extremely memory-bound! Fusion is critical.")

print("\n    Q3: Does fusion change numerical output?")
print("        Answer: No (if implemented correctly). ReLU is applied")
print("                to the same accumulated values, just in registers")
print("                instead of global memory.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command (verify fusion reduced global stores):")
print(f"    ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\\")
print(f"                smsp__inst_executed.sum \\")
print(f"        python ex01_relu_epilogue_FILL_IN.py")
print("\n    Look for:")
print("      - Reduced global memory stores for fused version")
print("      - Similar tensor core utilization")
print("      - Fewer total instructions (no separate ReLU kernel)")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: Why is epilogue fusion critical for inference performance?")
print("    A: Inference is often memory-bandwidth bound. Every fused")
print("       operation saves a global memory round-trip. For LLMs with")
print("       many layers, this compounds to significant speedup.")
print("       TensorRT-LLM fuses: GEMM + bias + activation + quantization.")

print("\n    Q: What operations can be fused into GEMM epilogue?")
print("    A: Element-wise ops that don't require reduction:")
print("       - Activations: ReLU, GELU, SiLU, SwiGLU")
print("       - Binary ops: add, multiply (bias, scaling)")
print("       - Quantization: clamp, round, cast")
print("       - LayerNorm (partial, with additional buffering)")

# C4: Production guidance
print("\nC4: Production Epilogue Fusion Tips")
print("    Always fuse:")
print("      - Bias addition (free, no reason not to)")
print("      - Activations in MLP blocks")
print("      - Output quantization (FP8/INT8)")
print("\n    Consider separate kernel when:")
print("      - Activation needs GEMM output for something else first")
print("      - Complex fusion exceeds register budget")
print("      - Debugging/verification needed")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
