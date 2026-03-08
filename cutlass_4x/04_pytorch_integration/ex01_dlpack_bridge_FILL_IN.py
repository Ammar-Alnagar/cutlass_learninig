"""
Module 04 — PyTorch Integration
Exercise 01 — DLPack Bridge: Zero-Copy torch ↔ cutlass Interop

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  Zero-copy tensor interchange between PyTorch and CUTLASS using DLPack.
  This is critical for integrating CUTLASS kernels into PyTorch workflows
  without unnecessary data movement.

OBJECTIVE:
  - Use from_dlpack to convert torch tensors to cutlass tensors
  - Understand memory ownership semantics
  - Verify zero-copy behavior
  - Build end-to-end torch → cutlass → torch pipeline
"""

import torch
import cutlass
from cutlass.cute.runtime import from_dlpack
import time
from dataclasses import dataclass
from typing import Tuple


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What happens to the original torch tensor after from_dlpack()?
#     Does it copy data or share memory?

# Q2: Can you modify the cutlass tensor and see changes in torch?
#     Why or why not?

# Q3: What's the performance cost of DLPack conversion?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions
M, K, N = 1024, 2048, 1024
dtype = torch.float16
device = torch.device("cuda")

# Create PyTorch tensors
A_torch = torch.randn(M, K, dtype=dtype, device=device)
B_torch = torch.randn(K, N, dtype=dtype, device=device)
C_torch = torch.zeros(M, N, dtype=dtype, device=device)

# Reference output (pure torch)
C_ref = torch.mm(A_torch, B_torch)

print("=" * 60)
print("DLPack Bridge: Zero-Copy torch ↔ cutlass")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"\nDLPack enables zero-copy tensor interchange:")
print("  torch.Tensor --from_dlpack()--> cutlass.Tensor")
print("  cutlass.Tensor --torch.from_dlpack()--> torch.Tensor")


# ==============================================================================
# FILL IN: Level 1 — DLPack Conversion
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: DLPack Conversion (Zero-Copy)")
print("=" * 60)

# TODO [EASY]: Convert torch tensors to cutlass using from_dlpack
# HINT:
#   - A_cutlass = from_dlpack(A_torch)
#   - This creates a view, not a copy
#   - Original torch tensor remains valid
# REF: cutlass/examples/python/CuTeDSL/dlpack_bridge.py

# TODO: Convert input tensors
# A_cutlass = from_dlpack(A_torch)
# B_cutlass = from_dlpack(B_torch)

# Placeholder (replace with implementation)
A_cutlass = A_torch
B_cutlass = B_torch

print(f"\n✓ Converted torch tensors to cutlass")
print(f"  A_cutlass shape: {A_cutlass.shape if hasattr(A_cutlass, 'shape') else 'N/A'}")
print(f"  B_cutlass shape: {B_cutlass.shape if hasattr(B_cutlass, 'shape') else 'N/A'}")


# TODO [EASY]: Verify zero-copy behavior
# HINT:
#   - Modify A_torch in place
#   - Check if A_cutlass reflects the change
#   - If zero-copy, changes should be visible

# Store original value for comparison
original_value = A_torch[0, 0].item()

# TODO: Modify torch tensor
# A_torch[0, 0] = 999.0

# TODO: Check if cutlass tensor sees the change
# cutlass_sees_change = (A_cutlass[0, 0] == 999.0) if hasattr(A_cutlass, '__getitem__') else False

# Placeholder test
cutlass_sees_change = False

# Restore original value
A_torch[0, 0] = original_value

print(f"\nZero-copy verification:")
print(f"  Modified A_torch[0,0] = 999.0")
print(f"  A_cutlass sees change: {'✓ Yes (zero-copy)' if cutlass_sees_change else '? (check implementation)'}")


# ==============================================================================
# FILL IN: Level 1 — Run CUTLASS GEMM via DLPack
# ==============================================================================

print("\n" + "=" * 60)
print("Running CUTLASS GEMM via DLPack")
print("=" * 60)

# TODO [EASY]: Run CUTLASS GEMM using DLPack-converted tensors
# HINT:
#   - Create cutlass.op.Gemm plan
#   - Run with cutlass tensors (from_dlpack)
#   - Convert result back to torch

# TODO: Create GEMM plan
# plan = cutlass.op.Gemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor,
# )

# TODO: Allocate output in cutlass
# C_cutlass = cutlass.zeros((M, N), dtype=cutlass.float16)

# TODO: Run GEMM
# plan.run(A_cutlass, B_cutlass, C_cutlass)

# TODO: Convert result back to torch
# C_torch_result = torch.from_dlpack(C_cutlass)

# Placeholder (replace with implementation)
C_torch_result = torch.zeros(M, N, dtype=dtype, device=device)

print(f"\n✓ CUTLASS GEMM completed via DLPack")
print(f"  Output shape: {C_torch_result.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

# TODO [EASY]: Verify correctness against torch reference
# is_correct = torch.allclose(C_torch_result, C_ref, rtol=..., atol=...)

is_correct = torch.allclose(C_torch_result, C_ref, rtol=1e-2, atol=1e-2)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (C_torch_result - C_ref).abs().max().item()
    print(f"  Max error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK: DLPack Overhead
# ==============================================================================

def benchmark_dlpack_conversion(tensor, num_iters=1000) -> float:
    """Benchmark DLPack conversion overhead."""
    # Warmup
    for _ in range(10):
        _ = from_dlpack(tensor)
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        converted = from_dlpack(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return (elapsed / num_iters) * 1e6  # microseconds


def benchmark_torch_gemm(A, B, C, num_warmup=10, num_iters=100) -> float:
    """Benchmark pure torch GEMM."""
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_cutlass_gemm_via_dlpack(A, B, C, num_warmup=10, num_iters=100) -> float:
    """Benchmark CUTLASS GEMM with DLPack conversion overhead."""
    # Convert once (not per iteration)
    A_cutlass = from_dlpack(A)
    B_cutlass = from_dlpack(B)
    C_cutlass = from_dlpack(C)
    
    # Create plan
    plan = cutlass.op.Gemm(
        element=cutlass.float16,
        layout=cutlass.LayoutType.RowMajor,
    )
    
    # Warmup
    for _ in range(num_warmup):
        plan.run(A_cutlass, B_cutlass, C_cutlass)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(A_cutlass, B_cutlass, C_cutlass)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: DLPack Overhead")
print("=" * 60)

# Benchmark DLPack conversion
dlpack_overhead_us = benchmark_dlpack_conversion(A_torch)
print(f"\nDLPack conversion overhead: {dlpack_overhead_us:.2f} μs per conversion")
print(f"  (One-time cost, amortized over kernel execution)")

# Benchmark GEMM
torch_latency = benchmark_torch_gemm(A_torch, B_torch, C_torch)
cutlass_latency = benchmark_cutlass_gemm_via_dlpack(A_torch, B_torch, C_torch)

print(f"\nGEMM Latency:")
print(f"  torch.mm:           {torch_latency:.3f} ms")
print(f"  CUTLASS (DLPack):   {cutlass_latency:.3f} ms")

if cutlass_latency > 0:
    speedup = torch_latency / cutlass_latency
    print(f"\n  Speedup: {speedup:.2f}×")


# ==============================================================================
# MEMORY OWNERSHIP SEMANTICS
# ==============================================================================

print("\n" + "=" * 60)
print("Memory Ownership Semantics")
print("=" * 60)

print("""
DLPack Memory Model:
  1. from_dlpack(tensor) creates a VIEW, not a copy
  2. Both tensors share the same underlying memory
  3. Modifications to one are visible in the other
  4. Original tensor remains valid and usable

  ┌─────────────┐      ┌─────────────┐
  │ torch.Tensor│      │cutlass.Tensor│
  │   (Python)  │      │   (Python)   │
  └──────┬──────┘      └──────┬──────┘
         │                    │
         └────────┬───────────┘
                  │
         ┌────────▼────────┐
         │  GPU Memory     │
         │  (shared data)  │
         └─────────────────┘

Important:
  - Don't free/resize original tensor while cutlass view is in use
  - DLPack conversion is O(1) - just creates descriptor
  - Data movement only happens if tensors are on different devices
""")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: What happens after from_dlpack()?")
print("        Answer: Creates a view (descriptor), not a copy.")
print("                Both tensors share GPU memory.")

print("\n    Q2: Can you modify and see changes?")
if cutlass_sees_change:
    print("        Answer: ✓ Yes - zero-copy confirmed!")
else:
    print("        Answer: Check implementation - should be yes for zero-copy.")

print("\n    Q3: DLPack performance cost?")
print(f"        Actual: ~{dlpack_overhead_us:.1f} μs per conversion")
print("        Negligible compared to kernel execution (ms scale)")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command (verify no extra memory copies):")
print(f"    ncu --metrics l2tex__t_bytes.sum,\\")
print(f"                gmem__transactions \\")
print(f"        python ex01_dlpack_bridge_FILL_IN.py")
print("\n    Look for:")
print("      - No unexpected memory transactions")
print("      - Same memory traffic as pure torch version")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: Why use DLPack instead of torch.cuda.FloatTensor()?")
print("    A: DLPack is framework-agnostic standard for tensor")
print("       interchange. Works with PyTorch, TensorFlow, JAX,")
print("       CuPy, etc. No data copy, just reinterprets existing")
print("       memory with different tensor descriptor.")

print("\n    Q: When would DLPack NOT be zero-copy?")
print("    A: When tensors are on different devices:")
print("       - CPU tensor → GPU cutlass = copy required")
print("       - Different GPUs (GPU0 → GPU1) = copy via PCIe/NVLink")
print("       Same device = always zero-copy.")

# C4: Production guidance
print("\nC4: Production DLPack Tips")
print("    Best practices:")
print("      1. Convert tensors once, reuse cutlass views")
print("      2. Keep original torch tensor alive during cutlass use")
print("      3. Convert back to torch only when needed")
print("      4. Use torch.from_dlpack() for cutlass→torch")
print("\n    Common patterns:")
print("      - PyTorch custom op with CUTLASS backend")
print("      - Mixed framework pipelines (torch → cutlass → jax)")
print("      - Zero-copy data loading for inference")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
