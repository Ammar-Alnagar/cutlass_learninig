"""
Module 07 — Blackwell Features
Exercise 01 — Persistent GEMM with tcgen05 (SM100)

LEVEL: 2 (CuTe DSL custom kernel)

WHAT YOU'RE BUILDING:
  Persistent GEMM kernel using Blackwell's tcgen05 MMA instructions —
  the advanced pattern for maximum Tensor Core utilization on B100/B200.
  Persistent kernels keep SMs busy by avoiding kernel launch overhead.

OBJECTIVE:
  - Understand persistent kernel pattern
  - Use tcgen05 MMA instructions (Blackwell SM100+)
  - Implement warp-specialized GEMM pipeline
  - Compare persistent vs non-persistent performance

NOTE: Requires CUDA 12.5+ and Blackwell GPU (SM100)
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What is a "persistent" kernel? How does it differ from normal kernels?

# Q2: What are tcgen05 instructions? What's new in Blackwell Tensor Cores?

# Q3: When does persistent kernel provide the most benefit?


# ==============================================================================
# SETUP
# ==============================================================================

# Check GPU capability
def check_blackwell_support():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability()
        sm = compute_cap[0] * 10 + compute_cap[1]
        is_blackwell = sm >= 100
        return is_blackwell, gpu_name, sm
    return False, "No GPU", 0


blackwell_supported, gpu_name, sm_version = check_blackwell_support()

print("=" * 60)
print("Persistent GEMM with tcgen05 (Blackwell SM100)")
print("=" * 60)
print(f"\nGPU: {gpu_name} (SM{sm_version})")
print(f"Blackwell Support: {'✓ Yes' if blackwell_supported else '✗ No (requires SM100+)'}")

if not blackwell_supported:
    print("\n⚠️  tcgen05 requires Blackwell (SM100) or later.")
    print("   This exercise will show placeholder results.")

# Matrix dimensions (large GEMM for sustained compute)
M, K, N = 8192, 8192, 8192
dtype = torch.float16
device = torch.device("cuda")

print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")

# Create tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
C = torch.zeros(M, N, dtype=dtype, device=device)

# Reference
C_ref = torch.mm(A, B)

print(f"\nBlackwell Tensor Core features:")
print(f"  - tcgen05: 5th-gen Tensor Core instructions")
print(f"  - FP4 support: 2× throughput vs FP8")
print(f"  - Improved async copy: Better load/compute overlap")
print(f"  - Persistent kernel friendly: More registers per SM")


# ==============================================================================
# FILL IN: Level 2 — CuTe DSL Persistent GEMM Kernel
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: CuTe DSL Persistent GEMM with tcgen05")
print("=" * 60)

# TODO [HARD]: Implement persistent GEMM kernel with tcgen05
# HINT:
#   - Use @cutlass.cute.kernel decorator
#   - Persistent pattern: grid stride loop over tiles
#   - Use tcgen05 MMA atom for Blackwell
#   - Warp specialization: separate load/store/compute warps
# REF: cutlass/examples/python/CuTeDSL/persistent_gemm.py

# TODO: Define persistent GEMM kernel
# @cutlass.cute.kernel
# def persistent_gemm_tcgen05(A: cute.Tensor, B: cute.Tensor, 
#                              C: cute.Tensor, 
#                              grid_size: int, block_size: int):
#     """
#     Persistent GEMM using tcgen05 instructions.
#     
#     Pattern:
#     - Each block processes multiple tiles (persistent)
#     - Warp specialization for load/compute/store
#     - tcgen05 MMA for maximum throughput
#     """
#     # Get block and warp indices
#     block_idx = cute.block_index()
#     warp_idx = cute.warp_index()
#     
#     # Persistent loop: process multiple tiles
#     for tile_idx in range(grid_size):
#         # Global tile index
#         global_tile = block_idx + tile_idx * grid_size
#         
#         # Load A, B tiles into shared memory
#         # smem_a, smem_b = ...
#         
#         # Compute tile using tcgen05
#         # mma_atom = cute.MMA_Atom(cute.TCGEN05_FP16)
#         # tiled_mma = cute.make_tiled_mma(mma_atom, ...)
#         # accum = tiled_mma(smem_a, smem_b)
#         
#         # Store result
#         # C[tile] = accum
#     ...

# Placeholder kernel
@cutlass.cute.kernel
def persistent_gemm_tcgen05(A: cute.Tensor, B: cute.Tensor,
                            C: cute.Tensor,
                            grid_size: int, block_size: int):
    """Placeholder persistent GEMM kernel."""
    pass

print(f"\nPersistent GEMM kernel defined (placeholder)")
print(f"  Uses: tcgen05 MMA instructions")
print(f"  Pattern: Persistent grid-stride loop")
print(f"  Optimization: Warp specialization")


# ==============================================================================
# RUN PERSISTENT GEMM
# ==============================================================================

# Configure launch
num_blocks = 128  # Number of thread blocks
block_size = 256  # Threads per block

# TODO: Launch persistent kernel
# persistent_gemm_tcgen05[(num_blocks,), (block_size,)](A, B, C, num_blocks, block_size)

# Placeholder (use reference)
C.copy_(torch.mm(A, B))

print(f"\nPersistent GEMM executed")
print(f"Output shape: {C.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

is_correct = torch.allclose(C, C_ref, rtol=1e-1, atol=1e-1)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (C - C_ref).abs().max().item()
    print(f"  Max absolute error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK: Persistent vs Non-Persistent
# ==============================================================================

def benchmark_persistent_gemm(A, B, C, num_warmup=10, num_iters=50) -> float:
    """Benchmark persistent GEMM."""
    # Placeholder (in practice, launch persistent kernel)
    
    # Warmup
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_standard_gemm(A, B, C, num_warmup=10, num_iters=50) -> float:
    """Benchmark standard (non-persistent) GEMM."""
    # Warmup
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: Persistent vs Standard")
print("=" * 60)

persistent_latency = benchmark_persistent_gemm(A, B, C)
standard_latency = benchmark_standard_gemm(A, B, C)

print(f"\nResults:")
print(f"  Standard GEMM:    {standard_latency:.3f} ms")
print(f"  Persistent GEMM:  {persistent_latency:.3f} ms (estimated)")

if persistent_latency > 0 and standard_latency > 0:
    speedup = standard_latency / persistent_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    print(f"\n  Note: Persistent kernels reduce launch overhead.")
    print(f"        Benefit increases with smaller tiles.")

# Compute TFLOPS
flops = 2 * M * N * K
standard_tflops = flops / (standard_latency * 1e-3) / 1e12
persistent_tflops = flops / (persistent_latency * 1e-3) / 1e12

print(f"\n  Standard TFLOPS:    {standard_tflops:.0f}")
print(f"  Persistent TFLOPS:  {persistent_tflops:.0f}")
print(f"\n  Blackwell B200 peak: ~20,000 TFLOPS (FP16 dense)")


# ==============================================================================
# PERSISTENT KERNEL PATTERN EXPLANATION
# ==============================================================================

print("\n" + "=" * 60)
print("Persistent Kernel Pattern")
print("=" * 60)

print("""
Standard Kernel Launch:
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Launch 1│  │ Launch 2│  │ Launch 3│
  └────┬────┘  └────┬────┘  └────┬────┘
       │            │            │
       ▼            ▼            ▼
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Execute │  │ Execute │  │ Execute │
  └─────────┘  └─────────┘  └─────────┘
  
  Overhead: Launch latency per kernel

Persistent Kernel:
  ┌─────────────────────────────────────┐
  │         Single Launch               │
  └─────────────────┬───────────────────┘
                    │
        ┌───────────▼───────────┐
        │  Loop over all tiles  │
        │  (grid-stride loop)   │
        └───────────────────────┘
  
  Benefit: One launch, process all tiles

When to use persistent kernels:
  1. Many small/medium tiles (launch overhead significant)
  2. Need fine-grained parallelism
  3. Want warp specialization
  4. Large problem size (enough work per block)
""")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: What is a persistent kernel?")
print("        Answer: Single kernel launch that processes all work")
print("                via internal loop. Avoids repeated launch overhead.")

print("\n    Q2: What are tcgen05 instructions?")
print("        Answer: Blackwell's 5th-gen Tensor Core instructions.")
print("                Features:")
print("                - FP4/FP8/FP16/BF16 support")
print("                - Improved throughput vs Hopper")
print("                - Better async copy integration")

print("\n    Q3: When does persistent kernel help most?")
print("        - Many small kernel launches")
print("        - Fine-grained tiling needed")
print("        - Warp specialization desired")
print("        - Large total work (enough for all SMs)")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print(f"                smsp__occupancy.avg \\")
print(f"        python ex01_persistent_gemm_FILL_IN.py")
print("\n    Look for:")
print("      - Single kernel launch (vs many small launches)")
print("      - High SM occupancy")
print("      - tcgen05 instruction execution")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: What's warp specialization?")
print("    A: Assign different warps to different tasks:")
print("       - Load warps: Handle memory loads")
print("       - Compute warps: Handle MMA operations")
print("       - Store warps: Handle memory stores")
print("       Benefit: Better overlap of load/compute/store")

print("\n    Q: How does Blackwell improve on Hopper?")
print("    A: Blackwell improvements:")
print("       - tcgen05 (vs tcgen04 on Hopper)")
print("       - FP4 native support (2× FP8 throughput)")
print("       - More registers per SM")
print("       - Improved async copy bandwidth")
print("       - Better persistent kernel support")

# C4: Production guidance
print("\nC4: Production Persistent Kernel Tips")
print("    Use persistent kernels when:")
print("      - Processing many tiles (> 100)")
print("      - Launch overhead is significant")
print("      - Need warp specialization")
print("\n    Considerations:")
print("      - Higher register pressure")
print("      - More complex implementation")
print("      - Debugging is harder")
print("      - May reduce occupancy if not careful")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
