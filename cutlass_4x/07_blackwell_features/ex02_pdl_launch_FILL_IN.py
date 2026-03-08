"""
Module 07 — Blackwell Features
Exercise 02 — Programmatic Dependent Launch (PDL)

LEVEL: 2 (CuTe DSL custom kernel)

WHAT YOU'RE BUILDING:
  Programmatic Dependent Launch (PDL) — Blackwell's new mechanism for 
  launching child kernels from parent kernels without CPU involvement.
  This enables dynamic parallelism for irregular workloads like MoE, 
  sparse attention, and adaptive computation.

OBJECTIVE:
  - Understand PDL pattern and use cases
  - Implement parent-child kernel launch
  - Compare PDL vs CPU launch overhead
  - Learn when dynamic parallelism is beneficial

NOTE: Requires CUDA 12.5+ and Blackwell GPU (SM100+)
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What is PDL? How does it differ from traditional dynamic parallelism?

# Q2: What are good use cases for PDL? When should you avoid it?

# Q3: What's the overhead of PDL vs CPU kernel launch?


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
print("Programmatic Dependent Launch (PDL)")
print("=" * 60)
print(f"\nGPU: {gpu_name} (SM{sm_version})")
print(f"Blackwell Support: {'✓ Yes' if blackwell_supported else '✗ No (requires SM100+)'}")

if not blackwell_supported:
    print("\n⚠️  PDL requires Blackwell (SM100) or later.")
    print("   This exercise will show placeholder results.")

# MoE-like configuration (irregular workload)
NUM_EXPERTS = 16
HIDDEN_SIZE = 2048
EXPERT_WIDTH = 4096

# Simulate imbalanced expert routing (some experts get more tokens)
torch.manual_seed(42)
tokens_per_expert = torch.randint(32, 512, (NUM_EXPERTS,), dtype=torch.int32)

dtype = torch.float16
device = torch.device("cuda")

print(f"\nMoE Configuration:")
print(f"  Num experts:     {NUM_EXPERTS}")
print(f"  Hidden size:     {HIDDEN_SIZE}")
print(f"  Expert width:    {EXPERT_WIDTH}")
print(f"\nTokens per expert (imbalanced):")
for i in range(NUM_EXPERTS):
    print(f"  Expert {i:2d}: {tokens_per_expert[i]:4d} tokens")

total_tokens = tokens_per_expert.sum().item()
print(f"\nTotal tokens: {total_tokens}")


# ==============================================================================
# CREATE EXPERT TENSORS
# ==============================================================================

# Create expert weights (same for all experts for simplicity)
expert_weights = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, EXPERT_WIDTH, 
                             dtype=dtype, device=device)

# Create input tokens (grouped by expert)
inputs = []
for i in range(NUM_EXPERTS):
    inp = torch.randn(tokens_per_expert[i].item(), HIDDEN_SIZE, 
                      dtype=dtype, device=device)
    inputs.append(inp)

# Reference outputs (one per expert)
reference_outputs = []
for i in range(NUM_EXPERTS):
    ref = torch.mm(inputs[i], expert_weights[i])
    reference_outputs.append(ref)


# ==============================================================================
# FILL IN: Level 2 — CuTe DSL PDL Kernel
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: CuTe DSL PDL Kernel")
print("=" * 60)

# TODO [HARD]: Implement PDL parent kernel that launches child GEMM kernels
# HINT:
#   - Parent kernel iterates over experts
#   - For each expert with work, launch child kernel via PDL
#   - Child kernel processes that expert's GEMM
#   - Use cutlass.pdl.launch() or similar API
# REF: cutlass/examples/python/CuTeDSL/pdl_launch.py

# TODO: Define child kernel (expert GEMM)
# @cutlass.cute.kernel
# def expert_gemm_kernel(A: cute.Tensor, B: cute.Tensor, 
#                        C: cute.Tensor, expert_id: int):
#     """Child kernel: Process single expert's GEMM."""
#     # Standard GEMM implementation
#     ...

# TODO: Define parent kernel (PDL launcher)
# @cutlass.cute.kernel  
# def moe_pdl_parent(expert_inputs: List[cute.Tensor],
#                    expert_weights: cute.Tensor,
#                    expert_outputs: List[cute.Tensor]):
#     """
#     Parent kernel that launches child kernels via PDL.
#     
#     For each expert:
#       1. Check if expert has tokens
#       2. Launch child kernel for that expert
#       3. Continue without CPU involvement
#     """
#     for expert_id in range(NUM_EXPERTS):
#         if tokens_per_expert[expert_id] > 0:
#             # Launch child kernel via PDL
#             # cutlass.pdl.launch(
#             #     expert_gemm_kernel,
#             #     args=(expert_inputs[expert_id], ...),
#             #     grid=...,
#             #     block=...
#             # )
#     ...

# Placeholder kernels
@cutlass.cute.kernel
def expert_gemm_kernel(A: cute.Tensor, B: cute.Tensor,
                       C: cute.Tensor, expert_id: int):
    """Placeholder child kernel."""
    pass


@cutlass.cute.kernel
def moe_pdl_parent(expert_inputs, expert_weights, expert_outputs):
    """Placeholder PDL parent kernel."""
    pass

print(f"\nPDL kernels defined (placeholder)")
print(f"  Parent: moe_pdl_parent (launches children)")
print(f"  Child:  expert_gemm_kernel (processes expert GEMM)")


# ==============================================================================
# RUN PDL KERNEL
# ==============================================================================

# Allocate outputs
expert_outputs = []
for i in range(NUM_EXPERTS):
    out = torch.zeros(tokens_per_expert[i].item(), EXPERT_WIDTH, 
                      dtype=dtype, device=device)
    expert_outputs.append(out)

# TODO: Launch PDL parent kernel
# moe_pdl_parent[(1,), (256,)](inputs, expert_weights, expert_outputs)

# Placeholder (use reference)
for i in range(NUM_EXPERTS):
    expert_outputs[i].copy_(torch.mm(inputs[i], expert_weights[i]))

print(f"\nPDL execution completed")
print(f"Processed {NUM_EXPERTS} experts")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

all_correct = True
for i in range(NUM_EXPERTS):
    is_correct = torch.allclose(expert_outputs[i], reference_outputs[i], 
                                 rtol=1e-1, atol=1e-1)
    if not is_correct:
        all_correct = False
        print(f"  Expert {i}: ✗ FAIL")

print(f"\nCorrectness check: {'✓ PASS' if all_correct else '✗ FAIL'}")


# ==============================================================================
# BENCHMARK: PDL vs CPU Launch
# ==============================================================================

def benchmark_pdl_launch(expert_inputs, expert_weights, expert_outputs,
                         num_warmup=10, num_iters=50) -> float:
    """Benchmark PDL launch (parent launches children)."""
    # Placeholder (in practice, launch PDL parent kernel)
    
    # Warmup
    for _ in range(num_warmup):
        for i in range(NUM_EXPERTS):
            if tokens_per_expert[i] > 0:
                expert_outputs[i].copy_(torch.mm(expert_inputs[i], expert_weights[i]))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        for i in range(NUM_EXPERTS):
            if tokens_per_expert[i] > 0:
                expert_outputs[i].copy_(torch.mm(expert_inputs[i], expert_weights[i]))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_cpu_sequential(expert_inputs, expert_weights, expert_outputs,
                             num_warmup=10, num_iters=50) -> float:
    """Benchmark CPU launching each expert kernel sequentially."""
    # Warmup
    for _ in range(num_warmup):
        for i in range(NUM_EXPERTS):
            if tokens_per_expert[i] > 0:
                # Simulate kernel launch overhead
                expert_outputs[i].copy_(torch.mm(expert_inputs[i], expert_weights[i]))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        for i in range(NUM_EXPERTS):
            if tokens_per_expert[i] > 0:
                # Each iteration has CPU launch overhead
                expert_outputs[i].copy_(torch.mm(expert_inputs[i], expert_weights[i]))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance: PDL vs CPU Launch")
print("=" * 60)

pdl_latency = benchmark_pdl_launch(inputs, expert_weights, expert_outputs)
cpu_latency = benchmark_cpu_sequential(inputs, expert_weights, expert_outputs)

print(f"\nResults:")
print(f"  CPU sequential launch: {cpu_latency:.3f} ms")
print(f"  PDL launch:            {pdl_latency:.3f} ms (estimated)")

if pdl_latency > 0 and cpu_latency > 0:
    speedup = cpu_latency / pdl_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    print(f"\n  Note: PDL eliminates CPU launch overhead.")
    print(f"        Benefit increases with more child kernels.")


# ==============================================================================
# PDL PATTERN EXPLANATION
# ==============================================================================

print("\n" + "=" * 60)
print("PDL Pattern Explanation")
print("=" * 60)

print("""
Traditional Dynamic Parallelism (CUDA):
  - cudaLaunchKernel from device code
  - High overhead (~10 μs per launch)
  - Limited use cases

Programmatic Dependent Launch (PDL):
  - cutlass.pdl.launch() from device code
  - Low overhead (~1 μs per launch)
  - Tight integration with CUTLASS

PDL Use Cases:
  1. MoE (Mixture of Experts)
     - Parent routes tokens, launches expert kernels
     
  2. Sparse Attention
     - Parent identifies active regions, launches attention kernels
     
  3. Adaptive Computation
     - Parent decides compute path, launches appropriate kernels
     
  4. Irregular Workloads
     - Parent balances load, launches variable-size kernels

When NOT to use PDL:
  - Regular, predictable workloads (use static launch)
  - Very small child kernels (overhead dominates)
  - Debugging complexity is concern
""")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: What is PDL?")
print("        Answer: Programmatic Dependent Launch.")
print("                Device code launches child kernels")
print("                with low overhead (~1 μs).")

print("\n    Q2: PDL vs traditional dynamic parallelism?")
print("        PDL: ~1 μs overhead, CUTLASS integrated")
print("        Traditional: ~10 μs overhead, general purpose")
print("        PDL is 10× faster for kernel launch")

print("\n    Q3: Good PDL use cases?")
print("        - MoE expert routing")
print("        - Sparse/irregular computation")
print("        - Adaptive compute paths")
print("        - Dynamic load balancing")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --set full --target-processes all \\")
print(f"        python ex02_pdl_launch_FILL_IN.py")
print("\n    Look for:")
print("      - Parent kernel launch (single)")
print("      - Child kernel launches (from device)")
print("      - Reduced CPU-GPU synchronization")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How does PDL help MoE implementation?")
print("    A: MoE with PDL:")
print("       1. Parent kernel receives all tokens")
print("       2. Routes tokens to experts")
print("       3. Launches expert kernels via PDL")
print("       4. All on GPU, no CPU involvement")
print("       Benefit: Low latency, high throughput")

print("\n    Q: What's the limitation of PDL?")
print("    A: PDL limitations:")
print("       - Blackwell only (SM100+)")
print("       - Limited to 1 level of nesting")
print("       - Child kernel must be known at compile time")
print("       - Debugging is more complex")

# C4: Production guidance
print("\nC4: Production PDL Tips")
print("    Use PDL when:")
print("      - Irregular workload (MoE, sparse)")
print("      - Need low-latency dispatch")
print("      - Targeting Blackwell GPUs")
print("\n    Avoid PDL when:")
print("      - Regular, predictable workload")
print("      - Pre-Blackwell deployment")
print("      - Simplicity is priority")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
