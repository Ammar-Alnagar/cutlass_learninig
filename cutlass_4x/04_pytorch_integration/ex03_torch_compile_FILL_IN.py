"""
Module 04 — PyTorch Integration
Exercise 03 — torch.compile Integration with CUTLASS Kernels

LEVEL: 1 → 2 (High-level op → torch.compile custom backend)

WHAT YOU'RE BUILDING:
  torch.compile integration for CUTLASS kernels — enabling PyTorch 2.0's 
  compilation to fuse your custom CUTLASS ops into the compiled graph.
  This is how you get end-to-end optimization with custom kernels.

OBJECTIVE:
  - Use @torch.compile decorator with CUTLASS ops
  - Understand torch.compile's Inductor backend
  - Register CUTLASS as custom backend operation
  - Measure compilation overhead vs runtime gain
"""

import torch
import cutlass
from cutlass.cute.runtime import from_dlpack
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: Will torch.compile speed up or slow down CUTLASS kernel execution?
#     Consider: compilation overhead vs kernel fusion benefits

# Q2: What's the difference between mode='reduce-overhead' and mode='max-autotune'?
#     Which is better for CUTLASS integration?

# Q3: Can torch.compile fuse multiple CUTLASS ops together?
#     Or does it treat them as opaque boundaries?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions (typical MLP layer)
M, K, N = 512, 4096, 4096
dtype = torch.float16
device = torch.device("cuda")

# Test tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
C = torch.randn(N, M, dtype=dtype, device=device)  # For chain

print("=" * 60)
print("torch.compile Integration with CUTLASS")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"\ntorch.compile modes:")
print("  - default: General optimization")
print("  - reduce-overhead: Minimize kernel launch overhead")
print("  - max-autotune: Aggressive operator fusion")


# ==============================================================================
# CUTLASS GEMM WRAPPER
# ==============================================================================

def cutlass_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    CUTLASS GEMM wrapper for torch.compile.
    
    TODO [MEDIUM]: Implement actual CUTLASS GEMM
    HINT: Use from_dlpack for zero-copy conversion
    """
    # Placeholder (replace with CUTLASS implementation)
    # A_cutlass = from_dlpack(A)
    # B_cutlass = from_dlpack(B)
    # C_cutlass = cutlass.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    # plan = cutlass.op.Gemm(element=cutlass.float16, layout=cutlass.LayoutType.RowMajor)
    # plan.run(A_cutlass, B_cutlass, C_cutlass)
    # return torch.from_dlpack(C_cutlass)
    
    return torch.mm(A, B)  # Placeholder


# ==============================================================================
# FILL IN: Level 1 — torch.compile with CUTLASS
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: torch.compile with CUTLASS Ops")
print("=" * 60)

# TODO [MEDIUM]: Create compiled function using torch.compile
# HINT:
#   - Use @torch.compile decorator or torch.compile() function
#   - Try different modes: default, reduce-overhead, max-autotune
#   - Measure compilation time vs runtime benefit
# REF: cutlass/examples/python/CuTeDSL/torch_compile.py

# Define function to compile
def gemm_function(A, B):
    """Simple GEMM function."""
    return cutlass_gemm(A, B)


# TODO: Compile with different modes
# compiled_default = torch.compile(gemm_function, mode='default')
# compiled_reduce = torch.compile(gemm_function, mode='reduce-overhead')
# compiled_autotune = torch.compile(gemm_function, mode='max-autotune')

# Placeholder (replace with torch.compile)
compiled_default = gemm_function
compiled_reduce = gemm_function
compiled_autotune = gemm_function

print(f"\n✓ Compiled functions created")
print(f"  - compiled_default (mode='default')")
print(f"  - compiled_reduce (mode='reduce-overhead')")
print(f"  - compiled_autotune (mode='max-autotune')")


# ==============================================================================
# TESTING: First Run (Includes Compilation)
# ==============================================================================

print("\n" + "=" * 60)
print("Testing: First Run (Includes Compilation)")
print("=" * 60)

def measure_first_run(compiled_fn, A, B):
    """Measure first run time (includes compilation)."""
    start = time.perf_counter()
    result = compiled_fn(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000, result


# Test each mode
print("\nFirst run latency (includes compilation):")

default_time, result_default = measure_first_run(compiled_default, A, B)
print(f"  default:         {default_time:.2f} ms")

reduce_time, result_reduce = measure_first_run(compiled_reduce, A, B)
print(f"  reduce-overhead: {reduce_time:.2f} ms")

autotune_time, result_autotune = measure_first_run(compiled_autotune, A, B)
print(f"  max-autotune:    {autotune_time:.2f} ms")

# Reference (no compilation)
torch_result = torch.mm(A, B)
print(f"\n  torch.mm (ref):  N/A (no compilation)")


# ==============================================================================
# TESTING: Subsequent Runs (After Compilation)
# ==============================================================================

print("\n" + "=" * 60)
print("Testing: Subsequent Runs (After Compilation)")
print("=" * 60)

def benchmark_compiled(compiled_fn, A, B, num_warmup=5, num_iters=50) -> float:
    """Benchmark compiled function (after compilation)."""
    # Warmup (ensures compilation is done)
    for _ in range(num_warmup):
        _ = compiled_fn(A, B)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = compiled_fn(A, B)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\nSteady-state latency (after compilation):")

default_latency = benchmark_compiled(compiled_default, A, B)
print(f"  default:         {default_latency:.3f} ms")

reduce_latency = benchmark_compiled(compiled_reduce, A, B)
print(f"  reduce-overhead: {reduce_latency:.3f} ms")

autotune_latency = benchmark_compiled(compiled_autotune, A, B)
print(f"  max-autotune:    {autotune_latency:.3f} ms")

# Reference
def benchmark_torch_mm(A, B, num_warmup=5, num_iters=50) -> float:
    C = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000

torch_latency = benchmark_torch_mm(A, B)
print(f"\n  torch.mm (ref):  {torch_latency:.3f} ms")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Check all results match reference
default_correct = torch.allclose(result_default, torch_result, rtol=1e-1, atol=1e-1)
reduce_correct = torch.allclose(result_reduce, torch_result, rtol=1e-1, atol=1e-1)
autotune_correct = torch.allclose(result_autotune, torch_result, rtol=1e-1, atol=1e-1)

print(f"  default correctness:         {'✓ PASS' if default_correct else '✗ FAIL'}")
print(f"  reduce-overhead correctness: {'✓ PASS' if reduce_correct else '✗ FAIL'}")
print(f"  max-autotune correctness:    {'✓ PASS' if autotune_correct else '✗ FAIL'}")


# ==============================================================================
# ADVANCED: Multi-Layer MLP with torch.compile
# ==============================================================================

print("\n" + "=" * 60)
print("Advanced: Multi-Layer MLP with torch.compile")
print("=" * 60)

# Define MLP using CUTLASS GEMM
class CutlassMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.fc1_bias = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight = torch.nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.fc2_bias = torch.nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        # Layer 1: GEMM + bias + ReLU
        x = cutlass_gemm(x, self.fc1_weight.T) + self.fc1_bias
        x = torch.relu(x)
        # Layer 2: GEMM + bias
        x = cutlass_gemm(x, self.fc2_weight.T) + self.fc2_bias
        return x


# Create and compile MLP
batch_size = 64
input_dim, hidden_dim, output_dim = 1024, 4096, 1024

mlp = CutlassMLP(input_dim, hidden_dim, output_dim).to(device)
x = torch.randn(batch_size, input_dim, dtype=dtype, device=device)

# TODO [MEDIUM]: Compile the entire MLP
# compiled_mlp = torch.compile(mlp, mode='max-autotune')

compiled_mlp = mlp  # Placeholder

print(f"\nMLP Configuration:")
print(f"  Input:  [{batch_size}, {input_dim}]")
print(f"  Hidden: [{batch_size}, {hidden_dim}]")
print(f"  Output: [{batch_size}, {output_dim}]")

# Benchmark uncompiled
def benchmark_mlp(model, x, num_warmup=5, num_iters=20):
    for _ in range(num_warmup):
        _ = model(x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


mlp_uncompiled_latency = benchmark_mlp(mlp, x)
mlp_compiled_latency = benchmark_mlp(compiled_mlp, x)

print(f"\nMLP Performance:")
print(f"  Uncompiled: {mlp_uncompiled_latency:.3f} ms")
print(f"  Compiled:   {mlp_compiled_latency:.3f} ms")

if mlp_compiled_latency > 0:
    speedup = mlp_uncompiled_latency / mlp_compiled_latency
    print(f"  Speedup:    {speedup:.2f}×")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: torch.compile effect on CUTLASS?")
print("        Answer: Depends on mode:")
print("        - default: Minimal overhead, modest gains")
print("        - reduce-overhead: Best for single ops")
print("        - max-autotune: Best for fusion opportunities")

print("\n    Q2: reduce-overhead vs max-autotune?")
print("        reduce-overhead:")
print("          - Uses CUDA graphs")
print("          - Reduces kernel launch overhead")
print("          - Best for repeated same-shape calls")
print("        max-autotune:")
print("          - Aggressive operator fusion")
print("          - May reorder operations")
print("          - Best for complex graphs")

print("\n    Q3: Can torch.compile fuse CUTLASS ops?")
print("        Answer: CUTLASS ops are opaque to Inductor.")
print("                torch.compile can fuse around them,")
print("                but not into them. For full fusion,")
print("                use CuTe DSL custom kernels.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --set full python ex03_torch_compile_FILL_IN.py")
print("\n    Look for:")
print("      - CUDA graph capture (reduce-overhead mode)")
print("      - Fewer kernel launches after compilation")
print("      - Same CUTLASS kernel execution")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: When should you use torch.compile with custom kernels?")
print("    A: Use torch.compile when:")
print("       - You have a graph of operations (not single op)")
print("       - Want to reduce Python dispatch overhead")
print("       - Need CUDA graph benefits")
print("    Skip torch.compile when:")
print("       - Single kernel dominates runtime")
print("       - Dynamic shapes prevent graph capture")
print("       - Compilation overhead > runtime benefit")

print("\n    Q: What's the compilation overhead?")
print("    A: First run: 100ms - 2s depending on graph size.")
print("       Subsequent runs: Near-zero (cached).")
print("       Use torch.compile with cache_dir for persistence.")

# C4: Production guidance
print("\nC4: Production torch.compile Tips")
print("    Best practices:")
print("      1. Compile at module level, not per-call")
print("      2. Use mode='reduce-overhead' for inference")
print("      3. Use mode='max-autotune' for training")
print("      4. Set TORCHINDUCTOR_CACHE_DIR for persistent cache")
print("      5. Profile first-run vs steady-state latency")
print("\n    Debugging:")
print("      - TORCH_LOGS=dynamo for compilation logs")
print("      - torch._dynamo.explain() for fusion analysis")
print("      - Check torch.compile compatibility list")

print("\n" + "=" * 60)
print("Exercise 03 Complete!")
print("=" * 60)
