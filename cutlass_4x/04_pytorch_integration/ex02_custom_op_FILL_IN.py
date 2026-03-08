"""
Module 04 — PyTorch Integration
Exercise 02 — Register CUTLASS as PyTorch Custom Operator

LEVEL: 1 → 2 (High-level op → Custom torch op registration)

WHAT YOU'RE BUILDING:
  PyTorch custom operator wrapping CUTLASS GEMM — the pattern used to 
  integrate custom kernels into PyTorch's dispatch system. This enables 
  seamless use of CUTLASS kernels in PyTorch models with autograd support.

OBJECTIVE:
  - Use torch.library to register custom op
  - Implement forward pass with CUTLASS backend
  - Add backward pass for autograd support
  - Use in PyTorch module like native op
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
# Q1: What's the benefit of registering as torch custom op vs plain function?
#     Hint: Think about dispatch, graph capture, and autograd

# Q2: Do you need to implement backward pass for inference-only models?
#     What about training?

# Q3: How does PyTorch know which device/dtype your custom op supports?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions
M, K, N = 512, 1024, 2048
dtype = torch.float16
device = torch.device("cuda")

# Test tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)

# Reference output
C_ref = torch.mm(A, B)

print("=" * 60)
print("PyTorch Custom Operator Registration")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"\nGoal: Register CUTLASS GEMM as torch.ops.cutlass.gemm")
print("  - Usable like native torch.mm")
print("  - Supports autograd (with backward)")
print("  - Integrates with torch.compile")


# ==============================================================================
# FILL IN: Level 1/2 — Custom Op Registration
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1/2: Custom Op with torch.library")
print("=" * 60)

# TODO [HARD]: Register custom GEMM operator
# HINT:
#   - Use torch.library.register_fake for abstract impl
#   - Use torch.library.impl for CUDA implementation
#   - Use torch.autograd.Function for backward pass
# REF: cutlass/examples/python/CuTeDSL/torch_custom_op.py
#      https://pytorch.org/docs/stable/library.html

# Step 1: Define the custom autograd Function
# class CutlassGemmFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A, B):
#         # TODO: Save tensors for backward
#         # ctx.save_for_backward(A, B)
#         
#         # TODO: Run CUTLASS GEMM forward
#         # C_cutlass = cutlass.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
#         # plan = cutlass.op.Gemm(element=..., layout=...)
#         # plan.run(from_dlpack(A), from_dlpack(B), C_cutlass)
#         # return torch.from_dlpack(C_cutlass)
#         pass
#     
#     @staticmethod
#     def backward(ctx, grad_output):
#         # TODO: Compute gradients using CUTLASS
#         # grad_A = grad_output @ B.T
#         # grad_B = A.T @ grad_output
#         # return grad_A, grad_B
#         pass

# Placeholder implementation (replace with CUTLASS)
class CutlassGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return torch.mm(A, B)  # Placeholder
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_output @ B.T
        grad_B = A.T @ grad_output
        return grad_A, grad_B


# Step 2: Register the operator
# torch.library.define("cutlass::gemm", "(Tensor A, Tensor B) -> Tensor")

# @torch.library.impl("cutlass::gemm", "CUDA")
# def gemm_cuda(A, B):
#     return CutlassGemmFunction.apply(A, B)

# @torch.library.impl_abstract("cutlass::gemm")
# def gemm_abstract(A, B):
#     # Abstract impl for torch.compile
#     return A.new_empty(A.shape[0], B.shape[1])

# For this exercise, we'll use the autograd Function directly
def cutlass_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUTLASS GEMM wrapper (use CutlassGemmFunction for autograd)."""
    return CutlassGemmFunction.apply(A, B)


print(f"\n✓ Custom operator registered")
print(f"  Usage: cutlass_gemm(A, B)")


# ==============================================================================
# TESTING: Forward Pass
# ==============================================================================

print("\n" + "=" * 60)
print("Testing: Forward Pass")
print("=" * 60)

# TODO [EASY]: Test forward pass
# C_custom = cutlass_gemm(A, B)

C_custom = cutlass_gemm(A, B)

print(f"Output shape: {C_custom.shape}")
print(f"Expected:     {C_ref.shape}")

# Verify
is_correct = torch.allclose(C_custom, C_ref, rtol=1e-2, atol=1e-2)
print(f"\nForward correctness: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (C_custom - C_ref).abs().max().item()
    print(f"  Max error: {max_error:.6f}")


# ==============================================================================
# TESTING: Backward Pass (Autograd)
# ==============================================================================

print("\n" + "=" * 60)
print("Testing: Backward Pass (Autograd)")
print("=" * 60)

# TODO [MEDIUM]: Test backward pass
# HINT:
#   - Set requires_grad=True on inputs
#   - Compute loss = sum(output)
#   - Call loss.backward()
#   - Check gradients

# Create tensors with grad
A_grad = A.clone().requires_grad_(True)
B_grad = B.clone().requires_grad_(True)

# Forward
C_grad = cutlass_gemm(A_grad, B_grad)
loss = C_grad.sum()

# Backward
loss.backward()

# Reference gradients
A_grad_ref = B.T.sum(dim=0).unsqueeze(0).expand_as(A)
B_grad_ref = A.T.sum(dim=0).unsqueeze(0).expand_as(B)

# Check gradients
A_grad_correct = torch.allclose(A_grad.grad, A_grad_ref, rtol=1e-1, atol=1e-1)
B_grad_correct = torch.allclose(B_grad.grad, B_grad_ref, rtol=1e-1, atol=1e-1)

print(f"grad_A correctness: {'✓ PASS' if A_grad_correct else '✗ FAIL'}")
print(f"grad_B correctness: {'✓ PASS' if B_grad_correct else '✗ FAIL'}")

if not A_grad_correct:
    max_error = (A_grad.grad - A_grad_ref).abs().max().item()
    print(f"  grad_A max error: {max_error:.6f}")

if not B_grad_correct:
    max_error = (B_grad.grad - B_grad_ref).abs().max().item()
    print(f"  grad_B max error: {max_error:.6f}")


# ==============================================================================
# USAGE: In PyTorch Module
# ==============================================================================

print("\n" + "=" * 60)
print("Usage: In PyTorch Module")
print("=" * 60)

# TODO [EASY]: Create PyTorch module using custom op
# class CutlassLinear(torch.nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
#         self.bias = torch.nn.Parameter(torch.zeros(out_features))
#     
#     def forward(self, x):
#         # x: [batch, in_features]
#         # weight: [out_features, in_features]
#         # Need: x @ weight.T = [batch, out_features]
#         return cutlass_gemm(x, self.weight.T) + self.bias

class CutlassLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return cutlass_gemm(x, self.weight.T) + self.bias


# Test module
batch_size = 32
linear = CutlassLinear(K, N).to(device)
x = torch.randn(batch_size, K, dtype=dtype, device=device)

output = linear(x)
print(f"\nModule output shape: {output.shape}")
print(f"Expected:            [{batch_size}, {N}]")

# Test training step
target = torch.randn(batch_size, N, dtype=dtype, device=device)
criterion = torch.nn.MSELoss()
loss = criterion(output, target)
loss.backward()

print(f"\nTraining step:")
print(f"  Loss: {loss.item():.4f}")
print(f"  Weight grad: {'✓ Computed' if linear.weight.grad is not None else '✗ None'}")
print(f"  Bias grad:   {'✓ Computed' if linear.bias.grad is not None else '✗ None'}")


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark_custom_op(A, B, num_warmup=10, num_iters=100) -> float:
    """Benchmark custom op."""
    # Warmup
    for _ in range(num_warmup):
        _ = cutlass_gemm(A, B)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = cutlass_gemm(A, B)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance")
print("=" * 60)

torch_latency = benchmark_torch_mm(A, B)
custom_latency = benchmark_custom_op(A, B)

print(f"\nResults:")
print(f"  torch.mm:        {torch_latency:.3f} ms")
print(f"  Custom CUTLASS:  {custom_latency:.3f} ms")

if custom_latency > 0:
    speedup = torch_latency / custom_latency
    print(f"\n  Speedup: {speedup:.2f}×")


def benchmark_torch_mm(A, B, num_warmup=10, num_iters=100) -> float:
    C = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Benefit of custom op vs plain function?")
print("        Answer:")
print("        - Integrates with torch.compile (graph capture)")
print("        - Proper dispatch (CPU/CUDA, different dtypes)")
print("        - Autograd support (backward pass)")
print("        - Appears in torchscript/profiler")

print("\n    Q2: Need backward for inference?")
print("        Answer: No for inference-only.")
print("                Yes for training (gradient computation).")
print("                Can implement forward-only for inference models.")

print("\n    Q3: How does PyTorch know supported device/dtype?")
print("        Answer: Via torch.library.impl registration:")
print("                impl('cutlass::gemm', 'CUDA') → CUDA support")
print("                impl('cutlass::gemm', 'CPU') → CPU support")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --set full python ex02_custom_op_FILL_IN.py")
print("\n    Look for:")
print("      - Same kernel execution as direct CUTLASS")
print("      - Minimal Python overhead")
print("      - Proper CUDA stream usage")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How do you integrate custom CUDA kernels into PyTorch?")
print("    A: Options:")
print("       1. torch.library (modern, recommended)")
print("       2. torch.autograd.Function (simple, works)")
print("       3. PyBind11 + CUDA extension (full control)")
print("       4. torch.utils.cpp_extension (build-time integration)")

print("\n    Q: What's the difference between define() and impl()?")
print("    A: define() declares the operator signature (abstract).")
print("       impl() provides the actual implementation for a device.")
print("       Multiple impl() calls for different devices.")

# C4: Production guidance
print("\nC4: Production Custom Op Tips")
print("    Registration best practices:")
print("      1. Use torch.library for new ops (PyTorch 1.12+)")
print("      2. Provide abstract impl for torch.compile")
print("      3. Implement backward only if needed for training")
print("      4. Register for multiple dtypes/devices as needed")
print("\n    Debugging:")
print("      - Use torch.profiler to verify op appears")
print("      - Check torch.compile compatibility")
print("      - Test gradient computation numerically")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
