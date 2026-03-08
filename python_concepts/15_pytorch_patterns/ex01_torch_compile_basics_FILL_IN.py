"""
Module 15 — PyTorch Patterns
Exercise 01 — torch.compile and Graph Capture

WHAT YOU'RE BUILDING:
  torch.compile (PyTorch 2.0+) JIT-compiles Python code to optimized kernels.
  vLLM uses this for fusion. Understanding capture, graphs, and dynamism
  is essential for debugging compilation issues.

OBJECTIVE:
  - Use @torch.compile decorator
  - Understand graph capture and recompilation
  - Know what breaks compilation (dynamic shapes, control flow)
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does torch.compile do to a function at runtime?
# Q2: What Python constructs cause recompilation (shape changes, branches)?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from typing import Optional

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Apply torch.compile to a simple kernel function.
#              Compare first run (compile) vs subsequent runs (cached).
# HINT: @torch.compile decorator, or torch.compile(fn) wrapper

def matmul_add_relu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Fused matmul + add + relu."""
    return torch.relu(torch.matmul(a, b) + c)

# TODO: Apply torch.compile to the function
# compiled_fn = torch.compile(matmul_add_relu)
compiled_fn = None  # TODO: replace

def test_compile_timing():
    """Compare compiled vs eager execution."""
    a = torch.randn(512, 512, device='cuda')
    b = torch.randn(512, 512, device='cuda')
    c = torch.randn(512, 512, device='cuda')
    
    # TODO: Time first run (includes compile)
    # TODO: Time subsequent runs (cached)
    pass

# TODO [MEDIUM]: Understand what breaks compilation.
#              Dynamic control flow causes recompilation.
# HINT: Python if statements on tensor values break graph capture

def conditional_kernel(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Kernel with dynamic control flow."""
    # TODO: Add Python if statement that checks x.mean() > threshold
    # This causes recompilation when condition changes
    # HINT: if x.mean().item() > threshold: return x * 2 else: return x
    pass

# TODO [EASY]: Use torch.compile with mode options.
#              mode='reduce-overhead' vs 'max-autotune' vs 'default'
# HINT: torch.compile(fn, mode='reduce-overhead')

def test_compile_modes():
    """Test different compilation modes."""
    # TODO: Compile with different modes and compare
    # modes: 'default', 'reduce-overhead', 'max-autotune'
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How much faster is compiled vs eager for the fused kernel?
# C2: What triggers recompilation? How do you avoid it?

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for this exercise")
        exit(1)

    print("Testing torch.compile timing...")
    test_compile_timing()

    print("\nTesting dynamic control flow...")
    x = torch.randn(100, 100, device='cuda')
    result = conditional_kernel(x, threshold=0.5)
    print(f"  Result shape: {result.shape}")

    print("\nTesting compile modes...")
    test_compile_modes()

    print("\nDone!")
