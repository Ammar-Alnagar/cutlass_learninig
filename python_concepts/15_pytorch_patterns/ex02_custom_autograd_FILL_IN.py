"""
Module 15 — PyTorch Patterns
Exercise 02 — Custom Autograd Functions

WHAT YOU'RE BUILDING:
  Custom autograd functions let you define custom forward/backward passes.
  vLLM uses this for custom attention, fused ops, and gradient checkpointing.
  Understanding this is key for implementing new ops.

OBJECTIVE:
  - Extend torch.autograd.Function
  - Implement forward and backward with context
  - Save tensors for backward pass
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between nn.Module and autograd.Function?
# Q2: Why do you need to save tensors in ctx.save_for_backward?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from torch.autograd.function import FunctionCtx

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Implement a custom ReLU autograd function.
#              This shows the basic forward/backward pattern.
# HINT: 
#   - forward(ctx, x) -> save input, return output
#   - backward(ctx, grad_output) -> return grad_input

class CustomReLU(torch.autograd.Function):
    """Custom ReLU with manual autograd."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        # TODO: save input for backward, return relu output
        # HINT: ctx.save_for_backward(x); return torch.relu(x)
        pass

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO: retrieve saved input, compute gradient
        # HINT: (x,) = ctx.saved_tensors; grad = grad_output * (x > 0)
        pass

# TODO [EASY]: Use the custom autograd function.
#              How does it compare to torch.nn.functional.relu?

def test_custom_relu():
    """Test custom ReLU with autograd."""
    x = torch.randn(10, 10, requires_grad=True)
    
    # TODO: Apply custom ReLU
    # HINT: y = CustomReLU.apply(x)
    pass

# TODO [MEDIUM]: Implement a custom GELU approximation with autograd.
#              GELU is used in transformers; approximations are faster.
# HINT: Use tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

class CustomGELU(torch.autograd.Function):
    """Custom GELU approximation."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        # TODO: save input, compute GELU approximation
        # HINT: Use the tanh approximation formula
        pass

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO: compute gradient of GELU approximation
        # HINT: grad = grad_output * d(gelu)/dx
        pass

# TODO [EASY]: Understand when to use custom autograd vs nn.Module.
#              Fill in:
#              - Use autograd.Function when: ______
#              - Use nn.Module when: ______

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why save tensors in forward? What happens if you don't?
# C2: How does custom autograd enable fused operations?

if __name__ == "__main__":
    print("Testing custom ReLU...")
    test_custom_relu()
    
    x = torch.randn(10, 10, requires_grad=True)
    y = CustomReLU.apply(x)
    loss = y.sum()
    loss.backward()
    print(f"  Input grad shape: {x.grad.shape}")

    print("\nTesting custom GELU...")
    x = torch.randn(10, 10, requires_grad=True)
    y = CustomGELU.apply(x)
    loss = y.sum()
    loss.backward()
    print(f"  Input grad shape: {x.grad.shape}")
    print(f"  Grad sample: {x.grad[0, :5]}")

    print("\nDone!")
