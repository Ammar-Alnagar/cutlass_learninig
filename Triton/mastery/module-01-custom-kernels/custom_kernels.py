"""
Module 01: Custom Kernel Development

This module teaches how to design and implement custom Triton kernels
from scratch for specialized operations.

LEARNING OBJECTIVES:
1. Design kernel architecture from requirements
2. Implement custom operations efficiently
3. Handle complex indexing patterns
4. Optimize for specific hardware
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# CUSTOM KERNEL 1: Polynomial Activation
# ============================================================================

@triton.jit
def polynomial_activation_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    COEF_0: tl.constexpr,
    COEF_1: tl.constexpr,
    COEF_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Custom polynomial activation: f(x) = c0 + c1*x + c2*x^2
    
    This demonstrates implementing a custom element-wise operation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Polynomial computation
    output = COEF_0 + COEF_1 * x + COEF_2 * x * x
    
    tl.store(output_ptr + offsets, output, mask=mask)


def polynomial_activation(x: torch.Tensor, c0: float = 0.0, c1: float = 1.0, c2: float = 0.1) -> torch.Tensor:
    """
    Apply polynomial activation to input tensor.
    """
    assert x.is_cuda, "Input must be on CUDA"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    polynomial_activation_kernel[grid](
        x, output, n_elements,
        c0, c1, c2,
        BLOCK_SIZE
    )
    
    return output


# ============================================================================
# CUSTOM KERNEL 2: Gated Linear Unit (GLU)
# ============================================================================

@triton.jit
def glu_kernel(
    x_ptr,
    gate_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gated Linear Unit: output = x * sigmoid(gate)
    
    This demonstrates a fused operation with two inputs.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Sigmoid: 1 / (1 + exp(-x))
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    
    output = x * sigmoid_gate
    
    tl.store(output_ptr + offsets, output, mask=mask)


def glu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Apply Gated Linear Unit.
    """
    assert x.is_cuda and gate.is_cuda
    assert x.shape == gate.shape
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    glu_kernel[grid](x, gate, output, n_elements, BLOCK_SIZE)
    
    return output


# ============================================================================
# CUSTOM KERNEL 3: Custom Reduction (L2 Norm)
# ============================================================================

@triton.jit
def l2_norm_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute L2 norm for each row.
    
    This demonstrates a reduction operation.
    """
    pid = tl.program_id(axis=0)
    
    # Each program handles one row
    row_start = pid * n_cols
    
    # Load row data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Compute squared values
    squared = row * row
    
    # Sum (reduction)
    sum_squared = tl.sum(squared, axis=0)
    
    # Square root
    norm = tl.sqrt(sum_squared)
    
    # Store result
    tl.store(output_ptr + pid, norm)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 norm for each row of input.
    """
    assert x.is_cuda
    assert x.dim() == 2
    
    n_rows, n_cols = x.shape
    
    # BLOCK_SIZE must be >= n_cols for single-pass reduction
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    output = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    grid = (n_rows,)
    
    l2_norm_kernel[grid](x, output, n_rows, n_cols, x.stride(0), BLOCK_SIZE)
    
    return output


# ============================================================================
# CUSTOM KERNEL 4: Complex Multiplication
# ============================================================================

@triton.jit
def complex_mul_kernel(
    real_a_ptr, imag_a_ptr,
    real_b_ptr, imag_b_ptr,
    real_out_ptr, imag_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    
    This demonstrates working with split complex tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load real and imaginary parts
    real_a = tl.load(real_a_ptr + offsets, mask=mask, other=0.0)
    imag_a = tl.load(imag_a_ptr + offsets, mask=mask, other=0.0)
    real_b = tl.load(real_b_ptr + offsets, mask=mask, other=0.0)
    imag_b = tl.load(imag_b_ptr + offsets, mask=mask, other=0.0)
    
    # Complex multiplication
    real_out = real_a * real_b - imag_a * imag_b
    imag_out = real_a * imag_b + imag_a * real_b
    
    # Store results
    tl.store(real_out_ptr + offsets, real_out, mask=mask)
    tl.store(imag_out_ptr + offsets, imag_out, mask=mask)


def complex_multiply(real_a: torch.Tensor, imag_a: torch.Tensor,
                     real_b: torch.Tensor, imag_b: torch.Tensor) -> tuple:
    """
    Multiply two complex tensors.
    """
    assert real_a.is_cuda and imag_a.is_cuda
    assert real_b.is_cuda and imag_b.is_cuda
    assert real_a.shape == imag_a.shape == real_b.shape == imag_b.shape
    
    n_elements = real_a.numel()
    BLOCK_SIZE = 1024
    
    real_out = torch.empty_like(real_a)
    imag_out = torch.empty_like(imag_a)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    complex_mul_kernel[grid](
        real_a, imag_a, real_b, imag_b,
        real_out, imag_out,
        n_elements, BLOCK_SIZE
    )
    
    return real_out, imag_out


# ============================================================================
# CUSTOM KERNEL 5: Batched Scale-Rotate
# ============================================================================

@triton.jit
def scale_rotate_kernel(
    x_ptr, y_ptr,
    out_x_ptr, out_y_ptr,
    scale,
    cos_angle,
    sin_angle,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply scale and rotation to 2D points.
    
    [out_x]   [scale * cos, -scale * sin] [x]
    [out_y] = [scale * sin,  scale * cos] [y]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Precompute rotation matrix elements
    sc = scale * cos_angle
    ss = scale * sin_angle
    
    # Apply transformation
    out_x = sc * x - ss * y
    out_y = ss * x + sc * y
    
    tl.store(out_x_ptr + offsets, out_x, mask=mask)
    tl.store(out_y_ptr + offsets, out_y, mask=mask)


if __name__ == "__main__":
    print("Custom Kernel Development Module")
    print("=" * 60)
    
    # Test polynomial activation
    print("\n1. Polynomial Activation")
    x = torch.randn(10000, device="cuda")
    output = polynomial_activation(x, c0=0.1, c1=1.0, c2=0.01)
    expected = 0.1 + 1.0 * x + 0.01 * x * x
    print(f"   {'✓ PASS' if torch.allclose(output, expected) else '✗ FAIL'}")
    
    # Test GLU
    print("\n2. Gated Linear Unit")
    x = torch.randn(10000, device="cuda")
    gate = torch.randn(10000, device="cuda")
    output = glu(x, gate)
    expected = x * torch.sigmoid(gate)
    print(f"   {'✓ PASS' if torch.allclose(output, expected) else '✗ FAIL'}")
    
    # Test L2 Norm
    print("\n3. L2 Norm")
    x = torch.randn(100, 64, device="cuda")
    output = l2_norm(x)
    expected = torch.norm(x, dim=1)
    print(f"   {'✓ PASS' if torch.allclose(output, expected) else '✗ FAIL'}")
    
    # Test Complex Multiplication
    print("\n4. Complex Multiplication")
    real_a = torch.randn(10000, device="cuda")
    imag_a = torch.randn(10000, device="cuda")
    real_b = torch.randn(10000, device="cuda")
    imag_b = torch.randn(10000, device="cuda")
    
    real_out, imag_out = complex_multiply(real_a, imag_a, real_b, imag_b)
    
    expected_real = real_a * real_b - imag_a * imag_b
    expected_imag = real_a * imag_b + imag_a * real_b
    
    real_ok = torch.allclose(real_out, expected_real)
    imag_ok = torch.allclose(imag_out, expected_imag)
    print(f"   {'✓ PASS' if real_ok and imag_ok else '✗ FAIL'}")
    
    # Test Scale-Rotate
    print("\n5. Scale-Rotate Transform")
    import math
    x = torch.randn(10000, device="cuda")
    y = torch.randn(10000, device="cuda")
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    
    scale = 2.0
    angle = math.pi / 4  # 45 degrees
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    grid = (triton.cdiv(10000, 1024),)
    scale_rotate_kernel[grid](x, y, out_x, out_y, scale, cos_a, sin_a, 10000, 1024)
    
    expected_x = scale * (cos_a * x - sin_a * y)
    expected_y = scale * (sin_a * x + cos_a * y)
    
    print(f"   {'✓ PASS' if torch.allclose(out_x, expected_x) and torch.allclose(out_y, expected_y) else '✗ FAIL'}")
    
    print("\n" + "=" * 60)
    print("CUSTOM KERNEL DESIGN CHECKLIST:")
    print("□ Define clear input/output specifications")
    print("□ Choose appropriate block size")
    print("□ Handle boundary conditions with masks")
    print("□ Consider memory access patterns")
    print("□ Test with various input sizes")
    print("□ Verify against reference implementation")
    print("□ Profile and optimize if needed")
