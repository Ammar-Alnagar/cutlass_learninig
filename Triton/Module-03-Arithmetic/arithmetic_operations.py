"""
Arithmetic Operations Example: Element-wise Computations
This example demonstrates various arithmetic and mathematical operations in Triton.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def multiply_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise multiplication kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def math_functions_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel demonstrating various mathematical functions"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply various mathematical functions
    sqrt_x = tl.sqrt(tl.abs(x) + 1e-8)  # Add small value to avoid sqrt of negative numbers
    exp_x = tl.exp(tl.clamp(x, -10, 10))  # Clamp to avoid overflow
    log_x = tl.log(tl.abs(x) + 1e-8)     # Add small value to avoid log(0)
    
    # Combine operations: sqrt(abs(x)) + exp(clamped x) + log(abs(x))
    output = sqrt_x + exp_x + log_x
    
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fused_multiply_add_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused multiply-add: output = a * b + c"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    output = a * b + c  # Fused multiply-add operation
    
    tl.store(output_ptr + offsets, output, mask=mask)


def element_wise_add(x, y):
    """Host function for element-wise addition"""
    assert x.shape == y.shape
    output = torch.empty_like(x)
    
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def element_wise_multiply(x, y):
    """Host function for element-wise multiplication"""
    assert x.shape == y.shape
    output = torch.empty_like(x)
    
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    multiply_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def apply_math_functions(x):
    """Host function to apply various mathematical functions"""
    output = torch.empty_like(x)
    
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    math_functions_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def fused_multiply_add(a, b, c):
    """Host function for fused multiply-add operation"""
    assert a.shape == b.shape == c.shape
    output = torch.empty_like(a)
    
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    fused_multiply_add_kernel[grid](a, b, c, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


# Example usage
if __name__ == "__main__":
    # Create sample tensors
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    z = torch.randn(1024, device='cuda')
    
    print("Testing element-wise operations...")
    
    # Test addition
    result_add = element_wise_add(x, y)
    expected_add = x + y
    print(f"Addition correct: {torch.allclose(result_add, expected_add, atol=1e-5)}")
    
    # Test multiplication
    result_mul = element_wise_multiply(x, y)
    expected_mul = x * y
    print(f"Multiplication correct: {torch.allclose(result_mul, expected_mul, atol=1e-5)}")
    
    # Test mathematical functions
    result_math = apply_math_functions(x)
    expected_math = torch.sqrt(torch.abs(x) + 1e-8) + torch.exp(torch.clamp(x, -10, 10)) + torch.log(torch.abs(x) + 1e-8)
    print(f"Math functions correct: {torch.allclose(result_math, expected_math, atol=1e-3)}")
    
    # Test fused multiply-add
    result_fma = fused_multiply_add(x, y, z)
    expected_fma = x * y + z
    print(f"Fused multiply-add correct: {torch.allclose(result_fma, expected_fma, atol=1e-5)}")
    
    # Performance comparison example
    print("\nPerformance note: Triton operations are typically faster for large tensors")
    print("because they're optimized for GPU parallel execution.")