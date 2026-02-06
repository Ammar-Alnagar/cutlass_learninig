"""
Advanced Memory Layouts and Optimizations Example
This example demonstrates various memory optimization techniques in Triton,
including coalesced access patterns and memory layout transformations.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def coalesced_copy_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Demonstrates coalesced memory access pattern
    In coalesced access, consecutive threads access consecutive memory locations
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced access: consecutive threads access consecutive memory
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)


@triton.jit
def strided_access_kernel(
    input_ptr, output_ptr,
    n_elements, stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Demonstrates strided memory access pattern
    Strided access can lead to poor performance due to non-coalesced access
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = indices * stride  # Strided access pattern
    mask = indices < n_elements
    
    # Strided access: threads access memory with gaps
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)


@triton.jit
def transpose_coalesced_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    Implements an optimized transpose using coalesced access patterns
    """
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate starting indices
    rm = pid_m * block_size_m
    rn = pid_n * block_size_n
    
    # Generate indices for this block
    offs_m = rm + tl.arange(0, block_size_m)
    offs_n = rn + tl.arange(0, block_size_n)
    
    # Create masks for boundaries
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Calculate linear offsets for input (coalesced access)
    input_offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    
    # For output, we need to transpose the data
    # Calculate linear offsets for output (transposed)
    output_offsets = offs_n[:, None] * n_rows + offs_m[None, :]
    tl.store(output_ptr + output_offsets.T, input_vals, mask=mask)


@triton.jit
def padding_kernel(
    input_ptr, output_ptr,
    n_elements, original_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Demonstrates padding to avoid bank conflicts in shared memory
    This kernel adds padding to align memory accesses
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < original_size  # Only process original elements
    
    # Load with mask to avoid out-of-bounds access
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, input_vals, mask=mask)


def coalesced_copy(input_tensor):
    """Host function for coalesced memory copy"""
    output = torch.empty_like(input_tensor)
    
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    coalesced_copy_kernel[grid](
        input_tensor, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def strided_copy(input_tensor, stride):
    """Host function for strided memory copy"""
    n_elements = input_tensor.numel()
    output_size = (n_elements + stride - 1) // stride  # Ceiling division
    output = torch.zeros(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(output_size, BLOCK_SIZE), )
    
    strided_access_kernel[grid](
        input_tensor, output,
        output_size, stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def optimized_transpose(input_matrix):
    """Host function for optimized transpose with coalesced access"""
    n_rows, n_cols = input_matrix.shape
    output_matrix = torch.empty((n_cols, n_rows), dtype=input_matrix.dtype, device=input_matrix.device)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = (
        triton.cdiv(n_rows, BLOCK_SIZE_M),
        triton.cdiv(n_cols, BLOCK_SIZE_N),
    )
    
    transpose_coalesced_kernel[grid](
        input_matrix, output_matrix,
        n_rows, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N,
    )
    
    return output_matrix


def padded_operation(input_tensor, pad_to_multiple=32):
    """Host function that demonstrates padding for memory optimization"""
    original_size = input_tensor.numel()
    # Pad to the next multiple of pad_to_multiple
    padded_size = ((original_size + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    padded_tensor = torch.zeros(padded_size, dtype=input_tensor.dtype, device=input_tensor.device)
    padded_tensor[:original_size] = input_tensor
    
    output = torch.empty_like(padded_tensor)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(padded_size, BLOCK_SIZE), )
    
    padding_kernel[grid](
        padded_tensor, output,
        padded_size, original_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return only the original elements
    return output[:original_size]


# Example usage
if __name__ == "__main__":
    # Create sample tensors
    size = 1024
    input_vec = torch.randn(size, device='cuda')
    input_mat = torch.randn(128, 128, device='cuda')
    
    print("Testing advanced memory layouts and optimizations...")
    
    # Test coalesced copy
    result_coalesced = coalesced_copy(input_vec)
    print(f"Coalesced copy correct: {torch.allclose(result_coalesced, input_vec)}")
    
    # Test strided copy
    result_strided = strided_copy(input_vec[:100], stride=2)
    print(f"Strided copy completed, output shape: {result_strided.shape}")
    
    # Test optimized transpose
    result_transpose = optimized_transpose(input_mat)
    expected_transpose = input_mat.T
    print(f"Optimized transpose correct: {torch.allclose(result_transpose, expected_transpose)}")
    
    # Test padded operation
    result_padded = padded_operation(input_vec[:100])
    print(f"Padded operation correct: {torch.allclose(result_padded, input_vec[:100])}")
    
    print("\nMemory optimization notes:")
    print("- Coalesced access patterns significantly improve memory bandwidth utilization")
    print("- Strided access can hurt performance due to non-contiguous memory access")
    print("- Padding can help avoid bank conflicts in shared memory")
    print("- Proper memory layout is crucial for optimal GPU performance")