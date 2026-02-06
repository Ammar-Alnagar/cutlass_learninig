"""
Memory Operations Example: Efficient Memory Loading and Storing
This example demonstrates efficient memory operations in Triton, including coalesced access patterns and boundary handling.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(
    input_ptr,    # Pointer to input tensor
    output_ptr,   # Pointer to output tensor
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    """
    Kernel that copies data from input to output tensor
    Demonstrates efficient memory loading and storing
    """
    # Calculate the starting index for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate the offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data from input tensor
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Store data to output tensor
    tl.store(output_ptr + offsets, input_vals, mask=mask)


@triton.jit
def strided_copy_kernel(
    input_ptr,    # Pointer to input tensor
    output_ptr,   # Pointer to output tensor
    n_elements,   # Total number of elements
    stride,       # Stride for accessing elements
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    """
    Kernel that copies data with a specific stride
    Demonstrates how stride affects memory access patterns
    """
    # Calculate the starting index for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate the offsets for this block (with stride)
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = indices * stride  # Apply stride to get actual memory addresses
    
    # Create a mask to handle boundary conditions
    mask = indices < n_elements
    
    # Load data from input tensor
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Store data to output tensor
    tl.store(output_ptr + offsets, input_vals, mask=mask)


@triton.jit
def masked_load_store_kernel(
    input_ptr,      # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    n_elements,     # Total number of elements
    condition_ptr,  # Pointer to condition tensor (bool values)
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    """
    Kernel that conditionally loads and stores data based on a boolean condition
    Demonstrates conditional memory operations
    """
    # Calculate the starting index for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate the offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle boundary conditions
    boundary_mask = offsets < n_elements
    
    # Load condition values
    conditions = tl.load(condition_ptr + offsets, mask=boundary_mask)
    
    # Combine boundary mask with condition mask
    mask = boundary_mask & conditions
    
    # Load data from input tensor only where condition is True
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Store data to output tensor only where condition is True
    tl.store(output_ptr + offsets, input_vals, mask=mask)


def copy_tensor(input_tensor):
    """Host function to copy a tensor using Triton"""
    output_tensor = torch.empty_like(input_tensor)
    
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    copy_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor


def strided_copy_tensor(input_tensor, stride):
    """Host function to copy a tensor with a specific stride"""
    # For strided copy, we need to adjust the output size
    n_elements = input_tensor.numel()
    output_size = n_elements * stride
    output_tensor = torch.zeros(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    strided_copy_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor


def conditional_copy(input_tensor, condition_tensor):
    """Host function to conditionally copy tensor elements"""
    output_tensor = torch.zeros_like(input_tensor)
    
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    masked_load_store_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        condition_tensor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor


# Example usage
if __name__ == "__main__":
    # Test basic copy
    print("Testing basic copy operation...")
    input_tensor = torch.randn(2048, device='cuda')
    copied_tensor = copy_tensor(input_tensor)
    print(f"Basic copy correct: {torch.allclose(input_tensor, copied_tensor)}")
    
    # Test strided copy
    print("\nTesting strided copy operation...")
    small_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device='cuda')
    strided_result = strided_copy_tensor(small_tensor, stride=3)
    print(f"Original: {small_tensor}")
    print(f"After strided copy (stride=3): {strided_result[:12]}")  # Show first 12 elements
    
    # Test conditional copy
    print("\nTesting conditional copy operation...")
    input_cond = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    condition = torch.tensor([True, False, True, False, True], device='cuda')
    cond_result = conditional_copy(input_cond, condition)
    expected_cond = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0], device='cuda')  # Only True positions preserved
    print(f"Input: {input_cond}")
    print(f"Condition: {condition}")
    print(f"Conditional copy result: {cond_result}")
    print(f"Expected: {expected_cond}")
    print(f"Conditional copy correct: {torch.allclose(cond_result, expected_cond)}")