"""
Basic Triton example: Vector Addition
This example demonstrates the most basic Triton operation - adding two vectors element-wise.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # pointer to the input tensor x
    y_ptr,  # pointer to the input tensor y
    output_ptr,  # pointer to the output tensor
    n_elements,  # number of elements in the input tensors
    BLOCK_SIZE: tl.constexpr,  # How many elements each program instance will process
):
    """
    Kernel function that adds two vectors element-wise
    Each instance of this kernel processes BLOCK_SIZE elements
    """
    # Calculate the starting index for this program instance
    pid = tl.program_id(axis=0)  # Get the program ID along axis 0
    block_start = pid * BLOCK_SIZE  # Starting index for this instance
    
    # Calculate indices for elements this instance will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle cases where we go beyond the tensor bounds
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the addition
    output = x + y
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def add_tensors(x: torch.Tensor, y: torch.Tensor):
    """
    Host function that launches the Triton kernel to add two tensors
    """
    # Ensure inputs are on the same device
    assert x.device == y.device, "Input tensors must be on the same device"
    
    # Ensure inputs have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate total number of elements
    n_elements = output.numel()
    
    # Define block size (number of elements each program instance handles)
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of program instances to launch)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    # Launch the kernel
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# Example usage
if __name__ == "__main__":
    # Create sample tensors on GPU
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    
    # Perform addition using our Triton kernel
    result = add_tensors(x, y)
    
    # Verify correctness by comparing with PyTorch
    expected = x + y
    print(f"Results match: {(result - expected).abs().max() < 1e-4}")
    
    # Print some values to verify
    print(f"Input x (first 5): {x[:5]}")
    print(f"Input y (first 5): {y[:5]}")
    print(f"Triton result (first 5): {result[:5]}")
    print(f"PyTorch result (first 5): {expected[:5]}")