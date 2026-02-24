"""
Module 01: Vector Addition - Missing Elements Challenge

CHALLENGE: This kernel has several missing elements that prevent it from working.
Your task is to identify and add the missing pieces to make the kernel functional.

MISSING ELEMENTS TO FIND:
1. Missing import statement
2. Missing @triton.jit decorator
3. Missing tl.load() operations
4. Missing tl.store() operation
5. Missing block pointer calculation
6. Missing mask for boundary handling
"""

# HINT 1: You need to import triton and triton.language

# HINT 2: The kernel function needs a decorator


def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # HINT 3: Calculate the starting index for this program
    pid = # TODO: Get program ID
    
    # HINT 4: Calculate block start index
    block_start = # TODO: Calculate block start
    
    # HINT 5: Create offsets for the block
    offsets = # TODO: Create offset range
    
    # HINT 6: Calculate actual indices
    indices = # TODO: Combine block_start with offsets
    
    # HINT 7: Create mask for boundary checking
    mask = # TODO: Create mask to handle boundaries
    
    # HINT 8: Load data from input pointers
    x = # TODO: Load from x_ptr
    y = # TODO: Load from y_ptr
    
    # HINT 9: Perform element-wise addition
    output = # TODO: Add x and y
    
    # HINT 10: Store result to output pointer
    # TODO: Store output to output_ptr


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Launch the vector addition kernel.
    
    MISSING ELEMENTS:
    1. Output tensor allocation
    2. Grid calculation
    3. Kernel launch with proper arguments
    4. BLOCK_SIZE specification
    """
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA"
    assert x.shape == y.shape, "Input tensors must have same shape"
    
    n_elements = x.numel()
    
    # HINT 11: Choose an appropriate block size (power of 2, typically 256-1024)
    BLOCK_SIZE = # TODO: Set block size
    
    # HINT 12: Allocate output tensor
    output = # TODO: Allocate output tensor
    
    # HINT 13: Calculate grid dimensions
    grid = # TODO: Calculate grid size
    
    # HINT 14: Launch the kernel
    # TODO: Call vector_add_kernel with proper arguments
    
    return output


if __name__ == "__main__":
    import torch
    
    # Test the kernel
    n = 10000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    
    try:
        output = vector_add(x, y)
        expected = x + y
        assert torch.allclose(output, expected, rtol=1e-5), "Results don't match!"
        print("✓ Vector addition kernel works correctly!")
    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        print("Review the hints and add the missing elements.")
