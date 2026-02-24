"""
Module 03: Softmax - Missing Elements Challenge

CHALLENGE: This softmax kernel has missing elements for the two-pass algorithm.
Softmax requires: 1) Find max, 2) Compute exp sum, 3) Normalize

MISSING ELEMENTS TO FIND:
1. Missing import statements
2. Missing @triton.jit decorator
3. Missing first pass: finding maximum value
4. Missing tl.maximum() for reduction
5. Missing second pass: computing exp and sum
6. Missing tl.exp() operation
7. Missing tl.sum() for normalization
8. Missing third pass: final normalization
9. Missing proper masking
"""

# HINT 1: You need to import triton and triton.language


# HINT 2: The kernel function needs a decorator


def softmax_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    stride_x,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel: output = softmax(x)
    
    Implements the numerically stable softmax:
    1. Subtract max for numerical stability
    2. Compute exp(x - max)
    3. Normalize by sum
    """
    # HINT 3: Get program ID (each program handles one row)
    pid = # TODO: Get program ID
    
    # HINT 4: Calculate row offset
    row_start = # TODO: Calculate row start index
    
    # HINT 5: Create column offsets
    col_offsets = # TODO: Create offset range
    
    # HINT 6: Create mask for valid elements
    mask = # TODO: Create mask
    
    # ===== FIRST PASS: Find Maximum =====
    # HINT 7: Load the row data
    row = # TODO: Load from x_ptr
    
    # HINT 8: Find maximum value in the row (for numerical stability)
    row_max = # TODO: Find maximum using tl.max
    
    # ===== SECOND PASS: Compute exp and sum =====
    # HINT 9: Subtract max and compute exp
    exp_row = # TODO: Compute exp(row - row_max)
    
    # HINT 10: Compute sum of exp values
    exp_sum = # TODO: Compute sum using tl.sum
    
    # ===== THIRD PASS: Normalize =====
    # HINT 11: Normalize by dividing by sum
    output = # TODO: Divide exp_row by exp_sum
    
    # HINT 12: Store result to output pointer
    # TODO: Store output to output_ptr


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Launch the softmax kernel.
    
    MISSING ELEMENTS:
    1. Output tensor allocation
    2. Grid calculation (one program per row)
    3. Kernel launch with stride arguments
    4. BLOCK_SIZE selection
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.dim() == 2, "Softmax kernel expects 2D tensor"
    
    n_rows, n_cols = x.shape
    
    # HINT 13: Choose block size (should cover n_cols or use multiple blocks)
    BLOCK_SIZE = # TODO: Set block size (use next power of 2 >= n_cols)
    
    # HINT 14: Allocate output tensor
    output = # TODO: Allocate output tensor
    
    # HINT 15: Calculate grid (one program per row)
    grid = # TODO: Calculate grid size
    
    # HINT 16: Launch kernel
    # TODO: Call softmax_kernel with proper arguments
    
    return output


if __name__ == "__main__":
    import torch
    
    # Test the kernel
    n_rows, n_cols = 32, 128
    x = torch.randn(n_rows, n_cols, device="cuda")
    
    try:
        output = softmax(x)
        expected = torch.softmax(x, dim=1)
        assert torch.allclose(output, expected, rtol=1e-4), "Results don't match!"
        print("✓ Softmax kernel works correctly!")
        
        # Verify softmax properties
        row_sums = output.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n_rows, device="cuda")), "Softmax rows should sum to 1!"
        print("✓ Softmax properties verified (rows sum to 1)!")
    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        print("Review the hints and add the missing elements.")
