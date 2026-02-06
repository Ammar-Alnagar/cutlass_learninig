"""
Block Operations and Tiling Example: 2D Matrix Processing
This example demonstrates how to work with 2D blocks and tiling concepts in Triton.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def tile_copy_2d_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    2D kernel that processes a tile of the matrix at a time
    Demonstrates 2D block operations and tiling
    """
    # Get program IDs for row and column dimensions
    pid_m = tl.program_id(axis=0)  # Row block index
    pid_n = tl.program_id(axis=1)  # Column block index
    
    # Calculate starting row and column for this block
    rm = pid_m * block_size_m  # Starting row
    rn = pid_n * block_size_n  # Starting column
    
    # Generate row and column indices for this block
    offs_m = rm + tl.arange(0, block_size_m)  # Row indices
    offs_n = rn + tl.arange(0, block_size_n)  # Column indices
    
    # Create masks to handle boundary conditions
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    
    # Combine masks for 2D access
    mask = mask_m[:, None] & mask_n[None, :]  # Broadcasting to create 2D mask
    
    # Calculate linear indices for memory access
    input_offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # Load and store data
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + input_offsets, input_vals, mask=mask)


@triton.jit
def element_wise_2d_kernel(
    input1_ptr, input2_ptr, output_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    2D kernel that performs element-wise operations on matrices
    Demonstrates 2D indexing and computation
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
    
    # Calculate linear offsets
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # Load data from both input matrices
    input1_vals = tl.load(input1_ptr + offsets, mask=mask)
    input2_vals = tl.load(input2_ptr + offsets, mask=mask)
    
    # Perform element-wise operation (addition in this case)
    output_vals = input1_vals + input2_vals
    
    # Store result
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.jit
def transpose_tile_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    2D kernel that transposes a tile of the matrix
    Demonstrates how to swap rows and columns in tiled fashion
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
    
    # Calculate linear offsets for input (row-major order)
    input_offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # Load input tile
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Transpose: swap row and column indices for output
    output_offsets = offs_n[:, None] + offs_m[None, :] * n_rows  # Note: dimensions swapped
    
    # Store transposed tile
    tl.store(output_ptr + output_offsets.T, input_vals, mask=mask)


def tile_copy_2d(input_matrix):
    """Host function to copy a 2D matrix using tiling"""
    output_matrix = torch.empty_like(input_matrix)
    
    n_rows, n_cols = input_matrix.shape
    
    # Define tile sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(n_rows, BLOCK_SIZE_M),
        triton.cdiv(n_cols, BLOCK_SIZE_N),
    )
    
    tile_copy_2d_kernel[grid](
        input_matrix, output_matrix,
        n_rows, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N,
    )
    
    return output_matrix


def element_wise_2d(input1, input2):
    """Host function for element-wise operation on 2D matrices"""
    assert input1.shape == input2.shape
    output = torch.empty_like(input1)
    
    n_rows, n_cols = input1.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = (
        triton.cdiv(n_rows, BLOCK_SIZE_M),
        triton.cdiv(n_cols, BLOCK_SIZE_N),
    )
    
    element_wise_2d_kernel[grid](
        input1, input2, output,
        n_rows, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N,
    )
    
    return output


def transpose_tile(input_matrix):
    """Host function to transpose a matrix using tiling"""
    n_rows, n_cols = input_matrix.shape
    # Output matrix has swapped dimensions
    output_matrix = torch.empty((n_cols, n_rows), dtype=input_matrix.dtype, device=input_matrix.device)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = (
        triton.cdiv(n_rows, BLOCK_SIZE_M),
        triton.cdiv(n_cols, BLOCK_SIZE_N),
    )
    
    transpose_tile_kernel[grid](
        input_matrix, output_matrix,
        n_rows, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N,
    )
    
    return output_matrix


# Example usage
if __name__ == "__main__":
    # Create sample 2D tensors
    M, N = 128, 128
    input1 = torch.randn(M, N, device='cuda')
    input2 = torch.randn(M, N, device='cuda')
    
    print("Testing 2D block operations...")
    
    # Test tile-based copy
    result_copy = tile_copy_2d(input1)
    print(f"Tile copy correct: {torch.allclose(result_copy, input1)}")
    
    # Test element-wise operation
    result_elementwise = element_wise_2d(input1, input2)
    expected_elementwise = input1 + input2
    print(f"Element-wise operation correct: {torch.allclose(result_elementwise, expected_elementwise)}")
    
    # Test transpose
    result_transpose = transpose_tile(input1)
    expected_transpose = input1.T
    print(f"Transpose correct: {torch.allclose(result_transpose, expected_transpose)}")
    
    print(f"\nInput shape: {input1.shape}")
    print(f"Transposed shape: {result_transpose.shape}")
    
    # Demonstrate tiling effect
    print(f"\nTiling parameters:")
    print(f"- Tile size: {32}x{32}")
    print(f"- Grid size: {triton.cdiv(M, 32)}x{triton.cdiv(N, 32)}")
    print(f"- Total tiles: {triton.cdiv(M, 32) * triton.cdiv(N, 32)}")