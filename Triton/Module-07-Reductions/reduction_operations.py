"""
Reduction Operations Example: Sum, Max, and ArgMax
This example demonstrates various reduction operations in Triton,
including sum, max, and argmax reductions along specific axes.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    REDUCE_SIZE: tl.constexpr,
):
    """
    Reduction kernel that computes the sum along the last dimension
    Each program reduces one row of the input tensor
    """
    # Get row index
    row_idx = tl.program_id(0)
    
    # Create pointers to the input row
    input_ptrs = input_ptr + row_idx * input_row_stride + tl.arange(0, REDUCE_SIZE)
    
    # Load the entire row
    mask = tl.arange(0, REDUCE_SIZE) < n_cols
    input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Perform the reduction (sum)
    output_val = tl.sum(input_vals, axis=0)
    
    # Store the result
    output_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_ptr, output_val)


@triton.jit
def max_reduction_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,  # Also compute argmax
    input_row_stride,
    output_row_stride,
    n_cols,
    REDUCE_SIZE: tl.constexpr,
):
    """
    Reduction kernel that computes the max and argmax along the last dimension
    """
    # Get row index
    row_idx = tl.program_id(0)
    
    # Create pointers to the input row
    input_ptrs = input_ptr + row_idx * input_row_stride + tl.arange(0, REDUCE_SIZE)
    
    # Load the entire row
    mask = tl.arange(0, REDUCE_SIZE) < n_cols
    input_vals = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    
    # Perform the reduction (max)
    output_val, max_idx = tl.max(input_vals, axis=0, return_indices=True)
    
    # Store the results
    output_row_ptr = output_ptr + row_idx * output_row_stride
    indices_row_ptr = indices_ptr + row_idx * output_row_stride
    
    tl.store(output_row_ptr, output_val)
    tl.store(indices_row_ptr, max_idx)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    Softmax kernel that applies softmax along the last dimension
    This demonstrates a more complex reduction followed by broadcasting
    """
    # Get row index
    row_idx = tl.program_id(0)
    
    # Calculate column indices for this block
    col_start = tl.program_id(1) * block_size_n
    cols = col_start + tl.arange(0, block_size_n)
    
    # Create pointers
    input_ptrs = input_ptr + row_idx * n_cols + cols
    output_ptrs = output_ptr + row_idx * n_cols + cols
    
    # Load input values
    mask = (row_idx < n_rows) & (cols < n_cols)
    input_vals = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability (first reduction)
    row_max = tl.max(input_vals, axis=0)
    # Broadcast max to all elements in the row
    adjusted_vals = input_vals - tl.broadcast_to(row_max, input_vals.shape)
    
    # Compute exp
    exp_vals = tl.exp(adjusted_vals)
    
    # Compute sum of exps (second reduction)
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / tl.broadcast_to(sum_exp, exp_vals.shape)
    
    # Store result
    tl.store(output_ptrs, softmax_vals, mask=mask)


def sum_reduction(input_tensor, dim=-1):
    """
    Host function to perform sum reduction along a specified dimension
    """
    if dim == -1:
        dim = input_tensor.dim() - 1  # Last dimension
    
    # Calculate output shape
    output_shape = list(input_tensor.shape)
    n_cols = output_shape[dim]
    output_shape[dim] = 1
    output_shape = tuple(output_shape)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten input and output for easier indexing
    if dim == input_tensor.dim() - 1:
        # Reduce along last dimension
        input_flat = input_tensor.view(-1, n_cols)
        output_flat = output.view(-1)
        n_rows = input_flat.size(0)
        
        # Calculate grid
        grid = (n_rows, )
        
        # Launch kernel
        REDUCE_SIZE = triton.next_power_of_2(n_cols)
        sum_reduction_kernel[grid](
            input_flat, output_flat,
            input_flat.stride(0), output_flat.stride(0),
            n_cols,
            REDUCE_SIZE=REDUCE_SIZE
        )
    else:
        # For other dimensions, we would need to transpose, reduce, and transpose back
        # This is a simplified implementation for the last dimension
        raise NotImplementedError("Only reduction along last dimension is implemented in this example")
    
    return output


def max_argmax_reduction(input_tensor, dim=-1):
    """
    Host function to perform max and argmax reduction along a specified dimension
    """
    if dim == -1:
        dim = input_tensor.dim() - 1  # Last dimension
    
    # Calculate output shape
    output_shape = list(input_tensor.shape)
    n_cols = output_shape[dim]
    output_shape[dim] = 1
    output_shape = tuple(output_shape)
    
    # Create output tensors
    max_values = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    argmax_indices = torch.empty(output_shape, dtype=torch.long, device=input_tensor.device)
    
    # Flatten input and output for easier indexing
    if dim == input_tensor.dim() - 1:
        input_flat = input_tensor.view(-1, n_cols)
        max_flat = max_values.view(-1)
        argmax_flat = argmax_indices.view(-1)
        n_rows = input_flat.size(0)
        
        # Calculate grid
        grid = (n_rows, )
        
        # Launch kernel
        REDUCE_SIZE = triton.next_power_of_2(n_cols)
        max_reduction_kernel[grid](
            input_flat, max_flat, argmax_flat,
            input_flat.stride(0), max_flat.stride(0),
            n_cols,
            REDUCE_SIZE=REDUCE_SIZE
        )
    else:
        raise NotImplementedError("Only reduction along last dimension is implemented in this example")
    
    return max_values, argmax_indices


def softmax(input_tensor, dim=-1):
    """
    Host function to apply softmax along a specified dimension
    """
    if dim == -1:
        dim = input_tensor.dim() - 1  # Last dimension
    
    # For simplicity, we'll implement softmax along the last dimension
    if dim != input_tensor.dim() - 1:
        raise NotImplementedError("Only softmax along last dimension is implemented in this example")
    
    n_rows, n_cols = input_tensor.shape[0:-1], input_tensor.shape[-1]
    n_rows_total = torch.prod(torch.tensor(n_rows)).item()
    
    # Reshape to 2D for easier processing
    input_2d = input_tensor.view(n_rows_total, n_cols)
    output_2d = torch.empty_like(input_2d)
    
    # Define block sizes
    BLOCK_SIZE_M = 1  # Process one row at a time
    BLOCK_SIZE_N = min(1024, n_cols)  # Process up to 1024 elements per block
    
    # Calculate grid
    grid = (
        n_rows_total,  # Number of rows
        triton.cdiv(n_cols, BLOCK_SIZE_N)  # Number of blocks per row
    )
    
    # Launch kernel
    softmax_kernel[grid](
        input_2d, output_2d,
        n_rows_total, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N
    )
    
    # Reshape back to original shape
    output = output_2d.view(*input_tensor.shape)
    
    return output


# Example usage
if __name__ == "__main__":
    # Create sample tensors
    input_vec = torch.randn(4, 8, device='cuda')
    input_large = torch.randn(2, 16, device='cuda')
    
    print("Testing reduction operations...")
    
    # Test sum reduction
    print("\nSum reduction:")
    result_sum = sum_reduction(input_vec).squeeze(-1)
    expected_sum = input_vec.sum(dim=-1)
    print(f"Sum reduction correct: {torch.allclose(result_sum, expected_sum, atol=1e-4)}")
    print(f"Input shape: {input_vec.shape}, Output shape: {result_sum.shape}")
    
    # Test max and argmax reduction
    print("\nMax and ArgMax reduction:")
    max_vals, argmax_idx = max_argmax_reduction(input_large)
    max_vals = max_vals.squeeze(-1)
    argmax_idx = argmax_idx.squeeze(-1)
    
    expected_max, expected_argmax = torch.max(input_large, dim=-1)
    print(f"Max reduction correct: {torch.allclose(max_vals, expected_max, atol=1e-4)}")
    print(f"ArgMax reduction correct: {torch.equal(argmax_idx, expected_argmax)}")
    
    # Test softmax
    print("\nSoftmax operation:")
    input_softmax = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    result_softmax = softmax(input_softmax)
    expected_softmax = torch.softmax(input_softmax, dim=-1)
    print(f"Softmax correct: {torch.allclose(result_softmax, expected_softmax, atol=1e-4)}")
    print(f"Softmax sums to 1: {torch.allclose(result_softmax.sum(dim=-1), torch.ones_like(result_softmax.sum(dim=-1)))}")
    
    print(f"\nInput: {input_softmax}")
    print(f"Triton Softmax: {result_softmax}")
    print(f"PyTorch Softmax: {expected_softmax}")
    
    print("\nReduction operations are fundamental for:")
    print("- Pooling operations in neural networks")
    print("- Normalization layers (softmax, layer norm)")
    print("- Loss function computations")
    print("- Attention mechanisms")