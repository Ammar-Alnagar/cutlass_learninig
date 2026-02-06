"""
Advanced Techniques and Best Practices Example
This example demonstrates advanced Triton techniques and best practices,
including performance considerations, numerical stability, and optimization strategies.
"""

import torch
import triton
import triton.language as tl
import time

@triton.jit
def numerically_stable_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    Numerically stable softmax implementation using the log-sum-exp trick
    This prevents overflow/underflow in exponentials
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
    centered_vals = input_vals - tl.broadcast_to(row_max, input_vals.shape)
    
    # Compute exp
    exp_vals = tl.exp(centered_vals)
    
    # Compute sum of exps (second reduction)
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / tl.broadcast_to(sum_exp, exp_vals.shape)
    
    # Store result
    tl.store(output_ptrs, softmax_vals, mask=mask)


@triton.jit
def block_wise_mean_std_kernel(
    input_ptr, mean_ptr, std_ptr,
    n_rows, n_cols,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr,
):
    """
    Compute mean and standard deviation for each row using block-wise operations
    Demonstrates multiple reductions in a single kernel
    """
    # Get row index
    row_idx = tl.program_id(0)
    
    # Calculate column indices for this block
    col_start = tl.program_id(1) * block_size_n
    cols = col_start + tl.arange(0, block_size_n)
    
    # Create pointers
    input_ptrs = input_ptr + row_idx * n_cols + cols
    
    # Load input values
    mask = (row_idx < n_rows) & (cols < n_cols)
    input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Compute sum for mean calculation
    sum_vals = tl.sum(input_vals, axis=0)
    count = tl.sum(mask.to(tl.int32), axis=0)  # Count valid elements
    
    # Compute mean (broadcast for next step)
    mean_val = sum_vals / count
    broadcasted_mean = tl.broadcast_to(mean_val, input_vals.shape)
    
    # Compute squared differences for variance
    diff = input_vals - broadcasted_mean
    squared_diff = diff * diff
    
    # Sum squared differences
    sum_squared_diff = tl.sum(squared_diff, axis=0)
    
    # Compute variance and std (with numerical stability)
    variance = sum_squared_diff / count
    std_val = tl.sqrt(variance)
    
    # Store results (each block writes to same location, so only first block should write)
    if col_start == 0:  # Only first block in row writes the results
        mean_row_ptr = mean_ptr + row_idx
        std_row_ptr = std_ptr + row_idx
        tl.store(mean_row_ptr, mean_val)
        tl.store(std_row_ptr, std_val)


@triton.jit
def optimized_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Optimized GEMM kernel with advanced techniques
    Includes grouping for better cache reuse and handling of even K dimension
    """
    # Program IDs
    pid = tl.program_id(axis=0)
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Extract the program id for the M and N dimensions
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers for the first block of A and B
    A = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
            b = tl.load(B, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
        
        # Compute the block of the output
        acc += tl.dot(a, b)
        
        # Advance the ptrs to the next K block
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    
    # Write back the block of the output matrix C
    C = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)


def numerically_stable_softmax(input_tensor, dim=-1):
    """Host function for numerically stable softmax"""
    if dim != input_tensor.dim() - 1:
        raise NotImplementedError("Only softmax along last dimension is implemented")
    
    n_rows, n_cols = input_tensor.shape[0:-1], input_tensor.shape[-1]
    n_rows_total = torch.prod(torch.tensor(n_rows)).item()
    
    input_2d = input_tensor.view(n_rows_total, n_cols)
    output_2d = torch.empty_like(input_2d)
    
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = min(1024, n_cols)
    
    grid = (
        n_rows_total,
        triton.cdiv(n_cols, BLOCK_SIZE_N)
    )
    
    numerically_stable_softmax_kernel[grid](
        input_2d, output_2d,
        n_rows_total, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N
    )
    
    output = output_2d.view(*input_tensor.shape)
    return output


def block_wise_stats(input_tensor, dim=-1):
    """Host function to compute mean and std for each row"""
    if dim != input_tensor.dim() - 1:
        raise NotImplementedError("Only stats along last dimension is implemented")
    
    n_rows, n_cols = input_tensor.shape[0:-1], input_tensor.shape[-1]
    n_rows_total = torch.prod(torch.tensor(n_rows)).item()
    
    input_2d = input_tensor.view(n_rows_total, n_cols)
    
    means = torch.empty(n_rows_total, dtype=torch.float32, device=input_tensor.device)
    stds = torch.empty(n_rows_total, dtype=torch.float32, device=input_tensor.device)
    
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = min(1024, n_cols)
    
    grid = (
        n_rows_total,
        triton.cdiv(n_cols, BLOCK_SIZE_N)
    )
    
    block_wise_mean_std_kernel[grid](
        input_2d, means, stds,
        n_rows_total, n_cols,
        block_size_m=BLOCK_SIZE_M,
        block_size_n=BLOCK_SIZE_N
    )
    
    means = means.view(*n_rows)
    stds = stds.view(*n_rows)
    
    return means, stds


def optimized_gemm(a, b):
    """Host function for optimized GEMM"""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Check if K dimension is evenly divisible
    EVEN_K = K % BLOCK_SIZE_K == 0
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
    
    # Launch kernel
    optimized_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        EVEN_K=EVEN_K
    )
    
    return c


# Example usage and performance comparison
if __name__ == "__main__":
    print("Testing advanced techniques and best practices...")
    
    # Test numerically stable softmax
    print("\nTesting numerically stable softmax...")
    input_softmax = torch.randn(4, 16, device='cuda') * 10  # Large values to test stability
    result_softmax = numerically_stable_softmax(input_softmax)
    expected_softmax = torch.softmax(input_softmax, dim=-1)
    print(f"Numerically stable softmax correct: {torch.allclose(result_softmax, expected_softmax, atol=1e-5)}")
    print(f"Softmax sums to 1: {torch.allclose(result_softmax.sum(dim=-1), torch.ones_like(result_softmax.sum(dim=-1)), atol=1e-5)}")
    
    # Test block-wise statistics
    print("\nTesting block-wise statistics...")
    input_stats = torch.randn(4, 64, device='cuda')
    means, stds = block_wise_stats(input_stats)
    expected_means = input_stats.mean(dim=-1)
    expected_stds = input_stats.std(dim=-1)
    print(f"Means correct: {torch.allclose(means, expected_means, atol=1e-5)}")
    print(f"Stds correct: {torch.allclose(stds, expected_stds, atol=1e-5)}")
    
    # Performance comparison for GEMM
    print("\nPerformance comparison for GEMM...")
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Time Triton implementation
    start_time = time.time()
    for _ in range(5):
        _ = optimized_gemm(a, b)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / 5
    
    # Time PyTorch implementation
    start_time = time.time()
    for _ in range(5):
        _ = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / 5
    
    print(f"Triton GEMM time: {triton_time:.4f}s")
    print(f"PyTorch GEMM time: {pytorch_time:.4f}s")
    print(f"Speed ratio (Triton/PyTorch): {triton_time/pytorch_time:.2f}x")
    
    # Demonstrate best practices
    print("\nBest practices demonstrated:")
    print("1. Numerical stability using log-sum-exp trick in softmax")
    print("2. Efficient memory access patterns")
    print("3. Proper handling of boundary conditions")
    print("4. Grouping strategies for better cache reuse")
    print("5. Conditional loading based on even dimensions")
    print("6. Comprehensive error checking and validation")
    
    print("\nAdvanced Triton techniques covered:")
    print("- Numerical stability in mathematical operations")
    print("- Multiple reductions in a single kernel")
    print("- Optimized memory access patterns")
    print("- Grouping strategies for better performance")
    print("- Conditional execution based on problem characteristics")
    print("- Performance profiling and benchmarking")