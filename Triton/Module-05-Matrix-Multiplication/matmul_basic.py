"""
Matrix Multiplication Example: Basic and Optimized Approaches
This example demonstrates matrix multiplication in Triton, starting with a basic implementation
and moving toward more optimized approaches.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_basic(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr` 
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Basic matrix multiplication kernel using tiling
    Computes C = A x B where A is (M, K), B is (K, N), and C is (M, N)
    """
    # Indices for the block of C it computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
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


@triton.jit
def matmul_kernel_optimized(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel with better thread cooperation
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


def matmul_basic(a, b):
    """
    Host function to perform matrix multiplication using Triton
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # 1D launch kernel where each block gets its own program
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel_basic[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=32,
    )
    
    return c


def matmul_optimized(a, b):
    """
    Host function to perform optimized matrix multiplication using Triton
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # 1D launch kernel where each block gets its own program
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
    
    # Launch kernel
    matmul_kernel_optimized[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    
    return c


# Example usage
if __name__ == "__main__":
    # Create sample matrices
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    print(f"Matrix dimensions: A({M}x{K}), B({K}x{N}), C({M}x{N})")
    
    # Test basic matmul
    print("\nTesting basic matrix multiplication...")
    result_basic = matmul_basic(a, b)
    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    print(f"Basic matmul correct: {torch.allclose(result_basic, expected, atol=1e-2)}")
    
    # Test optimized matmul
    print("\nTesting optimized matrix multiplication...")
    result_optimized = matmul_optimized(a, b)
    print(f"Optimized matmul correct: {torch.allclose(result_optimized, expected, atol=1e-2)}")
    
    # Performance comparison
    import time
    
    print("\nPerformance comparison (approximate):")
    
    # Time basic version
    start_time = time.time()
    for _ in range(10):
        _ = matmul_basic(a, b)
    torch.cuda.synchronize()
    basic_time = (time.time() - start_time) / 10
    print(f"Basic matmul average time: {basic_time:.4f}s")
    
    # Time optimized version
    start_time = time.time()
    for _ in range(10):
        _ = matmul_optimized(a, b)
    torch.cuda.synchronize()
    optimized_time = (time.time() - start_time) / 10
    print(f"Optimized matmul average time: {optimized_time:.4f}s")
    
    print(f"Speedup: {basic_time / optimized_time:.2f}x")
    
    # Compare with PyTorch
    start_time = time.time()
    for _ in range(10):
        _ = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / 10
    print(f"PyTorch matmul average time: {pytorch_time:.4f}s")