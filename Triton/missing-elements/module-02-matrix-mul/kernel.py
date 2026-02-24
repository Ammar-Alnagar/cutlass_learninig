"""
Module 02: Matrix Multiplication - Missing Elements Challenge

CHALLENGE: This matrix multiplication kernel has critical missing elements.
Fix all the missing pieces to make 2D matrix multiplication work correctly.

MISSING ELEMENTS TO FIND:
1. Missing import statements
2. Missing @triton.jit decorator  
3. Missing program ID calculations for 2D grid
4. Missing row and column index calculations
5. Missing accumulator initialization
6. Missing inner loop for K dimension
7. Missing tl.dot() operation
8. Missing boundary masks
9. Missing tl.load() with proper offsets
10. Missing tl.store() operation
"""

# HINT 1: You need to import triton and triton.language


# HINT 2: The kernel function needs a decorator


def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Matrix multiplication kernel: C = A @ B
    
    A: [M, K]
    B: [K, N]
    C: [M, N]
    """
    # HINT 3: Get program ID for 2D grid
    pid = # TODO: Get program ID
    
    # HINT 4: Calculate program ID for M dimension (with swizzling)
    pid_m = # TODO: Calculate pid_m
    
    # HINT 5: Calculate program ID for N dimension
    pid_n = # TODO: Calculate pid_n
    
    # HINT 6: Calculate starting row index
    start_m = # TODO: Calculate start_m
    
    # HINT 7: Calculate starting column index
    start_n = # TODO: Calculate start_n
    
    # HINT 8: Create row offsets
    offs_am = # TODO: Create row offsets
    
    # HINT 9: Create column offsets
    offs_bn = # TODO: Create column offsets
    
    # HINT 10: Initialize accumulator for dot product
    accumulator = # TODO: Initialize to zeros
    
    # HINT 11: Loop over K dimension in blocks of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # HINT 12: Create K offsets
        offs_k = # TODO: Create k offsets
        
        # HINT 13: Load block from A matrix
        a_block = # TODO: Load from A
        
        # HINT 14: Load block from B matrix
        b_block = # TODO: Load from B
        
        # HINT 15: Perform matrix multiplication for this block
        accumulator = # TODO: Accumulate dot product
    
    # HINT 16: Create output mask for boundaries
    output_mask = # TODO: Create mask for M and N boundaries
    
    # HINT 17: Store result to C matrix
    # TODO: Store accumulator to c_ptr


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Launch the matrix multiplication kernel.
    
    MISSING ELEMENTS:
    1. Output tensor allocation
    2. Grid calculation for 2D launch
    3. Kernel launch with all stride arguments
    4. Block size configurations
    """
    assert a.is_cuda and b.is_cuda, "Input tensors must be on CUDA"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
    
    M, K = a.shape
    K2, N = b.shape
    
    # HINT 18: Choose block sizes (typically powers of 2)
    BLOCK_SIZE_M = # TODO: Set block size M
    BLOCK_SIZE_N = # TODO: Set block size N
    BLOCK_SIZE_K = # TODO: Set block size K
    
    # HINT 19: Allocate output tensor with correct shape and dtype
    c = # TODO: Allocate output tensor
    
    # HINT 20: Calculate 2D grid
    grid = # TODO: Calculate grid (M, N)
    
    # HINT 21: Launch kernel with all required arguments
    # TODO: Call matmul_kernel with proper arguments including strides
    
    return c


if __name__ == "__main__":
    import torch
    
    # Test with small matrices
    M, N, K = 128, 128, 64
    
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    try:
        output = matmul(a, b)
        expected = a @ b
        assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2), "Results don't match!"
        print("✓ Matrix multiplication kernel works correctly!")
    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        print("Review the hints and add the missing elements.")
