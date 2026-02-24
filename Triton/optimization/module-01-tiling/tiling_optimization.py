"""
Module 01: Tiling Optimization

This module teaches tiling optimization techniques for Triton kernels.
Tiling is crucial for maximizing memory coalescing and cache utilization.

LEARNING OBJECTIVES:
1. Understand tiling fundamentals
2. Learn to choose optimal block sizes
3. Implement multi-level tiling
4. Measure performance improvements
"""

import triton
import triton.language as tl
import torch


@triton.jit
def vector_add_tiled(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector addition with optimized tiling.
    
    TILING CONCEPTS:
    - Each program processes BLOCK_SIZE elements
    - Memory accesses are coalesced within a block
    - Block size should be a power of 2 for efficiency
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets within the tile
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load, compute, store
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def benchmark_tiling():
    """
    Benchmark different tile sizes to find optimal configuration.
    """
    import time
    
    n_elements = 10_000_000
    x = torch.randn(n_elements, device="cuda")
    y = torch.randn(n_elements, device="cuda")
    output = torch.empty(n_elements, device="cuda")
    
    # Test different block sizes
    block_sizes = [64, 128, 256, 512, 1024, 2048]
    
    print("Tiling Optimization Benchmark")
    print("=" * 50)
    print(f"{'Block Size':<15} {'Time (ms)':<15} {'GB/s':<15}")
    print("-" * 50)
    
    for BLOCK_SIZE in block_sizes:
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Warmup
        vector_add_tiled[grid](x, y, output, n_elements, BLOCK_SIZE)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            vector_add_tiled[grid](x, y, output, n_elements, BLOCK_SIZE)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) / 100 * 1000  # ms
        bytes_processed = n_elements * 4 * 3  # 2 reads, 1 write, 4 bytes each
        bandwidth = bytes_processed / avg_time / 1e6  # GB/s
        
        print(f"{BLOCK_SIZE:<15} {avg_time:<15.3f} {bandwidth:<15.2f}")
    
    # Compare with PyTorch
    start = time.perf_counter()
    for _ in range(100):
        _ = x + y
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    torch_time = (end - start) / 100 * 1000
    print("-" * 50)
    print(f"{'PyTorch':<15} {torch_time:<15.3f}")


def matrix_mul_tiled(
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
) -> torch.Tensor:
    """
    Matrix multiplication with 2D tiling optimization.
    
    TILING STRATEGY:
    - BLOCK_SIZE_M x BLOCK_SIZE_N output tile
    - BLOCK_SIZE_K reduction tile
    - Each program computes one output tile
    """
    
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            
            a_block = tl.load(
                a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
                other=0.0
            )
            
            b_block = tl.load(
                b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
                other=0.0
            )
            
            accumulator += tl.dot(a_block, b_block)
        
        offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        tl.store(
            c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
            accumulator,
            mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        )
    
    M, K = a.shape
    K2, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c


if __name__ == "__main__":
    print("Running Tiling Optimization Module")
    print("=" * 60)
    
    # Run benchmark
    benchmark_tiling()
    
    print("\n" + "=" * 60)
    print("Testing Matrix Multiplication with Tiling")
    
    # Test matrix multiplication
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    output = matrix_mul_tiled(a, b)
    expected = a @ b
    
    if torch.allclose(output, expected, rtol=1e-2, atol=1e-2):
        print("✓ Matrix multiplication with tiling works correctly!")
    else:
        print("✗ Results don't match")
    
    print("\nKEY TAKEAWAYS:")
    print("1. Block size significantly impacts performance")
    print("2. Powers of 2 are generally optimal for block sizes")
    print("3. Larger blocks reduce kernel launch overhead")
    print("4. But too large blocks may reduce occupancy")
    print("5. Always benchmark for your specific hardware!")
