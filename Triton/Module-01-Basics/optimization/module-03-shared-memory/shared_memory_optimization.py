"""
Module 03: Shared Memory Optimization

This module teaches shared memory (SRAM) optimization techniques
for maximizing data reuse and minimizing global memory traffic.

LEARNING OBJECTIVES:
1. Understand shared memory hierarchy in GPUs
2. Learn to use tl.static_range for compile-time loops
3. Implement blocked algorithms with shared memory
4. Measure reduction in global memory accesses
"""

import triton
import triton.language as tl
import torch


@triton.jit
def matmul_shared(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Matrix multiplication using shared memory optimization.
    
    SHARED MEMORY CONCEPTS:
    - Load blocks of A and B into shared memory (SRAM)
    - Reuse shared data for multiple computations
    - Reduces global memory bandwidth requirements
    - Shared memory is ~100x faster than global memory
    
    In Triton:
    - tl.load with proper masking automatically uses shared memory
    - The compiler manages shared memory allocation
    - Focus on access patterns, not manual management
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Accumulator in registers
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    # Each iteration loads a tile into shared memory
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A tile - automatically cached in shared memory
        a_tile = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        
        # Load B tile - automatically cached in shared memory
        b_tile = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0
        )
        
        # Compute dot product using tiles
        accumulator = tl.dot(a_tile, b_tile, accumulator)
    
    # Store result
    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator,
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    )


@triton.jit
def reduction_shared_memory(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduction (sum) using shared memory for tree reduction.
    
    SHARED MEMORY REDUCTION PATTERN:
    1. Load data into shared memory
    2. Perform tree reduction in shared memory
    3. Store partial result
    4. Final reduction of partial results
    
    This is much faster than atomic operations for large reductions.
    """
    pid = tl.program_id(axis=0)
    
    # Shared memory for this block
    # In Triton, we use tl.where and tl.sum for reduction
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Tree reduction using tl.sum
    # Triton automatically optimizes this using shared memory
    block_sum = tl.sum(x, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + pid, block_sum)


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute sum using shared memory reduction.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # First level reduction
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_sums = torch.empty(grid_size, device=x.device, dtype=torch.float32)
    
    reduction_shared_memory[grid_size](
        x, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Second level reduction (if needed)
    n_partial = partial_sums.numel()
    if n_partial > BLOCK_SIZE:
        final_sum = torch.empty(1, device=x.device, dtype=torch.float32)
        reduction_shared_memory[triton.cdiv(n_partial, BLOCK_SIZE)](
            partial_sums, final_sum, n_partial, BLOCK_SIZE
        )
        # Final reduction on CPU or with another kernel
        return final_sum.sum()
    else:
        return partial_sums.sum()


def benchmark_shared_memory():
    """
    Benchmark shared memory optimized kernels.
    """
    import time
    
    print("Shared Memory Optimization Benchmark")
    print("=" * 60)
    
    # Matrix multiplication benchmark
    M, N, K = 2048, 2048, 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Warmup
    matmul_shared[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(50):
        matmul_shared[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / 50 * 1000
    flops = 2 * M * N * K
    gflops = flops / avg_time / 1e6
    
    print(f"\nMatrix Multiplication ({M}x{N}x{K}):")
    print(f"  Time: {avg_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    
    # Reduction benchmark
    n = 10_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    
    # Triton reduction
    triton_result = reduce_sum(x)
    
    # PyTorch reduction
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = x.sum()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    torch_time = (end - start) / 100 * 1000
    
    print(f"\nReduction (sum of {n} elements):")
    print(f"  PyTorch time: {torch_time:.4f} ms")
    print(f"  Triton result matches: {torch.allclose(triton_result, x.sum(), rtol=1e-4)}")


if __name__ == "__main__":
    print("Running Shared Memory Optimization Module")
    print("=" * 60)
    
    benchmark_shared_memory()
    
    print("\n" + "=" * 60)
    print("Testing Correctness")
    
    # Test matrix multiplication
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_shared[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    expected = a @ b
    if torch.allclose(c, expected, rtol=1e-2, atol=1e-2):
        print("✓ Matrix multiplication with shared memory works!")
    else:
        print("✗ Results don't match")
    
    # Test reduction
    x = torch.randn(100000, device="cuda")
    result = reduce_sum(x)
    if torch.allclose(result, x.sum(), rtol=1e-3):
        print("✓ Shared memory reduction works!")
    else:
        print("✗ Reduction results don't match")
    
    print("\nKEY TAKEAWAYS:")
    print("1. Shared memory (SRAM) is ~100x faster than global memory")
    print("2. Triton automatically uses shared memory for blocked algorithms")
    print("3. Focus on access patterns - the compiler handles the rest")
    print("4. Tree reduction in shared memory is faster than atomics")
    print("5. Block size affects shared memory utilization")
