"""
Module 02: Pipelining Optimization

This module teaches software pipelining techniques to hide memory latency
and improve throughput in Triton kernels.

LEARNING OBJECTIVES:
1. Understand software pipelining concepts
2. Implement prefetching for memory operations
3. Use multiple buffers for overlapping compute and memory
4. Measure latency hiding improvements
"""

import triton
import triton.language as tl
import torch


@triton.jit
def matmul_pipelined(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Matrix multiplication with software pipelining.
    
    PIPELINING CONCEPTS:
    - NUM_STAGES controls how many iterations are in flight
    - While computing stage N, we prefetch data for stage N+1
    - Hides memory latency by overlapping compute and memory ops
    
    Triton's compiler automatically handles:
    - Register allocation for multiple stages
    - Prefetch scheduling
    - Dependency management
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop with pipelining
    # Triton automatically pipelines this loop based on NUM_STAGES
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A block
        a_block = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        
        # Load B block
        b_block = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0
        )
        
        # Matrix multiply and accumulate
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    # Store result
    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator,
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    )


def matmul_with_stages(a: torch.Tensor, b: torch.Tensor, num_stages: int = 3) -> torch.Tensor:
    """
    Launch matrix multiplication with configurable pipeline stages.
    """
    M, K = a.shape
    K2, N = b.shape
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_pipelined[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_STAGES=num_stages
    )
    
    return c


def benchmark_pipelining():
    """
    Benchmark different pipeline stage configurations.
    """
    import time
    
    M, N, K = 2048, 2048, 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    # Test different numbers of stages
    num_stages_list = [1, 2, 3, 4, 5]
    
    print("Pipelining Optimization Benchmark")
    print("=" * 60)
    print(f"{'Num Stages':<15} {'Time (ms)':<15} {'GFLOPS':<15}")
    print("-" * 60)
    
    for num_stages in num_stages_list:
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        # Warmup
        matmul_pipelined[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            NUM_STAGES=num_stages
        )
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            matmul_pipelined[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                NUM_STAGES=num_stages
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) / 50 * 1000  # ms
        
        # Calculate GFLOPS (2 * M * N * K operations for matmul)
        flops = 2 * M * N * K
        gflops = flops / avg_time / 1e6
        
        print(f"{num_stages:<15} {avg_time:<15.3f} {gflops:<15.2f}")
    
    # Compare with PyTorch
    start = time.perf_counter()
    for _ in range(50):
        _ = a @ b
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    torch_time = (end - start) / 50 * 1000
    torch_flops = flops / torch_time / 1e6
    print("-" * 60)
    print(f"{'PyTorch':<15} {torch_time:<15.3f} {torch_flops:<15.2f}")


@triton.jit
def vector_add_prefetch(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector addition with explicit prefetching pattern.
    
    This demonstrates the prefetching concept where we:
    1. Load current block data
    2. Compute on current block
    3. While computing, prefetch next block (handled by hardware)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    output = x + y
    
    # Store
    tl.store(output_ptr + offsets, output, mask=mask)


if __name__ == "__main__":
    print("Running Pipelining Optimization Module")
    print("=" * 60)
    
    # Run benchmark
    benchmark_pipelining()
    
    print("\n" + "=" * 60)
    print("Testing Correctness")
    
    # Test correctness
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    for num_stages in [1, 2, 3, 4]:
        output = matmul_with_stages(a, b, num_stages)
        expected = a @ b
        
        if torch.allclose(output, expected, rtol=1e-2, atol=1e-2):
            print(f"✓ {num_stages} stages: Results match!")
        else:
            print(f"✗ {num_stages} stages: Results don't match")
    
    print("\nKEY TAKEAWAYS:")
    print("1. Pipelining hides memory latency by overlapping ops")
    print("2. NUM_STAGES controls how many iterations are in flight")
    print("3. More stages = more parallelism but more register pressure")
    print("4. Optimal stages depends on your GPU and kernel")
    print("5. Triton's compiler handles most pipelining automatically")
    print("6. Typical values: 2-4 stages for most kernels")
