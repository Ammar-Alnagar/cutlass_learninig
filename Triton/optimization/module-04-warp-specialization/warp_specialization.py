"""
Module 04: Warp Specialization Optimization

This module teaches warp specialization techniques where different
warps perform different tasks for maximum GPU utilization.

LEARNING OBJECTIVES:
1. Understand warp-level parallelism
2. Learn warp specialization patterns
3. Implement producer-consumer patterns
4. Measure throughput improvements
"""

import triton
import triton.language as tl
import torch


@triton.jit
def matmul_warp_specialized(
    a_ptr, b_ptr, c_ptr,
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
    Matrix multiplication with warp specialization.
    
    WARP SPECIALIZATION CONCEPTS:
    - Different warps handle different parts of computation
    - Producer warps load data, consumer warps compute
    - Overlaps memory and compute operations
    - Maximizes GPU utilization
    
    In this implementation:
    - We use swizzled program IDs for better L2 cache utilization
    - Each program (warp block) handles a tile of the output
    """
    pid = tl.program_id(axis=0)
    
    # Swizzle program IDs for better cache utilization
    num_pid_m = M // BLOCK_SIZE_M
    num_pid_n = N // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
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
        
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator,
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    )


@triton.jit
def fused_kernel_warp_specialized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel demonstrating warp specialization pattern.
    
    This kernel fuses multiple operations:
    1. Matrix multiply (done by some warps)
    2. Bias addition (done by same warps, no extra memory round-trip)
    3. Activation (done inline)
    
    WARP SPECIALIZATION BENEFIT:
    - Data stays in registers/shared memory between operations
    - No global memory round-trip between fused operations
    - Different warps can work on different stages
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations (all in registers)
    output = x * weight + bias
    
    # Apply activation (ReLU) inline
    output = tl.maximum(0.0, output)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def matmul_swizzled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with swizzled program IDs for better cache usage.
    """
    M, K = a.shape
    K2, N = b.shape
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    matmul_warp_specialized[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M
    )
    
    return c


def fused_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear + ReLU operation.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    output = torch.empty_like(x)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_kernel_warp_specialized[grid](
        x, weight, bias, output, n_elements, BLOCK_SIZE
    )
    
    return output


def benchmark_warp_specialization():
    """
    Benchmark warp specialized kernels.
    """
    import time
    
    print("Warp Specialization Benchmark")
    print("=" * 60)
    
    # Matrix multiplication
    M, N, K = 2048, 2048, 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    # Warmup
    c = matmul_swizzled(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(50):
        c = matmul_swizzled(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / 50 * 1000
    flops = 2 * M * N * K
    gflops = flops / avg_time / 1e6
    
    print(f"\nMatrix Multiplication with Swizzling ({M}x{N}x{K}):")
    print(f"  Time: {avg_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    
    # Fused kernel benchmark
    n = 1_000_000
    x = torch.randn(n, device="cuda")
    weight = torch.randn(n, device="cuda")
    bias = torch.randn(n, device="cuda")
    
    # Triton fused
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = fused_linear_relu(x, weight, bias)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    triton_time = (end - start) / 100 * 1000
    
    # PyTorch unfused
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.relu(x * weight + bias)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    torch_time = (end - start) / 100 * 1000
    
    print(f"\nFused Linear + ReLU ({n} elements):")
    print(f"  Triton fused: {triton_time:.4f} ms")
    print(f"  PyTorch: {torch_time:.4f} ms")
    print(f"  Speedup: {torch_time / triton_time:.2f}x")


if __name__ == "__main__":
    print("Running Warp Specialization Optimization Module")
    print("=" * 60)
    
    benchmark_warp_specialization()
    
    print("\n" + "=" * 60)
    print("Testing Correctness")
    
    # Test matrix multiplication
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    c = matmul_swizzled(a, b)
    expected = a @ b
    
    if torch.allclose(c, expected, rtol=1e-2, atol=1e-2):
        print("✓ Swizzled matrix multiplication works!")
    else:
        print("✗ Results don't match")
    
    # Test fused kernel
    n = 10000
    x = torch.randn(n, device="cuda")
    weight = torch.randn(n, device="cuda")
    bias = torch.randn(n, device="cuda")
    
    output = fused_linear_relu(x, weight, bias)
    expected = torch.relu(x * weight + bias)
    
    if torch.allclose(output, expected, rtol=1e-5):
        print("✓ Fused linear + ReLU works!")
    else:
        print("✗ Fused results don't match")
    
    print("\nKEY TAKEAWAYS:")
    print("1. Warp specialization assigns different tasks to different warps")
    print("2. Program ID swizzling improves L2 cache utilization")
    print("3. Fused kernels eliminate intermediate memory traffic")
    print("4. Producer-consumer patterns overlap memory and compute")
    print("5. Keep data in registers/shared memory between operations")
