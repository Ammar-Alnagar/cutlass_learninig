"""
Module 05: Advanced Optimizations

This module combines all optimization techniques for maximum performance.
Learn advanced patterns used in production Triton kernels.

LEARNING OBJECTIVES:
1. Combine multiple optimization techniques
2. Implement autotuning for optimal parameters
3. Use persistent kernels for maximum occupancy
4. Profile and analyze kernel performance
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# ADVANCED TECHNIQUE 1: Autotuning
# ============================================================================

def get_matmul_configs():
    """
    Generate configuration candidates for autotuning.
    """
    configs = []
    for BLOCK_SIZE_M in [32, 64, 128]:
        for BLOCK_SIZE_N in [32, 64, 128]:
            for BLOCK_SIZE_K in [32, 64]:
                for NUM_STAGES in [2, 3, 4]:
                    configs.append(
                        triton.Config({
                            'BLOCK_SIZE_M': BLOCK_SIZE_M,
                            'BLOCK_SIZE_N': BLOCK_SIZE_N,
                            'BLOCK_SIZE_K': BLOCK_SIZE_K,
                            'NUM_STAGES': NUM_STAGES,
                        })
                    )
    return configs


@triton.autotune(
    configs=get_matmul_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotuned(
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
    Matrix multiplication with autotuning.
    
    AUTOTUNING CONCEPTS:
    - Triton automatically tests different configurations
    - Finds optimal block sizes and pipeline stages
    - Caches results for future runs with same shapes
    """
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
        
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator,
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    )


# ============================================================================
# ADVANCED TECHNIQUE 2: Persistent Kernels
# ============================================================================

@triton.jit
def persistent_vector_add(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Persistent kernel that processes multiple blocks per program.
    
    PERSISTENT KERNEL CONCEPTS:
    - Fewer program launches (reduces launch overhead)
    - Each program processes multiple blocks in a grid-stride loop
    - Better for small-to-medium workloads
    - Reduces CPU-GPU synchronization
    """
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    # Grid-stride loop: each program handles multiple blocks
    block_start = pid * BLOCK_SIZE
    stride = num_programs * BLOCK_SIZE
    
    for start in range(block_start, n_elements, stride):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================================
# ADVANCED TECHNIQUE 3: Combined Optimizations
# ============================================================================

@triton.jit
def optimized_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Matrix multiplication combining all optimization techniques:
    1. Tiling (BLOCK_SIZE_M, N, K)
    2. Pipelining (NUM_STAGES)
    3. Shared memory (automatic via blocked loads)
    4. Warp specialization (GROUP_SIZE_M swizzling)
    """
    pid = tl.program_id(axis=0)
    
    # Swizzle for better cache utilization
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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


def benchmark_advanced():
    """
    Benchmark advanced optimization techniques.
    """
    import time
    
    print("Advanced Optimizations Benchmark")
    print("=" * 60)
    
    # Test autotuned matmul
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    print("\nAutotuned Matrix Multiplication")
    print("-" * 40)
    
    # First run triggers autotuning
    print("Running autotuning (this may take a moment)...")
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_autotuned[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    torch.cuda.synchronize()
    
    # Benchmark after tuning
    start = time.perf_counter()
    for _ in range(50):
        matmul_autotuned[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1)
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / 50 * 1000
    flops = 2 * M * N * K
    gflops = flops / avg_time / 1e6
    
    print(f"  Time: {avg_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    
    # Test persistent kernel
    n = 10_000_000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.empty(n, device="cuda")
    
    print("\nPersistent Kernel (Vector Add)")
    print("-" * 40)
    
    BLOCK_SIZE = 1024
    # Use fewer programs for persistent kernel
    grid_size = 128
    
    # Warmup
    persistent_vector_add[grid_size](x, y, output, n, BLOCK_SIZE)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        persistent_vector_add[grid_size](x, y, output, n, BLOCK_SIZE)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / 100 * 1000
    
    print(f"  Time: {avg_time:.4f} ms")
    
    # Compare with standard kernel
    grid_size_standard = triton.cdiv(n, BLOCK_SIZE)
    
    start = time.perf_counter()
    for _ in range(100):
        persistent_vector_add[grid_size_standard](x, y, output, n, BLOCK_SIZE)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    std_time = (end - start) / 100 * 1000
    print(f"  Standard kernel time: {std_time:.4f} ms")
    print(f"  Speedup: {std_time / avg_time:.2f}x")


if __name__ == "__main__":
    print("Running Advanced Optimizations Module")
    print("=" * 60)
    
    benchmark_advanced()
    
    print("\n" + "=" * 60)
    print("Testing Correctness")
    
    # Test optimized matmul
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    
    optimized_matmul[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        NUM_STAGES=3
    )
    
    expected = a @ b
    if torch.allclose(c, expected, rtol=1e-2, atol=1e-2):
        print("✓ Optimized matrix multiplication works!")
    else:
        print("✗ Results don't match")
    
    # Test persistent kernel
    n = 100000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.empty(n, device="cuda")
    
    persistent_vector_add[128](x, y, output, n, 1024)
    
    if torch.allclose(output, x + y):
        print("✓ Persistent kernel works!")
    else:
        print("✗ Persistent results don't match")
    
    print("\nKEY TAKEAWAYS:")
    print("1. Autotuning finds optimal parameters automatically")
    print("2. Persistent kernels reduce launch overhead")
    print("3. Combine techniques for maximum performance")
    print("4. Profile to identify bottlenecks")
    print("5. Different workloads need different optimizations")
    print("\nOPTIMIZATION CHECKLIST:")
    print("□ Choose optimal block sizes (tiling)")
    print("□ Enable pipelining (NUM_STAGES)")
    print("□ Use swizzling for large matrices")
    print("□ Fuse element-wise operations")
    print("□ Consider persistent kernels for small workloads")
    print("□ Use autotuning for production code")
