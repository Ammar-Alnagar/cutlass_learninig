"""
Module 03: Performance Profiling and Analysis

This module teaches how to profile Triton kernels to identify
performance bottlenecks and optimization opportunities.

LEARNING OBJECTIVES:
1. Use Triton's built-in profiling tools
2. Analyze memory bandwidth and compute utilization
3. Identify performance bottlenecks
4. Compare kernel performance
"""

import triton
import triton.language as tl
import torch
import time


# ============================================================================
# PROFILING TECHNIQUE 1: Basic Timing
# ============================================================================

def profile_kernel(kernel_func, grid, *args, warmup=10, repeat=100, **kwargs):
    """
    Profile a Triton kernel and return timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        kernel_func[grid](*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        kernel_func[grid](*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / repeat * 1000  # ms
    min_time = avg_time  # Could use more sophisticated measurement
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'total_time_ms': (end - start) * 1000,
    }


# ============================================================================
# PROFILING TECHNIQUE 2: Bandwidth Analysis
# ============================================================================

def calculate_bandwidth(n_bytes: int, time_ms: float) -> float:
    """
    Calculate memory bandwidth in GB/s.
    """
    return n_bytes / time_ms / 1e6


def analyze_memory_bandwidth():
    """
    Analyze memory bandwidth of vector operations.
    """
    print("Memory Bandwidth Analysis")
    print("=" * 60)
    
    @triton.jit
    def vector_add_kernel(
        x_ptr, y_ptr, output_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)
    
    # Test different sizes
    sizes = [10**6, 10**7, 10**8]
    
    print(f"{'Size':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20}")
    print("-" * 60)
    
    for n in sizes:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        output = torch.empty(n, device="cuda")
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        
        stats = profile_kernel(vector_add_kernel, grid, x, y, output, n, BLOCK_SIZE)
        
        # 2 reads + 1 write, 4 bytes per float
        bytes_transferred = n * 4 * 3
        bandwidth = calculate_bandwidth(bytes_transferred, stats['avg_time_ms'])
        
        print(f"{n:<15} {stats['avg_time_ms']:<15.3f} {bandwidth:<20.2f}")
    
    print("-" * 60)
    print("Note: Peak bandwidth varies by GPU")
    print("  - RTX 3090: ~936 GB/s")
    print("  - A100: ~1555 GB/s")
    print("  - V100: ~900 GB/s")


# ============================================================================
# PROFILING TECHNIQUE 3: Compute Analysis
# ============================================================================

def analyze_compute_performance():
    """
    Analyze compute performance of matrix multiplication.
    """
    print("\nCompute Performance Analysis (Matrix Multiplication)")
    print("=" * 60)
    
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
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
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
    
    # Test different sizes
    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    
    print(f"{'Size':<20} {'Time (ms)':<15} {'GFLOPS':<15}")
    print("-" * 60)
    
    for M, N, K in sizes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        stats = profile_kernel(matmul_kernel, grid,
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        
        # 2 * M * N * K operations for matmul
        flops = 2 * M * N * K
        gflops = flops / stats['avg_time_ms'] / 1e6
        
        print(f"{M}x{N}x{K:<10} {stats['avg_time_ms']:<15.3f} {gflops:<15.2f}")


# ============================================================================
# PROFILING TECHNIQUE 4: Bottleneck Identification
# ============================================================================

def identify_bottleneck():
    """
    Identify whether a kernel is memory-bound or compute-bound.
    """
    print("\nBottleneck Analysis")
    print("=" * 60)
    
    # Arithmetic intensity = FLOPs / bytes
    # High intensity = compute-bound
    # Low intensity = memory-bound
    
    operations = [
        ("Vector Add", 1, 12),      # 1 FLOP, 12 bytes (2 read + 1 write * 4 bytes)
        ("Vector Mul", 1, 12),
        ("MatMul 128", 2*128, 12*128),  # Simplified
        ("MatMul 256", 2*256, 12*256),
        ("MatMul 512", 2*512, 12*512),
    ]
    
    print(f"{'Operation':<20} {'Arith. Intensity':<20} {'Likely Bound':<15}")
    print("-" * 60)
    
    for name, flops, bytes_ in operations:
        intensity = flops / bytes_
        bound_type = "Compute" if intensity > 10 else "Memory"
        print(f"{name:<20} {intensity:<20.2f} {bound_type:<15}")
    
    print("\nRule of thumb:")
    print("  - Intensity < 5: Memory-bound (optimize bandwidth)")
    print("  - Intensity 5-20: Mixed")
    print("  - Intensity > 20: Compute-bound (optimize FLOPS)")


# ============================================================================
# PROFILING TECHNIQUE 5: Comparison Framework
# ============================================================================

def compare_implementations():
    """
    Compare Triton implementation with PyTorch baseline.
    """
    print("\nImplementation Comparison")
    print("=" * 60)
    
    @triton.jit
    def triton_add(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    
    n = 10_000_000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    
    # Triton
    output_triton = torch.empty(n, device="cuda")
    grid = (triton.cdiv(n, 1024),)
    
    triton_stats = profile_kernel(triton_add, grid, x, y, output_triton, n, 1024)
    
    # PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        output_torch = x + y
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    torch_time = (end - start) / 100 * 1000
    
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Relative':<15}")
    print("-" * 60)
    print(f"{'PyTorch':<20} {torch_time:<15.4f} {'1.00x':<15}")
    print(f"{'Triton':<20} {triton_stats['avg_time_ms']:<15.4f} {torch_time/triton_stats['avg_time_ms']:.2f}x")
    
    # Verify correctness
    if torch.allclose(output_triton, x + y):
        print("\n✓ Triton results match PyTorch")
    else:
        print("\n✗ WARNING: Results don't match!")


# ============================================================================
# PROFILING TECHNIQUE 6: Occupancy Analysis
# ============================================================================

def analyze_occupancy():
    """
    Analyze GPU occupancy for different block sizes.
    """
    print("\nOccupancy Analysis")
    print("=" * 60)
    
    # Simplified occupancy estimation
    # Real occupancy depends on registers, shared memory, etc.
    
    warps_per_sm = 64  # A100
    threads_per_warp = 32
    max_threads_per_sm = 2048  # A100
    
    block_sizes = [64, 128, 256, 512, 1024]
    
    print(f"{'Block Size':<15} {'Blocks/SM':<15} {'Occupancy':<15}")
    print("-" * 60)
    
    for block_size in block_sizes:
        blocks_per_sm = min(warps_per_sm * threads_per_warp // block_size, 
                           max_threads_per_sm // block_size)
        occupancy = blocks_per_sm * block_size / max_threads_per_sm
        
        print(f"{block_size:<15} {blocks_per_sm:<15} {occupancy:.0%}")
    
    print("\nNote: Real occupancy also depends on:")
    print("  - Register usage per thread")
    print("  - Shared memory per block")
    print("  - Hardware limits")


if __name__ == "__main__":
    print("Performance Profiling and Analysis Module")
    print("=" * 60)
    
    # Run all analyses
    analyze_memory_bandwidth()
    analyze_compute_performance()
    identify_bottleneck()
    compare_implementations()
    analyze_occupancy()
    
    print("\n" + "=" * 60)
    print("PROFILING CHECKLIST:")
    print("□ Measure baseline performance")
    print("□ Calculate achieved bandwidth")
    print("□ Calculate achieved GFLOPS")
    print("□ Identify bottleneck (memory vs compute)")
    print("□ Compare with theoretical peak")
    print("□ Test multiple configurations")
    print("□ Verify correctness before optimizing")
    print("\nPROFILING TOOLS:")
    print("  - time.perf_counter() for basic timing")
    print("  - torch.cuda.synchronize() for accurate timing")
    print("  - Nsight Compute for detailed analysis")
    print("  - triton.testing for benchmarking")
