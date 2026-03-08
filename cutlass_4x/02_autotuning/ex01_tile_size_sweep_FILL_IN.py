"""
Module 02 — Autotuning
Exercise 01 — Tile Size Sweep for GEMM

LEVEL: 1 → 2 (High-level op → CuTe DSL)

WHAT YOU'RE BUILDING:
  Manual tile size sweep to find optimal GEMM configuration — the foundation 
  of autotuning. This is how you find the best thread block and warp tile 
  sizes for your specific matrix shape before using automated tools.

OBJECTIVE:
  - Enumerate valid tile size configurations
  - Measure performance across configurations
  - Understand how tile sizes affect occupancy and throughput
  - Pick the best config for your matrix shape
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
from dataclasses import dataclass
from typing import List, Tuple


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What tile size would you expect to be optimal for M=512, K=1024, N=2048?
#     Consider: larger tiles = more compute, but may reduce occupancy

# Q2: Why can't we just always use the maximum tile size (e.g., 256×256)?
#     Hint: Think about register pressure and shared memory limits

# Q3: How does the M dimension being smaller (512) affect optimal tile choice?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class TileConfig:
    """Tile size configuration for GEMM."""
    block_m: int      # Thread block M dimension
    block_n: int      # Thread block N dimension  
    block_k: int      # Thread block K dimension
    warp_m: int       # Warp-level M dimension
    warp_n: int       # Warp-level N dimension
    warp_k: int       # Warp-level K dimension
    stages: int       # Pipeline stages (for async copy)
    
    def name(self) -> str:
        return f"M{self.block_m}N{self.block_n}K{self.block_k}_S{self.stages}"


# Matrix dimensions
M, K, N = 512, 1024, 2048
dtype = torch.float16
device = torch.device("cuda")

# Allocate tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
C = torch.zeros(M, N, dtype=dtype, device=device)

# Reference output
C_ref = torch.mm(A, B)

# Common valid tile configurations (these are CUTLASS defaults that work well)
TILE_CONFIGS = [
    # (block_m, block_n, block_k, warp_m, warp_n, warp_k, stages)
    TileConfig(128, 128, 32, 64, 64, 32, 3),    # Balanced
    TileConfig(128, 256, 32, 64, 128, 32, 3),   # N-heavy (good for N >> M)
    TileConfig(256, 128, 32, 128, 64, 32, 3),   # M-heavy (good for M >> N)
    TileConfig(64, 256, 32, 32, 128, 32, 4),    # Small M, large N
    TileConfig(128, 128, 64, 64, 64, 32, 4),    # Larger K tile
    TileConfig(256, 256, 32, 128, 128, 32, 3),  # Large tiles (high compute)
]


# ==============================================================================
# FILL IN: Level 1 — High-Level Op with Different Configs
# ==============================================================================

print("=" * 60)
print("Tile Size Sweep for GEMM")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"\nTesting {len(TILE_CONFIGS)} tile configurations...\n")


def benchmark_tile_config(config: TileConfig, A, B, C, 
                          num_warmup=10, num_iters=100) -> Tuple[float, bool]:
    """
    Benchmark a specific tile configuration.
    
    TODO [MEDIUM]: Configure GEMM with specific tile sizes
    HINT: 
      - cutlass.op.Gemm accepts tile_config parameter
      - Use cutlass.GemmCoord for tile sizes
    REF: cutlass/examples/python/CuTeDSL/gemm_tile_config.py
    """
    
    # TODO: Create GEMM plan with specific tile config
    # plan = cutlass.op.Gemm(
    #     element=cutlass.float16,
    #     layout=cutlass.LayoutType.RowMajor,
    #     tile_config=cutlass.GemmCoord(config.block_m, config.block_n, config.block_k),
    #     warp_config=cutlass.GemmCoord(config.warp_m, config.warp_n, config.warp_k),
    #     stages=config.stages,
    # )
    
    # TODO: Warmup and benchmark
    # for _ in range(num_warmup):
    #     plan.run(A, B, C)
    # torch.cuda.synchronize()
    
    # start = time.perf_counter()
    # for _ in range(num_iters):
    #     plan.run(A, B, C)
    # torch.cuda.synchronize()
    # elapsed = time.perf_counter() - start
    
    # avg_latency = (elapsed / num_iters) * 1000
    
    # Placeholder (replace with implementation)
    avg_latency = 0.0
    success = True
    
    return avg_latency, success


results: List[Tuple[TileConfig, float, bool]] = []

for config in TILE_CONFIGS:
    latency, success = benchmark_tile_config(config, A, B, C)
    results.append((config, latency, success))
    
    status = "✓" if success else "✗"
    print(f"{status} {config.name():<25} {latency:.3f} ms")


# ==============================================================================
# ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)

# Find best configuration
valid_results = [(cfg, lat) for cfg, lat, ok in results if ok and lat > 0]
if valid_results:
    best_config, best_latency = min(valid_results, key=lambda x: x[1])
    
    # Compute TFLOPS
    flops = 2 * M * N * K
    best_tflops = flops / (best_latency * 1e-3) / 1e12
    
    print(f"\nBest configuration: {best_config.name()}")
    print(f"  Latency: {best_latency:.3f} ms")
    print(f"  TFLOPS:  {best_tflops:.1f}")
    
    # Compare with worst
    worst_config, worst_latency = max(valid_results, key=lambda x: x[1])
    slowdown = worst_latency / best_latency
    print(f"\nWorst configuration: {worst_config.name()}")
    print(f"  Latency: {worst_latency:.3f} ms")
    print(f"  Slowdown vs best: {slowdown:.2f}×")
    
    # Print full ranking
    print("\n" + "=" * 60)
    print("Full Ranking (fastest to slowest)")
    print("=" * 60)
    sorted_results = sorted(valid_results, key=lambda x: x[1])
    for rank, (cfg, lat) in enumerate(sorted_results, 1):
        tflops = flops / (lat * 1e-3) / 1e12
        rel = lat / best_latency
        print(f"  {rank}. {cfg.name():<25} {lat:.3f} ms  {tflops:.1f} TFLOPS  ({rel:.2f}×)")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Optimal tile size for M=512, K=1024, N=2048?")
if valid_results:
    print(f"        Your prediction: ?")
        print(f"        Actual best:     {best_config.name()}")
print("\n    Q2: Why not always use max tile size?")
print("        Answer: Larger tiles increase register pressure,")
print("                reducing occupancy. Also may not fit in")
print("                shared memory for large K tiles.")

print("\n    Q3: How does small M affect tile choice?")
print("        Answer: When M is small, use smaller block_m to")
print("                avoid idle threads. M=512 fits 4×128 or 8×64")
print("                tiles well.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command (compare best vs worst):")
print(f"    ncu --metrics smsp__throughput.avg,\\")
print(f"                sm__warps_per_sm.avg,\\")
print(f"                l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \\")
print(f"        python ex01_tile_size_sweep_FILL_IN.py")
print("\n    Look for:")
print("      - Higher throughput for best config")
print("      - Occupancy (warps_per_sm) differences")
print("      - L1 cache hit rate variations")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How do you choose tile sizes for a new matrix shape?")
print("    A: 1. Start with CUTLASS defaults (128×128×32)")
print("       2. If M << N, try smaller block_m (64)")
print("       3. If N << M, try smaller block_n (64)")
print("       4. For very large matrices, try 256×256")
print("       5. Sweep 3-5 configs, pick best")
print("       6. Use automated autotuning for production")

print("\n    Q: What is pipeline staging?")
print("    A: Overlapping memory loads with compute using multiple")
print("       buffer 'stages'. While computing on stage N, load")
print("       data for stage N+1. More stages = more parallelism")
print("       but higher shared memory usage. Typical: 3-5 stages.")

# C4: Production guidance
print("\nC4: Production Tile Size Selection")
print("    Matrix Shape          Recommended Starting Config")
print("    Square (M≈N≈K)        128×128×32, 3 stages")
print("    M << N (decoder)      64×256×32, 4 stages")
print("    N << M (encoder)      256×64×32, 4 stages")
print("    K >> M,N (large K)    128×128×64, 4 stages")
print("    Very large (4K+)      256×256×32, 3 stages")
print("\n    Rule: Always sweep for critical kernels, use defaults otherwise.")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
