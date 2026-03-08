"""
Module 02 — Autotuning
Exercise 02 — Automated GEMM Autotuning with cutlass.autotune

LEVEL: 1 (High-level op API with autotuner)

WHAT YOU'RE BUILDING:
  Automated autotuning using CUTLASS's built-in autotuner — the production 
  approach for finding optimal GEMM configurations. This is how TensorRT-LLM 
  and vLLM tune kernels for specific hardware and workloads.

OBJECTIVE:
  - Use cutlass.autotune to search configuration space
  - Define search space with constraints
  - Run autotuning with warmup and measurement
  - Export and cache optimal configuration
"""

import torch
import cutlass
from cutlass import autotune
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: How long do you expect autotuning to take for 50 configurations?
#     Consider: warmup + measurement iterations per config

# Q2: What's the risk of overfitting to a specific matrix shape during autotuning?
#     Hint: Consider real workloads with variable sequence lengths

# Q3: Why would you cache autotuning results? What changes would invalidate cache?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class AutotuneConfig:
    """Configuration for autotuning run."""
    matrix_shape: tuple
    dtype: str
    num_warmup: int = 10
    num_iters: int = 50
    search_space_size: int = 50


# Matrix dimensions (typical transformer attention shape)
M, K, N = 1024, 4096, 4096  # Q @ K^T or attention output projection
dtype = torch.float16
device = torch.device("cuda")

# Allocate tensors
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
C = torch.zeros(M, N, dtype=dtype, device=device)

# Reference
C_ref = torch.mm(A, B)

# Cache directory for autotuning results
CACHE_DIR = Path("./autotune_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / f"gemm_{M}x{K}x{N}_fp16.json"


# ==============================================================================
# FILL IN: Level 1 — Automated Autotuning
# ==============================================================================

print("=" * 60)
print("Automated GEMM Autotuning")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Dtype: {dtype}")
print(f"Cache: {CACHE_FILE}")

# Check if cached result exists
if CACHE_FILE.exists():
    print(f"\n✓ Found cached autotuning result")
    with open(CACHE_FILE, 'r') as f:
        cached_result = json.load(f)
    print(f"  Best config: {cached_result.get('best_config', 'unknown')}")
    print(f"  Best latency: {cached_result.get('best_latency_ms', 0):.3f} ms")
    print("\nSkipping autotuning (delete cache file to re-run)")
else:
    print(f"\nNo cache found, running autotuning...")


# TODO [HARD]: Set up and run automated autotuning
# HINT:
#   - Use cutlass.autotune.GemmTuner or similar API
#   - Define search space (tile sizes, stages, etc.)
#   - Run tuner.search() or tuner.tune()
#   - Get best configuration
# REF: cutlass/examples/python/CuTeDSL/autotune_gemm.py

def run_autotuning(A, B, C, num_warmup=10, num_iters=50) -> Dict[str, Any]:
    """
    Run automated GEMM autotuning.
    
    Returns dict with:
      - best_config: optimal configuration
      - best_latency_ms: achieved latency
      - all_results: full benchmark results
    """
    
    # TODO: Create autotuner
    # tuner = cutlass.autotune.GemmTuner(
    #     element=cutlass.float16,
    #     layout=cutlass.LayoutType.RowMajor,
    #     search_space=cutlass.autotune.default_gemm_search_space,
    # )
    
    # TODO: Run search
    # results = tuner.search(
    #     A, B, C,
    #     num_warmup=num_warmup,
    #     num_iters=num_iters,
    # )
    
    # TODO: Get best config
    # best_config = results.best_config
    # best_latency = results.best_latency
    
    # Placeholder (replace with implementation)
    best_config = {"block_m": 128, "block_n": 128, "block_k": 32, "stages": 3}
    best_latency = 0.0
    all_results = []
    
    return {
        "best_config": best_config,
        "best_latency_ms": best_latency,
        "all_results": all_results,
    }


if not CACHE_FILE.exists():
    autotune_result = run_autotuning(A, B, C)
    
    # Cache the result
    cache_data = {
        "matrix_shape": (M, K, N),
        "dtype": str(dtype),
        "best_config": autotune_result["best_config"],
        "best_latency_ms": autotune_result["best_latency_ms"],
        "timestamp": time.time(),
    }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"\n✓ Autotuning complete")
    print(f"  Best config: {autotune_result['best_config']}")
    print(f"  Best latency: {autotune_result['best_latency_ms']:.3f} ms")
    print(f"  Results cached to: {CACHE_FILE}")
else:
    with open(CACHE_FILE, 'r') as f:
        autotune_result = json.load(f)


# ==============================================================================
# BENCHMARK: Autotuned vs Default
# ==============================================================================

def benchmark_with_config(config: Dict[str, Any], A, B, C, 
                          num_warmup=10, num_iters=100) -> float:
    """Benchmark GEMM with specific configuration."""
    
    # TODO: Create GEMM plan with autotuned config
    # plan = cutlass.op.Gemm(
    #     element=cutlass.float16,
    #     layout=cutlass.LayoutType.RowMajor,
    #     tile_config=cutlass.GemmCoord(
    #         config.get('block_m', 128),
    #         config.get('block_n', 128),
    #         config.get('block_k', 32),
    #     ),
    #     stages=config.get('stages', 3),
    # )
    
    # TODO: Benchmark
    # for _ in range(num_warmup):
    #     plan.run(A, B, C)
    # torch.cuda.synchronize()
    
    # start = time.perf_counter()
    # for _ in range(num_iters):
    #     plan.run(A, B, C)
    # torch.cuda.synchronize()
    
    # return (time.perf_counter() - start) / num_iters * 1000
    
    return 0.0


print("\n" + "=" * 60)
print("Benchmark: Autotuned vs Default Configuration")
print("=" * 60)

# Benchmark with autotuned config
autotuned_latency = benchmark_with_config(
    autotune_result.get("best_config", {}), A, B, C
)

# Benchmark with default config (no explicit tile config)
default_latency = benchmark_with_config({}, A, B, C)

print(f"\nResults:")
print(f"  Autotuned config: {autotuned_latency:.3f} ms")
print(f"  Default config:   {default_latency:.3f} ms")

if autotuned_latency > 0 and default_latency > 0:
    speedup = default_latency / autotuned_latency
    print(f"\n  Speedup from autotuning: {speedup:.2f}×")
    
    # Compute TFLOPS
    flops = 2 * M * N * K
    autotuned_tflops = flops / (autotuned_latency * 1e-3) / 1e12
    default_tflops = flops / (default_latency * 1e-3) / 1e12
    
    print(f"\n  Autotuned TFLOPS: {autotuned_tflops:.1f}")
    print(f"  Default TFLOPS:   {default_tflops:.1f}")


# ==============================================================================
# VARIABLE SHAPE GENERALIZATION
# ==============================================================================

print("\n" + "=" * 60)
print("Generalization to Variable Shapes")
print("=" * 60)

# Test autotuned config on nearby shapes
test_shapes = [
    (512, K, N),    # Half batch
    (2048, K, N),   # Double batch
    (M, K, N // 2), # Half output
    (M, K // 2, N), # Half hidden
]

print("\nTesting autotuned config on nearby shapes:")
print("(Note: optimal config may differ for each shape)")

for test_m, test_k, test_n in test_shapes:
    A_test = torch.randn(test_m, test_k, dtype=dtype, device=device)
    B_test = torch.randn(test_k, test_n, dtype=dtype, device=device)
    C_test = torch.zeros(test_m, test_n, dtype=dtype, device=device)
    
    latency = benchmark_with_config(
        autotune_result.get("best_config", {}), 
        A_test, B_test, C_test
    )
    
    flops = 2 * test_m * test_n * test_k
    tflops = flops / (latency * 1e-3) / 1e12 if latency > 0 else 0
    
    print(f"  M={test_m:4d}, K={test_k:4d}, N={test_n:4d}: "
          f"{latency:.3f} ms ({tflops:.1f} TFLOPS)")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Autotuning time?")
print("        Answer: Depends on #configs × iterations.")
print("                50 configs × 50 iters × 1ms = ~2.5 seconds")
print("                Plus warmup overhead.")

print("\n    Q2: Overfitting risk?")
print("        Answer: High if tuning for single shape.")
print("                Solution: Tune on representative shape distribution")
print("                or use shape buckets (powers of 2).")

print("\n    Q3: Cache invalidation?")
print("        Answer: Invalidate cache when:")
print("        - GPU architecture changes")
print("        - CUTLASS version changes")
print("        - CUDA driver changes significantly")
print("        - Matrix shape changes substantially")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --set full --target-processes all \\")
print(f"        python ex02_autotune_gemm_FILL_IN.py")
print("\n    Look for:")
print("      - SM efficiency comparison (autotuned vs default)")
print("      - Memory throughput differences")
print("      - Occupancy improvements")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How does TensorRT-LLM use autotuning?")
print("    A: During model import/build phase, TensorRT-LLM:")
print("       1. Identifies all GEMM shapes in the model")
print("       2. Groups similar shapes into buckets")
print("       3. Runs autotuning for each bucket")
print("       4. Caches results in engine file")
print("       5. Loads cached configs at runtime")

print("\n    Q: What's the alternative to per-shape autotuning?")
print("    A: Model-based autotuning (e.g., Ansor, TVM):")
print("       - Build cost model from benchmark data")
print("       - Predict performance without running")
print("       - Faster for large search spaces")
print("       - Less accurate than measurement-based")

# C4: Production guidance
print("\nC4: Production Autotuning Best Practices")
print("    1. Cache results persistently (per GPU + shape)")
print("    2. Use shape buckets for variable-length workloads")
print("    3. Set timeout limits (don't tune forever)")
print("    4. Fall back to defaults if tuning fails")
print("    5. Re-tune periodically (driver/CUTLASS updates)")
print("    6. Consider multi-GPU tuning for large models")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
