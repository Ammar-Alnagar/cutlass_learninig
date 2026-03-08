"""
Project 01 — Fused MHA Full Implementation
File: benchmark.py

Benchmark FMHA implementations against FlashAttention reference.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class BenchmarkConfig:
    batch_sizes: list
    seq_lens: list
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    causal: bool
    num_warmup: int
    num_iters: int


config = BenchmarkConfig(
    batch_sizes=[1, 8, 32, 128],
    seq_lens=[128, 512, 1024, 2048, 4096],
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
    causal=True,
    num_warmup=10,
    num_iters=50,
)


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def benchmark_flash_attention(Q, K, V, causal, num_warmup=10, num_iters=50):
    """Benchmark FlashAttention reference."""
    try:
        from flash_attn import flash_attn_func
        
        # Warmup
        for _ in range(num_warmup):
            _ = flash_attn_func(Q, K, V, causal=causal)
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = flash_attn_func(Q, K, V, causal=causal)
        torch.cuda.synchronize()
        
        return (time.perf_counter() - start) / num_iters * 1000
    except ImportError:
        return None


def benchmark_torch_sdpa(Q, K, V, causal, num_warmup=10, num_iters=50):
    """Benchmark torch scaled_dot_product_attention."""
    O = torch.zeros_like(Q)
    
    for _ in range(num_warmup):
        O.copy_(torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        O.copy_(torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_custom_fmha(fmha_fn, Q, K, V, causal, num_warmup=10, num_iters=50):
    """Benchmark custom FMHA implementation."""
    O = torch.zeros_like(Q)
    
    for _ in range(num_warmup):
        fmha_fn(Q, K, V, O, causal=causal)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        fmha_fn(Q, K, V, O, causal=causal)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================

def run_benchmark():
    import math
    from dataclasses import dataclass
    
    print("=" * 70)
    print("Fused MHA Benchmark")
    print("=" * 70)
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability()
    sm = compute_cap[0] * 10 + compute_cap[1]
    
    print(f"\nGPU: {gpu_name} (SM{sm})")
    print(f"Dtype: {config.dtype}, Causal: {config.causal}")
    
    results = []
    
    for B in config.batch_sizes:
        for S in config.seq_lens:
            H, D = config.num_heads, config.head_dim
            
            # Skip invalid configurations
            if B * S > 65536:  # Memory limit
                continue
            
            # Create tensors
            Q = torch.randn(B, H, S, D, dtype=config.dtype, device='cuda')
            K = torch.randn(B, H, S, D, dtype=config.dtype, device='cuda')
            V = torch.randn(B, H, S, D, dtype=config.dtype, device='cuda')
            
            # Benchmark FlashAttention
            fa_latency = benchmark_flash_attention(
                Q, K, V, config.causal, config.num_warmup, config.num_iters
            )
            
            # Benchmark torch SDPA
            torch_latency = benchmark_torch_sdpa(
                Q, K, V, config.causal, config.num_warmup, config.num_iters
            )
            
            # Compute tokens/sec
            tokens_per_iter = B * S
            if fa_latency and fa_latency > 0:
                tokens_sec = tokens_per_iter / (fa_latency * 1e-3)
            else:
                tokens_sec = 0
            
            results.append({
                'batch': B,
                'seq_len': S,
                'flash_attn': fa_latency,
                'torch_sdpa': torch_latency,
                'tokens_sec': tokens_sec,
            })
            
            print(f"\nB={B:3d}, S={S:4d}: "
                  f"FlashAttn={fa_latency:.3f}ms if fa_latency else 'N/A':>10} "
                  f"torch={torch_latency:.3f}ms "
                  f"Tokens/sec={tokens_sec/1000:.1f}K")
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"{'Batch':<8} {'Seq':<8} {'FlashAttn (ms)':<16} {'torch (ms)':<12} {'Tokens/sec'}")
    print("-" * 70)
    
    for r in results:
        fa_str = f"{r['flash_attn']:.3f}" if r['flash_attn'] else "N/A"
        torch_str = f"{r['torch_sdpa']:.3f}" if r['torch_sdpa'] else "N/A"
        print(f"{r['batch']:<8} {r['seq_len']:<8} {fa_str:<16} {torch_str:<12} {r['tokens_sec']/1000:.1f}K")


if __name__ == "__main__":
    run_benchmark()
