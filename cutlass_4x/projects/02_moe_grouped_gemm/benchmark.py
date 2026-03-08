"""
Project 02 — MoE Grouped GEMM
File: benchmark.py

Benchmark MoE implementation vs naive expert loop.
"""

import torch
import time
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    num_experts: list
    hidden_size: int
    expert_width: int
    top_k: int
    total_tokens: list
    dtype: torch.dtype
    num_warmup: int
    num_iters: int


config = BenchmarkConfig(
    num_experts=[8, 16, 32],
    hidden_size=1024,
    expert_width=2048,
    top_k=2,
    total_tokens=[1024, 4096, 16384],
    dtype=torch.float16,
    num_warmup=10,
    num_iters=50,
)


def benchmark_naive_moe(tokens, expert_weights, expert_indices, top_k,
                        num_warmup=10, num_iters=50):
    """Benchmark naive MoE (loop over experts)."""
    num_experts = expert_weights.shape[0]
    
    # Group tokens (simplified)
    grouped = []
    for e in range(num_experts):
        mask = (expert_indices == e).any(dim=-1)
        grouped.append(tokens[mask])
    
    # Warmup
    for _ in range(num_warmup):
        outputs = []
        for e in range(num_experts):
            if grouped[e].shape[0] > 0:
                outputs.append(torch.mm(grouped[e], expert_weights[e]))
        torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        outputs = []
        for e in range(num_experts):
            if grouped[e].shape[0] > 0:
                outputs.append(torch.mm(grouped[e], expert_weights[e]))
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_grouped_moe(tokens, expert_weights, expert_indices, top_k,
                          num_warmup=10, num_iters=50):
    """Benchmark GroupedGEMM MoE."""
    # TODO: Implement with cutlass.op.GroupedGemm
    return benchmark_naive_moe(tokens, expert_weights, expert_indices, top_k,
                               num_warmup, num_iters)


def run_benchmark():
    print("=" * 70)
    print("MoE Grouped GEMM Benchmark")
    print("=" * 70)
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    
    results = []
    
    for num_experts in config.num_experts:
        for total_tokens in config.total_tokens:
            # Create test data
            tokens = torch.randn(total_tokens, config.hidden_size,
                                 dtype=config.dtype, device=device)
            expert_weights = torch.randn(num_experts, config.hidden_size, config.expert_width,
                                          dtype=config.dtype, device=device)
            
            # Simple routing (random assignment)
            expert_indices = torch.randint(0, num_experts, 
                                            (total_tokens, top_k), device=device)
            
            # Benchmark
            naive_latency = benchmark_naive_moe(
                tokens, expert_weights, expert_indices, config.top_k,
                config.num_warmup, config.num_iters
            )
            
            grouped_latency = benchmark_grouped_moe(
                tokens, expert_weights, expert_indices, config.top_k,
                config.num_warmup, config.num_iters
            )
            
            speedup = naive_latency / grouped_latency if grouped_latency > 0 else 0
            
            # Tokens/sec
            token_expert_pairs = total_tokens * config.top_k
            tokens_per_sec = token_expert_pairs / (grouped_latency * 1e-3)
            
            results.append({
                'experts': num_experts,
                'tokens': total_tokens,
                'naive_ms': naive_latency,
                'grouped_ms': grouped_latency,
                'speedup': speedup,
                'tokens_per_sec': tokens_per_sec,
            })
            
            print(f"E={num_experts:2d}, T={total_tokens:5d}: "
                  f"Naive={naive_latency:.3f}ms, Grouped={grouped_latency:.3f}ms, "
                  f"Speedup={speedup:.2f}×, Tokens/sec={tokens_per_sec/1e6:.1f}M")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Experts':<10} {'Tokens':<10} {'Naive (ms)':<12} {'Grouped (ms)':<14} {'Speedup':<10} {'Tokens/sec'}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['experts']:<10} {r['tokens']:<10} {r['naive_ms']:<12.3f} "
              f"{r['grouped_ms']:<14.3f} {r['speedup']:<10.2f} {r['tokens_per_sec']/1e6:.1f}M")


if __name__ == "__main__":
    run_benchmark()
