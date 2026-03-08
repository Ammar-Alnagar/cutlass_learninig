"""
Project 06 — Fused Attention Variants Benchmark
"""

import torch
import time
import math

SEQ_LENS = [512, 1024, 2048, 4096]
NUM_HEADS = 8
HEAD_DIM = 64
GQA_RATIO = 4  # Query heads per KV head


def benchmark_mha(seq_len, num_runs=100):
    """Benchmark standard MHA."""
    Q = torch.randn((NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    K = torch.randn((NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    V = torch.randn((NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    start = time.perf_counter()
    for _ in range(num_runs):
        scores = torch.einsum('hqk,hkd->hqk', Q.float(), K.float()) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('hqk,hkd->hqk', attn, V.float())
    torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / num_runs) * 1000
    
    return avg_time_ms


def benchmark_gqa(seq_len, num_runs=100):
    """Benchmark GQA."""
    num_kv_heads = NUM_HEADS // GQA_RATIO
    
    Q = torch.randn((NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    K = torch.randn((num_kv_heads, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    V = torch.randn((num_kv_heads, seq_len, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Expand KV for reference
    K_exp = K.repeat_interleave(GQA_RATIO, dim=0)
    V_exp = V.repeat_interleave(GQA_RATIO, dim=0)
    
    start = time.perf_counter()
    for _ in range(num_runs):
        scores = torch.einsum('hqk,hkd->hqk', Q.float(), K_exp.float()) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('hqk,hkd->hqk', attn, V_exp.float())
    torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / num_runs) * 1000
    
    return avg_time_ms


def print_table(results):
    print("\n" + "=" * 80)
    print("  Project 06 — Fused Attention Variants Benchmark")
    print("=" * 80)
    
    print("\n┌────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Sequence Len   │  MHA         │  GQA         │  Speedup     │")
    print("│                │  Time (ms)   │  Time (ms)   │  (vs MHA)    │")
    print("├────────────────┼──────────────┼──────────────┼──────────────┤")
    
    for seq_len, mha_time, gqa_time in results:
        speedup = mha_time / gqa_time if gqa_time > 0 else 0
        kv_savings = (1 - 1/GQA_RATIO) * 100
        print(f"│ {seq_len:>14}  │ {mha_time:>10.2f}   │ {gqa_time:>10.2f}   │ {speedup:>10.2f}×    │")
    
    print("└────────────────┴──────────────┴──────────────┴──────────────┘")
    print(f"\n  GQA KV cache savings: {kv_savings:.0f}% (vs MHA)")
    print("=" * 80 + "\n")


def run():
    print("\n" + "=" * 60)
    print("  Project 06 — Fused Attention Variants Benchmark")
    print("=" * 60)
    
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Configuration: Heads={NUM_HEADS}, Dim={HEAD_DIM}, GQA ratio={GQA_RATIO}")
    
    results = []
    
    for seq_len in SEQ_LENS:
        print(f"\n  Sequence length: {seq_len}")
        
        mha_time = benchmark_mha(seq_len)
        print(f"    MHA: {mha_time:.2f} ms")
        
        gqa_time = benchmark_gqa(seq_len)
        print(f"    GQA: {gqa_time:.2f} ms")
        
        results.append((seq_len, mha_time, gqa_time))
    
    print_table(results)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
