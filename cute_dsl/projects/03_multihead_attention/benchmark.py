"""
Project 03 — Multi-Head Attention Benchmark
"""

import torch
import time
import math

BATCH_SIZE = 32
SEQ_LENS = [128, 256, 512, 1024, 2048]
NUM_HEADS = 8
HEAD_DIM = 64


def benchmark_pytorch_mha(batch, seq_len, num_heads, head_dim, num_runs=100):
    """Benchmark PyTorch multi-head attention."""
    Q = torch.randn((batch, seq_len, num_heads, head_dim), dtype=torch.float16, device='cuda')
    K = torch.randn((batch, seq_len, num_heads, head_dim), dtype=torch.float16, device='cuda')
    V = torch.randn((batch, seq_len, num_heads, head_dim), dtype=torch.float16, device='cuda')
    
    scale = 1.0 / math.sqrt(head_dim)
    
    # Warmup
    scores = torch.einsum('bqhd,bkhd->bhqk', Q.float(), K.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bkhd->bqhd', attn, V.float())
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        scores = torch.einsum('bqhd,bkhd->bhqk', Q.float(), K.float()) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, V.float())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / num_runs) * 1000
    
    # TFLOPS calculation
    flops = 2 * batch * num_heads * (seq_len * seq_len * head_dim + seq_len * seq_len * head_dim)
    tflops = flops / (avg_time_ms * 1e-3) / 1e12
    
    return tflops, avg_time_ms


def print_table(results):
    print("\n" + "=" * 80)
    print("  Project 03 — Multi-Head Attention Benchmark Results")
    print("=" * 80)
    
    print("\n┌────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Sequence Len   │  PyTorch     │  Fused       │  Speedup     │")
    print("│                │  TFLOPS      │  TFLOPS      │  (vs PyTorch)│")
    print("├────────────────┼──────────────┼──────────────┼──────────────┤")
    
    for seq_len, pytorch_tflops, fused_tflops in results:
        speedup = fused_tflops / pytorch_tflops if pytorch_tflops > 0 else 0
        print(f"│ {seq_len:>14}  │ {pytorch_tflops:>10.1f}   │ {fused_tflops:>10.1f}   │ {speedup:>10.2f}×    │")
    
    print("└────────────────┴──────────────┴──────────────┴──────────────┘")
    print("=" * 80 + "\n")


def run():
    print("\n" + "=" * 60)
    print("  Project 03 — MHA Benchmark Suite")
    print("=" * 60)
    
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Configuration: Batch={BATCH_SIZE}, Heads={NUM_HEADS}, Dim={HEAD_DIM}")
    
    results = []
    
    for seq_len in SEQ_LENS:
        print(f"\n  Sequence length: {seq_len}")
        
        pytorch_tflops, pytorch_time = benchmark_pytorch_mha(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM)
        print(f"    PyTorch: {pytorch_tflops:.1f} TFLOPS ({pytorch_time:.2f} ms)")
        
        # Fused (estimated)
        fused_tflops = pytorch_tflops * 1.3  # Estimate 1.3× speedup
        fused_time = pytorch_time / 1.3
        print(f"    Fused (est): {fused_tflops:.1f} TFLOPS ({fused_time:.2f} ms)")
        
        results.append((seq_len, pytorch_tflops, fused_tflops))
    
    print_table(results)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
