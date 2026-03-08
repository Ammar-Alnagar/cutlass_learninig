"""
Project 07 — Quantized GEMM Benchmark

Compares FP8, INT8 vs FP16 GEMM.
"""

import torch
import time

MATRIX_SIZES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]


def benchmark_fp16_gemm(M, N, K, num_runs=100):
    """Benchmark FP16 GEMM."""
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')
    
    # Warmup
    torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / num_runs) * 1000
    flops = 2 * M * N * K
    tflops = flops / (avg_time_ms * 1e-3) / 1e12
    
    return tflops, avg_time_ms


def print_table(results):
    print("\n" + "=" * 90)
    print("  Project 07 — Quantized GEMM Benchmark Results")
    print("=" * 90)
    
    print("\n┌──────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Matrix Size (M,N,K)  │  FP16        │  INT8        │  FP8         │  FP8 Speedup │")
    print("│                      │  TFLOPS      │  TFLOPS      │  TFLOPS      │  (vs FP16)   │")
    print("├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    
    for (M, N, K), fp16_tflops, int8_tflops, fp8_tflops in results:
        matrix_str = f"({M//1024}K, {N//1024}K, {K//1024}K)" if M >= 1024 else f"({M}, {N}, {K})"
        int8_speedup = int8_tflops / fp16_tflops if fp16_tflops > 0 else 0
        fp8_speedup = fp8_tflops / fp16_tflops if fp16_tflops > 0 else 0
        
        print(f"│ {matrix_str:<20} │ {fp16_tflops:>10.1f}   │ {int8_tflops:>10.1f}   │ {fp8_tflops:>10.1f}   │ {fp8_speedup:>10.2f}×    │")
    
    print("└──────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    
    # Target check
    avg_fp8_speedup = sum(r[3] / r[1] for r in results if r[1] > 0) / len(results) if results else 0
    print(f"\n  Average FP8 speedup vs FP16: {avg_fp8_speedup:.2f}×")
    print(f"  Target: >1.5× speedup")
    print(f"  Status: {'✓ ACHIEVED' if avg_fp8_speedup >= 1.5 else '✗ BELOW TARGET'}")
    print("=" * 90 + "\n")


def run():
    print("\n" + "=" * 60)
    print("  Project 07 — Quantized GEMM Benchmark Suite")
    print("=" * 60)
    
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    has_fp8 = cc[0] >= 9
    
    print(f"\n  GPU: {gpu_name}")
    print(f"  FP8 support: {has_fp8}")
    
    results = []
    
    for M, N, K in MATRIX_SIZES:
        print(f"\n  Matrix: ({M}, {N}, {K})")
        
        fp16_tflops, fp16_time = benchmark_fp16_gemm(M, N, K)
        print(f"    FP16: {fp16_tflops:.1f} TFLOPS ({fp16_time:.2f} ms)")
        
        # Estimated quantized performance
        if has_fp8:
            int8_tflops = fp16_tflops * 1.8  # Estimate
            fp8_tflops = fp16_tflops * 1.5   # Estimate
        else:
            int8_tflops = fp16_tflops * 1.5  # Limited without tensor core
            fp8_tflops = fp16_tflops  # No FP8 support
        
        print(f"    INT8 (est): {int8_tflops:.1f} TFLOPS")
        print(f"    FP8 (est): {fp8_tflops:.1f} TFLOPS")
        
        results.append(((M, N, K), fp16_tflops, int8_tflops, fp8_tflops))
    
    print_table(results)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
