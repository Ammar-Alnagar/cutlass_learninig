"""
Project 04 — Benchmarks Master
File: roofline.py

Generate roofline analysis charts for all kernel implementations.
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path


@dataclass
class KernelBenchmark:
    name: str
    flops: int
    bytes_read: int
    bytes_write: int
    latency_ms: float
    
    @property
    def arithmetic_intensity(self) -> float:
        """Ops per byte."""
        total_bytes = self.bytes_read + self.bytes_write
        return self.flops / total_bytes if total_bytes > 0 else 0
    
    @property
    def achieved_tflops(self) -> float:
        """Achieved TFLOPS."""
        if self.latency_ms <= 0:
            return 0
        return self.flops / (self.latency_ms * 1e-3) / 1e12


@dataclass
class GPU specs:
    name: str
    sm: int
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    memory_bandwidth_gbps: float


# GPU specifications
GPU_SPECS = {
    "A100": GPU specs(
        name="A100", sm=80,
        peak_fp16_tflops=312,  # Dense FP16 Tensor Core
        peak_fp32_tflops=19.5,
        memory_bandwidth_gbps=1555,
    ),
    "H100": GPU specs(
        name="H100", sm=90,
        peak_fp16_tflops=989,  # Dense FP16 Tensor Core
        peak_fp32_tflops=67,
        memory_bandwidth_gbps=2000,
    ),
    "B200": GPU specs(
        name="B200", sm=100,
        peak_fp16_tflops=20000,  # FP4 dense (FP16 equivalent lower)
        peak_fp32_tflops=90,
        memory_bandwidth_gbps=8000,
    ),
}


def benchmark_gemm(M: int, K: int, N: int, dtype=torch.float16, 
                   num_warmup=10, num_iters=50) -> KernelBenchmark:
    """Benchmark GEMM and return metrics."""
    device = torch.device("cuda")
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = torch.zeros(M, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    latency_ms = (time.perf_counter() - start) / num_iters * 1000
    
    # Compute metrics
    flops = 2 * M * K * N  # 2 ops per FMA
    bytes_read = (M * K + K * N) * 2  # FP16 = 2 bytes
    bytes_write = M * N * 2
    
    return KernelBenchmark(
        name=f"GEMM {M}x{K}x{N}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
        latency_ms=latency_ms,
    )


def benchmark_attention(B: int, H: int, S: int, D: int, 
                        dtype=torch.float16, num_warmup=10, num_iters=50) -> KernelBenchmark:
    """Benchmark attention and return metrics."""
    device = torch.device("cuda")
    
    Q = torch.randn(B, H, S, D, dtype=dtype, device=device)
    K = torch.randn(B, H, S, D, dtype=dtype, device=device)
    V = torch.randn(B, H, S, D, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    
    latency_ms = (time.perf_counter() - start) / num_iters * 1000
    
    # Compute metrics (attention is O(B * H * S² * D))
    flops = 2 * B * H * S * S * D * 2  # Q@K^T + weights@V
    bytes_read = (B * H * S * D * 3) * 2  # Q, K, V
    bytes_write = (B * H * S * D) * 2  # Output
    
    return KernelBenchmark(
        name=f"Attention B{B}H{H}S{S}D{D}",
        flops=flops,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
        latency_ms=latency_ms,
    )


def generate_roofline_chart(benchmarks: List[KernelBenchmark], 
                            gpu: GPU specs,
                            output_path: str = "roofline_chart.png"):
    """Generate roofline analysis chart."""
    
    # Compute roofline boundaries
    max_ai = max(b.arithmetic_intensity for b in benchmarks) if benchmarks else 100
    ai_range = np.logspace(0, np.log10(max_ai * 10), 100)
    
    # Memory-bound region: TFLOPS = bandwidth × AI
    memory_bound = gpu.memory_bandwidth_gbps * ai_range / 1e3  # GB/s → TFLOPS
    
    # Compute-bound region: flat at peak TFLOPS
    compute_bound = np.full_like(ai_range, gpu.peak_fp16_tflops)
    
    # Roofline is minimum of both
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot roofline
    ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
    ax.fill_between(ai_range, roofline, alpha=0.2, label='Achievable Region')
    
    # Plot benchmarks
    for b in benchmarks:
        ax.scatter(b.arithmetic_intensity, b.achieved_tflops, 
                   s=100, marker='o', label=b.name)
        # Add annotation
        ax.annotate(b.name[:20], 
                    (b.arithmetic_intensity, b.achieved_tflops),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    # Add efficiency lines (distance from roofline)
    for efficiency in [0.25, 0.5, 0.75]:
        ax.loglog(ai_range, roofline * efficiency, 'k--', alpha=0.3, 
                  label=f'{efficiency*100:.0f}% Efficiency' if efficiency == 0.5 else '')
    
    # Labels and title
    ax.set_xlabel('Arithmetic Intensity (ops/byte)', fontsize=12)
    ax.set_ylabel('Achieved TFLOPS', fontsize=12)
    ax.set_title(f'Roofline Analysis - {gpu.name} (SM{gpu.sm})', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    
    # Add GPU specs text box
    specs_text = f"GPU: {gpu.name}\n"
    specs_text += f"Peak FP16: {gpu.peak_fp16_tflops:.0f} TFLOPS\n"
    specs_text += f"Bandwidth: {gpu.memory_bandwidth_gbps:.0f} GB/s"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Roofline chart saved to: {output_path}")
    
    return fig, ax


def run_benchmarks():
    """Run all benchmarks and generate roofline chart."""
    
    print("=" * 70)
    print("Benchmarks Master - Roofline Analysis")
    print("=" * 70)
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability()
    sm = compute_cap[0] * 10 + compute_cap[1]
    
    # Find matching GPU spec
    gpu_spec = None
    for name, spec in GPU_SPECS.items():
        if name.lower() in gpu_name.lower():
            gpu_spec = spec
            break
    
    if gpu_spec is None:
        # Use closest match
        if sm >= 100:
            gpu_spec = GPU_SPECS["B200"]
        elif sm >= 90:
            gpu_spec = GPU_SPECS["H100"]
        else:
            gpu_spec = GPU_SPECS["A100"]
    
    print(f"\nGPU: {gpu_name} (SM{sm})")
    print(f"Using specs: {gpu_spec.name}")
    print(f"  Peak FP16: {gpu_spec.peak_fp16_tflops} TFLOPS")
    print(f"  Bandwidth: {gpu_spec.memory_bandwidth_gbps} GB/s")
    
    # Run GEMM benchmarks
    print("\nRunning GEMM benchmarks...")
    benchmarks = []
    
    gemm_shapes = [
        (512, 1024, 2048),
        (1024, 2048, 4096),
        (2048, 4096, 8192),
        (4096, 8192, 8192),
    ]
    
    for M, K, N in gemm_shapes:
        try:
            b = benchmark_gemm(M, K, N)
            benchmarks.append(b)
            print(f"  {b.name}: {b.achieved_tflops:.1f} TFLOPS "
                  f"(AI={b.arithmetic_intensity:.1f} ops/byte)")
        except Exception as e:
            print(f"  {M}x{K}x{N}: Failed - {e}")
    
    # Run attention benchmarks
    print("\nRunning attention benchmarks...")
    
    attention_configs = [
        (1, 32, 512, 128),
        (8, 32, 512, 128),
        (32, 32, 1024, 128),
        (32, 32, 2048, 128),
    ]
    
    for B, H, S, D in attention_configs:
        try:
            b = benchmark_attention(B, H, S, D)
            benchmarks.append(b)
            print(f"  {b.name}: {b.achieved_tflops:.1f} TFLOPS "
                  f"(AI={b.arithmetic_intensity:.1f} ops/byte)")
        except Exception as e:
            print(f"  B{B}H{H}S{S}D{D}: Failed - {e}")
    
    # Generate roofline chart
    print("\nGenerating roofline chart...")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"roofline_{gpu_spec.name.lower()}.png"
    
    generate_roofline_chart(benchmarks, gpu_spec, str(output_path))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\n{'Kernel':<30} {'TFLOPS':<12} {'AI (ops/B)':<12} {'Efficiency'}")
    print("-" * 70)
    
    for b in benchmarks:
        efficiency = b.achieved_tflops / gpu_spec.peak_fp16_tflops * 100
        print(f"{b.name:<30} {b.achieved_tflops:<12.1f} {b.arithmetic_intensity:<12.1f} {efficiency:.1f}%")
    
    print(f"\nRoofline chart saved to: {output_path}")


if __name__ == "__main__":
    run_benchmarks()
