"""
Project 08 — Roofline Chart Generator

Generates roofline performance charts from benchmark data.
"""

import torch
import os

# Roofline data for common GPUs
ROOFLINE_DATA = {
    (8, 0): {  # A100
        'name': 'A100',
        'fp16_peak': 312.0,
        'fp32_peak': 19.5,
        'memory_bw': 1555.0,
    },
    (9, 0): {  # H100
        'name': 'H100',
        'fp16_peak': 989.0,
        'fp32_peak': 67.0,
        'memory_bw': 3350.0,
    },
    (10, 0): {  # B200 (estimated)
        'name': 'B200',
        'fp16_peak': 2250.0,
        'fp32_peak': 100.0,
        'memory_bw': 8000.0,
    },
}

# Benchmark results (to be filled from actual runs)
BENCHMARK_RESULTS = {
    'gemm_ampere': {'tflops': 0.0, 'arithmetic_intensity': 100.0},
    'gemm_hopper': {'tflops': 0.0, 'arithmetic_intensity': 100.0},
    'fa2_prefill': {'tflops': 0.0, 'arithmetic_intensity': 50.0},
    'fa3_warp_specialized': {'tflops': 0.0, 'arithmetic_intensity': 50.0},
    'softmax': {'bw_gbs': 0.0, 'bw_util_pct': 0.0},
}


def get_gpu_roofline():
    """Get roofline data for current GPU."""
    try:
        cc = torch.cuda.get_device_capability(0)
        key = (cc[0], cc[1])
        if key in ROOFLINE_DATA:
            return ROOFLINE_DATA[key]
        # Try to find closest match
        for roof_key, roof_data in ROOFLINE_DATA.items():
            if roof_key[0] == cc[0]:
                return roof_data
    except:
        pass
    
    # Default to A100
    return ROOFLINE_DATA[(8, 0)]


def print_roofline_ascii(roofline):
    """Print ASCII roofline chart."""
    peak = roofline['fp16_peak']
    bw = roofline['memory_bw']
    
    print("\n" + "=" * 70)
    print(f"  Roofline Model — {roofline['name']}")
    print("=" * 70)
    
    # ASCII chart
    print("\n  Performance (TFLOPS)")
    print("  ^")
    print(f"  │ {peak:>6.0f} ───────────────────────────── Memory Bound")
    print("  │                         ╱")
    print(f"  │                        ╱")
    print(f"  │                       ╱  Compute Bound")
    print(f"  │                      ╱")
    print(f"  │                     ╱")
    print(f"  │                    ╱")
    print(f"  │ {peak/10:>6.0f} ──────────────╱")
    print("  │                  ╱")
    print(f"  │                 ╱")
    print(f"  │ {peak/100:>6.0f} ────────────╱")
    print("  │              ╱")
    print("  │             ╱")
    print(f"  │ {peak/1000:>6.0f} ─────────╱")
    print("  │           ╱")
    print("  │          ╱")
    print("  │         ╱")
    print("  └────────┴─────────┴─────────┴──────────> Arithmetic Intensity")
    print("         0.01      0.1       1        10   (FLOP/byte)")
    print(f"\n  Peak Compute: {peak:.0f} TFLOPS (FP16 Tensor)")
    print(f"  Peak Memory:  {bw:.0f} GB/s")
    print(f"  Ridge Point:  {peak / bw * 1000:.1f} FLOP/byte")
    print("=" * 70)


def print_benchmark_table(results, roofline):
    """Print benchmark results table."""
    print("\n" + "=" * 80)
    print(f"  Benchmark Results — {roofline['name']}")
    print("=" * 80)
    
    print("\n┌──────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Kernel                   │  TFLOPS      │  % Roofline  │  AI (FLOP/B) │")
    print("├──────────────────────────┼──────────────┼──────────────┼──────────────┤")
    
    kernels = [
        ('Tiled GEMM (Ampere)', results.get('gemm_ampere', {})),
        ('Tiled GEMM (Hopper)', results.get('gemm_hopper', {})),
        ('FlashAttention-2', results.get('fa2_prefill', {})),
        ('FlashAttention-3', results.get('fa3_warp_specialized', {})),
    ]
    
    for name, data in kernels:
        tflops = data.get('tflops', 0.0)
        roofline_pct = (tflops / roofline['fp16_peak']) * 100 if roofline['fp16_peak'] > 0 else 0
        ai = data.get('arithmetic_intensity', 0.0)
        
        status = '✓' if roofline_pct >= 75 else '✗'
        print(f"│ {name:<24} │ {tflops:>10.1f}   │ {roofline_pct:>10.1f}%    {status} │ {ai:>10.1f}   │")
    
    print("└──────────────────────────┴──────────────┴──────────────┴──────────────┘")
    
    # Memory bandwidth results
    print("\n┌──────────────────────────┬──────────────┬──────────────┐")
    print("│ Kernel                   │  GB/s        │  % Peak BW   │")
    print("├──────────────────────────┼──────────────┼──────────────┤")
    
    softmax_data = results.get('softmax', {})
    bw_gbs = softmax_data.get('bw_gbs', 0.0)
    bw_pct = softmax_data.get('bw_util_pct', 0.0)
    status = '✓' if bw_pct >= 85 else '✗'
    print(f"│ {'Online Softmax':<24} │ {bw_gbs:>10.1f}   │ {bw_pct:>10.1f}%    {status} │")
    
    print("└──────────────────────────┴──────────────┴──────────────┘")
    print("=" * 80 + "\n")


def save_results_csv(results, roofline):
    """Save results to CSV."""
    os.makedirs('results', exist_ok=True)
    
    csv_path = 'results/benchmark_summary.csv'
    with open(csv_path, 'w') as f:
        f.write('Kernel,TFLOPS,Roofline_Pct,Arithmetic_Intensity\n')
        
        kernels = ['gemm_ampere', 'gemm_hopper', 'fa2_prefill', 'fa3_warp_specialized']
        for kernel in kernels:
            data = results.get(kernel, {})
            tflops = data.get('tflops', 0.0)
            roofline_pct = (tflops / roofline['fp16_peak']) * 100 if roofline['fp16_peak'] > 0 else 0
            ai = data.get('arithmetic_intensity', 0.0)
            f.write(f'{kernel},{tflops:.1f},{roofline_pct:.1f},{ai:.1f}\n')
    
    print(f"  Results saved to: {csv_path}")


def run():
    """Generate roofline charts and benchmark summary."""
    
    print("\n" + "=" * 60)
    print("  Project 08 — Roofline Chart Generator")
    print("=" * 60)
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n  GPU: {gpu_name}")
    
    roofline = get_gpu_roofline()
    print(f"  Roofline model: {roofline['name']}")
    
    print_roofline_ascii(roofline)
    print_benchmark_table(BENCHMARK_RESULTS, roofline)
    save_results_csv(BENCHMARK_RESULTS, roofline)
    
    print("\n  → Update BENCHMARK_RESULTS with actual measurements")
    print("  → Re-run to generate final roofline chart")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
