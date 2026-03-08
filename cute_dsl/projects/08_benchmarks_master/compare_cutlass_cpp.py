"""
Project 08 — CuTe DSL vs CuTe C++ Performance Comparison

Compares performance between CuTe DSL (Python) and CuTe C++ implementations.
"""

import os

# Benchmark comparison data (to be filled from actual runs)
COMPARISON_DATA = {
    'Tiled GEMM (SM80)': {
        'cpp_tflops': 241.0,
        'dsl_tflops': 198.0,
        'dsl_efficiency': 0.0,  # Computed
    },
    'Tiled GEMM (SM90)': {
        'cpp_tflops': 750.0,
        'dsl_tflops': 620.0,
        'dsl_efficiency': 0.0,
    },
    'FlashAttention-2': {
        'cpp_tflops': 180.0,
        'dsl_tflops': 150.0,
        'dsl_efficiency': 0.0,
    },
    'FlashAttention-3': {
        'cpp_tflops': 700.0,
        'dsl_tflops': 580.0,
        'dsl_efficiency': 0.0,
    },
    'Online Softmax': {
        'cpp_gbs': 1400.0,
        'dsl_gbs': 1180.0,
        'dsl_efficiency': 0.0,
    },
}


def compute_efficiency():
    """Compute DSL efficiency vs C++."""
    for kernel, data in COMPARISON_DATA.items():
        if 'tflops' in kernel.lower() or 'gemm' in kernel.lower() or 'attention' in kernel.lower():
            cpp = data.get('cpp_tflops', 0)
            dsl = data.get('dsl_tflops', 0)
        else:
            cpp = data.get('cpp_gbs', 0)
            dsl = data.get('dsl_gbs', 0)
        
        data['dsl_efficiency'] = (dsl / cpp * 100) if cpp > 0 else 0


def print_comparison_table():
    """Print CuTe DSL vs C++ comparison table."""
    
    compute_efficiency()
    
    print("\n" + "=" * 90)
    print("  CuTe DSL vs CuTe C++ Performance Comparison")
    print("=" * 90)
    
    print("\n┌──────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Kernel                   │  CuTe C++    │  CuTe DSL    │  DSL Eff.    │")
    print("│                          │  (Baseline)  │  (Python)    │  (% of C++)  │")
    print("├──────────────────────────┼──────────────┼──────────────┼──────────────┤")
    
    for kernel, data in COMPARISON_DATA.items():
        if 'gemm' in kernel.lower() or 'attention' in kernel.lower():
            cpp_val = data.get('cpp_tflops', 0)
            dsl_val = data.get('dsl_tflops', 0)
            unit = 'TFLOPS'
        else:
            cpp_val = data.get('cpp_gbs', 0)
            dsl_val = data.get('dsl_gbs', 0)
            unit = 'GB/s'
        
        efficiency = data.get('dsl_efficiency', 0)
        status = '✓' if efficiency >= 80 else '⚠'
        
        print(f"│ {kernel:<24} │ {cpp_val:>10.1f} {unit:<6} │ {dsl_val:>10.1f} {unit:<6} │ {efficiency:>10.1f}%    {status} │")
    
    print("└──────────────────────────┴──────────────┴──────────────┴──────────────┘")
    
    # Summary
    avg_efficiency = sum(d['dsl_efficiency'] for d in COMPARISON_DATA.values()) / len(COMPARISON_DATA)
    
    print(f"\n  Average DSL Efficiency: {avg_efficiency:.1f}% of C++ performance")
    print(f"  Target: >80% of C++ performance")
    print(f"  Status: {'✓ ACHIEVED' if avg_efficiency >= 80 else '⚠ BELOW TARGET'}")
    
    print("\n  Notes:")
    print("    - CuTe DSL JIT compilation has minimal overhead")
    print("    - Most performance difference comes from tuning parameters")
    print("    - Python host launch overhead is negligible for large kernels")
    print("=" * 90 + "\n")


def save_comparison_csv():
    """Save comparison to CSV."""
    os.makedirs('results', exist_ok=True)
    
    csv_path = 'results/dsl_vs_cpp_comparison.csv'
    with open(csv_path, 'w') as f:
        f.write('Kernel,CPP_Value,DSL_Value,DSL_Efficiency_Pct,Unit\n')
        
        for kernel, data in COMPARISON_DATA.items():
            if 'gemm' in kernel.lower() or 'attention' in kernel.lower():
                cpp_val = data.get('cpp_tflops', 0)
                dsl_val = data.get('dsl_tflops', 0)
                unit = 'TFLOPS'
            else:
                cpp_val = data.get('cpp_gbs', 0)
                dsl_val = data.get('dsl_gbs', 0)
                unit = 'GB/s'
            
            efficiency = data.get('dsl_efficiency', 0)
            f.write(f'{kernel},{cpp_val},{dsl_val},{efficiency:.1f},{unit}\n')
    
    print(f"  Comparison saved to: {csv_path}")


def run():
    """Generate CuTe DSL vs C++ comparison."""
    
    print("\n" + "=" * 60)
    print("  Project 08 — CuTe DSL vs C++ Comparison")
    print("=" * 60)
    
    print_comparison_table()
    save_comparison_csv()
    
    print("\n  → Update COMPARISON_DATA with actual benchmark measurements")
    print("  → Re-run to generate final comparison report")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
