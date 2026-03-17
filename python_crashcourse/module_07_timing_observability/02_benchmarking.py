"""
Module 07 - Exercise 02: Benchmarking

Scenario: You're comparing different implementations of a data processing
function to find the fastest one. Proper benchmarking requires multiple
runs, statistical analysis, and avoiding common pitfalls.

Topics covered:
- timeit module for micro-benchmarks
- Warmup runs and statistical analysis
- Variance, percentiles, outlier detection
- Avoiding GC and caching pitfalls
"""

import timeit
import statistics
import random
from typing import List, Callable
import gc


# =============================================================================
# Part 1: timeit Basics
# =============================================================================

def benchmark_with_timeit():
    """Use timeit to benchmark a simple operation."""
    results = {}
    
    # List comprehension
    results['list_comp'] = timeit.timeit(
        stmt='[x*x for x in range(1000)]',
        number=10000
    )
    
    # TODO: Benchmark map with lambda
    results['map'] = None
    
    # For loop function
    def for_loop():
        result = []
        for x in range(1000):
            result.append(x * x)
        return result
    
    results['for_loop'] = None
    
    return results


def benchmark_function_call():
    """Benchmark a function call with setup code."""
    setup = "import random; data = list(range(1000)); random.shuffle(data)"
    
    time_sorted = timeit.timeit(
        stmt='sorted(data)',
        setup=setup,
        number=1000
    )
    
    # TODO: Benchmark list.sort() (in-place sort)
    time_sort = None
    
    return time_sorted, time_sort


# =============================================================================
# Part 2: Statistical Benchmarking
# =============================================================================

def run_benchmark_stats(func: Callable, iterations: int = 100) -> dict:
    """Run a benchmark with statistical analysis."""
    times = []
    
    # TODO: Run function 'iterations' times, recording each elapsed time
    for _ in range(iterations):
        start = timeit.default_timer()
        func()
        elapsed = timeit.default_timer() - start
        times.append(elapsed)
    
    # TODO: Calculate statistics
    stats = {
        'mean': None,
        'median': None,
        'stdev': None,
        'min': None,
        'max': None,
        'p95': None,
        'p99': None,
    }
    
    return stats


def compare_implementations():
    """Compare multiple implementations with proper statistics."""
    def method_list_comp(data):
        return [x for x in data if x > 0]
    
    def method_filter(data):
        return list(filter(lambda x: x > 0, data))
    
    def method_loop(data):
        result = []
        for x in data:
            if x > 0:
                result.append(x)
        return result
    
    test_data = [random.randint(-100, 100) for _ in range(1000)]
    
    results = {}
    
    results['list_comp'] = run_benchmark_stats(
        lambda: method_list_comp(test_data),
        iterations=500
    )
    
    # TODO: Benchmark method_filter and method_loop
    results['filter'] = None
    results['loop'] = None
    
    return results


# =============================================================================
# Part 3: Warmup and GC Considerations
# =============================================================================

def benchmark_with_warmup(func: Callable, warmup_runs: int = 10, timed_runs: int = 100):
    """Benchmark with warmup runs to stabilize performance."""
    # TODO: Run warmup iterations (don't measure)
    for _ in range(warmup_runs):
        func()
    
    # TODO: Run timed iterations
    times = []
    
    return {
        'mean_us': statistics.mean(times) * 1_000_000 if times else 0,
        'median_us': statistics.median(times) * 1_000_000 if times else 0,
        'p99_us': sorted(times)[int(len(times) * 0.99)] * 1_000_000 if times else 0,
    }


def benchmark_gc_impact():
    """Demonstrate the impact of garbage collection on benchmarks."""
    def allocate_memory():
        return [list(range(1000)) for _ in range(100)]
    
    results = {}
    
    gc.enable()
    results['with_gc'] = timeit.timeit(allocate_memory, number=100)
    
    # TODO: Run benchmark with GC disabled
    gc.disable()
    results['without_gc'] = None
    gc.enable()
    
    return results


# =============================================================================
# Part 4: Benchmarking ML Operations
# =============================================================================

def benchmark_matrix_operations():
    """Benchmark different matrix operation approaches."""
    import numpy as np
    
    size = 1000
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    
    results = {}
    
    results['matmul'] = timeit.timeit(
        stmt='a @ b',
        globals={'a': a, 'b': b},
        number=10
    )
    
    # TODO: Benchmark element-wise and dot product
    results['elementwise'] = None
    results['dot'] = None
    
    return results


def benchmark_inference_latency():
    """Simulate benchmarking inference latency."""
    def fake_inference():
        time.sleep(random.uniform(0.01, 0.05))
        return sum(i * i for i in range(1000))
    
    # TODO: Run benchmark with warmup
    results = benchmark_with_warmup(fake_inference, warmup_runs=5, timed_runs=50)
    
    return results


# =============================================================================
# Part 5: Avoiding Benchmarking Pitfalls
# =============================================================================

def common_pitfalls_demo():
    """Demonstrate common benchmarking mistakes."""
    pitfalls = {}
    
    def bad_benchmark():
        data = list(range(1000))
        return sum(data)
    
    def good_benchmark():
        data = list(range(1000))
        return sum(data)
    
    pitfalls['bad_includes_setup'] = timeit.timeit(bad_benchmark, number=1000)
    pitfalls['good_setup_outside'] = timeit.timeit(
        stmt='sum(data)',
        setup='data = list(range(1000))',
        number=1000
    )
    
    def quick_func():
        return 1 + 1
    
    pitfalls['few_iterations'] = timeit.timeit(quick_func, number=10)
    pitfalls['many_iterations'] = timeit.timeit(quick_func, number=100000)
    
    return pitfalls


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 07 - Exercise 02: Self-Check")
    print("=" * 60)
    
    # Check 1: timeit basics
    print("\nBenchmarking basic operations...")
    results = benchmark_with_timeit()
    for method, time_taken in results.items():
        assert time_taken > 0, f"{method} should have positive time"
        print(f"  {method}: {time_taken:.4f}s")
    print("[PASS] benchmark_with_timeit")
    
    # Check 2: Function call benchmark
    time_sorted, time_sort = benchmark_function_call()
    assert time_sorted > 0 and time_sort > 0
    print(f"[PASS] benchmark_function_call")
    
    # Check 3: Statistical benchmarking
    def test_func():
        return sum(range(1000))
    
    stats = run_benchmark_stats(test_func, iterations=50)
    assert all(v is not None for v in stats.values())
    print(f"[PASS] run_benchmark_stats")
    
    # Check 4: Compare implementations
    print("\nComparing filter implementations...")
    comparison = compare_implementations()
    for method, method_stats in comparison.items():
        assert method_stats is not None, f"{method} should have stats"
    print("[PASS] compare_implementations")
    
    # Check 5: GC impact
    print("\nGC impact demonstration...")
    gc_results = benchmark_gc_impact()
    assert 'with_gc' in gc_results and 'without_gc' in gc_results
    print(f"  With GC: {gc_results['with_gc']:.4f}s")
    print(f"  Without GC: {gc_results['without_gc']:.4f}s")
    print("[PASS] benchmark_gc_impact")
    
    # Check 6: Common pitfalls
    print("\nCommon pitfalls demonstration...")
    pitfalls = common_pitfalls_demo()
    assert pitfalls['bad_includes_setup'] > pitfalls['good_setup_outside']
    print("[PASS] common_pitfalls_demo")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
