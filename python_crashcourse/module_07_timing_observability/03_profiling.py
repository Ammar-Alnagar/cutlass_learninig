"""
Module 07 - Exercise 03: Profiling

Scenario: Your inference pipeline is slower than expected. You need to
find the actual bottleneck rather than guessing. Profiling shows exactly
where time and memory are spent.

Topics covered:
- cProfile for function-level profiling
- line_profiler for line-by-line analysis
- memory_profiler for memory usage
- Interpreting profiling results

Prerequisites:
    pip install memory-profiler line-profiler
"""

import cProfile
import pstats
import io
import time
from typing import List
import random


# =============================================================================
# Part 1: cProfile Basics
# =============================================================================

def slow_function():
    """Simulate a slow function."""
    time.sleep(0.1)
    return sum(i * i for i in range(10000))


def medium_function():
    """Simulate a medium-speed function."""
    result = 0
    for i in range(1000):
        result += i * i
    time.sleep(0.05)
    return result


def fast_function():
    """Simulate a fast function."""
    return sum(range(100))


def pipeline():
    """Simulate a processing pipeline with multiple steps."""
    results = []
    
    data = list(range(10000))
    preprocessed = [x * 2 for x in data]
    
    for _ in range(5):
        results.append(slow_function())
    
    for _ in range(10):
        results.append(medium_function())
    
    for _ in range(100):
        results.append(fast_function())
    
    return sum(results)


def profile_with_cprofile(func, sort_by='cumulative'):
    """Profile a function using cProfile."""
    # TODO: Create a profiler
    profiler = None  # cProfile.Profile()
    
    # TODO: Run the function under profiler
    
    # TODO: Create stats object and format output
    
    # TODO: Capture stats to string
    
    return "Not implemented"


# =============================================================================
# Part 2: Analyzing Profile Results
# =============================================================================

def find_bottlenecks(profile_output: str) -> List[str]:
    """Parse profile output to identify bottlenecks."""
    lines = profile_output.split('\n')
    bottlenecks = []
    
    # TODO: Parse the profile output and find top time consumers
    
    return bottlenecks


def profile_and_analyze(func):
    """Profile a function and automatically identify bottlenecks."""
    # TODO: Profile the function
    profile_output = profile_with_cprofile(func)
    
    # TODO: Find bottlenecks
    bottlenecks = find_bottlenecks(profile_output)
    
    return {
        'profile_output': profile_output,
        'bottlenecks': bottlenecks,
    }


# =============================================================================
# Part 3: Line-by-Line Profiling
# =============================================================================

def process_data_with_loops(data: List[int]) -> List[int]:
    """Process data with multiple loops - candidate for line profiling."""
    result = []
    
    for i in range(len(data)):
        squared = data[i] * data[i]
        result.append(squared)
    
    filtered = []
    for val in result:
        if val > 0:
            filtered.append(val)
    
    total = sum(filtered)
    normalized = []
    for val in filtered:
        norm_val = val / total if total > 0 else 0
        normalized.append(norm_val)
    
    return normalized


def run_line_profiler(func, *args):
    """Run line_profiler on a function."""
    # For this exercise, return simulated output
    output = """
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    10                                           def process_data_with_loops(data):
    11         1          2.0      2.0      0.1      result = []
    13      1001        150.0      0.1      7.5      for i in range(len(data)):
    14      1000        200.0      0.2     10.0          squared = data[i] * data[i]
    15      1000        180.0      0.2      9.0          result.append(squared)
    24      1001        800.0      0.8     40.0      for val in filtered:
    25       999        350.0      0.4     17.5          norm_val = val / total if total > 0 else 0
"""
    return output


# =============================================================================
# Part 4: Memory Profiling
# =============================================================================

def memory_intensive_operation(n: int) -> List[List[int]]:
    """Operation that uses significant memory."""
    result = []
    for i in range(n):
        inner_list = list(range(i * 100))
        result.append(inner_list)
    return result


def profile_memory(func, *args, **kwargs):
    """Profile memory usage of a function."""
    try:
        from memory_profiler import memory_usage
        
        mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=1)
        
        return {
            'max_memory_mb': max(mem_usage),
            'min_memory_mb': min(mem_usage),
            'memory_increase_mb': max(mem_usage) - min(mem_usage),
        }
    except ImportError:
        return {'error': 'memory_profiler not installed'}


# =============================================================================
# Part 5: Profiling ML Pipeline
# =============================================================================

class MLPipelineProfiler:
    """Profiler for ML inference pipelines."""
    
    def __init__(self):
        self.timings = {}
    
    def time_step(self, step_name: str, func, *args):
        """Time a single pipeline step."""
        start = time.perf_counter()
        result = func(*args)
        elapsed = (time.perf_counter() - start) * 1000
        
        # TODO: Store timing
        self.timings[step_name] = elapsed
        
        return result
    
    def run_pipeline(self, data):
        """Run and profile a complete ML pipeline."""
        preprocessed = self.time_step('preprocessing', self.preprocess, data)
        features = self.time_step('feature_extraction', self.extract_features, preprocessed)
        predictions = self.time_step('inference', self.infer, features)
        result = self.time_step('postprocessing', self.postprocess, predictions)
        
        return result, self.generate_report()
    
    def preprocess(self, data):
        time.sleep(0.05)
        return [x / 255.0 for x in data]
    
    def extract_features(self, data):
        time.sleep(0.1)
        return [x * 2 for x in data]
    
    def infer(self, features):
        time.sleep(0.2)
        return [sum(features) / len(features)]
    
    def postprocess(self, predictions):
        time.sleep(0.02)
        return [round(p, 4) for p in predictions]
    
    def generate_report(self) -> str:
        """Generate timing report."""
        report = "Pipeline Timing Report\n"
        report += "=" * 40 + "\n"
        total = 0
        for step, time_ms in self.timings.items():
            report += f"{step}: {time_ms:.2f} ms\n"
            total += time_ms
        report += "=" * 40 + "\n"
        report += f"Total: {total:.2f} ms\n"
        return report


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 07 - Exercise 03: Self-Check")
    print("=" * 60)
    
    # Check 1: cProfile basic usage
    print("\nProfiling pipeline...")
    profile_output = profile_with_cprofile(pipeline)
    assert profile_output != "Not implemented"
    print("Profile output (first 500 chars):")
    print(profile_output[:500])
    print("[PASS] profile_with_cprofile")
    
    # Check 2: Find bottlenecks
    bottlenecks = find_bottlenecks(profile_output)
    print(f"Identified bottlenecks: {bottlenecks}")
    print("[PASS] find_bottlenecks")
    
    # Check 3: Line profiler simulation
    print("\nLine profiler output:")
    line_output = run_line_profiler(process_data_with_loops, list(range(1000)))
    assert 'Line #' in line_output
    print(line_output)
    print("[PASS] run_line_profiler")
    
    # Check 4: Memory profiling
    print("\nMemory profiling...")
    mem_results = profile_memory(memory_intensive_operation, 100)
    if 'error' not in mem_results:
        assert mem_results['max_memory_mb'] > mem_results['min_memory_mb']
        print(f"Memory increase: {mem_results['memory_increase_mb']:.2f} MB")
    else:
        print(f"Skipping: {mem_results['error']}")
    print("[PASS] profile_memory")
    
    # Check 5: ML Pipeline Profiler
    print("\nML Pipeline profiling...")
    profiler = MLPipelineProfiler()
    test_data = list(range(1000))
    result, report = profiler.run_pipeline(test_data)
    assert result is not None
    assert 'Total:' in report
    print(report)
    print("[PASS] MLPipelineProfiler")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
