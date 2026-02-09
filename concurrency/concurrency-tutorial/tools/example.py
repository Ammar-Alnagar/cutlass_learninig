# Tools and Profiling - Hands-on Example (Python Version)
# This example demonstrates how to profile and analyze concurrent code in Python

import threading
import time
import queue
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
import cProfile
import pstats
from pstats import SortKey
import io
import psutil
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Helper class for performance measurement
class Profiler:
    def __init__(self):
        self.measurements = {}
    
    def start_timer(self, name: str):
        if name not in self.measurements:
            self.measurements[name] = []
        setattr(self, f"_{name}_start", time.perf_counter())
    
    def stop_timer(self, name: str):
        end_time = time.perf_counter()
        start_time = getattr(self, f"_{name}_start", None)
        if start_time is not None:
            duration = (end_time - start_time) * 1000000  # Convert to microseconds
            self.measurements[name].append(duration)
    
    def print_summary(self):
        print("\n=== Performance Summary ===")
        for name, measurements in self.measurements.items():
            if measurements:
                avg = sum(measurements) / len(measurements)
                print(f"{name}: {len(measurements)} runs, avg: {avg:.2f} μs")
    
    def print_detailed_stats(self):
        print("\n=== Detailed Statistics ===")
        for name, measurements in self.measurements.items():
            if not measurements:
                continue
            
            min_val = min(measurements)
            max_val = max(measurements)
            avg = sum(measurements) / len(measurements)
            stdev = statistics.stdev(measurements) if len(measurements) > 1 else 0
            
            print(f"{name}:")
            print(f"  Runs: {len(measurements)}")
            print(f"  Min: {min_val:.2f} μs")
            print(f"  Max: {max_val:.2f} μs")
            print(f"  Avg: {avg:.2f} μs")
            print(f"  Std Dev: {stdev:.2f} μs")

# Example 1: Measuring lock contention
class ContentionBenchmark:
    def __init__(self, profiler: Profiler):
        self.profiler = profiler
        self.mtx = threading.Lock()
        self.atomic_counter = 0  # Python doesn't have true atomics, using lock instead
        self.regular_counter = 0
    
    def benchmark_mutex(self):
        self.profiler.start_timer("mutex_benchmark")
        
        def worker():
            for _ in range(10000):
                with self.mtx:
                    self.regular_counter += 1
        
        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.profiler.stop_timer("mutex_benchmark")
    
    def benchmark_atomic(self):
        # In Python, we'll simulate atomic operations using a lock
        # (Python doesn't have true atomic operations in the standard library)
        self.profiler.start_timer("atomic_benchmark")
        
        def worker():
            for _ in range(10000):
                with self.mtx:  # Using lock to simulate atomicity
                    self.atomic_counter += 1
        
        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.profiler.stop_timer("atomic_benchmark")

# Example 2: False sharing demonstration (conceptual in Python)
def false_sharing_demo():
    print("\n=== False Sharing Demo (Conceptual) ===")
    print("Python's GIL limits true parallelism, so false sharing effects are minimized")
    print("However, we can demonstrate the concept with memory layout:")
    
    # In a real system, we'd demonstrate how adjacent data can cause cache line conflicts
    # In Python, this is less relevant due to the GIL and object layout
    print("In C/C++/Java, placing frequently modified variables in the same cache line causes false sharing")
    print("Solution: Add padding between frequently modified variables")

# Example 3: Thread pool performance analysis
class ThreadPoolProfiler:
    def __init__(self, profiler: Profiler):
        self.profiler = profiler
        self.tasks_executed = 0
        self.lock = threading.Lock()
    
    def benchmark_throughput(self, num_threads: int, num_tasks: int):
        label = f"thread_pool_throughput_{num_threads}_threads"
        self.profiler.start_timer(label)
        
        def worker():
            # Simulate work
            time.sleep(0.0001)  # 100 microseconds of work
            with self.lock:
                self.tasks_executed += 1
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_tasks)]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
        
        self.profiler.stop_timer(label)

# Example 4: Lock contention analysis
class LockContentionAnalyzer:
    def __init__(self, profiler: Profiler):
        self.profiler = profiler
        self.shared_resource_lock = threading.Lock()
        self.resource_value = 0
    
    def analyze_contention(self, num_threads: int, iterations: int):
        label = f"contention_{num_threads}_threads"
        self.profiler.start_timer(label)
        
        def worker():
            for _ in range(iterations):
                with self.shared_resource_lock:
                    self.resource_value += 1
                    # Simulate some work holding the lock
                    time.sleep(0.00001)  # 10 microseconds
        
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.profiler.stop_timer(label)

# Example 5: Memory access pattern analysis
def memory_access_analysis():
    print("\n=== Memory Access Pattern Analysis ===")
    
    size = 100000
    data = list(range(size))
    
    # Sequential access (cache-friendly)
    start_seq = time.perf_counter()
    seq_sum = sum(data)  # Built-in sum is optimized
    end_seq = time.perf_counter()
    seq_duration = (end_seq - start_seq) * 1000000  # Convert to microseconds
    
    # Random access (cache-unfriendly)
    indices = list(range(size))
    random.shuffle(indices)
    
    start_rand = time.perf_counter()
    rand_sum = sum(data[i] for i in indices)  # Less optimized due to random access
    end_rand = time.perf_counter()
    rand_duration = (end_rand - start_rand) * 1000000  # Convert to microseconds
    
    print(f"Sequential access: {seq_duration:.2f} μs")
    print(f"Random access: {rand_duration:.2f} μs")
    if seq_duration > 0:
        print(f"Random access overhead: {rand_duration / seq_duration:.2f}x")

# Example 6: Using cProfile for concurrency analysis
def profile_concurrent_code():
    print("\n=== cProfile Analysis ===")
    
    def cpu_bound_task(n):
        # Simulate CPU-intensive work
        result = 0
        for i in range(n * 1000):
            result += i * i
        return result
    
    def run_concurrent_tasks():
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_bound_task, 1000) for _ in range(8)]
            results = [future.result() for future in futures]
        return results
    
    # Profile the concurrent execution
    pr = cProfile.Profile()
    pr.enable()
    results = run_concurrent_tasks()
    pr.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(10)  # Print top 10 functions
    
    print(s.getvalue())

# Example 7: System resource monitoring
def system_monitoring():
    print("\n=== System Resource Monitoring ===")
    
    # Get current process info
    process = psutil.Process(os.getpid())
    
    print(f"Current process ID: {process.pid}")
    print(f"CPU percent: {process.cpu_percent(interval=1)}%")
    print(f"Memory info: {process.memory_info()}")
    print(f"Number of threads: {process.num_threads()}")
    
    # Monitor system-wide stats
    print(f"System CPU percent: {psutil.cpu_percent(percpu=True)}")
    print(f"System memory percent: {psutil.virtual_memory().percent}%")

# Example 8: Benchmarking different synchronization primitives
def sync_primitive_benchmark():
    print("\n=== Synchronization Primitive Benchmark ===")
    
    profiler = Profiler()
    
    # Test with Lock
    shared_counter = [0]
    lock = threading.Lock()
    
    def lock_worker():
        for _ in range(10000):
            with lock:
                shared_counter[0] += 1
    
    profiler.start_timer("lock_sync")
    threads = [threading.Thread(target=lock_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    profiler.stop_timer("lock_sync")
    
    # Test with RLock
    shared_counter[0] = 0
    rlock = threading.RLock()
    
    def rlock_worker():
        for _ in range(10000):
            with rlock:
                shared_counter[0] += 1
    
    profiler.start_timer("rlock_sync")
    threads = [threading.Thread(target=rlock_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    profiler.stop_timer("rlock_sync")
    
    # Print results
    for name, measurements in profiler.measurements.items():
        if measurements:
            avg = sum(measurements) / len(measurements)
            print(f"{name}: avg: {avg:.2f} μs")

if __name__ == "__main__":
    print("Tools and Profiling - Hands-on Example (Python)")
    
    profiler = Profiler()
    
    # Example 1: Compare different synchronization methods
    bench = ContentionBenchmark(profiler)
    bench.benchmark_mutex()
    bench.benchmark_atomic()
    
    # Example 2: False sharing demonstration
    false_sharing_demo()
    
    # Example 3: Thread pool throughput analysis
    pool_profiler = ThreadPoolProfiler(profiler)
    pool_profiler.benchmark_throughput(2, 50)
    pool_profiler.benchmark_throughput(4, 50)
    pool_profiler.benchmark_throughput(8, 50)
    
    # Example 4: Lock contention analysis with different thread counts
    analyzer = LockContentionAnalyzer(profiler)
    analyzer.analyze_contention(2, 5000)
    analyzer.analyze_contention(4, 5000)
    analyzer.analyze_contention(8, 5000)
    
    # Example 5: Memory access pattern analysis
    memory_access_analysis()
    
    # Example 6: cProfile analysis
    profile_concurrent_code()
    
    # Example 7: System monitoring
    system_monitoring()
    
    # Example 8: Synchronization primitive benchmark
    sync_primitive_benchmark()
    
    # Print results
    profiler.print_summary()
    profiler.print_detailed_stats()
    
    print("\nAll profiling examples completed!")