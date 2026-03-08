"""
Module 14 — Threading & Multiprocessing
Exercise 03 — Parallel Benchmark Harness

WHAT YOU'RE BUILDING:
  Combine threading and multiprocessing for a real benchmark harness.
  Use multiprocessing for parallel kernel evaluation (CPU-bound),
  threading for concurrent result logging (I/O-bound).

OBJECTIVE:
  - Build a parallel benchmark runner
  - Combine multiprocessing with threading
  - Handle results from multiple workers
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How would you structure a benchmark that uses both threads and processes?
# Q2: What synchronization primitives might you need (Lock, Event, etc.)?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import multiprocessing as mp
import threading
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from queue import Queue

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config: Tuple[int, int, int]
    time_ms: float
    worker_id: int

def run_single_benchmark(config: Tuple[int, int, int], worker_id: int) -> BenchmarkResult:
    """Simulate running a single benchmark."""
    M, N, K = config
    # Simulate compute time proportional to problem size
    compute_time = (M * N * K) / 1e9
    time.sleep(compute_time)
    return BenchmarkResult(config, compute_time * 1000, worker_id)

# TODO [MEDIUM]: Build a parallel benchmark runner using Pool.
#              Distribute configs across workers, collect results.
# HINT: Use pool.starmap or pool.map with enumerate for worker_id

def run_parallel_benchmarks(configs: List[Tuple[int, int, int]], num_workers: int = 4) -> List[BenchmarkResult]:
    """Run benchmarks in parallel across multiple processes."""
    results = []
    
    # TODO: Use mp.Pool to run benchmarks in parallel
    # HINT: Create args list with (config, worker_id) pairs
    # Use pool.starmap(run_single_benchmark, args_list)
    pass

# TODO [EASY]: Add a logging thread that writes results as they arrive.
#              Use a Queue to communicate between workers and logger.
# HINT: Logger thread runs while queue is not empty or running flag is set

def logger_thread(result_queue: Queue, stop_event: threading.Event):
    """Background thread that logs results as they arrive."""
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=0.1)
            # TODO: Log the result (print or write to file)
            print(f"  Logged: config={result.config}, time={result.time_ms:.2f}ms")
        except:
            continue

# TODO [MEDIUM]: Combine parallel execution with logging thread.
#              This is the full benchmark harness pattern.

def run_benchmark_with_logging(configs: List[Tuple[int, int, int]]) -> List[BenchmarkResult]:
    """Run benchmarks with concurrent logging."""
    result_queue = Queue()
    stop_event = threading.Event()
    
    # TODO: Start logger thread
    # HINT: threading.Thread(target=logger_thread, args=(...))
    
    # TODO: Run benchmarks (could use Pool or ThreadPool)
    # For simplicity, simulate workers putting to queue
    
    # TODO: Signal logger to stop and wait for it
    # HINT: stop_event.set(); logger_thread.join()
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How much speedup did parallel benchmarking provide?
# C2: What are the challenges of shared state in parallel programs?

if __name__ == "__main__":
    print("Running parallel benchmarks...")
    configs = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (256, 512, 1024),
    ]
    
    results = run_parallel_benchmarks(configs, num_workers=2)
    print(f"\nResults from {len(results)} benchmarks:")
    for r in results:
        print(f"  {r.config}: {r.time_ms:.2f}ms (worker {r.worker_id})")

    print("\nRunning with logging...")
    results_logged = run_benchmark_with_logging(configs[:2])
    print(f"Logged {len(results_logged)} results")

    print("\nDone!")
