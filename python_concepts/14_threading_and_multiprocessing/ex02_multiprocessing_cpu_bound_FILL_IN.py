"""
Module 14 — Threading & Multiprocessing
Exercise 02 — Multiprocessing for CPU-Bound Tasks

WHAT YOU'RE BUILDING:
  Multiprocessing bypasses the GIL for CPU-bound work. For kernel
  benchmarking, you might run CPU preprocessing on multiple cores
  while GPU computes, or parallelize hyperparameter sweeps.

OBJECTIVE:
  - Use multiprocessing.Pool for parallel CPU work
  - Understand Process vs Pool
  - Know when to use multiprocessing vs threading
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How does multiprocessing bypass the GIL?
# Q2: What's the overhead of multiprocessing vs threading?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import multiprocessing as mp
import time
from typing import List, Tuple

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# CPU-bound task: compute-intensive
def compute_intensive(n: int) -> int:
    """Simulate CPU-bound computation."""
    result = 0
    for i in range(n * 1000000):
        result += i ** 2
    return result

# TODO [EASY]: Use multiprocessing.Pool to parallelize CPU work.
#              Pool.map applies function to iterable in parallel.
# HINT: with mp.Pool() as pool: results = pool.map(func, iterable)

def test_multiprocessing_pool():
    """Test Pool for parallel CPU-bound work."""
    inputs = [10, 20, 30, 40]
    
    # TODO: Run with Pool.map and compare to sequential
    # Sequential first
    start = time.time()
    sequential_results = [compute_intensive(n) for n in inputs]
    sequential_time = time.time() - start
    
    # TODO: Parallel with Pool
    # with mp.Pool() as pool:
    #     parallel_results = pool.map(compute_intensive, inputs)
    pass

# TODO [MEDIUM]: Use Process for more control than Pool.
#              Useful when workers need different functions or state.
# HINT: p = mp.Process(target=func, args=(...)); p.start(); p.join()

def worker_process(name: str, duration: float):
    """Worker process that does independent work."""
    print(f"  {name} starting...")
    time.sleep(duration)
    print(f"  {name} finished")

def test_processes():
    """Test individual Process control."""
    # TODO: Create and start 3 processes
    # HINT: processes = [Process(target=worker_process, args=(...)) for ...]
    pass

# TODO [EASY]: Compare threading vs multiprocessing for different workloads.
#              Fill in when to use each:
#              - I/O-bound (file, network): use ______
#              - CPU-bound (computation): use ______

def choose_parallelism_strategy(workload_type: str) -> str:
    """Return recommended parallelism strategy."""
    # TODO: return "threading" or "multiprocessing" based on workload
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Was Pool faster than sequential for CPU-bound work? By how much?
# C2: What's the memory overhead of multiprocessing vs threading?

if __name__ == "__main__":
    print("Testing multiprocessing Pool...")
    test_multiprocessing_pool()

    print("\nTesting individual Processes...")
    test_processes()

    print("\nStrategy guide...")
    print(f"  I/O-bound: use {choose_parallelism_strategy('io')}")
    print(f"  CPU-bound: use {choose_parallelism_strategy('cpu')}")

    print("\nDone!")
