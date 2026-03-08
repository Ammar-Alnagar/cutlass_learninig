"""
Module 13 — Iterators & Generators
Exercise 02 — yield and send()

WHAT YOU'RE BUILDING:
  Generators with yield can receive values via send(). This enables
  coroutines for data pipelines — feeding data incrementally to a
  processor. Useful for streaming kernel benchmarks or online learning.

OBJECTIVE:
  - Use yield to produce values lazily
  - Use send() to communicate with a generator
  - Build a data pipeline with generator chaining
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between return and yield?
# Q2: What does generator.send(value) do? What does it return?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Generator, Tuple, List
import time

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Write a generator function that yields benchmark results.
#              This is simpler than a class-based iterator.
# HINT: Use yield inside a for loop

def benchmark_results(shapes: List[Tuple[int, int, int]]) -> Generator[Tuple[int, int, int, float], None, None]:
    """Yield simulated benchmark results (M, N, K, time_ms)."""
    for M, N, K in shapes:
        simulated_time = (M * N * K) / 1e9  # Fake timing
        # TODO: yield (M, N, K, simulated_time)
        pass

# TODO [MEDIUM]: Write a generator that accepts values via send().
#              This is a running average calculator — it yields
#              the current average after each new value.
# HINT: value = yield current_average (first yield returns None)

def running_average() -> Generator[float, float, None]:
    """Coroutine that computes running average via send()."""
    total = 0.0
    count = 0
    
    # TODO: infinite loop yielding current_average, receiving new value
    # HINT: new_value = yield current_average
    pass

# TODO [EASY]: Chain generators to build a pipeline.
#              Generate shapes → compute benchmark → filter slow results.
# HINT: Use generator expressions or nested loops

def shape_generator() -> Generator[Tuple[int, int, int], None, None]:
    """Generate benchmark shapes."""
    for size in [256, 512, 1024, 2048]:
        yield (size, size, size)

def filter_fast_results(results: Generator[Tuple[int, int, int, float], None, None], 
                        threshold: float) -> Generator[Tuple[int, int, int, float], None, None]:
    """Filter results faster than threshold."""
    # TODO: yield only results where time < threshold
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why is a generator more memory-efficient than a list for large datasets?
# C2: What's a practical use case for send() in ML pipelines?

if __name__ == "__main__":
    print("Testing benchmark_results generator...")
    shapes = [(100, 100, 100), (200, 200, 200), (300, 300, 300)]
    for result in benchmark_results(shapes):
        print(f"  {result}")

    print("\nTesting running_average with send()...")
    avg_gen = running_average()
    next(avg_gen)  # Prime the generator (returns None)
    
    for value in [10, 20, 30, 40, 50]:
        avg = avg_gen.send(value)
        print(f"  After {value}: average = {avg}")

    print("\nTesting generator pipeline...")
    shapes_gen = shape_generator()
    results_gen = benchmark_results(list(shapes_gen))  # Note: need to re-create
    fast_results = filter_fast_results(
        benchmark_results([(s, s, s) for s in [256, 512, 1024, 2048]]),
        threshold=1.0
    )
    for result in fast_results:
        print(f"  Fast: {result}")

    print("\nDone!")
