"""
Module 07 - Exercise 01: Timing Basics

Scenario: You're optimizing an inference pipeline and need to measure
latency accurately. Using the wrong timing function or pattern can
give misleading results.

Topics covered:
- time.perf_counter vs time.time vs time.monotonic
- Writing a reusable @timer decorator
- Context-manager-based timing
- Measuring async functions
"""

import time
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Callable, Any
import asyncio


# =============================================================================
# Part 1: Timing Functions
# =============================================================================

def measure_with_time():
    """Demonstrate time.time() for wall-clock timestamps."""
    # TODO: Get current timestamp using time.time()
    now = None
    
    # TODO: Calculate timestamp from 10 seconds ago
    ten_seconds_ago = None
    
    return now, ten_seconds_ago


def measure_duration_perf():
    """Measure a duration using time.perf_counter()."""
    # TODO: Record start time with time.perf_counter()
    start = None
    
    time.sleep(0.1)
    
    # TODO: Record end time and calculate elapsed milliseconds
    end = None
    elapsed_ms = None
    
    return elapsed_ms


def measure_duration_monotonic():
    """Measure duration using time.monotonic()."""
    # TODO: Record start with time.monotonic()
    start = None
    
    result = sum(i * i for i in range(10000))
    
    # TODO: Record end and calculate elapsed time
    end = None
    elapsed = None
    
    return elapsed, result


def compare_timing_functions():
    """Compare the three timing functions."""
    results = {}
    
    # With perf_counter
    start_perf = time.perf_counter()
    time.sleep(0.05)
    results['perf_counter_ms'] = None
    
    # With monotonic
    start_mono = time.monotonic()
    time.sleep(0.05)
    results['monotonic_ms'] = None
    
    # With time
    start_time = time.time()
    time.sleep(0.05)
    results['time_ms'] = None
    
    return results


# =============================================================================
# Part 2: Timer Decorator
# =============================================================================

def timer(func: Callable) -> Callable:
    """Decorator that times function execution and prints the result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Record start time using perf_counter()
        start = None
        
        # TODO: Call the original function
        result = None
        
        # TODO: Record end time and calculate elapsed
        end = None
        elapsed_ms = None
        
        print(f"{func.__name__} took {elapsed_ms:.2f} ms")
        
        return result
    
    return wrapper


def timer_with_logging(func: Callable) -> Callable:
    """Enhanced timer decorator with configurable logging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # TODO: Calculate elapsed time in milliseconds
            elapsed_ms = None
            
            # TODO: Print log message with function name and elapsed time
            print(f"[TIMED] {func.__name__} completed in {elapsed_ms:.2f} ms")
    
    return wrapper


# =============================================================================
# Part 3: Timer Context Manager
# =============================================================================

@contextmanager
def timed_block(label: str = "Block"):
    """Context manager for timing a code block."""
    # TODO: Record start time
    start = None
    
    try:
        yield
    finally:
        # TODO: Record end time, calculate elapsed, and print
        end = None
        elapsed_ms = None
        print(f"[TIMED] {label}: {elapsed_ms:.2f} ms")


class Timer:
    """Reusable timer class for complex timing scenarios."""
    
    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._elapsed_ms: Optional[float] = None
    
    def start(self):
        """Start the timer."""
        # TODO: Record start time using perf_counter()
        self._start = None
    
    def stop(self):
        """Stop the timer and calculate elapsed time."""
        # TODO: Record end time and calculate elapsed milliseconds
        self._end = None
        self._elapsed_ms = None
    
    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds."""
        return self._elapsed_ms
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =============================================================================
# Part 4: Timing Async Functions
# =============================================================================

async def async_timer(func: Callable) -> Callable:
    """Decorator for timing async functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # TODO: Record start time
        start = None
        
        # TODO: Await the original function
        result = await func(*args, **kwargs)
        
        # TODO: Record end time and calculate elapsed
        end = None
        elapsed_ms = None
        
        print(f"[ASYNC] {func.__name__} took {elapsed_ms:.2f} ms")
        
        return result
    
    return wrapper


async def measure_async_operation():
    """Demonstrate timing of async operations."""
    async def async_sleep(delay: float):
        await asyncio.sleep(delay)
    
    start = time.perf_counter()
    
    # TODO: Run three async sleeps sequentially (0.05s each)
    
    # TODO: Calculate total elapsed time
    elapsed = None
    
    return elapsed


async def measure_async_parallel():
    """Demonstrate timing of parallel async operations."""
    async def async_sleep(delay: float):
        await asyncio.sleep(delay)
    
    start = time.perf_counter()
    
    # TODO: Run three async sleeps in parallel using asyncio.gather
    
    # TODO: Calculate elapsed time
    elapsed = None
    
    return elapsed


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 07 - Exercise 01: Self-Check")
    print("=" * 60)
    
    # Check 1: time.time for timestamps
    now, past = measure_with_time()
    assert now > past
    print("[PASS] measure_with_time")
    
    # Check 2: perf_counter for duration
    elapsed = measure_duration_perf()
    assert 90 <= elapsed <= 120, f"Expected ~100ms, got {elapsed:.2f}ms"
    print(f"[PASS] measure_duration_perf ({elapsed:.2f}ms)")
    
    # Check 3: monotonic duration
    elapsed, result = measure_duration_monotonic()
    assert elapsed > 0
    print(f"[PASS] measure_duration_monotonic ({elapsed:.4f}s)")
    
    # Check 4: Compare timing functions
    results = compare_timing_functions()
    for key, value in results.items():
        assert 40 <= value <= 70, f"{key}: Expected ~50ms, got {value:.2f}ms"
    print("[PASS] compare_timing_functions")
    
    # Check 5: Timer decorator
    @timer
    def test_func():
        time.sleep(0.05)
    test_func()
    print("[PASS] timer decorator")
    
    # Check 6: Timer context manager
    print("Testing timed_block:")
    with timed_block("Test block"):
        time.sleep(0.05)
    print("[PASS] timed_block")
    
    # Check 7: Timer class
    with Timer() as t:
        time.sleep(0.05)
    assert t.elapsed_ms is not None
    print(f"[PASS] Timer class ({t.elapsed_ms:.2f}ms)")
    
    # Check 8: Async timing
    async def run_async_checks():
        elapsed_seq = await measure_async_operation()
        assert elapsed_seq >= 0.14, f"Sequential async should take >140ms"
        print(f"[PASS] measure_async_operation ({elapsed_seq:.3f}s)")
        
        elapsed_par = await measure_async_parallel()
        assert elapsed_par < 0.1, f"Parallel async should take <100ms"
        print(f"[PASS] measure_async_parallel ({elapsed_par:.3f}s)")
    
    asyncio.run(run_async_checks())
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
