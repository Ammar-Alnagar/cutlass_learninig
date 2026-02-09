# Testing Concurrency - Hands-on Example (Python Version)
# This example demonstrates various techniques for testing concurrent code

import threading
import time
import random
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import multiprocessing
import sys
from typing import List, Tuple
import statistics

# A simple thread-safe counter for testing
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            temp = self._value
            # Add a tiny delay to increase chance of race conditions if not properly synchronized
            time.sleep(0.000001)
            self._value = temp + 1
    
    def get(self):
        with self._lock:
            return self._value
    
    def reset(self):
        with self._lock:
            self._value = 0

# A buggy counter to demonstrate race conditions
class BuggyCounter:
    def __init__(self):
        self._value = 0
        # No lock - deliberately not thread-safe
    
    def increment(self):
        # Deliberately not thread-safe to demonstrate race conditions
        temp = self._value
        time.sleep(0.000001)  # Increase chance of race condition
        self._value = temp + 1
    
    def get(self):
        return self._value
    
    def reset(self):
        self._value = 0

# Test 1: Basic stress test for thread safety
def test_thread_safety():
    print("\n=== Test 1: Thread Safety Stress Test ===")
    
    num_threads = 4
    increments_per_thread = 100000
    expected_total = num_threads * increments_per_thread
    
    counter = ThreadSafeCounter()
    
    start_time = time.time()
    
    def worker():
        for _ in range(increments_per_thread):
            counter.increment()
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    
    result = counter.get()
    print(f"Expected: {expected_total}, Got: {result}")
    assert result == expected_total, f"Expected {expected_total}, got {result}"
    print(f"Thread safety test PASSED! Time: {end_time - start_time:.3f}s")

# Test 2: Demonstrate race condition with buggy counter
def test_race_condition():
    print("\n=== Test 2: Race Condition Demonstration ===")
    
    num_threads = 4
    increments_per_thread = 10000
    expected_total = num_threads * increments_per_thread
    
    counter = BuggyCounter()
    
    start_time = time.time()
    
    def worker():
        for _ in range(increments_per_thread):
            counter.increment()
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    
    result = counter.get()
    print(f"Expected: {expected_total}, Got: {result}")
    
    if result != expected_total:
        print("Race condition detected! Result differs from expected.")
    else:
        print("No race condition detected this time (but it's still there!).")
    
    print(f"Race condition test completed in: {end_time - start_time:.3f}s")

# Test 3: Test with different thread counts to find scaling issues
def test_scalability():
    print("\n=== Test 3: Scalability Test ===")
    
    increments_per_thread = 50000
    thread_counts = [1, 2, 4, 8]
    
    results = []
    
    for num_threads in thread_counts:
        counter = ThreadSafeCounter()
        expected_total = num_threads * increments_per_thread
        
        start_time = time.time()
        
        def worker():
            for _ in range(increments_per_thread):
                counter.increment()
        
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
        result = counter.get()
        print(f"Threads: {num_threads}, Expected: {expected_total}, "
              f"Got: {result}, Duration: {duration:.3f}s")
        
        assert result == expected_total
        results.append((num_threads, duration))
    
    # Print scalability analysis
    if len(results) > 1:
        speedup = results[0][1] / results[-1][1] if results[-1][1] > 0 else float('inf')
        print(f"Speedup from 1 to {results[-1][0]} threads: {speedup:.2f}x")

# Test 4: Test with randomized delays to expose timing issues
def test_with_random_delays():
    print("\n=== Test 4: Random Delay Test ===")
    
    num_threads = 4
    operations_per_thread = 10000
    expected_total = num_threads * operations_per_thread
    
    counter = ThreadSafeCounter()
    
    def worker():
        for _ in range(operations_per_thread):
            counter.increment()
            # Add random delay to increase chance of exposing race conditions
            time.sleep(random.uniform(0.0000001, 0.000001))
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    result = counter.get()
    print(f"Expected: {expected_total}, Got: {result}")
    assert result == expected_total
    print("Random delay test PASSED!")

# Test 5: Test for deadlocks using condition variables
def test_deadlock():
    print("\n=== Test 5: Deadlock Test ===")
    
    # Create two resources with locks
    resource1 = threading.Lock()
    resource2 = threading.Lock()
    
    resource1_value = [100]  # Using list to make it mutable
    resource2_value = [100]
    
    def transfer_1_to_2(amount, delay=False):
        with resource1:
            if delay:
                time.sleep(0.01)  # Increase chance of deadlock
            with resource2:
                resource1_value[0] -= amount
                resource2_value[0] += amount
    
    def transfer_2_to_1(amount, delay=False):
        with resource2:  # Different order than transfer_1_to_2!
            if delay:
                time.sleep(0.01)  # Increase chance of deadlock
            with resource1:  # Different order than transfer_1_to_2!
                resource2_value[0] -= amount
                resource1_value[0] += amount
    
    # Attempt to create a potential deadlock situation
    print("Attempting to trigger deadlock...")
    
    # Use a timeout to prevent hanging indefinitely
    def run_with_timeout(func, timeout=5):
        result_container = [None]
        exception_container = [None]
        
        def target():
            try:
                func()
                result_container[0] = True
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            print("Potential deadlock detected (operation timed out)!")
            return False
        elif exception_container[0]:
            raise exception_container[0]
        else:
            return result_container[0]
    
    # Try to trigger deadlock
    success1 = run_with_timeout(lambda: transfer_1_to_2(10, delay=True))
    success2 = run_with_timeout(lambda: transfer_2_to_1(5, delay=True))
    
    if success1 and success2:
        print("Operations completed without deadlock.")
    else:
        print("Deadlock likely occurred.")
    
    print(f"Resource values: {resource1_value[0]}, {resource2_value[0]}")

# Test 6: Property-based testing concept demonstration
def test_property_based():
    print("\n=== Test 6: Property-Based Testing Concept ===")
    
    # Property: The sum of increments equals the final counter value
    num_threads = 3
    increments_per_thread = 50000
    expected_total = num_threads * increments_per_thread
    
    counter = ThreadSafeCounter()
    
    def worker():
        for _ in range(increments_per_thread):
            counter.increment()
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    final_value = counter.get()
    
    # Property check: final value equals expected total
    property_holds = (final_value == expected_total)
    print(f"Property 'final_value == expected_total' holds: {property_holds}")
    print(f"Expected: {expected_total}, Actual: {final_value}")
    
    # Additional property: value is non-negative
    non_negative = (final_value >= 0)
    print(f"Property 'final_value >= 0' holds: {non_negative}")
    
    assert property_holds and non_negative
    print("Property-based tests PASSED!")

# Test 7: Repeat test multiple times to catch intermittent failures
def test_repeat_for_flakiness():
    print("\n=== Test 7: Repeat Test for Intermittent Issues ===")
    
    num_repeats = 10
    num_threads = 4
    increments_per_thread = 25000
    expected_total = num_threads * increments_per_thread
    
    results = []
    all_passed = True
    
    for repeat in range(num_repeats):
        counter = ThreadSafeCounter()
        
        def worker():
            for _ in range(increments_per_thread):
                counter.increment()
        
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        result = counter.get()
        passed = (result == expected_total)
        
        print(f"Repeat {repeat + 1}: {'PASS' if passed else 'FAIL'} "
              f"(Expected: {expected_total}, Got: {result})")
        
        results.append(passed)
        if not passed:
            all_passed = False
    
    if all_passed:
        print("All repeats PASSED!")
    else:
        failed_count = results.count(False)
        print(f"Some repeats FAILED - {failed_count}/{num_repeats} failed - potential flakiness detected!")

# Test 8: Performance testing under load
def test_performance_under_load():
    print("\n=== Test 8: Performance Testing Under Load ===")
    
    thread_counts = [1, 2, 4, 8, 16]
    operations_per_thread = 10000
    
    results = []
    
    for num_threads in thread_counts:
        counter = ThreadSafeCounter()
        
        start_time = time.time()
        
        def worker():
            for _ in range(operations_per_thread):
                counter.increment()
        
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        end_time = time.time()
        duration = end_time - start_time
        total_ops = num_threads * operations_per_thread
        ops_per_second = total_ops / duration if duration > 0 else float('inf')
        
        results.append({
            'threads': num_threads,
            'duration': duration,
            'ops_per_second': ops_per_second
        })
        
        print(f"Threads: {num_threads}, Duration: {duration:.3f}s, "
              f"Ops/sec: {ops_per_second:.0f}")
        
        # Verify correctness
        assert counter.get() == total_ops
    
    # Analyze performance scaling
    if len(results) > 1:
        baseline = results[0]['ops_per_second']
        for result in results[1:]:
            efficiency = result['ops_per_second'] / (baseline * result['threads'])
            print(f"Efficiency with {result['threads']} threads: {efficiency:.2%}")

# Test 9: Testing with ThreadPoolExecutor
def test_with_threadpool():
    print("\n=== Test 9: Testing with ThreadPoolExecutor ===")
    
    def increment_operation(start, end, counter):
        for _ in range(start, end):
            counter.increment()
        return end - start
    
    num_threads = 4
    total_operations = 100000
    operations_per_thread = total_operations // num_threads
    expected_total = total_operations
    
    counter = ThreadSafeCounter()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * operations_per_thread
            end = start + operations_per_thread if i < num_threads - 1 else total_operations
            future = executor.submit(increment_operation, start, end, counter)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            ops_completed = future.result()
            # Each thread should complete operations_per_thread operations
            assert ops_completed == operations_per_thread
    
    end_time = time.time()
    
    result = counter.get()
    print(f"Expected: {expected_total}, Got: {result}")
    print(f"ThreadPoolExecutor test completed in: {end_time - start_time:.3f}s")
    
    assert result == expected_total
    print("ThreadPoolExecutor test PASSED!")

if __name__ == "__main__":
    print("Testing Concurrency - Hands-on Example (Python)")
    
    test_thread_safety()
    test_race_condition()
    test_scalability()
    test_with_random_delays()
    test_deadlock()
    test_property_based()
    test_repeat_for_flakiness()
    test_performance_under_load()
    test_with_threadpool()
    
    print("\nAll concurrency testing examples completed!")