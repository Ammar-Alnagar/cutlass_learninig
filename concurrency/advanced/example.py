# Advanced Concurrency Topics - Hands-on Example (Python Version)
# This example demonstrates advanced concurrency concepts including atomic-like operations

import threading
import time
import queue
import random
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import deque
import sys

# Since Python's GIL limits true parallelism for CPU-bound tasks,
# we'll focus on advanced threading patterns and async techniques

# Lock-free counter simulation using threading.Lock (since Python doesn't have true atomics)
class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            # Simulating atomic increment
            temp = self._value
            # Add a small delay to make race conditions more visible if not properly synchronized
            time.sleep(0.000001)  
            self._value = temp + 1
    
    def get(self):
        with self._lock:
            return self._value
    
    def reset(self):
        with self._lock:
            self._value = 0

# Lock-free stack simulation using threading.Lock
class LockFreeStack:
    def __init__(self):
        self._data = []
        self._lock = threading.Lock()
    
    def push(self, item):
        with self._lock:
            self._data.append(item)
    
    def pop(self):
        with self._lock:
            if self._data:
                return self._data.pop()
            return None
    
    def empty(self):
        with self._lock:
            return len(self._data) == 0

# Lock-free queue simulation using threading.Lock
class LockBasedQueue:
    def __init__(self):
        self._data = deque()
        self._lock = threading.Lock()
    
    def put(self, item):
        with self._lock:
            self._data.append(item)
    
    def get(self):
        with self._lock:
            if self._data:
                return self._data.popleft()
            return None
    
    def empty(self):
        with self._lock:
            return len(self._data) == 0

# Example demonstrating atomic operations simulation
def atomic_operations_example():
    print("\n=== Atomic Operations Example (Simulated) ===")
    
    # Python doesn't have true atomic operations, but we can simulate them
    counter = AtomicCounter()
    
    # Increment the counter multiple times
    for i in range(5):
        counter.increment()
        print(f"After increment {i+1}: {counter.get()}")

# Example demonstrating lock-free stack simulation
def lock_free_stack_example():
    print("\n=== Lock-Based Stack Example (Simulated Lock-Free) ===")
    
    stack = LockFreeStack()
    
    # Push some values
    for i in range(1, 6):
        stack.push(i * 10)
        print(f"Pushed: {i * 10}")
    
    # Pop values
    while not stack.empty():
        val = stack.pop()
        if val is not None:
            print(f"Popped: {val}")

# Performance comparison: lock-based counter vs other approaches
def performance_comparison():
    print("\n=== Performance Comparison: Threading Approaches ===")
    
    num_threads = 4
    increments_per_thread = 100000  # Reduced for Python due to GIL
    
    # Test lock-based counter
    atomic_counter = AtomicCounter()
    start_time = time.time()
    
    def incrementer():
        for _ in range(increments_per_thread):
            atomic_counter.increment()
    
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=incrementer)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    atomic_duration = end_time - start_time
    
    print(f"Lock-based counter: {atomic_counter.get()} in {atomic_duration:.3f} s")
    
    # Test with queue for comparison
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def worker():
        local_sum = 0
        while True:
            item = work_queue.get()
            if item is None:
                break
            local_sum += item
            work_queue.task_done()
        result_queue.put(local_sum)
    
    # Fill work queue
    for _ in range(num_threads):
        for _ in range(increments_per_thread):
            work_queue.put(1)
    
    # Start workers
    start_time = time.time()
    workers = []
    for i in range(num_threads):
        w = threading.Thread(target=worker)
        w.start()
        workers.append(w)
    
    # Signal workers to finish
    for _ in range(num_threads):
        work_queue.put(None)
    
    # Wait for completion
    work_queue.join()
    
    # Collect results
    total = 0
    while not result_queue.empty():
        total += result_queue.get()
    
    for w in workers:
        w.join()
    
    end_time = time.time()
    queue_duration = end_time - start_time
    
    print(f"Queue-based counter: {total} in {queue_duration:.3f} s")

# Producer-Consumer with condition variables (more advanced synchronization)
def producer_consumer_advanced():
    print("\n=== Advanced Producer-Consumer with Condition Variables ===")
    
    buffer = deque()
    max_buffer_size = 5
    buffer_lock = threading.Lock()
    buffer_not_empty = threading.Condition(buffer_lock)
    buffer_not_full = threading.Condition(buffer_lock)
    
    def producer(prod_id, num_items):
        for i in range(num_items):
            with buffer_not_full:
                while len(buffer) >= max_buffer_size:
                    print(f"Producer {prod_id} waiting for space...")
                    buffer_not_full.wait()
                
                item = f"P{prod_id}-Item{i}"
                buffer.append(item)
                print(f"Producer {prod_id} produced: {item} (Buffer size: {len(buffer)})")
                
                buffer_not_empty.notify_all()  # Notify consumers
            
            time.sleep(0.1)  # Simulate production time
    
    def consumer(cons_id):
        while True:
            with buffer_not_empty:
                while len(buffer) == 0:
                    print(f"Consumer {cons_id} waiting for items...")
                    buffer_not_empty.wait()
                
                item = buffer.popleft()
                print(f"Consumer {cons_id} consumed: {item} (Buffer size: {len(buffer)})")
                
                buffer_not_full.notify_all()  # Notify producers
                
                # Stop condition - if it's a termination signal
                if "TERM" in item:
                    break
            
            time.sleep(0.15)  # Simulate consumption time
    
    # Start producers and consumers
    producers = [threading.Thread(target=producer, args=(i, 3)) for i in range(2)]
    consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(2)]
    
    for p in producers:
        p.start()
    for c in consumers:
        c.start()
    
    # Wait for producers to finish
    for p in producers:
        p.join()
    
    # Send termination signals
    for c in consumers:
        with buffer_not_full:
            buffer.append("TERM-1")
            buffer_not_full.notify_all()
    
    # Wait for consumers to finish
    for c in consumers:
        c.join()
    
    print("Advanced Producer-Consumer completed.")

# Async example demonstrating advanced patterns
async def async_advanced_patterns():
    print("\n=== Advanced Async Patterns ===")
    
    # Async generator with async context manager
    async def async_number_generator(n):
        for i in range(n):
            await asyncio.sleep(0.1)  # Simulate async work
            yield i * i
    
    # Async context manager
    class AsyncTimer:
        def __init__(self, name):
            self.name = name
            self.start_time = None
        
        async def __aenter__(self):
            self.start_time = asyncio.get_event_loop().time()
            print(f"Starting {self.name}")
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            end_time = asyncio.get_event_loop().time()
            print(f"{self.name} took {end_time - self.start_time:.3f}s")
    
    # Use async generator and context manager together
    async with AsyncTimer("Number Processing"):
        async for num in async_number_generator(5):
            print(f"Generated: {num}")
    
    # Task grouping and cancellation
    async def cancellable_task(task_id, duration):
        try:
            print(f"Task {task_id} starting...")
            await asyncio.sleep(duration)
            print(f"Task {task_id} completed")
            return f"Result from task {task_id}"
        except asyncio.CancelledError:
            print(f"Task {task_id} was cancelled")
            raise
    
    # Create and manage tasks
    tasks = [
        asyncio.create_task(cancellable_task(1, 2)),
        asyncio.create_task(cancellable_task(2, 3)),  # This will be cancelled
        asyncio.create_task(cancellable_task(3, 1))
    ]
    
    # Cancel one task after a delay
    await asyncio.sleep(1.5)
    tasks[1].cancel()
    
    # Wait for all tasks to complete (or be cancelled)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, asyncio.CancelledError):
            print(f"Task {i+1} was cancelled")
        else:
            print(f"Task {i+1} result: {result}")

# Barrier synchronization example
def barrier_example():
    print("\n=== Barrier Synchronization Example ===")
    
    def worker(barrier, worker_id):
        print(f"Worker {worker_id} starting...")
        time.sleep(random.uniform(0.5, 2.0))  # Random work time
        print(f"Worker {worker_id} reached barrier")
        
        # Wait at barrier
        barrier.wait()
        
        print(f"Worker {worker_id} passed barrier")
    
    # Create a barrier for 3 workers
    barrier = threading.Barrier(3)
    
    # Start worker threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(barrier, i))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print("All workers passed the barrier")

# Run all examples
def main():
    print("Advanced Concurrency Topics - Hands-on Example")
    
    atomic_operations_example()
    lock_free_stack_example()
    performance_comparison()
    producer_consumer_advanced()
    
    # Run async example
    asyncio.run(async_advanced_patterns())
    
    barrier_example()
    
    print("\nAll advanced examples completed!")

if __name__ == "__main__":
    main()