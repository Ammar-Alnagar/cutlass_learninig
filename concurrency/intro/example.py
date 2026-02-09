# Introduction to Concurrency - Hands-on Example (Python Version)
# This example demonstrates the difference between sequential and concurrent execution

import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

def simulate_work(task_id, duration):
    """Function that simulates work by sleeping for a duration"""
    print(f"Task {task_id} starting...")
    time.sleep(duration)
    print(f"Task {task_id} completed.")

def sequential_execution():
    """Execute tasks sequentially"""
    print("=== Sequential Execution ===")
    start_time = time.time()
    
    # Sequential execution - tasks run one after another
    simulate_work(1, 1)  # 1 second task
    simulate_work(2, 1)  # 1 second task
    simulate_work(3, 1)  # 1 second task
    
    end_time = time.time()
    sequential_duration = end_time - start_time
    print(f"Sequential execution took: {sequential_duration:.2f} seconds")
    return sequential_duration

def concurrent_execution_threads():
    """Execute tasks concurrently using threads"""
    print("\n=== Concurrent Execution (Threading) ===")
    start_time = time.time()
    
    # Concurrent execution - tasks run in parallel using threads
    threads = []
    for i in range(1, 4):
        thread = threading.Thread(target=simulate_work, args=(i, 1))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    concurrent_duration = end_time - start_time
    print(f"Concurrent execution (Threading) took: {concurrent_duration:.2f} seconds")
    return concurrent_duration

def concurrent_execution_pool():
    """Execute tasks concurrently using thread pool"""
    print("\n=== Concurrent Execution (ThreadPool) ===")
    start_time = time.time()
    
    # Concurrent execution using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(simulate_work, i, 1) for i in range(1, 4)]
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    end_time = time.time()
    concurrent_duration = end_time - start_time
    print(f"Concurrent execution (ThreadPool) took: {concurrent_duration:.2f} seconds")
    return concurrent_duration

async def async_simulate_work(task_id, duration):
    """Async version of simulate_work"""
    print(f"Async Task {task_id} starting...")
    await asyncio.sleep(duration)
    print(f"Async Task {task_id} completed.")

async def concurrent_execution_async():
    """Execute tasks concurrently using async/await"""
    print("\n=== Concurrent Execution (Async/Await) ===")
    start_time = time.time()
    
    # Concurrent execution using asyncio
    tasks = [async_simulate_work(i, 1) for i in range(1, 4)]
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    concurrent_duration = end_time - start_time
    print(f"Concurrent execution (Async/Await) took: {concurrent_duration:.2f} seconds")
    return concurrent_duration

if __name__ == "__main__":
    # Run all execution methods
    seq_time = sequential_execution()
    thr_time = concurrent_execution_threads()
    pool_time = concurrent_execution_pool()
    
    # Run async version
    async def run_async():
        return await concurrent_execution_async()
    
    async_time = asyncio.run(run_async())
    
    print(f"\nSpeedup comparison:")
    print(f"Threading speedup: {seq_time/thr_time:.2f}x")
    print(f"ThreadPool speedup: {seq_time/pool_time:.2f}x")
    print(f"Async speedup: {seq_time/async_time:.2f}x")