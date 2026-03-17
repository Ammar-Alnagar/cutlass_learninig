"""
Module 05 - Exercise 02: Multiprocessing

Scenario: You're building a data preprocessing pipeline that needs to
parallelize CPU-bound transformations across multiple cores. Python's
GIL prevents threading from helping with CPU-bound work, so you need
the multiprocessing module.

Topics covered:
- Process vs Pool vs ProcessPoolExecutor
- Queue and Pipe for inter-process communication
- Event, Lock, Semaphore for synchronization
- Worker lifecycle management

Note: All multiprocessing code should be in `if __name__ == "__main__":`
blocks on Windows to avoid infinite process spawning.
"""

from multiprocessing import Process, Pool, Queue, Pipe, Event, Lock, Value, Array
from concurrent.futures import ProcessPoolExecutor
import time
import os


# =============================================================================
# Part 1: Basic Process Creation
# =============================================================================

def worker_function(n):
    """
    A simple worker function for process execution.
    
    Args:
        n: Number to square
        
    Returns:
        int: Square of n
    """
    # TODO: Return the square of n
    pass


def run_single_process(target_func, args):
    """
    Run a function in a separate process.
    
    Use case: Isolating crashes, running CPU-bound tasks in parallel.
    
    Args:
        target_func: Function to run
        args: Tuple of arguments
        
    Returns:
        None: Note that return values are NOT captured with Process
    """
    # TODO: Create a Process with target=target_func and args=args
    # Start the process and join it
    proc = None  # Create Process
    proc.start()
    proc.join()


def run_multiple_processes(func, arg_list):
    """
    Run the same function with different arguments in parallel processes.
    
    Args:
        func: Function to run in each process
        arg_list: List of argument tuples
        
    Returns:
        list: Results from each process (requires Queue for collection)
    """
    queue = Queue()
    processes = []
    
    # TODO: Create and start processes
    
    # TODO: Wait for all processes to complete
    
    # TODO: Collect results from queue
    results = []
    
    return results


# =============================================================================
# Part 2: Process Pool
# =============================================================================

def pool_worker(x):
    """
    Worker function for pool execution.
    
    Args:
        x: Input value
        
    Returns:
        int: x squared plus x
    """
    # TODO: Return x*x + x
    pass


def use_process_pool(func, iterable, num_workers=4):
    """
    Use a Process Pool to parallelize function execution.
    
    Pool.map() automatically distributes work and collects results.
    
    Args:
        func: Function to apply
        iterable: Input values
        num_workers: Number of worker processes
        
    Returns:
        list: Results in same order as input
    """
    with Pool(processes=num_workers) as pool:
        result = None
        return result


def use_process_pool_executor(func, iterable, num_workers=4):
    """
    Use ProcessPoolExecutor (newer API, works with context manager).
    
    This is the recommended modern approach for pool-based parallelism.
    
    Args:
        func: Function to apply
        iterable: Input values
        num_workers: Number of worker processes
        
    Returns:
        list: Results in same order as input
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        result = None
        return result


# =============================================================================
# Part 3: Inter-Process Communication
# =============================================================================

def producer_worker(queue, items):
    """
    Producer that puts items in a queue.
    
    Args:
        queue: multiprocessing.Queue
        items: Items to produce
    """
    # TODO: Put each item in the queue
    for item in items:
        pass  # Put item in queue
    
    # Signal end of production
    queue.put(None)


def consumer_worker(queue):
    """
    Consumer that gets items from a queue.
    
    Args:
        queue: multiprocessing.Queue
        
    Returns:
        list: Consumed items
    """
    results = []
    while True:
        # TODO: Get item from queue
        item = None
        if item is None:  # Sentinel value
            break
        results.append(item)
    return results


def use_pipe_communication():
    """
    Demonstrate two-way communication using Pipe.
    
    Pipe() returns two Connection objects for bidirectional communication.
    
    Returns:
        tuple: (parent_received, child_received)
    """
    # TODO: Create a Pipe (duplex=True for bidirectional)
    parent_conn, child_conn = None  # Create Pipe
    
    def child_process(conn):
        conn.close()  # Close parent's end
        # TODO: Receive from parent
        received = None
        # TODO: Send response back
        conn.send(f"Child received: {received}")
        conn.close()
    
    # TODO: Create and start child process
    
    parent_conn.send("Hello from parent")
    # TODO: Receive response from child
    
    parent_conn.close()
    
    return None, None  # Return both messages


# =============================================================================
# Part 4: Synchronization Primitives
# =============================================================================

def worker_with_lock(lock, shared_counter):
    """
    Worker that increments a shared counter with lock protection.
    
    Without the lock, increments would be lost due to race conditions.
    
    Args:
        lock: multiprocessing.Lock
        shared_counter: multiprocessing.Value (shared memory)
    """
    with lock:
        # TODO: Increment shared_counter.value by 1
        pass


def worker_with_event(event, results, worker_id):
    """
    Worker that waits for an event signal before starting.
    
    Use case: Coordinating startup of multiple workers.
    
    Args:
        event: multiprocessing.Event
        results: List to store result
        worker_id: ID of this worker
    """
    # TODO: Wait for event to be set
    event.wait()
    # TODO: Add result after event is signaled
    results.append(f"Worker {worker_id} started")


def worker_with_semaphore(semaphore, active_count):
    """
    Worker that limits concurrent access using semaphore.
    
    Use case: Limiting database connections, API rate limiting.
    
    Args:
        semaphore: multiprocessing.Semaphore with max count
        active_count: Shared counter for active workers
    """
    with semaphore:
        # TODO: Increment active_count
        # Simulate work
        time.sleep(0.1)
        # TODO: Decrement active_count
        pass


# =============================================================================
# Part 5: Shared Memory (Value and Array)
# =============================================================================

def compute_with_shared_array(shared_array, start, end):
    """
    Compute values into a shared array.
    
    Args:
        shared_array: multiprocessing.Array (shared memory)
        start: Start index for this worker
        end: End index for this worker
    """
    # TODO: Fill shared_array[i] = i*i for i in range(start, end)
    for i in range(start, end):
        pass


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 05 - Exercise 02: Self-Check")
    print("=" * 60)
    
    # Check 1: worker_function
    result = worker_function(5)
    assert result == 25, f"worker_function: expected 25, got {result}"
    print("[PASS] worker_function")
    
    # Check 2: use_process_pool
    result = use_process_pool(pool_worker, [1, 2, 3, 4, 5])
    assert result == [2, 6, 12, 20, 30], f"use_process_pool: expected [2,6,12,20,30], got {result}"
    print("[PASS] use_process_pool")
    
    # Check 3: use_process_pool_executor
    result = use_process_pool_executor(pool_worker, [1, 2, 3, 4, 5])
    assert result == [2, 6, 12, 20, 30], f"use_process_pool_executor: expected [2,6,12,20,30], got {result}"
    print("[PASS] use_process_pool_executor")
    
    # Check 4: Queue communication
    queue = Queue()
    producer_worker(queue, [1, 2, 3, 4, 5])
    print("[PASS] producer_worker (Queue)")
    
    # Check 5: Pipe communication
    parent_msg, child_msg = use_pipe_communication()
    assert parent_msg == "Child received: Hello from parent", f"use_pipe_communication: got {parent_msg}"
    print("[PASS] use_pipe_communication")
    
    # Check 6: Lock synchronization
    lock = Lock()
    counter = Value('i', 0)
    processes = []
    for _ in range(4):
        p = Process(target=worker_with_lock, args=(lock, counter))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    assert counter.value == 4, f"worker_with_lock: expected 4, got {counter.value}"
    print("[PASS] worker_with_lock")
    
    # Check 7: Shared array
    shared_array = Array('d', 100)
    p = Process(target=compute_with_shared_array, args=(shared_array, 0, 10))
    p.start()
    p.join()
    expected = [float(i*i) for i in range(10)]
    actual = list(shared_array[:10])
    assert actual == expected, f"compute_with_shared_array: expected {expected}, got {actual}"
    print("[PASS] compute_with_shared_array")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
