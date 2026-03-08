"""
Module 14 — Threading & Multiprocessing
Exercise 01 — Threading Basics

WHAT YOU'RE BUILDING:
  Threading handles I/O-bound tasks (downloading datasets, reading files).
  For kernel benchmarking, you might fetch data while GPU computes.
  Python threads share memory but are limited by the GIL for CPU work.

OBJECTIVE:
  - Create threads with threading.Thread
  - Use Queue for thread-safe communication
  - Understand when threading helps (I/O) vs hurts (CPU)
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the GIL and how does it affect threading?
# Q2: When is threading better than sequential execution?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import threading
import time
from typing import List
from queue import Queue

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Create a thread that simulates downloading a dataset.
#              Use threading.Thread(target=function, args=(...))
# HINT: thread.start() to begin, thread.join() to wait

def download_dataset(name: str, duration: float):
    """Simulate downloading a dataset."""
    print(f"  Starting download: {name}")
    time.sleep(duration)  # Simulate I/O
    print(f"  Finished download: {name}")

def test_threading():
    """Test basic threading with parallel downloads."""
    # TODO: Create and start 3 threads for parallel downloads
    # HINT: threads = [Thread(target=download_dataset, args=(...)) for ...]
    pass

# TODO [MEDIUM]: Use a Queue for thread-safe result collection.
#              Workers put results, main thread gets them.
# HINT: queue.put(result), queue.get()

def worker_with_queue(worker_id: int, task_queue: Queue, result_queue: Queue):
    """Worker that gets tasks from queue and puts results."""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
        # Simulate work
        result = task * 2
        result_queue.put((worker_id, task, result))
        task_queue.task_done()

def test_queue_communication():
    """Test thread-safe queue communication."""
    task_queue = Queue()
    result_queue = Queue()
    
    # TODO: Add tasks to queue
    # HINT: task_queue.put(item)
    for i in range(5):
        pass  # TODO: add tasks
    
    # TODO: Start workers
    # HINT: threading.Thread(target=worker_with_queue, args=(...))
    workers = []
    
    # TODO: Wait for tasks to complete
    # HINT: task_queue.join()
    
    # TODO: Send poison pills to stop workers
    # HINT: task_queue.put(None) for each worker
    
    # TODO: Collect results
    # HINT: while not result_queue.empty(): result_queue.get()
    results = []
    
    print(f"  Results: {results}")

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Did parallel downloads finish faster than sequential? Why?
# C2: Why doesn't threading speed up CPU-bound Python code?

if __name__ == "__main__":
    print("Testing basic threading...")
    test_threading()

    print("\nTesting queue communication...")
    test_queue_communication()

    print("\nDone!")
