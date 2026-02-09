# Synchronization Primitives - Hands-on Example (Python Version)
# This example demonstrates mutexes, semaphores, and condition variables

import threading
import time
import queue
import random
from collections import deque

# Shared data structure for producer-consumer example
class SharedBuffer:
    def __init__(self, max_size=5):
        self.buffer = deque()
        self.max_size = max_size
        self.lock = threading.RLock()  # Reentrant lock
        self.condition = threading.Condition(self.lock)
        self.finished = False

# Producer function using mutex and condition variable
def producer(shared_buffer, producer_id, items_to_produce):
    for i in range(items_to_produce):
        with shared_buffer.condition:
            # Wait if buffer is full
            while len(shared_buffer.buffer) >= shared_buffer.max_size and not shared_buffer.finished:
                shared_buffer.condition.wait()
                
            if shared_buffer.finished:
                break  # Exit if consumer finished
                
            # Produce item
            item = producer_id * 100 + i
            shared_buffer.buffer.append(item)
            print(f"Producer {producer_id} produced item: {item} (Buffer size: {len(shared_buffer.buffer)})")
            
            shared_buffer.condition.notify_all()  # Notify consumers
        
        # Simulate production time
        time.sleep(0.1)

# Consumer function using mutex and condition variable
def consumer(shared_buffer, consumer_id):
    while True:
        item = None
        with shared_buffer.condition:
            # Wait if buffer is empty and producers haven't finished
            while len(shared_buffer.buffer) == 0 and not shared_buffer.finished:
                shared_buffer.condition.wait()
                
            if len(shared_buffer.buffer) == 0 and shared_buffer.finished:
                break  # Exit if no more items and producers finished
                
            # Consume item
            item = shared_buffer.buffer.popleft()
            print(f"Consumer {consumer_id} consumed item: {item} (Buffer size: {len(shared_buffer.buffer)})")
            
            shared_buffer.condition.notify_all()  # Notify producers
        
        # Simulate consumption time
        time.sleep(0.15)
    
    print(f"Consumer {consumer_id} finished.")

# Example demonstrating mutex (lock) usage
def mutex_example():
    print("\n=== Mutex (Lock) Example ===")
    
    shared_counter = [0]  # Using list to make it mutable
    counter_lock = threading.Lock()
    
    def incrementer(thread_id):
        for i in range(100):
            with counter_lock:  # Acquire lock
                # Critical section - only one thread can access at a time
                temp = shared_counter[0]
                time.sleep(0.0001)  # Simulate work
                shared_counter[0] = temp + 1
        print(f"Thread {thread_id} completed increments.")
    
    threads = []
    for i in range(5):
        thread = threading.Thread(target=incrementer, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Final counter value: {shared_counter[0]} (Expected: 500)")

# Example demonstrating semaphore usage
def semaphore_example():
    print("\n=== Semaphore Example ===")
    
    # Create a semaphore with 3 available resources
    resource_semaphore = threading.Semaphore(3)
    output_lock = threading.Lock()
    
    def worker(worker_id):
        with output_lock:
            print(f"Thread {worker_id} requesting resource...")
        
        # Acquire resource (will block if none available)
        with resource_semaphore:
            with output_lock:
                print(f"Thread {worker_id} acquired resource.")
            
            # Simulate using the resource
            time.sleep(0.5)
            
            with output_lock:
                print(f"Thread {worker_id} released resource.")
    
    threads = []
    # Create more threads than available resources
    for i in range(8):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Example demonstrating condition variables
def condition_variable_example():
    print("\n=== Condition Variable Example (Producer-Consumer) ===")
    
    shared_buffer = SharedBuffer(max_size=5)
    
    # Create producer and consumer threads
    prod1 = threading.Thread(target=producer, args=(shared_buffer, 1, 5))
    prod2 = threading.Thread(target=producer, args=(shared_buffer, 2, 5))
    cons1 = threading.Thread(target=consumer, args=(shared_buffer, 1))
    cons2 = threading.Thread(target=consumer, args=(shared_buffer, 2))
    
    # Start all threads
    prod1.start()
    prod2.start()
    cons1.start()
    cons2.start()
    
    # Wait for producers to finish
    prod1.join()
    prod2.join()
    
    # Signal that production is finished
    with shared_buffer.condition:
        shared_buffer.finished = True
        shared_buffer.condition.notify_all()  # Wake up all waiting consumers
    
    # Wait for consumers to finish
    cons1.join()
    cons2.join()
    
    print("Producer-Consumer example completed.")

# Example demonstrating RLock (Reentrant Lock)
def rlock_example():
    print("\n=== RLock (Reentrant Lock) Example ===")
    
    class CountingClass:
        def __init__(self):
            self._value = 0
            self._lock = threading.RLock()
        
        def increment(self):
            with self._lock:
                self._value += 1
        
        def add(self, amount):
            with self._lock:
                for _ in range(amount):
                    self.increment()  # This would cause deadlock with regular Lock
                print(f"Value after adding {amount}: {self._value}")
    
    counting_obj = CountingClass()
    
    def worker():
        counting_obj.add(5)
    
    threads = [threading.Thread(target=worker) for _ in range(3)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    print("Synchronization Primitives - Hands-on Example")
    
    mutex_example()
    semaphore_example()
    condition_variable_example()
    rlock_example()
    
    print("\nAll synchronization examples completed!")