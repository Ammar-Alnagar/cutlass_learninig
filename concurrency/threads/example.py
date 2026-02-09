# Thread Basics - Hands-on Example (Python Version)
# This example demonstrates thread creation, lifecycle, and basic operations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import random

def basic_thread_example():
    """Demonstrate basic thread creation and joining"""
    print("\n=== Basic Thread Creation ===")
    
    def worker_function():
        thread_id = threading.current_thread().ident
        print(f"Hello from thread {thread_id}")
        time.sleep(0.5)
        print(f"Thread {thread_id} finishing...")
    
    # Create and start a thread
    main_thread_id = threading.current_thread().ident
    print(f"Main thread ID: {main_thread_id}")
    
    thread = threading.Thread(target=worker_function)
    thread.start()
    print(f"Created thread ID: {thread.ident}")
    
    # Wait for the thread to complete
    thread.join()
    print("Thread joined successfully!")

def multiple_threads_example():
    """Demonstrate multiple threads"""
    print("\n=== Multiple Threads ===")
    
    num_threads = 5
    threads = []
    
    def worker_function(thread_num):
        thread_id = threading.current_thread().ident
        print(f"Thread {thread_num} (ID: {thread_id}) starting...")
        
        # Simulate some work
        time.sleep(0.1 * (thread_num + 1))
        
        print(f"Thread {thread_num} (ID: {thread_id}) completed.")
    
    # Create multiple threads
    for i in range(num_threads):
        thread = threading.Thread(target=worker_function, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All threads joined!")

def thread_arguments_example():
    """Demonstrate thread arguments"""
    print("\n=== Thread Arguments ===")
    
    def worker_function(msg, val):
        thread_id = threading.current_thread().ident
        print(f"Received message: {msg}")
        print(f"Received value: {val}")
        print(f"Processing in thread: {thread_id}")
    
    message = "Hello from thread!"
    value = 42
    
    thread = threading.Thread(target=worker_function, args=(message, value))
    thread.start()
    thread.join()
    
    print("Thread with arguments completed!")

def thread_pool_example():
    """Demonstrate using thread pools"""
    print("\n=== Thread Pool ===")
    
    def worker_function(task_id):
        thread_id = threading.current_thread().ident
        print(f"Task {task_id} running on thread {thread_id}")
        
        # Simulate variable work time
        time.sleep(random.uniform(0.1, 0.5))
        
        result = f"Result from task {task_id}"
        print(f"Task {task_id} completed on thread {thread_id}")
        return result
    
    # Using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit multiple tasks
        futures = [executor.submit(worker_function, i) for i in range(5)]
        
        # Collect results
        results = [future.result() for future in futures]
        
    print(f"All tasks completed. Results: {results}")

def daemon_thread_example():
    """Demonstrate daemon threads"""
    print("\n=== Daemon Threads ===")
    
    def daemon_worker():
        thread_id = threading.current_thread().ident
        print(f"Daemon thread {thread_id} starting...")
        
        for i in range(5):
            print(f"Daemon thread {thread_id} working... ({i+1}/5)")
            time.sleep(0.5)
        
        print(f"Daemon thread {thread_id} finishing...")
    
    # Create a daemon thread
    daemon = threading.Thread(target=daemon_worker)
    daemon.daemon = True  # Set as daemon thread
    daemon.start()
    
    print("Main thread continuing...")
    time.sleep(2)  # Main thread sleeps for 2 seconds
    print("Main thread ending...")
    # Daemon thread will be killed when main thread ends

if __name__ == "__main__":
    print("Thread Basics - Hands-on Example")
    
    basic_thread_example()
    multiple_threads_example()
    thread_arguments_example()
    thread_pool_example()
    daemon_thread_example()
    
    print("\nAll thread examples completed!")