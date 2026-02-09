# Real-World Case Studies - Hands-on Example (Python Version)
# This example implements simplified versions of real-world concurrent systems

import threading
import time
import queue
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from dataclasses import dataclass

# Case Study 1: Simplified Web Server with Thread Pool
class ThreadPool:
    def __init__(self, num_threads: int):
        self.task_queue = queue.Queue()
        self.threads = []
        self.shutdown = False
        
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)
    
    def worker(self):
        while not self.shutdown:
            try:
                task = self.task_queue.get(timeout=1)
                task()
                self.task_queue.task_done()
            except queue.Empty:
                continue
    
    def submit(self, func, *args, **kwargs):
        def task():
            func(*args, **kwargs)
        self.task_queue.put(task)
    
    def shutdown_pool(self):
        self.shutdown = True
        for t in self.threads:
            t.join()

# Simplified HTTP request handler
def handle_request(request_id: int):
    print(f"Processing request {request_id} on thread {threading.current_thread().ident}")
    
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))
    
    print(f"Completed request {request_id}")

# Case Study 2: Database Connection Pool
class DatabaseConnection:
    def __init__(self, conn_id: int):
        self.conn_id = conn_id
        self.in_use = False
        self.last_used = time.time()
    
    def execute_query(self, query: str):
        print(f"Connection {self.conn_id} executing: {query}")
        time.sleep(0.1)  # Simulate query execution
        print(f"Connection {self.conn_id} completed query")
        self.last_used = time.time()

class ConnectionPool:
    def __init__(self, max_connections: int):
        self.max_connections = max_connections
        self.connections = [DatabaseConnection(i) for i in range(max_connections)]
        self.available_connections = queue.Queue()
        self.pool_lock = threading.Lock()
        self.condition = threading.Condition(self.pool_lock)
        self.active_count = 0
        
        # Initially all connections are available
        for conn in self.connections:
            self.available_connections.put(conn)
    
    def get_connection(self, timeout: float = 5.0):
        with self.condition:
            start_time = time.time()
            while self.available_connections.empty():
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None  # Timeout
                remaining = timeout - elapsed
                self.condition.wait(timeout=min(remaining, 0.1))
            
            conn = self.available_connections.get()
            conn.in_use = True
            self.active_count += 1
            return conn
    
    def return_connection(self, conn: DatabaseConnection):
        with self.condition:
            conn.in_use = False
            self.available_connections.put(conn)
            self.active_count -= 1
            self.condition.notify()
    
    def print_stats(self):
        with self.pool_lock:
            available = self.available_connections.qsize()
            print(f"Pool stats: Available: {available}, Active: {self.active_count}, Max: {self.max_connections}")

# Case Study 3: Concurrent Cache with Lock Striping
import hashlib

class ConcurrentCache:
    def __init__(self, num_buckets: int = 16):
        self.buckets = [{'data': {}, 'lock': threading.RLock()} for _ in range(num_buckets)]
        self.num_buckets = num_buckets
    
    def _get_bucket_index(self, key):
        return hash(key) % self.num_buckets
    
    def put(self, key, value):
        bucket_idx = self._get_bucket_index(key)
        with self.buckets[bucket_idx]['lock']:
            self.buckets[bucket_idx]['data'][key] = value
    
    def get(self, key):
        bucket_idx = self._get_bucket_index(key)
        with self.buckets[bucket_idx]['lock']:
            return self.buckets[bucket_idx]['data'].get(key)
    
    def remove(self, key):
        bucket_idx = self._get_bucket_index(key)
        with self.buckets[bucket_idx]['lock']:
            return self.buckets[bucket_idx]['data'].pop(key, None) is not None
    
    def size(self):
        total_size = 0
        for bucket in self.buckets:
            with bucket['lock']:
                total_size += len(bucket['data'])
        return total_size

# Case Study 4: Parallel Data Processor
class ParallelDataProcessor:
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
    
    def process_in_parallel(self, data_list, func):
        futures = [self.executor.submit(func, item) for item in data_list]
        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions

# Case Study 5: Producer-Consumer with Bounded Buffer
class BoundedBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
    
    def put(self, item):
        with self.not_full:
            while len(self.buffer) >= self.capacity:
                self.not_full.wait()
            
            self.buffer.append(item)
            self.not_empty.notify()
    
    def get(self):
        with self.not_empty:
            while len(self.buffer) == 0:
                self.not_empty.wait()
            
            item = self.buffer.pop(0)
            self.not_full.notify()
            return item
    
    def put_with_timeout(self, item, timeout: float):
        with self.not_full:
            start_time = time.time()
            while len(self.buffer) >= self.capacity:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                remaining = timeout - elapsed
                self.not_full.wait(timeout=min(remaining, 0.1))
            
            self.buffer.append(item)
            self.not_empty.notify()
            return True
    
    def get_with_timeout(self, timeout: float):
        with self.not_empty:
            start_time = time.time()
            while len(self.buffer) == 0:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
                remaining = timeout - elapsed
                self.not_empty.wait(timeout=min(remaining, 0.1))
            
            item = self.buffer.pop(0)
            self.not_full.notify()
            return item

# Demonstration of Case Study 1: Web Server
def web_server_demo():
    print("\n=== Case Study 1: Web Server with Thread Pool ===")
    
    pool = ThreadPool(4)  # 4 worker threads
    
    # Simulate incoming requests
    for i in range(1, 11):
        pool.submit(handle_request, i)
    
    # Give time for all requests to process
    time.sleep(2)
    pool.shutdown_pool()
    print("Web server demo completed")

# Demonstration of Case Study 2: Connection Pool
def connection_pool_demo():
    print("\n=== Case Study 2: Database Connection Pool ===")
    
    pool = ConnectionPool(3)  # Pool with 3 connections
    
    def client_worker(client_id):
        conn = pool.get_connection()
        if conn:
            print(f"Client {client_id} got connection {conn.conn_id}")
            
            # Use the connection
            conn.execute_query(f"SELECT * FROM users WHERE id = {client_id}")
            
            # Return the connection to the pool
            print(f"Client {client_id} returning connection")
            pool.return_connection(conn)
        else:
            print(f"Client {client_id} could not get connection")
    
    # Simulate multiple clients requesting connections
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(client_worker, i) for i in range(1, 6)]
        
        # Print stats periodically
        for _ in range(3):
            time.sleep(0.3)
            pool.print_stats()
        
        # Wait for all clients to finish
        for future in as_completed(futures):
            future.result()
    
    pool.print_stats()
    print("Connection pool demo completed")

# Demonstration of Case Study 3: Concurrent Cache
def concurrent_cache_demo():
    print("\n=== Case Study 3: Concurrent Cache ===")
    
    cache = ConcurrentCache(8)  # 8 buckets
    
    def populate_cache(worker_id):
        for j in range(10):
            key = worker_id * 10 + j
            cache.put(key, f"Value_{key}")
            time.sleep(0.01)
    
    def query_cache(worker_id):
        for j in range(20):
            value = cache.get(j)
            if value:
                print(f"Found: {j} -> {value}")
            time.sleep(0.015)
    
    # Populate cache from multiple threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        populate_futures = [executor.submit(populate_cache, i) for i in range(4)]
        
        # Query cache from multiple threads
        query_futures = [executor.submit(query_cache, i) for i in range(2)]
        
        # Wait for all threads to complete
        for future in as_completed(populate_futures + query_futures):
            future.result()
    
    print(f"Cache size: {cache.size()}")
    print("Concurrent cache demo completed")

# Demonstration of Case Study 4: Parallel Data Processor
def parallel_processor_demo():
    print("\n=== Case Study 4: Parallel Data Processor ===")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        processor = ParallelDataProcessor(executor)
        
        # Create a list of numbers to process
        data = list(range(1, 21))  # [1, 2, 3, ..., 20]
        
        print(f"Original data: {data}")
        
        # Process data in parallel (square each number)
        def process_item(n):
            # Simulate some work
            time.sleep(0.05)
            squared = n * n
            print(f"Processed: {n} -> {squared} on thread {threading.current_thread().ident}")
            return squared
        
        # Update data in place
        results = []
        processor.process_in_parallel(data, lambda x: results.append(process_item(x)))
        
        # Wait a bit for all processing to complete
        time.sleep(1)
        
        print(f"Processed data: {sorted([x for x in results if x is not None])}")
        print("Parallel processor demo completed")

# Demonstration of Case Study 5: Producer-Consumer
def producer_consumer_demo():
    print("\n=== Case Study 5: Producer-Consumer with Bounded Buffer ===")
    
    buffer = BoundedBuffer(5)  # Buffer with capacity 5
    
    def producer():
        for i in range(1, 11):
            print(f"Producer: putting {i}")
            buffer.put(i)
            time.sleep(0.1)
        print("Producer: finished")
    
    def consumer():
        for i in range(10):
            value = buffer.get()
            print(f"Consumer: got {value}")
            time.sleep(0.15)
        print("Consumer: finished")
    
    # Run producer and consumer concurrently
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    
    producer_thread.start()
    consumer_thread.start()
    
    producer_thread.join()
    consumer_thread.join()
    
    print("Producer-consumer demo completed")

# Case Study 6: Async Web Scraper (demonstrating real-world async usage)
async def async_web_scraper_demo():
    print("\n=== Case Study 6: Async Web Scraper ===")
    
    # This is a simulation since we can't actually make HTTP requests without external dependencies
    # In a real implementation, we would use aiohttp or similar
    
    async def fetch_url(url, delay):
        print(f"Fetching {url}...")
        # Simulate network delay
        await asyncio.sleep(delay)
        result = f"Content from {url}"
        print(f"Completed {url}")
        return result
    
    urls = [
        ("http://example.com/page1", 0.5),
        ("http://example.com/page2", 0.3),
        ("http://example.com/page3", 0.7),
        ("http://example.com/page4", 0.2),
        ("http://example.com/page5", 0.4)
    ]
    
    start_time = time.time()
    
    # Execute all requests concurrently
    tasks = [fetch_url(url, delay) for url, delay in urls]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"Scraped {len(results)} pages in {end_time - start_time:.2f} seconds")
    print("Async web scraper demo completed")

if __name__ == "__main__":
    print("Real-World Case Studies - Hands-on Example (Python)")
    
    web_server_demo()
    connection_pool_demo()
    concurrent_cache_demo()
    parallel_processor_demo()
    producer_consumer_demo()
    
    # Run async example
    asyncio.run(async_web_scraper_demo())
    
    print("\nAll real-world case study examples completed!")