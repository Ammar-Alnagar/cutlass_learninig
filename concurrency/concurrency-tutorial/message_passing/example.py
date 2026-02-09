# Message Passing - Hands-on Example (Python Version)
# This example demonstrates message passing using queues and the actor model

import threading
import queue
import time
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable
import multiprocessing as mp

# Producer-Consumer pattern using queues
def producer_consumer_example():
    print("\n=== Producer-Consumer with Queues ===")
    
    # Create a bounded queue (capacity 5)
    ch = queue.Queue(maxsize=5)
    
    def producer():
        for i in range(1, 11):
            print(f"Producer: sending {i}")
            ch.put(i)  # Blocks if queue is full
            time.sleep(0.1)  # Simulate work
        ch.put(None)  # Sentinel value to signal end
        print("Producer: finished")
    
    def consumer():
        while True:
            value = ch.get()  # Blocks if queue is empty
            if value is None:  # Check for sentinel
                ch.put(None)  # Put sentinel back for other consumers if any
                break
            print(f"Consumer: received {value}")
            time.sleep(0.15)  # Simulate processing
        print("Consumer: finished")
    
    # Start producer and consumer threads
    prod_thread = threading.Thread(target=producer)
    cons_thread = threading.Thread(target=consumer)
    
    prod_thread.start()
    cons_thread.start()
    
    prod_thread.join()
    cons_thread.join()

# Worker pool pattern using message passing
def worker_pool_example():
    print("\n=== Worker Pool with Message Passing ===")
    
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def worker(worker_id):
        while True:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                task_queue.put(None)  # Propagate shutdown signal
                break
            
            task_id = task
            print(f"Worker {worker_id} processing task {task_id}")
            
            # Simulate work
            time.sleep(0.2)
            
            # Send result
            result = task_id * task_id  # Simple computation
            result_queue.put((worker_id, result))
            print(f"Worker {worker_id} completed task {task_id}, result: {result}")
        
        print(f"Worker {worker_id} shutting down")
    
    # Create worker threads
    num_workers = 3
    workers = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        workers.append(t)
        t.start()
    
    # Send tasks to workers
    for i in range(1, 13):
        task_queue.put(i)
        time.sleep(0.05)
    
    # Send shutdown signals
    for _ in range(num_workers):
        task_queue.put(None)
    
    print("Dispatcher: all tasks sent")
    
    # Collect results
    results_received = 0
    expected_results = 12
    while results_received < expected_results:
        worker_id, result = result_queue.get()
        print(f"Collector: received result from worker {worker_id}, value: {result}")
        results_received += 1
    
    print(f"Collector: received all {results_received} results")
    
    # Wait for all workers to finish
    for w in workers:
        w.join()

# Simple Actor model implementation
class Actor:
    def __init__(self, name, mailbox_size=10):
        self.name = name
        self.mailbox = queue.Queue(maxsize=mailbox_size)
        self.running = True
        self.thread = threading.Thread(target=self.process_messages)
        self.thread.start()
    
    def send_message(self, msg):
        if self.running:
            self.mailbox.put(msg)
    
    def process_messages(self):
        while self.running:
            try:
                # Wait for a message with timeout to allow checking running flag
                msg = self.mailbox.get(timeout=0.1)
                if callable(msg):
                    msg()  # Execute the message (function)
                elif msg == "STOP":
                    self.running = False
                self.mailbox.task_done()
            except queue.Empty:
                continue  # Check running flag again
    
    def stop(self):
        self.running = False
        self.send_message("STOP")
        if self.thread.is_alive():
            self.thread.join()

# Math actor for performing calculations
class MathActor(Actor):
    def __init__(self, name, result_queue):
        super().__init__(name)
        self.result_queue = result_queue
    
    def perform_calculation(self, a, b, op):
        def calc_func():
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
            elif op == '/':
                result = a // b if b != 0 else 0
            else:
                result = 0
            
            print(f"{self.name}: {a} {op} {b} = {result}")
            self.result_queue.put(result)
        
        self.send_message(calc_func)

def actor_model_example():
    print("\n=== Actor Model Example ===")
    
    result_queue = queue.Queue()
    
    # Create math actors
    calc1 = MathActor("Calculator-1", result_queue)
    calc2 = MathActor("Calculator-2", result_queue)
    
    # Send calculation requests to actors
    calc1.perform_calculation(10, 5, '+')
    calc1.perform_calculation(20, 4, '*')
    calc2.perform_calculation(100, 25, '/')
    calc2.perform_calculation(50, 30, '-')
    
    # Collect results
    for _ in range(4):
        result = result_queue.get()
        print(f"Main: received calculation result: {result}")
    
    # Stop actors
    calc1.stop()
    calc2.stop()

# Pipeline pattern using queues
def pipeline_example():
    print("\n=== Pipeline Pattern ===")
    
    stage1_to_stage2 = queue.Queue()
    stage2_to_stage3 = queue.Queue()
    
    def generator():
        for i in range(1, 6):
            print(f"Generator: producing {i}")
            stage1_to_stage2.put(i)
            time.sleep(0.1)
        stage1_to_stage2.put(None)  # Sentinel to signal end
        print("Generator: finished")
    
    def processor():
        while True:
            value = stage1_to_stage2.get()
            if value is None:  # Check for sentinel
                stage1_to_stage2.put(None)  # Propagate sentinel
                break
            print(f"Processor: received {value}, processing...")
            # Simulate processing
            time.sleep(0.15)
            processed = value * 2
            print(f"Processor: sending {processed}")
            stage2_to_stage3.put(processed)
        stage2_to_stage3.put(None)  # Sentinel for next stage
        print("Processor: finished")
    
    def finalizer():
        while True:
            value = stage2_to_stage3.get()
            if value is None:  # Check for sentinel
                break
            print(f"Finalizer: received {value}, finalizing...")
            # Simulate finalization
            time.sleep(0.1)
            print(f"Finalizer: final result is {value + 10}")
        print("Finalizer: finished")
    
    # Start pipeline threads
    gen_thread = threading.Thread(target=generator)
    proc_thread = threading.Thread(target=processor)
    fin_thread = threading.Thread(target=finalizer)
    
    gen_thread.start()
    proc_thread.start()
    fin_thread.start()
    
    gen_thread.join()
    proc_thread.join()
    fin_thread.join()

# Demonstrating multiprocessing with message passing
def multiprocessing_example():
    print("\n=== Multiprocessing with Message Passing ===")
    
    def worker_process(recv_conn, worker_id):
        while True:
            try:
                # Receive a task
                task = recv_conn.recv()
                if task is None:  # Shutdown signal
                    break
                
                task_id, data = task
                print(f"Process {worker_id}: processing task {task_id} with data {data}")
                
                # Simulate work
                time.sleep(0.3)
                
                # Send result back
                result = data * data
                recv_conn.send((worker_id, task_id, result))
                print(f"Process {worker_id}: completed task {task_id}, result: {result}")
            except EOFError:
                break
        
        print(f"Process {worker_id}: shutting down")
    
    # Create parent-child connections
    parent_conn, child_conn = mp.Pipe()
    
    # Start worker process
    worker = mp.Process(target=worker_process, args=(child_conn, 1))
    worker.start()
    
    # Send tasks
    for i in range(1, 4):
        task = (i, i * 10)
        parent_conn.send(task)
        print(f"Parent: sent task {task}")
    
    # Receive results
    for i in range(3):
        worker_id, task_id, result = parent_conn.recv()
        print(f"Parent: received result from process {worker_id}, task {task_id}, result: {result}")
    
    # Send shutdown signal
    parent_conn.send(None)
    
    # Close connections and wait for worker to finish
    parent_conn.close()
    child_conn.close()
    worker.join()

# Demonstrating asyncio with queues for message passing
import asyncio

async def async_message_passing_example():
    print("\n=== AsyncIO Message Passing ===")
    
    # Create async queue
    async_queue = asyncio.Queue(maxsize=5)
    
    async def async_producer(name, n_items):
        for i in range(n_items):
            item = f"{name}-item-{i}"
            print(f"Producer {name}: putting {item}")
            await async_queue.put(item)
            await asyncio.sleep(0.1)
        print(f"Producer {name}: finished")
    
    async def async_consumer(name):
        while True:
            item = await async_queue.get()
            print(f"Consumer {name}: got {item}")
            await asyncio.sleep(0.15)
            async_queue.task_done()
    
    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(async_producer("A", 5))
    consumer_task = asyncio.create_task(async_consumer("X"))
    
    await producer_task  # Wait for producer to finish
    await asyncio.sleep(0.2)  # Give consumer time to process remaining items
    consumer_task.cancel()  # Cancel consumer after producer finishes

def main():
    print("Message Passing - Hands-on Example (Python)")
    
    producer_consumer_example()
    worker_pool_example()
    actor_model_example()
    pipeline_example()
    multiprocessing_example()
    
    # Run async example
    asyncio.run(async_message_passing_example())
    
    print("\nAll message passing examples completed!")

if __name__ == "__main__":
    main()