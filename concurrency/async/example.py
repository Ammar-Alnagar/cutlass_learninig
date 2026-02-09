# Asynchronous Programming - Hands-on Example (Python Version)
# This example demonstrates async/await, futures, and async execution

import asyncio
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Async function that simulates an async operation
async def simulate_async_operation(op_id, delay):
    print(f"Async operation {op_id} starting...")
    
    # Simulate async work with sleep (non-blocking)
    await asyncio.sleep(delay)
    
    # Return result
    result = op_id * op_id  # Just a simple computation
    print(f"Async operation {op_id} completed with result: {result}")
    
    return result

# Example using asyncio.gather for concurrent execution
async def gather_example():
    print("\n=== Gather Example ===")
    
    # Schedule multiple async operations concurrently
    tasks = [
        simulate_async_operation(1, 0.8),
        simulate_async_operation(2, 0.6),
        simulate_async_operation(3, 1.0)
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    print(f"Gather results: {results}")

# Example using asyncio.create_task
async def create_task_example():
    print("\n=== Create Task Example ===")
    
    # Create tasks explicitly
    task1 = asyncio.create_task(simulate_async_operation(4, 0.7))
    task2 = asyncio.create_task(simulate_async_operation(5, 0.5))
    
    # Do other work while tasks run
    print("Main coroutine doing other work...")
    await asyncio.sleep(0.2)
    
    # Wait for tasks to complete
    result1 = await task1
    result2 = await task2
    
    print(f"Task results: {result1}, {result2}")

# Example demonstrating async with threading
def blocking_operation(op_id, delay):
    """Simulates a blocking operation"""
    print(f"Blocking operation {op_id} starting...")
    time.sleep(delay)  # Blocking sleep
    result = op_id * 10
    print(f"Blocking operation {op_id} completed with result: {result}")
    return result

async def thread_integration_example():
    print("\n=== Thread Integration Example ===")
    
    loop = asyncio.get_event_loop()
    
    # Run blocking operations in a thread pool
    with ThreadPoolExecutor() as executor:
        # Submit blocking operations to thread pool
        future1 = loop.run_in_executor(executor, blocking_operation, 1, 0.8)
        future2 = loop.run_in_executor(executor, blocking_operation, 2, 0.6)
        
        # Do other async work while blocking operations run
        print("Doing other async work...")
        await asyncio.sleep(0.3)
        
        # Wait for blocking operations to complete
        result1 = await future1
        result2 = await future2
        
        print(f"Thread results: {result1}, {result2}")

# Example demonstrating async generators
async def async_generator():
    """An async generator that yields values over time"""
    for i in range(5):
        await asyncio.sleep(0.2)  # Simulate async work
        yield i * i

async def async_generator_example():
    print("\n=== Async Generator Example ===")
    
    async for value in async_generator():
        print(f"Generated value: {value}")

# Example demonstrating exception handling in async
async def error_prone_operation(op_id):
    print(f"Operation {op_id} starting...")
    await asyncio.sleep(0.4)
    
    if op_id == 99:  # Simulate an error condition
        raise ValueError(f"Operation {op_id} failed!")
    
    return op_id * 2

async def exception_handling_example():
    print("\n=== Exception Handling Example ===")
    
    tasks = [
        error_prone_operation(1),
        error_prone_operation(99),  # This will raise an exception
        error_prone_operation(3)
    ]
    
    # Handle exceptions in async context
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            print(f"Success: {result}")
        except ValueError as e:
            print(f"Caught exception: {e}")

# Example showing async pipeline
async def async_pipeline():
    print("\n=== Async Pipeline Example ===")
    
    # Stage 1: Generate data
    async def stage1():
        print("Stage 1: Generating data")
        await asyncio.sleep(0.2)
        return [1, 2, 3, 4, 5]
    
    # Stage 2: Process data
    async def stage2(raw_data):
        data = await raw_data  # Wait for stage 1
        print("Stage 2: Processing data")
        processed = []
        for item in data:
            await asyncio.sleep(0.1)  # Simulate processing time
            processed.append(item * 2)
        print("Stage 2: Data processed")
        return processed
    
    # Stage 3: Finalize data
    async def stage3(processed_data):
        data = await processed_data  # Wait for stage 2
        print("Stage 3: Finalizing data")
        result = sum(data)
        print("Stage 3: Sum calculated")
        return result
    
    # Chain the stages
    raw_data_task = asyncio.create_task(stage1())
    processed_data_task = asyncio.create_task(stage2(raw_data_task))
    final_result_task = asyncio.create_task(stage3(processed_data_task))
    
    final_result = await final_result_task
    print(f"Pipeline final result: {final_result}")

# Example demonstrating asyncio queues for producer-consumer
async def queue_example():
    print("\n=== Queue Example ===")
    
    queue = asyncio.Queue()
    
    async def producer(queue, n_items):
        for i in range(n_items):
            item = f"item-{i}"
            print(f"Producing {item}")
            await queue.put(item)
            await asyncio.sleep(0.1)  # Simulate time between productions
        
        # Signal completion
        await queue.put(None)
        print("Producer finished")
    
    async def consumer(queue, consumer_id):
        while True:
            item = await queue.get()
            if item is None:  # Sentinel value
                # Put sentinel back for other consumers
                await queue.put(None)
                break
            
            print(f"Consumer {consumer_id} got {item}")
            await asyncio.sleep(0.2)  # Simulate processing time
            queue.task_done()
        
        print(f"Consumer {consumer_id} finished")
    
    # Create producer and consumers
    producer_task = asyncio.create_task(producer(queue, 5))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, 1)),
        asyncio.create_task(consumer(queue, 2))
    ]
    
    # Wait for all tasks to complete
    await producer_task
    await asyncio.gather(*consumer_tasks)

async def main():
    print("Asynchronous Programming - Hands-on Example")
    
    await gather_example()
    await create_task_example()
    await thread_integration_example()
    await async_generator_example()
    await exception_handling_example()
    await async_pipeline()
    await queue_example()
    
    print("\nAll async examples completed!")

if __name__ == "__main__":
    asyncio.run(main())