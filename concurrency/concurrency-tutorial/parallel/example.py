# Parallel Algorithms - Hands-on Example (Python Version)
# This example demonstrates parallel execution using Python's multiprocessing

import multiprocessing as mp
from multiprocessing import Pool
import time
import random
from functools import reduce
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import operator

# Function to generate a large dataset for testing
def generate_dataset(size):
    return [random.randint(1, 1000) for _ in range(size)]

# Function to perform a computationally intensive operation
def heavy_computation(value):
    # Simulate a computation that takes some time
    result = value
    for i in range(100):
        result = (result * result + value) % 10007
    return result

def sequential_vs_parallel_map():
    print("\n=== Sequential vs Parallel Map ===")
    
    data_size = 1000000  # 1 million elements
    data = generate_dataset(data_size)
    
    # Sequential map
    start_time = time.time()
    sequential_results = list(map(heavy_computation, data))
    seq_duration = time.time() - start_time
    
    # Parallel map using ProcessPool
    start_time = time.time()
    with Pool() as pool:
        parallel_results = pool.map(heavy_computation, data)
    par_duration = time.time() - start_time
    
    print(f"Sequential map: {seq_duration:.3f} s")
    print(f"Parallel map: {par_duration:.3f} s")
    print(f"Speedup: {seq_duration/par_duration:.2f}x")

def parallel_reduce_example():
    print("\n=== Parallel Reduce Example ===")
    
    data_size = 5000000  # 5 million elements
    data = generate_dataset(data_size)
    
    # Sequential reduce
    start_time = time.time()
    seq_sum = reduce(operator.add, data, 0)
    seq_duration = time.time() - start_time
    
    # Parallel reduce using chunking
    def parallel_sum_chunk(chunk):
        return sum(chunk)
    
    start_time = time.time()
    chunk_size = len(data) // mp.cpu_count()
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with Pool() as pool:
        partial_sums = pool.map(parallel_sum_chunk, chunks)
    
    par_sum = sum(partial_sums)
    par_duration = time.time() - start_time
    
    print(f"Sequential reduce: {seq_duration:.3f} s, sum: {seq_sum}")
    print(f"Parallel reduce: {par_duration:.3f} s, sum: {par_sum}")
    print(f"Speedup: {seq_duration/par_duration:.2f}x")

def parallel_filter_example():
    print("\n=== Parallel Filter Example ===")
    
    data_size = 2000000  # 2 million elements
    data = generate_dataset(data_size)
    
    # Define a predicate function
    def is_prime_like(n):
        if n < 2:
            return False
        for i in range(2, min(int(n**0.5) + 1, 100)):  # Limit range for performance
            if n % i == 0:
                return False
        return True
    
    # Sequential filter
    start_time = time.time()
    seq_filtered = list(filter(is_prime_like, data))
    seq_duration = time.time() - start_time
    
    # Parallel filter using chunking
    def parallel_filter_chunk(chunk):
        return list(filter(is_prime_like, chunk))
    
    start_time = time.time()
    chunk_size = len(data) // mp.cpu_count()
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with Pool() as pool:
        partial_results = pool.map(parallel_filter_chunk, chunks)
    
    par_filtered = [item for sublist in partial_results for item in sublist]
    par_duration = time.time() - start_time
    
    print(f"Sequential filter: {seq_duration:.3f} s, found {len(seq_filtered)} items")
    print(f"Parallel filter: {par_duration:.3f} s, found {len(par_filtered)} items")
    print(f"Speedup: {seq_duration/par_duration:.2f}x")

def parallel_sort_example():
    print("\n=== Parallel Sort Example ===")
    
    data_size = 2000000  # 2 million elements
    data = generate_dataset(data_size)
    
    # Sequential sort
    data_copy = data.copy()
    start_time = time.time()
    data_copy.sort()
    seq_duration = time.time() - start_time
    
    # For parallel sort, we'll use a divide-and-conquer approach
    def merge_sorted_lists(lists):
        result = []
        indices = [0] * len(lists)
        
        while any(indices[i] < len(lists[i]) for i in range(len(lists))):
            min_val = float('inf')
            min_idx = -1
            
            for i, lst in enumerate(lists):
                if indices[i] < len(lst) and lst[indices[i]] < min_val:
                    min_val = lst[indices[i]]
                    min_idx = i
            
            if min_idx != -1:
                result.append(min_val)
                indices[min_idx] += 1
        
        return result
    
    def parallel_sort(data):
        if len(data) <= 10000:  # Base case: use sequential sort for small arrays
            data.sort()
            return data
        
        # Divide the data into chunks
        num_chunks = mp.cpu_count()
        chunk_size = len(data) // num_chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Sort each chunk in parallel
        with Pool() as pool:
            sorted_chunks = pool.map(lambda chunk: sorted(chunk), chunks)
        
        # Merge the sorted chunks
        return merge_sorted_lists(sorted_chunks)
    
    # Parallel sort
    start_time = time.time()
    par_sorted = parallel_sort(data.copy())
    par_duration = time.time() - start_time
    
    print(f"Sequential sort: {seq_duration:.3f} s")
    print(f"Parallel sort: {par_duration:.3f} s")
    print(f"Speedup: {seq_duration/par_duration:.2f}x")

def numpy_parallel_example():
    print("\n=== NumPy Vectorized Operations (Implicit Parallelism) ===")
    
    data_size = 10000000  # 10 million elements
    data = np.random.randint(1, 1000, size=data_size)
    
    # NumPy operations are optimized and can use multiple cores
    start_time = time.time()
    # Perform a computation that benefits from vectorization
    result = np.sum(np.power(data, 2) + data)  # Element-wise operations
    numpy_duration = time.time() - start_time
    
    # Equivalent sequential operation for comparison
    start_time = time.time()
    seq_result = sum(x*x + x for x in data[:100000])  # Smaller sample for time
    seq_duration = time.time() - start_time
    
    print(f"NumPy vectorized (large dataset): {numpy_duration:.3f} s")
    print(f"Sequential equivalent (small dataset): {seq_duration:.3f} s (scaled estimate would be much slower)")
    print("Note: NumPy operations use optimized C libraries and can leverage SIMD and multi-threading")

def custom_parallel_algorithm():
    print("\n=== Custom Parallel Algorithm: Parallel Prefix Sum ===")
    
    def sequential_prefix_sum(arr):
        result = [0] * len(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = result[i-1] + arr[i]
        return result
    
    def parallel_prefix_sum(arr):
        n = len(arr)
        if n <= 10000:  # Use sequential for small arrays
            return sequential_prefix_sum(arr)
        
        # Divide into chunks
        num_processes = mp.cpu_count()
        chunk_size = n // num_processes
        chunks = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]
        
        # Compute prefix sums for each chunk in parallel
        with Pool() as pool:
            chunk_prefix_sums = pool.map(sequential_prefix_sum, chunks)
        
        # Compute the cumulative offset for each chunk
        offsets = [0]
        for i in range(len(chunks) - 1):
            last_val = chunk_prefix_sums[i][-1] + offsets[-1]
            offsets.append(last_val)
        
        # Adjust each chunk with the appropriate offset
        result = []
        for i, chunk_ps in enumerate(chunk_prefix_sums):
            if i == 0:
                result.extend(chunk_ps)
            else:
                offset = offsets[i]
                adjusted_chunk = [x + offset for x in chunk_ps]
                result.extend(adjusted_chunk)
        
        return result
    
    data_size = 1000000  # 1 million elements
    data = generate_dataset(data_size)
    
    # Sequential prefix sum
    start_time = time.time()
    seq_result = sequential_prefix_sum(data)
    seq_duration = time.time() - start_time
    
    # Parallel prefix sum
    start_time = time.time()
    par_result = parallel_prefix_sum(data)
    par_duration = time.time() - start_time
    
    print(f"Sequential prefix sum: {seq_duration:.3f} s")
    print(f"Parallel prefix sum: {par_duration:.3f} s")
    print(f"Speedup: {seq_duration/par_duration:.2f}x")
    
    # Verify results are the same (check first and last few elements)
    print(f"Results match: {seq_result[:5] == par_result[:5] and seq_result[-5:] == par_result[-5:]}")

if __name__ == "__main__":
    print("Parallel Algorithms - Hands-on Example (Python)")
    
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    sequential_vs_parallel_map()
    parallel_reduce_example()
    parallel_filter_example()
    parallel_sort_example()
    numpy_parallel_example()
    custom_parallel_algorithm()
    
    print("\nAll parallel algorithm examples completed!")