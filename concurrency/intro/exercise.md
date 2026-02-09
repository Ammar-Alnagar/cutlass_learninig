# Exercise: Introduction to Concurrency

## Objective
Understand the basic differences between sequential and concurrent execution by modifying and experimenting with the provided examples.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++11 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Modify the examples**:
   - Change the number of tasks from 3 to 5
   - Vary the execution times (some tasks 1s, others 2s)
   - Observe how the speedup changes

3. **Analyze the results**:
   - Compare the execution times between sequential and concurrent approaches
   - Note the theoretical vs actual speedup
   - Consider why the actual speedup might be less than the theoretical maximum

4. **Experiment with different workloads**:
   - Replace the sleep operations with actual computation (e.g., calculating primes)
   - Measure the performance difference again

## Questions to Think About

1. Why might the concurrent execution time not be exactly 1/3 of the sequential time even with 3 tasks?
2. What factors could affect the actual speedup achieved?
3. In what scenarios would concurrent execution not provide benefits?
4. What are the costs associated with creating and managing threads?

## Solution Notes

The solution demonstrates that concurrent execution can significantly improve performance for I/O-bound tasks. However, the actual speedup depends on:
- Number of CPU cores available
- Nature of the workload (CPU-bound vs I/O-bound)
- Overhead of thread creation and synchronization