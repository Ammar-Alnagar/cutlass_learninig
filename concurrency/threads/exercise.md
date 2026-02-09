# Exercise: Thread Basics

## Objective
Practice creating, managing, and working with threads in different scenarios to understand thread lifecycle and basic operations.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++11 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Thread Producer-Consumer Simulation**:
   - Create a producer thread that generates numbers and puts them in a shared queue
   - Create a consumer thread that takes numbers from the queue and prints them
   - Run both threads for 10 iterations
   - Use a simple list/array as a shared buffer (we'll add synchronization later)

3. **Thread Performance Analysis**:
   - Create a function that calculates prime numbers up to N
   - Compare execution time between:
     - Single-threaded calculation
     - Multi-threaded calculation (split work among multiple threads)
   - Analyze the performance difference

4. **Thread Priority Experiment** (if supported by your language/runtime):
   - Create threads with different priorities
   - Observe how this affects execution order

## Advanced Challenge

Create a thread pool implementation from scratch:
- Create a fixed number of worker threads
- Implement a task queue
- Allow submitting tasks to the pool
- Handle thread lifecycle management

## Questions to Think About

1. What happens if you forget to join a thread?
2. What is the difference between join() and detach() in C++?
3. When would you use daemon threads vs regular threads?
4. What are the trade-offs of creating many threads vs few threads?
5. How does the operating system schedule threads?

## Solution Notes

This exercise introduces the fundamental concepts of thread management. Key takeaways include:
- Understanding thread lifecycle (creation, execution, joining)
- Recognizing the overhead of thread creation
- Learning about thread safety concerns (which will be addressed in the next module)
- Understanding when to use different threading approaches