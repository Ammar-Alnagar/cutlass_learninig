# Exercise: Advanced Concurrency Topics

## Objective
Explore advanced concurrency concepts including lock-free programming, memory models, and sophisticated synchronization patterns.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++11 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Implement a Lock-Free Queue**:
   - Research and implement the Michael & Scott lock-free queue algorithm
   - Test it with multiple producer and consumer threads
   - Compare performance with a mutex-protected queue

3. **Hazard Pointer Implementation**:
   - Implement a simple hazard pointer system for safe memory reclamation
   - Use it with your lock-free data structure
   - Demonstrate how it prevents the ABA problem

4. **Memory Model Exploration**:
   - Write code that behaves differently under different memory orderings
   - Experiment with acquire/release vs sequential consistency
   - Document the observed differences

## Advanced Challenge

Design and implement a lock-free hash table supporting:
- Concurrent insertions, deletions, and lookups
- Proper memory management to avoid ABA issues
- Good performance characteristics under high contention
- Compare its performance with a mutex-protected hash table

## Questions to Think About

1. When is lock-free programming beneficial vs detrimental?
2. What are the main challenges in implementing lock-free data structures?
3. How do memory barriers affect performance and correctness?
4. What is the relationship between lock-freedom and wait-freedom?
5. How do you test and verify lock-free algorithms?

## Solution Notes

This exercise explores the cutting edge of concurrent programming. Key takeaways include:
- Understanding the trade-offs of lock-free programming
- Learning about memory models and their impact
- Appreciating the complexity of advanced concurrency techniques
- Recognizing when simpler approaches may be preferable