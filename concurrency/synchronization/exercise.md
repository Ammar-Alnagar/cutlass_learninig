# Exercise: Synchronization Primitives

## Objective
Practice using different synchronization primitives to solve classic concurrency problems and understand their trade-offs.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++17 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Bank Account Simulation**:
   - Create a BankAccount class with balance and deposit/withdraw methods
   - Use mutexes to protect the balance during operations
   - Create multiple threads that perform deposits and withdrawals
   - Verify that the final balance is correct despite concurrent access

3. **Reader-Writer Problem**:
   - Implement a shared data structure that supports multiple readers OR one writer
   - Use condition variables to coordinate between readers and writers
   - Create multiple reader threads and writer threads
   - Ensure readers can access simultaneously, but writers have exclusive access

4. **Dining Philosophers Problem**:
   - Implement the dining philosophers problem using mutexes
   - Prevent deadlock by having philosophers pick up forks in a specific order
   - Track how many times each philosopher eats

## Advanced Challenge

Implement a thread-safe bounded buffer (circular buffer) using:
- Mutex for mutual exclusion
- Condition variables to signal when buffer is not empty/full
- Semaphore to track available spaces and items

## Questions to Think About

1. What is the difference between a mutex and a binary semaphore?
2. When would you use a recursive mutex vs a regular mutex?
3. What are the potential issues with condition variables?
4. How do you prevent deadlock when using multiple locks?
5. What are the performance implications of different synchronization primitives?

## Solution Notes

This exercise demonstrates the importance of proper synchronization in concurrent programs. Key takeaways include:
- Understanding when and how to use different synchronization primitives
- Recognizing common concurrency patterns and problems
- Learning to prevent race conditions and deadlocks
- Appreciating the trade-offs between different synchronization approaches