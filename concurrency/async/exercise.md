# Exercise: Asynchronous Programming

## Objective
Practice using async programming constructs to handle concurrent operations efficiently and understand the differences from traditional threading.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++11 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Web Scraper Simulation**:
   - Create an async function that "fetches" data from multiple URLs (simulate with delays)
   - Use async/await to handle multiple requests concurrently
   - Compare performance with a synchronous version

3. **Async Producer-Consumer**:
   - Implement a producer that generates items asynchronously
   - Implement consumers that process items from a shared async queue
   - Measure throughput and compare with threaded version

4. **Async Pipeline**:
   - Create a multi-stage async pipeline (fetch -> process -> store)
   - Each stage should be an async function
   - Chain them together and measure end-to-end latency

## Advanced Challenge

Build an async HTTP client that:
- Makes multiple concurrent requests to different endpoints
- Handles timeouts and retries
- Implements connection pooling
- Measures and reports performance metrics

## Questions to Think About

1. When should you use async programming vs traditional threading?
2. What are the limitations of async programming?
3. How do you handle CPU-bound tasks in an async context?
4. What is the difference between concurrency and parallelism in async programming?
5. How do you properly handle cancellation in async operations?

## Solution Notes

This exercise demonstrates the power of async programming for I/O-bound operations. Key takeaways include:
- Understanding when async provides benefits over threading
- Learning to structure code for async execution
- Recognizing the differences in error handling
- Appreciating the scalability benefits of async for I/O-bound tasks