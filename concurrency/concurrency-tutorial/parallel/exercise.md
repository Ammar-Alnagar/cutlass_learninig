# Exercise: Parallel Algorithms

## Objective
Practice using parallel algorithms to solve computational problems and understand their performance characteristics.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++17 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Parallel Matrix Multiplication**:
   - Implement matrix multiplication using parallel algorithms
   - Compare performance with sequential implementation
   - Experiment with different execution policies

3. **Parallel Search Algorithm**:
   - Implement a parallel search that looks for multiple targets simultaneously
   - Use parallel execution policies for better performance
   - Measure speedup compared to sequential search

4. **Custom Parallel Algorithm**:
   - Design and implement a parallel version of an algorithm of your choice
   - Compare performance with the sequential version
   - Document the conditions under which parallelization is beneficial

## Advanced Challenge

Implement a parallel version of merge sort:
- Divide the array into chunks
- Sort each chunk in parallel
- Merge the sorted chunks
- Compare performance with std::sort and sequential merge sort

## Questions to Think About

1. When is parallel execution not beneficial?
2. What types of algorithms are well-suited for parallelization?
3. How do execution policies affect performance?
4. What are the overhead costs of parallel execution?
5. How do you determine the optimal granularity for parallel tasks?

## Solution Notes

This exercise demonstrates the power and limitations of parallel algorithms. Key takeaways include:
- Understanding when parallelization provides benefits
- Learning to measure and analyze performance
- Recognizing the overhead costs of parallel execution
- Appreciating the complexity of designing efficient parallel algorithms