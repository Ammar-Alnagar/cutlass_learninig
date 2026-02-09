# Parallel Algorithms

## Overview

Parallel algorithms leverage multiple processing units to solve computational problems more efficiently than sequential algorithms. Modern programming languages provide built-in support for parallel execution of standard algorithms, making it easier to achieve performance gains without manual thread management.

## Key Concepts

### Parallel Execution Policies
C++17 introduced execution policies that allow specifying how standard algorithms should be executed:
- `std::execution::seq`: Sequential execution (default behavior)
- `std::execution::par`: Parallel execution
- `std::execution::par_unseq`: Parallel and vectorized execution

### Types of Parallelism
- **Data Parallelism**: Same operation applied to multiple data elements
- **Task Parallelism**: Different operations executed concurrently
- **Pipeline Parallelism**: Stages of a computation executed concurrently

### Amdahl's Law
Describes the theoretical speedup in latency of the execution of a task at fixed workload that can be expected of a system whose resources are improved. It highlights the limitations of parallelization.

### Gustafson's Law
Provides a counterpoint to Amdahl's law, suggesting that as computing power increases, the size of problems tends to increase to fill the available computing power.

## Standard Parallel Algorithms

### Numeric Operations
- `std::reduce`: Generalization of sum that applies a binary operation
- `std::transform_reduce`: Combines transformation and reduction
- `std::inclusive_scan`: Prefix sum operation
- `std::exclusive_scan`: Similar to inclusive scan but shifted

### Modifying Operations
- `std::for_each`: Applies function to each element
- `std::transform`: Applies function and stores result
- `std::sort`: Sorting algorithm that can be parallelized
- `std::stable_sort`: Stable sorting algorithm

### Non-modifying Operations
- `std::find`: Finds elements in a range
- `std::any_of`, `std::all_of`, `std::none_of`: Logical operations
- `std::count`: Counts occurrences of a value

## Performance Considerations

### Granularity
The amount of work done per parallel task affects performance. Too fine-grained tasks create overhead; too coarse-grained tasks lead to load imbalance.

### Load Balancing
Ensuring that work is evenly distributed among processing units to maximize efficiency.

### Memory Access Patterns
Parallel algorithms perform better when memory access patterns are predictable and minimize cache misses.

## Limitations and Challenges

1. **Algorithm Compatibility**: Not all algorithms can be parallelized effectively
2. **Side Effects**: Functions used with parallel algorithms should be free of side effects
3. **Exception Safety**: Exceptions in parallel execution require special handling
4. **Overhead**: Parallel execution has overhead that may not be worthwhile for small datasets