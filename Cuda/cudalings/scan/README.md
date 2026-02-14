# Scan (Parallel Prefix Sum) Optimization Challenge

## Concept Overview
Scan (also called parallel prefix sum) is a fundamental parallel algorithm that computes all prefixes of an array: output[i] = op(input[0], input[1], ..., input[i-1]) for exclusive scan or output[i] = op(input[0], input[1], ..., input[i]) for inclusive scan.

## Naive Implementation
The provided `scan_naive.cu` implements an exclusive scan where each thread independently computes the sum of all preceding elements, resulting in O(n²) complexity.

## Current Performance Characteristics
- Time complexity: O(n²) due to redundant computation
- Memory access pattern: Each thread reads many elements
- Scalability: Poor - performance degrades rapidly with input size
- Efficiency: Very low due to repeated work

## Optimization Challenges

### Level 1: Work-Efficient Algorithm
- Implement a work-efficient scan algorithm with O(n) total work
- Use a tree-based approach (up-sweep/down-sweep)

### Level 2: Shared Memory Usage
- Use shared memory to reduce global memory accesses
- Implement efficient intra-block scanning

### Level 3: Bank Conflict Elimination
- Address shared memory bank conflicts in the tree-based algorithm
- Use padding or reordering to eliminate conflicts

### Level 4: Hierarchical Scanning
- Scan within blocks using shared memory
- Perform inter-block scanning for final results
- Handle carry propagation between blocks

### Level 5: Warp-Level Optimizations
- Use warp shuffle operations for efficient intra-warp scans
- Minimize synchronization overhead

### Level 6: Optimized Libraries
- Compare against optimized implementations like CUB::DeviceScan
- Understand the techniques used in production libraries

## Expected Improvements
- Achieve O(n) total work instead of O(n²)
- Significant performance improvement for larger datasets
- Optimize for your specific GPU architecture

## Performance Metrics to Track
- Execution time (wall clock and kernel time)
- Total work performed (should be O(n) for optimized version)
- Memory bandwidth utilization
- Speedup compared to naive implementation
- Scalability with different input sizes

## Compilation and Execution
```bash
nvcc -o scan_naive scan_naive.cu
./scan_naive [num_elements]
```

## Profiling Commands
```bash
# Basic profiling
nvprof ./scan_naive 16384

# Memory access pattern analysis
nvprof --metrics gld_transactions,gst_transactions,shared_efficiency ./scan_naive 16384

# Compute capability metrics
nvprof --metrics sm_efficiency,achieved_occupancy,instruction_throughput ./scan_naive 16384

# Detailed memory statistics
nvprof --metrics dram_read_throughput,dram_write_throughput,l2_tex_hit_rate,shared_transfers ./scan_naive 16384
```