# CUDAlings: CUDA Optimization Learning Project

Welcome to CUDAlings, a comprehensive collection of CUDA concepts designed for learning and practicing GPU optimization techniques. Each module contains a naive implementation that you can optimize as much as possible.

## Project Structure

```
├── fundamentals/          # Basic CUDA concepts and fundamentals
├── vector_addition/       # Vector addition with optimization challenges
├── matrix_multiplication/ # Matrix multiplication with tiling challenges
├── reduction/             # Reduction operations with tree-based optimizations
├── convolution/           # 2D convolution with shared memory challenges
├── transpose/             # Matrix transpose with memory access optimizations
├── histogram/             # Histogram computation with atomic operation challenges
├── prefix_sum/            # Prefix sum (scan) with work-efficient algorithms
├── sort/                  # Sorting algorithms with parallel approaches
├── scan/                  # Exclusive/inclusive scan operations
├── spmv/                  # Sparse matrix-vector multiplication
└── utils/                 # Common utilities for timing and profiling
```

## Getting Started

Each concept directory contains:
- `.cu` file with naive implementation
- `README.md` with optimization challenges and expected improvements
- Test cases for correctness and performance measurement

## Optimization Process

For each concept, follow these steps:

1. **Compile and Run**: Compile the naive implementation and run it to establish baseline performance
2. **Profile**: Use `nvprof` to identify bottlenecks
3. **Optimize**: Apply optimization techniques based on the README suggestions
4. **Measure**: Compare performance against the naive implementation
5. **Iterate**: Continue optimizing until you reach the expected performance level

## Common Profiling Commands

```bash
# Basic profiling
nvprof ./program

# Memory access analysis
nvprof --metrics gld_transactions,gst_transactions,shared_efficiency ./program

# Occupancy and compute metrics
nvprof --metrics achieved_occupancy,sm_efficiency,instruction_throughput ./program

# Bandwidth and throughput
nvprof --metrics dram_read_throughput,dram_write_throughput ./program

# Detailed analysis
nvprof --print-gpu-trace ./program
```

## Performance Metrics to Track

- **Execution Time**: Wall clock and kernel execution time
- **Memory Bandwidth**: Achieved vs. theoretical peak
- **Occupancy**: Percentage of active warps
- **GFLOPS**: Billions of floating-point operations per second
- **Speedup**: Improvement factor over naive implementation

## Expected Learning Outcomes

By working through these exercises, you will gain expertise in:
- Memory access optimization (coalescing, shared memory usage)
- Computational optimization (reducing divergent warps, maximizing occupancy)
- Algorithmic optimization (work-efficient algorithms)
- Profiling and performance analysis
- Architecture-aware optimization techniques

## Utilities

The `utils/` directory contains common headers with:
- Error checking macros
- Timing utilities (both CPU and GPU timers)
- GPU property printing functions
- Occupancy calculation helpers

## Contributing

Feel free to add more optimization challenges or create additional CUDA concepts following the same pattern!

Happy optimizing!