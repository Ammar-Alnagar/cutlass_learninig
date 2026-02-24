# Module 03: Performance Profiling and Analysis

## Learning Objectives
1. Use Triton's built-in profiling tools
2. Analyze memory bandwidth and compute utilization
3. Identify performance bottlenecks
4. Compare kernel performance

## Why Profile?

Profiling helps you:
- **Identify bottlenecks** - Memory or compute bound?
- **Measure improvements** - Did optimization help?
- **Compare implementations** - Triton vs PyTorch
- **Tune parameters** - Find optimal block sizes

## Key Metrics

### 1. Execution Time
```python
import time

start = time.perf_counter()
kernel[grid](...)
torch.cuda.synchronize()
end = time.perf_counter()

time_ms = (end - start) * 1000
```

### 2. Memory Bandwidth
```python
# For vector add: 2 reads + 1 write
bytes_transferred = n_elements * 4 * 3
bandwidth_gb_s = bytes_transferred / time_ms / 1e6
```

### 3. Compute Throughput (GFLOPS)
```python
# For matmul: 2 * M * N * K FLOPs
flops = 2 * M * N * K
gflops = flops / time_ms / 1e6
```

## Bottleneck Analysis

### Memory-Bound Kernels
- **Characteristics**: Low arithmetic intensity
- **Examples**: Vector add, element-wise ops
- **Optimize**: Coalesced access, shared memory

### Compute-Bound Kernels
- **Characteristics**: High arithmetic intensity
- **Examples**: Large matrix multiply
- **Optimize**: Tensor cores, pipelining

### Arithmetic Intensity
```
Arithmetic Intensity = FLOPs / Bytes Accessed

< 5:   Memory-bound
5-20:  Mixed
> 20:  Compute-bound
```

## Profiling Tools

### 1. Basic Timing
```python
def profile(kernel, *args):
    # Warmup
    for _ in range(10):
        kernel(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        kernel(*args)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / 100
```

### 2. Triton Testing Utilities
```python
import triton.testing

triton.testing.test_perf(
    kernel,
    configs=[...],
    input_size=[...],
)
```

### 3. Nsight Compute
```bash
# Install: sudo apt install nsight-compute
ncu --set full python kernel.py
```

## Occupancy Analysis

Occupancy = Active warps / Maximum warps

Higher occupancy helps hide latency but isn't always better.

```
Block Size | Blocks/SM | Occupancy
-----------|-----------|----------
64         | 32        | 100%
128        | 16        | 100%
256        | 8         | 100%
512        | 4         | 100%
1024       | 2         | 100%
```

## Exercises

### Exercise 1: Profile Vector Add
Measure bandwidth for different vector sizes.

### Exercise 2: Analyze MatMul
Calculate GFLOPS for different matrix sizes.

### Exercise 3: Find Optimal Block Size
Benchmark multiple configurations.

## Performance Targets

| GPU | Peak Bandwidth | Peak FP16 TFLOPS |
|-----|----------------|------------------|
| V100 | 900 GB/s | 125 |
| A100 | 1555 GB/s | 312 |
| H100 | 3350 GB/s | 989 |

## Common Issues

### 1. Inaccurate Timing
```python
# WRONG: No synchronization
start = time.time()
kernel()
end = time.time()  # Kernel may still be running!

# CORRECT:
start = time.perf_counter()
kernel()
torch.cuda.synchronize()
end = time.perf_counter()
```

### 2. Not Enough Warmup
```python
# WRONG: First run includes compilation
time_kernel()

# CORRECT:
kernel()  # Warmup
torch.cuda.synchronize()
time_kernel()
```

### 3. Too Few Iterations
```python
# WRONG: High variance
for _ in range(5): time_kernel()

# CORRECT: Low variance
for _ in range(100): time_kernel()
```

## Next Steps
After mastering profiling, move to Module 04: Memory Debugging for advanced memory analysis.
