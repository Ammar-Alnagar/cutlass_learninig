# Module 02: Pipelining Optimization

## Learning Objectives
1. Understand software pipelining concepts
2. Learn how pipelining hides memory latency
3. Configure NUM_STAGES for optimal performance
4. Measure throughput improvements

## What is Pipelining?

**Software Pipelining** is an optimization technique that:
- Overlaps memory operations with compute operations
- Keeps the GPU pipeline full by having multiple iterations in flight
- Hides memory latency by working on different data simultaneously

## How It Works

```
Without Pipelining:
[Load iter 1] -> [Compute iter 1] -> [Load iter 2] -> [Compute iter 2]

With Pipelining (2 stages):
[Load iter 1] -> [Load iter 2] -> [Load iter 3]
                [Compute iter 1] -> [Compute iter 2] -> [Compute iter 3]
```

## NUM_STAGES Parameter

Triton's `NUM_STAGES` parameter controls pipeline depth:

```python
@triton.jit
def kernel(..., NUM_STAGES: tl.constexpr):
    for i in range(N):
        # Triton automatically pipelines this loop
        data = tl.load(...)
        result = compute(data)
        tl.store(...)
```

### Recommended Values

| Scenario | NUM_STAGES | Reason |
|----------|-----------|--------|
| Memory-bound kernels | 3-5 | Hide memory latency |
| Compute-bound kernels | 2-3 | Less benefit from pipelining |
| Small problems | 1-2 | Overhead may not be worth it |
| Large problems | 3-5 | More iterations to pipeline |

## Key Concepts

### 1. Latency Hiding
- GPU memory access takes hundreds of cycles
- While waiting for data, work on other iterations
- Keeps compute units busy

### 2. Register Pressure
- More stages = more registers needed
- Too many stages can reduce occupancy
- Balance pipeline depth with resource usage

### 3. Automatic Pipelining
- Triton's compiler handles scheduling
- No manual buffer management needed
- Focus on algorithm, not micro-optimization

## Exercises

### Exercise 1: Benchmark Different Stage Counts
```bash
python pipelining_optimization.py
```

Observe how performance changes with different NUM_STAGES values.

### Exercise 2: Find Optimal Stages for Your GPU
- Run the benchmark on your specific hardware
- Note the optimal number of stages
- Compare with theoretical expectations

### Exercise 3: Analyze Memory vs Compute Bound
- Which kernels benefit most from pipelining?
- How can you tell if a kernel is memory-bound?

## Performance Tips

1. **Start with 3 stages** as a default
2. **Benchmark 2-5 stages** for your specific kernel
3. **Watch for register pressure** - too many stages hurts occupancy
4. **Memory-bound kernels** benefit most from pipelining
5. **Combine with tiling** for maximum effect

## Common Patterns

### Matrix Multiplication
```python
matmul_kernel[grid](
    ...,
    NUM_STAGES=3  # Good default for matmul
)
```

### Element-wise Operations
```python
elementwise_kernel[grid](
    ...,
    NUM_STAGES=2  # Less benefit, lower overhead
)
```

### Reduction Operations
```python
reduction_kernel[grid](
    ...,
    NUM_STAGES=4  # Can help with multi-pass reductions
)
```

## Next Steps
After mastering pipelining, move to Module 03: Shared Memory for advanced data reuse.
