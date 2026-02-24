# Module 03: Shared Memory Optimization

## Learning Objectives
1. Understand GPU memory hierarchy
2. Learn shared memory access patterns
3. Implement blocked algorithms for data reuse
4. Optimize reduction operations

## GPU Memory Hierarchy

```
┌─────────────────────────────────────────┐
│         Global Memory (HBM/DRAM)        │
│         ~1TB/s bandwidth                │
│         High latency (~400 cycles)      │
└─────────────────────────────────────────┘
                    ↓↑
┌─────────────────────────────────────────┐
│         L2 Cache                        │
│         Shared across all SMs           │
└─────────────────────────────────────────┘
                    ↓↑
┌─────────────────────────────────────────┐
│    L1 Cache / Shared Memory (SRAM)      │
│    ~100x faster than global memory      │
│    Per-SM, configurable size            │
└─────────────────────────────────────────┘
                    ↓↑
┌─────────────────────────────────────────┐
│         Registers (per thread)          │
│         Fastest, lowest latency         │
└─────────────────────────────────────────┘
```

## Shared Memory Concepts

### What is Shared Memory?
- Fast on-chip memory shared by threads in a block
- ~100x faster than global memory
- Limited size (~48-164 KB per SM depending on GPU)
- Must be explicitly managed in CUDA, automatic in Triton

### How Triton Uses Shared Memory
```python
# In Triton, shared memory is used automatically
# when you load data in a blocked pattern:

for k in range(0, K, BLOCK_SIZE_K):
    # This load automatically uses shared memory
    block = tl.load(ptr + offsets, mask=mask)
    result = compute(block)
```

## Optimization Patterns

### 1. Blocked Matrix Multiplication
```
Global Memory → Shared Memory → Registers → Compute
     ↓              ↓              ↓
   Load tile    Reuse for      Multiple
   from HBM     multiple ops   dot products
```

### 2. Tree Reduction
```
Step 1: Each block loads data into shared memory
Step 2: Tree reduction within shared memory
Step 3: Store partial result
Step 4: Reduce partial results
```

### 3. Data Reuse
```python
# Without reuse (bad):
for i in range(N):
    for j in range(N):
        a = load_A[i]  # Loaded N times!
        b = load_B[j]
        compute(a, b)

# With reuse (good):
for i in range(N):
    a = load_A[i]  # Loaded once!
    for j in range(N):
        b = load_B[j]
        compute(a, b)
```

## Exercises

### Exercise 1: Benchmark Shared Memory Matmul
```bash
python shared_memory_optimization.py
```

### Exercise 2: Compare Reduction Strategies
- Implement atomic-based reduction
- Compare with shared memory reduction
- Measure the difference

### Exercise 3: Experiment with Block Sizes
- Try different BLOCK_SIZE_K values
- Observe impact on performance
- Find optimal for your GPU

## Performance Tips

1. **Maximize data reuse** - Load once, use many times
2. **Choose block sizes wisely** - Balance shared memory usage
3. **Avoid bank conflicts** - Use power-of-2 strides
4. **Combine with pipelining** - Overlap load and compute
5. **Profile memory traffic** - Use Nsight Compute

## Memory Bandwidth Analysis

```
For matrix multiplication C = A @ B:
- Naive: Read A and B once per output element
- Blocked: Read A and B once per block

Bandwidth savings = O(BLOCK_SIZE) reduction in global memory traffic
```

## Common Patterns

### Matrix Multiplication
```python
BLOCK_SIZE_M = 64  # Rows
BLOCK_SIZE_N = 64  # Columns  
BLOCK_SIZE_K = 32  # Reduction
```

### Reduction
```python
BLOCK_SIZE = 1024  # Elements per block
# Use tree reduction in shared memory
```

### Convolution
```python
# Load input window into shared memory
# Reuse for multiple output positions
```

## Next Steps
After mastering shared memory, move to Module 04: Warp Specialization for advanced parallelism.
