# Module 04: Warp Specialization

## Learning Objectives
1. Understand warp-level GPU architecture
2. Learn warp specialization patterns
3. Implement program ID swizzling
4. Create fused kernels for maximum efficiency

## GPU Warp Architecture

```
GPU
├── Streaming Multiprocessors (SMs)
│   ├── Warp 0 (32 threads)
│   ├── Warp 1 (32 threads)
│   ├── Warp 2 (32 threads)
│   └── ...
```

### What is a Warp?
- Basic execution unit on NVIDIA GPUs
- 32 threads that execute in lockstep (SIMT)
- All threads in a warp execute the same instruction
- Different warps can execute different instructions

## Warp Specialization Concepts

### Definition
**Warp Specialization** is when different warps perform different tasks:
- Producer warps: Load data from global memory
- Consumer warps: Perform computation
- Storage warps: Handle memory stores

### Benefits
1. **Overlapping operations** - Memory and compute happen simultaneously
2. **Better resource utilization** - Different warps use different resources
3. **Reduced latency** - Hide memory latency with compute

## Program ID Swizzling

Swizzling rearranges program IDs to improve cache utilization:

```python
# Standard mapping
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Swizzled mapping (better L2 cache usage)
group_id = pid // num_pid_in_group
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### Why Swizzle?
- Adjacent program IDs process nearby data
- Improves L2 cache hit rate
- Reduces memory bandwidth pressure

## Fused Kernel Pattern

### Before (Unfused)
```python
# Multiple kernel launches
temp1 = matmul(x, weight)     # Global memory write
temp2 = temp1 + bias          # Global memory read + write
output = relu(temp2)          # Global memory read + write
```

### After (Fused)
```python
# Single kernel - data stays in registers
x = load(input)
w = load(weight)
b = load(bias)
output = relu(x @ w + b)      # All in registers!
store(output)
```

### Benefits
- **3x less global memory traffic**
- **No intermediate allocations**
- **Better register utilization**

## Exercises

### Exercise 1: Benchmark Swizzled Matmul
```bash
python warp_specialization.py
```

### Exercise 2: Create a Fused Kernel
Fuse these operations:
1. Layer normalization
2. Linear transformation
3. GELU activation

### Exercise 3: Experiment with GROUP_SIZE
Try different GROUP_SIZE_M values (4, 8, 16) and measure impact.

## Performance Tips

1. **Use swizzling for large matrices** - Improves L2 cache hit rate
2. **Fuse element-wise operations** - Keep data in registers
3. **Balance warp workload** - Avoid warp divergence
4. **Consider register pressure** - Fused kernels use more registers
5. **Profile with Nsight Compute** - Identify bottlenecks

## Common Patterns

### Matrix Multiplication with Swizzling
```python
GROUP_SIZE_M = 8  # Good default
grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
```

### Fused Operations
```python
# Fuse: Linear + Bias + Activation
@triton.jit
def fused_linear(x_ptr, w_ptr, b_ptr, out_ptr, ...):
    x = tl.load(...)
    w = tl.load(...)
    b = tl.load(...)
    out = activation(x * w + b)  # All in registers
    tl.store(...)
```

### Producer-Consumer Pattern
```python
# Conceptual (actual implementation varies)
if warp_id == 0:
    # Producer: Load next tile
    prefetch(next_data)
else:
    # Consumer: Compute on current tile
    result = compute(current_data)
```

## Advanced: Warp Matrix Operations

NVIDIA Tensor Cores can be accessed via `tl.dot()`:

```python
# Warp-level matrix multiply-accumulate
accumulator = tl.dot(a_block, b_block, accumulator)
```

This uses Tensor Cores when:
- Input dtype is float16 or bfloat16
- Block sizes are appropriate (16x16, 32x32, etc.)

## Next Steps
After mastering warp specialization, move to Module 05: Advanced Optimizations for cutting-edge techniques.
