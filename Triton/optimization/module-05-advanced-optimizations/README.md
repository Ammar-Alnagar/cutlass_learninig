# Module 05: Advanced Optimizations

## Learning Objectives
1. Combine multiple optimization techniques
2. Implement autotuning for optimal parameters
3. Use persistent kernels for maximum occupancy
4. Profile and analyze kernel performance

## Optimization Techniques Summary

| Technique | Purpose | Typical Speedup |
|-----------|---------|-----------------|
| Tiling | Memory coalescing | 2-10x |
| Pipelining | Hide latency | 1.2-2x |
| Shared Memory | Data reuse | 2-5x |
| Warp Specialization | Resource utilization | 1.2-1.5x |
| Fused Kernels | Reduce memory traffic | 2-3x |
| Autotuning | Find optimal params | 1.5-3x |
| Persistent Kernels | Reduce launch overhead | 1.1-1.5x |

**Combined potential speedup: 10-100x over naive implementation**

## Technique 1: Autotuning

Autotuning automatically finds the best configuration:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1),
    ],
    key=['n_elements'],  # Retune when this changes
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

### How It Works
1. First run: Tests all configurations
2. Caches best configuration for input shape
3. Subsequent runs: Use cached optimal config

### Best Practices
- Include diverse configurations
- Use `key` to retune for different problem sizes
- Cache results for production use

## Technique 2: Persistent Kernels

Persistent kernels use grid-stride loops:

```python
@triton.jit
def persistent_kernel(..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Grid-stride loop
    for start in range(pid * BLOCK_SIZE, n_elements, num_programs * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        # Process block...
```

### When to Use
- Small-to-medium workloads
- Launch overhead is significant
- Want maximum GPU occupancy

### Benefits
- Fewer kernel launches
- Better CPU-GPU overlap
- Reduced synchronization

## Technique 3: Combined Optimizations

Production kernels combine all techniques:

```python
@triton.autotune(configs=...)
@triton.jit
def optimized_matmul(
    ...,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Swizzling for cache efficiency
    pid = swizzle_program_id(...)
    
    # Tiled computation
    for k in range(0, K, BLOCK_SIZE_K):
        # Pipelined loads
        a = tl.load(...)
        b = tl.load(...)
        # Compute
        acc = tl.dot(a, b, acc)
```

## Profiling Tools

### 1. Triton Profiler
```python
import triton.testing

triton.testing.test_perf(
    kernel,
    configs=[...],
    input_size=[...],
)
```

### 2. Nsight Compute
```bash
ncu --set full python kernel.py
```

### 3. PyTorch Profiler
```python
with torch.profiler.profile() as prof:
    kernel()
print(prof.key_averages().table())
```

## Exercises

### Exercise 1: Autotune a Kernel
Add autotuning to the vector addition kernel. Test different block sizes.

### Exercise 2: Create a Persistent Kernel
Convert the softmax kernel to use grid-stride loops.

### Exercise 3: Profile and Optimize
1. Profile a kernel with Nsight Compute
2. Identify the bottleneck
3. Apply appropriate optimization
4. Measure improvement

## Optimization Checklist

Before deploying a kernel:

- [ ] **Tiling**: Block sizes are powers of 2
- [ ] **Pipelining**: NUM_STAGES is tuned (2-4)
- [ ] **Shared Memory**: Data is reused effectively
- [ ] **Fused Operations**: Element-wise ops are combined
- [ ] **Autotuning**: Configuration is optimized
- [ ] **Profiling**: Bottlenecks are identified
- [ ] **Correctness**: Results match reference

## Performance Targets

| Operation | Target Performance |
|-----------|-------------------|
| Vector Add | >80% memory bandwidth |
| Matrix Mul (FP16) | >50 TFLOPS (A100) |
| Softmax | >500 GB/s |
| LayerNorm | >300 GB/s |
| Conv2D | >30 TFLOPS |

## Common Pitfalls

1. **Too many registers** - Reduces occupancy
2. **Bank conflicts** - Slows shared memory
3. **Warp divergence** - Serializes execution
4. **Uncoalesced access** - Wastes bandwidth
5. **Excessive tuning** - Long compile times

## Next Steps

You've completed the Optimization track! 

Next: Debugging track to learn how to fix issues when they arise.
