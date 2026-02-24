# Module 02: Fused Operations

## Learning Objectives
1. Understand fusion benefits and tradeoffs
2. Implement common fused operations
3. Design custom fused kernels
4. Measure fusion performance gains

## Why Fuse Operations?

### Memory Traffic Reduction
```
Unfused:
  Load x  → Multiply → Store temp1
  Load temp1 → Add → Store temp2
  Load temp2 → ReLU → Store output
  
  Total: 3 loads, 3 stores

Fused:
  Load x → Multiply → Add → ReLU → Store output
  
  Total: 1 load, 1 store
```

### Benefits
- **Reduced memory bandwidth** - Fewer global memory accesses
- **Better cache utilization** - Data stays in registers
- **Lower latency** - No intermediate synchronization
- **Higher throughput** - More compute per memory access

## Common Fusion Patterns

### Pattern 1: Linear + Bias + Activation
```python
@triton.jit
def fused_linear_activation(x_ptr, w_ptr, b_ptr, out_ptr, n, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # All operations in registers
    result = x * w + b
    result = activation(result)  # ReLU, GELU, etc.
    
    tl.store(out_ptr + offsets, result, mask=mask)
```

### Pattern 2: Normalization + Linear
```python
@triton.jit
def fused_norm_linear(x_ptr, norm_w, norm_b, lin_w, lin_b, out_ptr, n, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Normalization (e.g., LayerNorm)
    mean = tl.sum(x) / n
    var = tl.sum((x - mean) ** 2) / n
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_norm = x_norm * norm_w + norm_b
    
    # Linear transformation
    out = x_norm * lin_w + lin_b
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

### Pattern 3: Activation + Dropout
```python
@triton.jit
def fused_activation_dropout(x_ptr, out_ptr, n, scale, seed, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Activation (GELU)
    gelu = 0.5 * x * (1 + tl.math.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))
    
    # Dropout
    random = tl.rand(seed, offsets)
    mask = random < 0.5
    out = tl.where(mask, gelu * scale, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

### Pattern 4: QKV Projection
```python
@triton.jit
def fused_qkv(x_ptr, wq, wk, wv, bq, bk, bv, q_out, k_out, v_out, n, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # All three projections in one pass
    q = x * wq + bq
    k = x * wk + bk
    v = x * wv + bv
    
    tl.store(q_out + offsets, q, mask=mask)
    tl.store(k_out + offsets, k, mask=mask)
    tl.store(v_out + offsets, v, mask=mask)
```

## Fusion Tradeoffs

### When to Fuse
✅ Operations on same data
✅ Element-wise operations
✅ Small intermediate tensors
✅ Same thread handles all ops

### When NOT to Fuse
❌ Operations need different block sizes
❌ Register pressure too high
❌ Complex control flow
❌ Different data types

## Performance Analysis

### Arithmetic Intensity
```
Fusion increases arithmetic intensity:

Unfused: FLOPs / (2 * bytes)
Fused:   FLOPs / bytes

Higher intensity = better GPU utilization
```

### Register Usage
```
Fused kernels use more registers:
- Each intermediate value needs storage
- Too many registers → reduced occupancy

Balance fusion depth with register pressure.
```

## Exercises

### Exercise 1: Fused SiLU
Implement: `silu(x) = x * sigmoid(x)`

### Exercise 2: Fused Residual
Implement: `output = LayerNorm(x + Dropout(x))`

### Exercise 3: Fused Attention Input
Implement: `Q, K, V = fused_qkv_projection(x)`

## Best Practices

1. **Profile before and after** - Verify improvement
2. **Check register usage** - Don't over-fuse
3. **Test correctness** - Compare with unfused
4. **Consider maintainability** - Don't over-complicate
5. **Start with hot paths** - Fuse most-used operations first

## Next Steps
After mastering fused operations, move to Module 03: Attention Kernels for transformer-specific optimizations.
