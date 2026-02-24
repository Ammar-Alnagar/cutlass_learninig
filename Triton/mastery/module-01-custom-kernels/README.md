# Module 01: Custom Kernel Development

## Learning Objectives
1. Design kernel architecture from requirements
2. Implement custom operations efficiently
3. Handle complex indexing patterns
4. Optimize for specific hardware

## Kernel Design Process

### 1. Define Requirements
```
Input: What data does the kernel need?
Output: What should it produce?
Operation: What computation is performed?
Constraints: Any special requirements?
```

### 2. Choose Block Size
```python
# Element-wise: 512-1024
BLOCK_SIZE = 1024

# Reduction: Match input dimension
BLOCK_SIZE = triton.next_power_of_2(n_cols)

# Matrix ops: 32-128
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
```

### 3. Design Indexing
```python
# 1D: Simple linear indexing
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# 2D: Row-major indexing
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
```

## Custom Kernel Examples

### Polynomial Activation
```python
# f(x) = c0 + c1*x + c2*x^2
@triton.jit
def poly_kernel(x_ptr, out_ptr, n, c0, c1, c2, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = c0 + c1 * x + c2 * x * x
    tl.store(out_ptr + offsets, out, mask=mask)
```

### Gated Linear Unit
```python
# GLU(x, gate) = x * sigmoid(gate)
@triton.jit
def glu_kernel(x_ptr, gate_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)
    
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    out = x * sigmoid_gate
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

### L2 Norm (Reduction)
```python
# norm = sqrt(sum(x^2))
@triton.jit
def l2_norm_kernel(x_ptr, out_ptr, n_rows, n_cols, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # One program per row
    row_start = pid * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row = tl.load(x_ptr + row_start + col_offsets, mask=mask)
    squared = row * row
    sum_sq = tl.sum(squared)
    norm = tl.sqrt(sum_sq)
    
    tl.store(out_ptr + pid, norm)
```

### Complex Multiplication
```python
# (a+bi)(c+di) = (ac-bd) + (ad+bc)i
@triton.jit
def complex_mul_kernel(re_a, im_a, re_b, im_b, re_out, im_out, n, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    ra = tl.load(re_a + offsets, mask=mask)
    ia = tl.load(im_a + offsets, mask=mask)
    rb = tl.load(re_b + offsets, mask=mask)
    ib = tl.load(im_b + offsets, mask=mask)
    
    tl.store(re_out + offsets, ra * rb - ia * ib, mask=mask)
    tl.store(im_out + offsets, ra * ib + ia * rb, mask=mask)
```

## Design Patterns

### Pattern 1: Element-wise
```python
@triton.jit
def elementwise_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(in_ptr + offsets, mask=mask)
    out = custom_operation(x)
    tl.store(out_ptr + offsets, out, mask=mask)
```

### Pattern 2: Reduction
```python
@triton.jit
def reduction_kernel(in_ptr, out_ptr, n_rows, n_cols, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    row = tl.load(in_ptr + row_start + offsets, mask=mask)
    result = tl.sum(row)  # or tl.max, tl.min
    tl.store(out_ptr + pid, result)
```

### Pattern 3: Multi-input
```python
@triton.jit
def multi_input_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    out = fused_operation(a, b, c)
    tl.store(out_ptr + offsets, out, mask=mask)
```

## Exercises

### Exercise 1: Custom Activation
Implement Swish activation: `swish(x) = x * sigmoid(x)`

### Exercise 2: Batch Normalization
Implement batch norm forward pass.

### Exercise 3: Custom Loss Function
Implement Huber loss kernel.

## Best Practices

1. **Start simple** - Get basic version working first
2. **Test thoroughly** - Compare with reference
3. **Profile early** - Identify bottlenecks
4. **Document parameters** - Explain constexpr values
5. **Handle edge cases** - Test various input sizes

## Next Steps
After mastering custom kernels, move to Module 02: Fused Operations for combining multiple operations.
