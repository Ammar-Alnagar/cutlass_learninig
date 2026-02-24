# Module 04: Memory Debugging

## Learning Objectives
1. Detect out-of-bounds memory access
2. Debug alignment issues
3. Identify memory corruption
4. Use memory sanitizers and tools

## Common Memory Issues

### 1. Out-of-Bounds Access
```python
# PROBLEM: Accessing beyond tensor bounds
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offsets)  # No mask!

# FIX: Always use mask
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 2. Misaligned Access
```python
# Suboptimal: Misaligned starting position
x = tensor[1:]  # Offset by 1
kernel(x)

# Better: Aligned tensor
x = tensor.contiguous()
kernel(x)
```

### 3. Memory Corruption
```python
# PROBLEM: Writing beyond bounds
tl.store(ptr + offsets, value)  # No mask!

# FIX: Use mask on stores too
tl.store(ptr + offsets, value, mask=mask)
```

## Bounds Checking

### Always Use Masks
```python
@triton.jit
def safe_kernel(ptr, output, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask
    mask = offsets < n
    
    # Load with mask
    x = tl.load(ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    output_val = x * 2.0
    
    # Store with mask
    tl.store(output + offsets, output_val, mask=mask)
```

### Test Boundary Cases
```python
test_sizes = [
    1,              # Single element
    BLOCK_SIZE - 1, # Just under
    BLOCK_SIZE,     # Exact
    BLOCK_SIZE + 1, # Just over
]
```

## Alignment Issues

### What is Alignment?
Memory is aligned when its address is a multiple of the data size.

```
Aligned:   Address 0, 4, 8, 12, ...  (for float32)
Misaligned: Address 1, 5, 9, 13, ...
```

### Impact
- Misaligned access may be slower
- Triton handles it automatically
- But aligned is always better

### Best Practices
```python
# Good: Contiguous tensor
x = tensor.contiguous()

# Good: Proper slicing
x = tensor[:n]  # Keeps alignment

# Avoid: Misaligned slicing
x = tensor[1:n+1]  # May be misaligned
```

## Corruption Detection

### Guard Values
```python
# Add sentinel values around data
x = torch.cat([
    torch.full((10,), -99999.0),  # Guard before
    actual_data,
    torch.full((10,), -99999.0),  # Guard after
])

# After kernel, check guards
if (x[:10] != -99999.0).any():
    print("Memory corruption detected!")
```

### Output Validation
```python
def validate_output(output, expected):
    # Check shape
    assert output.shape == expected.shape
    
    # Check for special values
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Check values
    assert torch.allclose(output, expected)
```

## Memory Access Patterns

### Coalesced (Good)
```
Thread 0 -> Address 0
Thread 1 -> Address 1
Thread 2 -> Address 2
...
```

### Strided (Bad)
```
Thread 0 -> Address 0
Thread 1 -> Address 100
Thread 2 -> Address 200
...
```

### 2D Access Pattern
```python
# Good: Row-major, coalesced
offs_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
data = tl.load(ptr + row_idx * stride + offs_col)

# Bad: Column-major, strided
offs_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
data = tl.load(ptr + offs_row * stride + col_idx)
```

## Debugging Tools

### 1. cuda-memcheck
```bash
cuda-memcheck python kernel.py
```

### 2. Nsight Compute
```bash
ncu --metrics l1tex__data_pipe_lookup_mem_accesses python kernel.py
```

### 3. Custom Debugger
```python
class MemoryDebugger:
    def check_tensor(self, tensor):
        assert tensor.is_cuda
        assert tensor.is_contiguous()
        assert not torch.isnan(tensor).any()
```

## Exercises

### Exercise 1: Add Bounds Checking
Add proper masks to an unmasked kernel.

### Exercise 2: Test Boundaries
Create test cases for boundary conditions.

### Exercise 3: Detect Corruption
Implement guard value checking.

## Memory Debugging Checklist

- [ ] All loads use masks
- [ ] All stores use masks
- [ ] Tensors are contiguous
- [ ] Tensors are on CUDA
- [ ] No NaN/Inf in output
- [ ] Output shape is correct
- [ ] Boundary cases tested
- [ ] Access patterns are coalesced

## Next Steps
You've completed the Debugging track!

Next: Mastery track for advanced Triton techniques.
