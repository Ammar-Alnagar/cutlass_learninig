# Module 02: Error Handling and Recovery

## Learning Objectives
1. Implement robust input validation
2. Handle edge cases gracefully
3. Create fallback mechanisms
4. Debug complex kernel failures

## Input Validation

### Why Validate?
- Catch errors early (before kernel launch)
- Provide meaningful error messages
- Prevent undefined behavior
- Make debugging easier

### Validation Checklist

```python
def validate_inputs(*tensors):
    for tensor in tensors:
        # 1. Type check
        assert isinstance(tensor, torch.Tensor)
        
        # 2. Device check
        assert tensor.is_cuda, "Must be on CUDA"
        
        # 3. Contiguity check
        assert tensor.is_contiguous(), "Must be contiguous"
        
        # 4. Dtype check
        assert tensor.dtype in [torch.float16, torch.float32]
        
        # 5. Shape check
        assert tensor.dim() >= 1, "Must have at least 1 dimension"
```

## Safe Kernel Launch Pattern

```python
class KernelError(Exception):
    pass

def safe_launch(kernel, grid, *args):
    try:
        # Pre-launch validation
        validate_inputs(*args)
        
        # Launch
        kernel[grid](*args)
        
        # Post-launch sync (catches CUDA errors)
        torch.cuda.synchronize()
        
        return True
    except Exception as e:
        raise KernelError(f"Kernel failed: {e}")
```

## Fallback Strategies

### 1. Size-Based Fallback
```python
def matmul(a, b):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    
    # Small matrices: use PyTorch
    if M < 16 or N < 16 or K < 16:
        return a @ b
    
    # Large matrices: use Triton
    return triton_matmul(a, b)
```

### 2. Try-Except Fallback
```python
def robust_kernel(x):
    try:
        return triton_kernel(x)
    except Exception:
        return pytorch_fallback(x)
```

### 3. Adaptive Configuration
```python
def adaptive_kernel(x):
    n = x.numel()
    
    if n < 1000:
        BLOCK_SIZE = 64
    elif n < 100000:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    return kernel(x, BLOCK_SIZE)
```

## Debug Mode Pattern

```python
DEBUG = False

def set_debug(enabled):
    global DEBUG
    DEBUG = enabled

@triton.jit
def kernel(..., DEBUG_MODE: tl.constexpr = False):
    # Normal computation
    output = compute(x)
    
    if DEBUG_MODE:
        # Extra checks in debug mode
        tl.device_print("output = ", output)
    
    return output

# Usage
set_debug(True)
kernel(x, DEBUG_MODE=DEBUG)
```

## Common Error Patterns

### 1. Dimension Mismatch
```python
# ERROR
a = torch.randn(10, 20)
b = torch.randn(30, 40)
matmul(a, b)  # Will fail!

# FIX
assert a.shape[1] == b.shape[0], "Incompatible dimensions"
```

### 2. dtype Mismatch
```python
# ERROR
a = torch.randn(10, dtype=torch.float32)
b = torch.randn(10, dtype=torch.float16)
kernel(a, b)  # May fail!

# FIX
assert a.dtype == b.dtype, "dtype mismatch"
```

### 3. Device Mismatch
```python
# ERROR
a = torch.randn(10).cuda()
b = torch.randn(10)  # CPU!
kernel(a, b)  # Will fail!

# FIX
assert a.device == b.device, "Device mismatch"
```

## Error Recovery Strategies

### 1. Graceful Degradation
```python
try:
    result = optimized_kernel(x)
except:
    result = simple_kernel(x)  # Slower but reliable
```

### 2. Retry with Different Config
```python
for config in configs:
    try:
        return kernel(x, **config)
    except:
        continue  # Try next config
```

### 3. Partial Results
```python
def process_large_input(x):
    results = []
    for chunk in chunks(x):
        try:
            results.append(kernel(chunk))
        except:
            results.append(zero_like(chunk))  # Partial failure OK
    return concat(results)
```

## Exercises

### Exercise 1: Add Input Validation
Add comprehensive validation to the softmax kernel.

### Exercise 2: Implement Fallback
Create a fallback mechanism for the layer norm kernel.

### Exercise 3: Debug Mode
Add debug mode to an existing kernel with sanity checks.

## Best Practices

1. **Validate early** - Check inputs before any computation
2. **Fail fast** - Don't continue with invalid state
3. **Clear messages** - Explain what went wrong and why
4. **Test error paths** - Verify fallbacks work correctly
5. **Log failures** - Record errors for debugging

## Next Steps
After mastering error handling, move to Module 03: Performance Profiling for optimization debugging.
