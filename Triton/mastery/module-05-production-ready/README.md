# Module 05: Production-Ready Kernels

## Learning Objectives
1. Write robust, production-quality kernels
2. Implement comprehensive testing
3. Create benchmarks and performance tests
4. Document kernels properly

## Production Code Characteristics

### 1. Robustness
- Input validation
- Error handling
- Edge case handling
- Numerical stability

### 2. Maintainability
- Clear documentation
- Type hints
- Consistent style
- Modular design

### 3. Testability
- Unit tests
- Integration tests
- Performance tests
- Regression tests

### 4. Performance
- Benchmarks
- Profiling support
- Configurable parameters
- Optimization options

## Documentation Standards

### Kernel Docstring Template
```python
@triton.jit
def kernel_name(
    # Pointer arguments
    input_ptr,
    output_ptr,
    # Size arguments
    n_elements,
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,
):
    """
    Brief description of what the kernel does.
    
    Computes: output = f(input)
    
    Args:
        input_ptr: Description of input pointer
        output_ptr: Description of output pointer
        n_elements: Description of size parameter
        BLOCK_SIZE: Description of block size
    
    Returns:
        Description of output (via output_ptr)
    
    Notes:
        - Important implementation details
        - Performance characteristics
        - Limitations or constraints
    """
```

### Wrapper Docstring Template
```python
def kernel_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    High-level wrapper for kernel_name kernel.
    
    Args:
        x: First input tensor
        y: Second input tensor
    
    Returns:
        output: Result tensor
    
    Raises:
        KernelError: If inputs are invalid or kernel fails
    
    Example:
        >>> x = torch.randn(1000, device='cuda')
        >>> y = torch.randn(1000, device='cuda')
        >>> output = kernel_wrapper(x, y)
    """
```

## Input Validation

### Validation Checklist
```python
def validate_inputs(*tensors, **kwargs):
    for tensor in tensors:
        # Type check
        assert isinstance(tensor, torch.Tensor)
        
        # Device check
        assert tensor.is_cuda
        
        # Contiguity check
        assert tensor.is_contiguous()
        
        # Dtype check
        assert tensor.dtype in allowed_dtypes
        
        # Shape check
        assert tensor.shape == expected_shape
    
    # Cross-tensor checks
    assert all(t.device == tensors[0].device for t in tensors)
    assert all(t.dtype == tensors[0].dtype for t in tensors)
```

## Error Handling

### Custom Exception
```python
class TritonKernelError(Exception):
    """Base exception for Triton kernel errors."""
    pass

class InputValidationError(TritonKernelError):
    """Raised when input validation fails."""
    pass

class KernelExecutionError(TritonKernelError):
    """Raised when kernel execution fails."""
    pass
```

### Error Handling Pattern
```python
def safe_kernel_launch(kernel, grid, *args):
    try:
        # Pre-launch validation
        validate_inputs(*args)
        
        # Launch
        kernel[grid](*args)
        
        # Check for CUDA errors
        torch.cuda.synchronize()
        
    except triton.CompilationError as e:
        raise KernelExecutionError(f"Compilation failed: {e}")
    except torch.cuda.CUDAError as e:
        raise KernelExecutionError(f"CUDA error: {e}")
    except Exception as e:
        raise KernelExecutionError(f"Unexpected error: {e}")
```

## Testing Strategy

### Unit Tests
```python
def test_kernel_correctness():
    # Small test case
    x = torch.randn(100, device='cuda')
    y = torch.randn(100, device='cuda')
    
    output = kernel_wrapper(x, y)
    expected = x + y
    
    assert torch.allclose(output, expected, rtol=1e-5)
```

### Edge Case Tests
```python
def test_edge_cases():
    # Empty tensor
    test_kernel(torch.tensor([], device='cuda'))
    
    # Single element
    test_kernel(torch.ones(1, device='cuda'))
    
    # Boundary sizes
    test_kernel(torch.randn(BLOCK_SIZE - 1, device='cuda'))
    test_kernel(torch.randn(BLOCK_SIZE, device='cuda'))
    test_kernel(torch.randn(BLOCK_SIZE + 1, device='cuda'))
```

### Property Tests
```python
def test_properties():
    # Commutativity
    assert allclose(add(a, b), add(b, a))
    
    # Associativity
    assert allclose(add(add(a, b), c), add(a, add(b, c)))
    
    # Identity
    assert allclose(add(x, zeros), x)
```

## Benchmarking

### Benchmark Template
```python
def benchmark_kernel():
    sizes = [10**4, 10**5, 10**6, 10**7]
    
    for n in sizes:
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        
        # Warmup
        for _ in range(10):
            kernel(x, y)
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        for _ in range(100):
            kernel(x, y)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) / 100 * 1000
        print(f"n={n}: {avg_time:.4f} ms")
```

### Performance Metrics
- Execution time (ms)
- Throughput (elements/s, GFLOPS)
- Memory bandwidth (GB/s)
- Comparison with baseline

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Triton Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: gpu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run tests
        run: pytest tests/
      
      - name: Run benchmarks
        run: python benchmarks/
```

## Best Practices

1. **Always validate inputs** - Catch errors early
2. **Use meaningful error messages** - Help debugging
3. **Test thoroughly** - Cover edge cases
4. **Document everything** - Future maintainers
5. **Benchmark regularly** - Catch regressions
6. **Version your kernels** - Track changes
7. **Profile before optimizing** - Find real bottlenecks

## Project Structure

```
project/
├── kernels/
│   ├── __init__.py
│   ├── vector_ops.py
│   └── matmul.py
├── tests/
│   ├── test_vector_ops.py
│   └── test_matmul.py
├── benchmarks/
│   ├── bench_vector_ops.py
│   └── bench_matmul.py
├── utils/
│   ├── validation.py
│   └── testing.py
└── README.md
```

## Final Checklist

Before deploying a kernel:

- [ ] Input validation implemented
- [ ] Error handling with clear messages
- [ ] Unit tests passing
- [ ] Edge cases tested
- [ ] Benchmarks created
- [ ] Documentation complete
- [ ] Type hints added
- [ ] Code reviewed
- [ ] Performance verified
- [ ] Memory usage checked

## Congratulations!

You've completed the Triton Mastery track! You now have the skills to:
- Design and implement custom kernels
- Fuse operations for performance
- Implement attention mechanisms
- Create training kernels
- Write production-ready code

Keep practicing and building!
