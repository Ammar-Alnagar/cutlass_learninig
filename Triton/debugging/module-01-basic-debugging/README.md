# Module 01: Basic Debugging Techniques

## Learning Objectives
1. Understand common Triton error types
2. Use print debugging in kernels
3. Validate kernel outputs
4. Debug indexing and boundary issues

## Common Error Types

### 1. Compilation Errors
These occur when Triton compiles your kernel:

```python
# ERROR: Missing tl.constexpr
def kernel(..., BLOCK_SIZE):  # Should be: BLOCK_SIZE: tl.constexpr

# ERROR: Invalid operation in JIT kernel
result = some_python_function(x)  # Can't call Python functions

# ERROR: Type mismatch
x = tl.load(...)  # Returns float32
y = tl.load(...).to(torch.int8)  # Incompatible types for operation
```

### 2. Runtime Errors
These occur when launching or running the kernel:

```python
# ERROR: Out of bounds access
tl.load(ptr + offsets)  # Without mask for boundary elements

# ERROR: Wrong grid size
grid = (n // BLOCK_SIZE,)  # Misses remainder elements
# FIX: grid = (triton.cdiv(n, BLOCK_SIZE),)

# ERROR: Device mismatch
x = torch.randn(n)  # CPU tensor
kernel(x.cuda())    # Kernel expects CUDA
```

### 3. Logic Errors
Kernel runs but produces wrong results:

```python
# ERROR: Wrong indexing
pid = tl.program_id(axis=1)  # Should be axis=0 for 1D grid

# ERROR: Missing mask
tl.store(ptr + offsets, value)  # Without mask for boundaries

# ERROR: Wrong accumulation
accumulator = tl.dot(a, b)  # Should be: accumulator = tl.dot(a, b, accumulator)
```

## Debugging Techniques

### 1. Print Debugging

#### tl.static_print (Compile-time)
```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    tl.static_print("BLOCK_SIZE = ", BLOCK_SIZE)
```

#### tl.device_print (Runtime)
```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    tl.device_print("pid = ", pid, pid == 0)  # Conditional print
```

### 2. Output Validation

```python
def validate_output(actual, expected, atol=1e-5, rtol=1e-5):
    # Check for NaN/Inf
    assert not torch.isnan(actual).any(), "Output contains NaN"
    assert not torch.isinf(actual).any(), "Output contains Inf"
    
    # Check shape
    assert actual.shape == expected.shape, f"Shape mismatch"
    
    # Check values
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), "Values don't match"
```

### 3. Boundary Testing

Test edge cases:
```python
test_sizes = [
    1,              # Single element
    BLOCK_SIZE - 1, # Just under block size
    BLOCK_SIZE,     // Exactly block size
    BLOCK_SIZE + 1, // Just over block size
]
```

### 4. Incremental Development

1. Start with a simple kernel
2. Test with small inputs
3. Add features one at a time
4. Test after each change

## Debugging Workflow

```
1. Read the error message carefully
   └─→ Identify error type (compile/runtime/logic)

2. For compilation errors:
   └─→ Check tl.constexpr annotations
   └─→ Remove Python function calls
   └─→ Verify type compatibility

3. For runtime errors:
   └─→ Check tensor devices (all CUDA?)
   └─→ Verify grid size calculation
   └─→ Check mask usage

4. For logic errors:
   └─→ Add tl.device_print statements
   └─→ Compare with reference implementation
   └─→ Test with known inputs
```

## Exercises

### Exercise 1: Fix the Broken Kernel
The kernel has intentional bugs. Find and fix them.

### Exercise 2: Add Debug Prints
Add debug prints to understand data flow in a kernel.

### Exercise 3: Create Test Cases
Write comprehensive tests for a kernel including edge cases.

## Debugging Tools

### 1. Triton Error Messages
```
triton.errors.CompilationError: Error at line 15:
  x = y + z
      ^
Type mismatch: float32 and int32
```

### 2. PyTorch Assertions
```python
assert x.is_cuda, "Tensor must be on CUDA"
assert x.dim() == 2, "Expected 2D tensor"
```

### 3. Numerical Checks
```python
torch.isnan(output).any()
torch.isinf(output).any()
(output - expected).abs().max()
```

## Common Fixes

| Problem | Fix |
|---------|-----|
| Compilation error | Add `tl.constexpr` |
| Out of bounds | Add mask to load/store |
| Wrong results | Check indexing math |
| NaN output | Check for division by zero |
| Shape mismatch | Verify grid calculation |

## Next Steps
After mastering basic debugging, move to Module 02: Error Handling for advanced techniques.
