"""
Module 02: Error Handling and Recovery

This module teaches advanced error handling techniques for Triton kernels.
Learn how to gracefully handle errors and implement recovery strategies.

LEARNING OBJECTIVES:
1. Implement robust error checking
2. Handle edge cases gracefully
3. Create fallback mechanisms
4. Debug complex kernel failures
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# ERROR HANDLING TECHNIQUE 1: Input Validation
# ============================================================================

def validate_inputs(*tensors, **kwargs):
    """
    Comprehensive input validation for Triton kernels.
    """
    errors = []
    
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"Argument {i}: Expected torch.Tensor, got {type(tensor)}")
            continue
        
        if not tensor.is_cuda:
            errors.append(f"Argument {i}: Tensor must be on CUDA device")
        
        if not tensor.is_contiguous():
            errors.append(f"Argument {i}: Tensor must be contiguous (call .contiguous())")
    
    # Check dtype consistency
    if len(tensors) > 1:
        dtypes = [t.dtype for t in tensors if isinstance(t, torch.Tensor)]
        if len(set(dtypes)) > 1:
            errors.append(f"Dtype mismatch: Found {set(dtypes)}")
    
    # Check device consistency
    if len(tensors) > 1:
        devices = [t.device for t in tensors if isinstance(t, torch.Tensor)]
        if len(set(devices)) > 1:
            errors.append(f"Device mismatch: Found {set(devices)}")
    
    # Custom validations from kwargs
    if 'min_size' in kwargs:
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor) and tensor.numel() < kwargs['min_size']:
                errors.append(f"Argument {i}: Size {tensor.numel()} < min_size {kwargs['min_size']}")
    
    if 'required_shape' in kwargs:
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor) and tensor.shape != kwargs['required_shape']:
                errors.append(f"Argument {i}: Shape {tensor.shape} != required {kwargs['required_shape']}")
    
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# ============================================================================
# ERROR HANDLING TECHNIQUE 2: Safe Kernel Wrapper
# ============================================================================

class KernelError(Exception):
    """Custom exception for kernel errors."""
    pass


def safe_kernel_launch(kernel_func, grid, *args, **kwargs):
    """
    Safely launch a Triton kernel with error handling.
    """
    try:
        # Pre-launch checks
        if not kernel_func:
            raise KernelError("Kernel function is None")
        
        if not grid or len(grid) == 0:
            raise KernelError("Invalid grid specification")
        
        # Launch kernel
        kernel_func[grid](*args, **kwargs)
        
        # Check for CUDA errors
        torch.cuda.synchronize()
        
        return True
        
    except triton.CompilationError as e:
        raise KernelError(f"Compilation failed: {e}")
    except triton.OutOfResources as e:
        raise KernelError(f"Out of resources: {e}")
    except torch.cuda.CUDAError as e:
        raise KernelError(f"CUDA error: {e}")
    except Exception as e:
        raise KernelError(f"Unexpected error: {e}")


# ============================================================================
# ERROR HANDLING TECHNIQUE 3: Fallback Implementations
# ============================================================================

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Standard matrix multiplication kernel."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        a_block = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        
        b_block = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0
        )
        
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator,
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    )


def matmul_with_fallback(a: torch.Tensor, b: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """
    Matrix multiplication with fallback to PyTorch if Triton fails.
    """
    try:
        if use_triton:
            # Validate inputs
            validate_inputs(a, b, min_size=1)
            
            M, K = a.shape
            K2, N = b.shape
            
            if K != K2:
                raise KernelError(f"Dimension mismatch: {K} != {K2}")
            
            # Small matrix fallback
            if M < 16 or N < 16 or K < 16:
                print("Using PyTorch for small matrix")
                return a @ b
            
            BLOCK_SIZE_M = 64
            BLOCK_SIZE_N = 64
            BLOCK_SIZE_K = 32
            
            c = torch.empty((M, N), device=a.device, dtype=torch.float32)
            grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
            
            safe_kernel_launch(
                matmul_kernel, grid,
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
            )
            
            return c
            
    except KernelError as e:
        print(f"Triton failed: {e}")
        print("Falling back to PyTorch")
    
    # Fallback to PyTorch
    return a @ b


# ============================================================================
# ERROR HANDLING TECHNIQUE 4: Debug Mode
# ============================================================================

DEBUG_MODE = False

def set_debug_mode(enabled: bool):
    """Enable or disable debug mode."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


@triton.jit
def debug_vector_add(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    """
    Vector addition with optional debug mode.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    if DEBUG:
        # Debug: Check for NaN/Inf in inputs
        x_is_nan = tl.math.isnan(x)
        y_is_nan = tl.math.isnan(y)
        tl.device_print("x has NaN: ", x_is_nan, pid == 0)
        tl.device_print("y has NaN: ", y_is_nan, pid == 0)
    
    output = x + y
    
    if DEBUG:
        # Debug: Check output
        output_is_nan = tl.math.isnan(output)
        tl.device_print("output has NaN: ", output_is_nan, pid == 0)
    
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add_with_debug(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vector addition with debug mode support.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 256
    
    output = torch.empty(n_elements, device="cuda")
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    debug_vector_add[grid](x, y, output, n_elements, BLOCK_SIZE, DEBUG=DEBUG_MODE)
    
    if DEBUG_MODE:
        # Additional Python-side checks
        if torch.isnan(output).any():
            print("WARNING: Output contains NaN!")
        if torch.isinf(output).any():
            print("WARNING: Output contains Inf!")
    
    return output


# ============================================================================
# ERROR HANDLING TECHNIQUE 5: Error Recovery Strategies
# ============================================================================

def adaptive_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with adaptive block sizes based on input size.
    """
    M, K = a.shape
    K2, N = b.shape
    
    # Choose block sizes based on problem size
    if M <= 64 and N <= 64:
        # Small: Use smaller blocks
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    elif M <= 256 and N <= 256:
        # Medium: Use medium blocks
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    else:
        # Large: Use larger blocks
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    try:
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        return c
    except Exception as e:
        print(f"Adaptive matmul failed: {e}")
        return a @ b


if __name__ == "__main__":
    print("Error Handling and Recovery Module")
    print("=" * 60)
    
    # Test input validation
    print("\n1. Testing Input Validation")
    print("-" * 40)
    
    try:
        x_cpu = torch.randn(100)
        validate_inputs(x_cpu)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test safe kernel launch
    print("\n2. Testing Safe Kernel Launch")
    print("-" * 40)
    
    n = 1000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.empty(n, device="cuda")
    
    try:
        safe_kernel_launch(
            debug_vector_add,
            (triton.cdiv(n, 256),),
            x, y, output, n, 256
        )
        print("✓ Kernel launched successfully")
    except KernelError as e:
        print(f"✗ Kernel error: {e}")
    
    # Test fallback
    print("\n3. Testing Fallback Mechanism")
    print("-" * 40)
    
    a = torch.randn(10, 10, device="cuda", dtype=torch.float16)
    b = torch.randn(10, 10, device="cuda", dtype=torch.float16)
    
    result = matmul_with_fallback(a, b)
    expected = a @ b
    
    if torch.allclose(result, expected, rtol=1e-2):
        print("✓ Fallback matmul works correctly")
    
    # Test adaptive block sizes
    print("\n4. Testing Adaptive Block Sizes")
    print("-" * 40)
    
    for size in [32, 128, 512]:
        a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        
        result = adaptive_matmul(a, b)
        expected = a @ b
        
        status = "✓" if torch.allclose(result, expected, rtol=1e-2) else "✗"
        print(f"  {status} Size {size}x{size}")
    
    print("\n" + "=" * 60)
    print("ERROR HANDLING CHECKLIST:")
    print("□ Validate all inputs before kernel launch")
    print("□ Check for CUDA/device consistency")
    print("□ Implement fallback for edge cases")
    print("□ Use try/except for kernel launches")
    print("□ Provide meaningful error messages")
    print("□ Test error paths thoroughly")
    print("□ Consider adaptive strategies for different inputs")
