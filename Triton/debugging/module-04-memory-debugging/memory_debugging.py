"""
Module 04: Memory Debugging

This module teaches how to debug memory-related issues in Triton kernels
including out-of-bounds access, misaligned access, and memory corruption.

LEARNING OBJECTIVES:
1. Detect out-of-bounds memory access
2. Debug alignment issues
3. Identify memory corruption
4. Use memory sanitizers and tools
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# MEMORY DEBUGGING TECHNIQUE 1: Bounds Checking
# ============================================================================

@triton.jit
def safe_load_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr = True,
):
    """
    Kernel with explicit bounds checking.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Always use mask for safety
    mask = offsets < n_elements
    
    if CHECK_BOUNDS:
        # Extra safety: clamp indices
        safe_offsets = tl.maximum(0, tl.minimum(offsets, n_elements - 1))
        x = tl.load(x_ptr + safe_offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    output = x * 2.0
    tl.store(output_ptr + offsets, output, mask=mask)


def check_memory_bounds():
    """
    Test kernel with various boundary conditions.
    """
    print("Memory Bounds Checking")
    print("=" * 60)
    
    BLOCK_SIZE = 128
    test_cases = [
        (1, "Single element"),
        (127, "Just under block size"),
        (128, "Exactly block size"),
        (129, "Just over block size"),
        (1000, "Irregular size"),
    ]
    
    for n, description in test_cases:
        x = torch.randn(n, device="cuda")
        output = torch.empty(n, device="cuda")
        
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        
        try:
            safe_load_kernel[grid](x, output, n, BLOCK_SIZE, CHECK_BOUNDS=True)
            
            expected = x * 2.0
            if torch.allclose(output, expected, rtol=1e-5):
                print(f"✓ {description} (n={n}): PASS")
            else:
                print(f"✗ {description} (n={n}): FAIL - Values mismatch")
                max_err = (output - expected).abs().max().item()
                print(f"    Max error: {max_err}")
        except Exception as e:
            print(f"✗ {description} (n={n}): ERROR - {e}")


# ============================================================================
# MEMORY DEBUGGING TECHNIQUE 2: Alignment Checking
# ============================================================================

@triton.jit
def aligned_access_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that checks for aligned memory access.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x + 1.0
    tl.store(output_ptr + offsets, output, mask=mask)


def check_memory_alignment():
    """
    Test kernel with misaligned tensors.
    """
    print("\nMemory Alignment Checking")
    print("=" * 60)
    
    # Create tensors with different alignments
    base = torch.randn(10000, device="cuda")
    
    alignments = [
        (0, "Aligned (offset 0)"),
        (1, "Misaligned (offset 1)"),
        (2, "Misaligned (offset 2)"),
        (3, "Misaligned (offset 3)"),
    ]
    
    BLOCK_SIZE = 256
    
    for offset, description in alignments:
        x = base[offset:offset + 1024].clone()
        output = torch.empty_like(x)
        
        grid = (triton.cdiv(1024, BLOCK_SIZE),)
        
        try:
            aligned_access_kernel[grid](x, output, 1024, BLOCK_SIZE)
            
            expected = x + 1.0
            if torch.allclose(output, expected, rtol=1e-5):
                print(f"✓ {description}: PASS")
            else:
                print(f"✗ {description}: FAIL")
        except Exception as e:
            print(f"✗ {description}: ERROR - {e}")
    
    print("\nNote: Triton handles misaligned access automatically,")
    print("but aligned access is generally faster.")


# ============================================================================
# MEMORY DEBUGGING TECHNIQUE 3: Corruption Detection
# ============================================================================

def detect_memory_corruption():
    """
    Detect memory corruption by checking guard values.
    """
    print("\nMemory Corruption Detection")
    print("=" * 60)
    
    @triton.jit
    def kernel_with_guards(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=-99999.0)  # Sentinel
        output = x * 2.0
        tl.store(output_ptr + offsets, output, mask=mask)
    
    # Test with guard values
    n = 100
    BLOCK_SIZE = 32
    
    # Create tensor with guard values
    x = torch.randn(n, device="cuda")
    output = torch.empty(n, device="cuda")
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel_with_guards[grid](x, output, n, BLOCK_SIZE)
    
    # Check for corruption
    expected = x * 2.0
    
    # Check for sentinel values in output (indicates bug)
    has_sentinel = (output == -99999.0 * 2.0).any()
    
    if has_sentinel:
        print("✗ WARNING: Sentinel values detected - possible corruption!")
    else:
        print("✓ No corruption detected")
    
    if torch.allclose(output, expected, rtol=1e-5):
        print("✓ Output values correct")
    else:
        print("✗ Output values incorrect")


# ============================================================================
# MEMORY DEBUGGING TECHNIQUE 4: Pattern Analysis
# ============================================================================

def analyze_memory_access_pattern():
    """
    Analyze memory access patterns for efficiency.
    """
    print("\nMemory Access Pattern Analysis")
    print("=" * 60)
    
    patterns = [
        ("Coalesced (sequential)", True, "Optimal"),
        ("Strided access", False, "Suboptimal"),
        ("Random access", False, "Poor"),
    ]
    
    print(f"{'Pattern':<25} {'Coalesced':<15} {'Recommendation':<20}")
    print("-" * 60)
    
    for pattern, coalesced, recommendation in patterns:
        status = "Yes" if coalesced else "No"
        print(f"{pattern:<25} {status:<15} {recommendation:<20}")
    
    print("\nCoalesced Access Pattern:")
    print("  Thread 0 -> Address A")
    print("  Thread 1 -> Address A+1")
    print("  Thread 2 -> Address A+2")
    print("  ...")
    
    print("\nStrided Access Pattern (avoid):")
    print("  Thread 0 -> Address A")
    print("  Thread 1 -> Address A+stride")
    print("  Thread 2 -> Address A+2*stride")
    print("  ...")


# ============================================================================
# MEMORY DEBUGGING TECHNIQUE 5: Debug Utilities
# ============================================================================

class MemoryDebugger:
    """
    Utility class for memory debugging.
    """
    
    def __init__(self):
        self.errors = []
    
    def check_tensor(self, tensor: torch.Tensor, name: str = "tensor"):
        """Check tensor properties."""
        issues = []
        
        if not tensor.is_cuda:
            issues.append(f"{name}: Not on CUDA device")
        
        if not tensor.is_contiguous():
            issues.append(f"{name}: Not contiguous")
        
        if tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            issues.append(f"{name}: Unusual dtype {tensor.dtype}")
        
        if torch.isnan(tensor).any():
            issues.append(f"{name}: Contains NaN")
        
        if torch.isinf(tensor).any():
            issues.append(f"{name}: Contains Inf")
        
        if issues:
            self.errors.extend(issues)
            return False
        return True
    
    def check_kernel_output(self, output: torch.Tensor, expected: torch.Tensor, 
                           atol: float = 1e-5, rtol: float = 1e-5):
        """Check kernel output against expected values."""
        if output.shape != expected.shape:
            self.errors.append(f"Shape mismatch: {output.shape} vs {expected.shape}")
            return False
        
        if not torch.allclose(output, expected, atol=atol, rtol=rtol):
            max_err = (output - expected).abs().max().item()
            self.errors.append(f"Values mismatch: max error = {max_err}")
            return False
        
        return True
    
    def report(self):
        """Print error report."""
        if not self.errors:
            print("✓ No memory issues detected")
            return True
        
        print("Memory Issues Found:")
        for error in self.errors:
            print(f"  - {error}")
        return False
    
    def clear(self):
        """Clear error list."""
        self.errors = []


def debug_kernel_launch(kernel_func, grid, args, expected_output=None):
    """
    Debug a kernel launch with comprehensive checks.
    """
    debugger = MemoryDebugger()
    
    # Check inputs
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            debugger.check_tensor(arg, f"arg[{i}]")
    
    # Launch kernel
    try:
        kernel_func[grid](*args)
        torch.cuda.synchronize()
    except Exception as e:
        debugger.errors.append(f"Kernel launch failed: {e}")
        return debugger.report()
    
    # Check output if expected provided
    if expected_output is not None:
        output = args[-1] if isinstance(args[-1], torch.Tensor) else None
        if output is not None:
            debugger.check_kernel_output(output, expected_output)
    
    return debugger.report()


if __name__ == "__main__":
    print("Memory Debugging Module")
    print("=" * 60)
    
    # Run all checks
    check_memory_bounds()
    check_memory_alignment()
    detect_memory_corruption()
    analyze_memory_access_pattern()
    
    # Test debugger utility
    print("\n" + "=" * 60)
    print("Testing Memory Debugger Utility")
    print("-" * 60)
    
    n = 1000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.empty(n, device="cuda")
    
    @triton.jit
    def test_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    
    grid = (triton.cdiv(n, 256),)
    test_kernel[grid](x, y, output, n, 256)
    
    expected = x + y
    debug_kernel_launch(test_kernel, grid, (x, y, output, n, 256), expected)
    
    print("\n" + "=" * 60)
    print("MEMORY DEBUGGING CHECKLIST:")
    print("□ Check tensor is on CUDA device")
    print("□ Check tensor is contiguous")
    print("□ Use masks for boundary elements")
    print("□ Verify output shape matches expected")
    print("□ Check for NaN/Inf in output")
    print("□ Use guard values for corruption detection")
    print("□ Prefer coalesced memory access")
    print("□ Test with various input sizes")
    print("\nMEMORY DEBUGGING TOOLS:")
    print("  - cuda-memcheck for CUDA error detection")
    print("  - Nsight Compute for memory analysis")
    print("  - Custom guard values for corruption detection")
