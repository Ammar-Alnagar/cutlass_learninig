"""
Module 01: Basic Debugging Techniques

This module teaches fundamental debugging techniques for Triton kernels.
Learn how to identify and fix common kernel errors.

LEARNING OBJECTIVES:
1. Understand common Triton error types
2. Use print debugging in kernels
3. Validate kernel outputs
4. Debug indexing and boundary issues
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# DEBUGGING TECHNIQUE 1: Print Debugging
# ============================================================================

@triton.jit
def vector_add_debug(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector addition with debug prints.
    
    DEBUGGING WITH tl.static_print:
    - Prints values at compile time (constants)
    - Use tl.device_print for runtime values
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Debug: Print program ID and block start
    tl.static_print("Program ID: ", pid, " Block start: ", block_start)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Debug: Print loaded values (first element of first block)
    tl.device_print("x[0] = ", x[0], " y[0] = ", y[0], pid == 0)
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================================
# DEBUGGING TECHNIQUE 2: Output Validation
# ============================================================================

def validate_kernel_output(kernel_func, inputs, expected_output, atol=1e-5, rtol=1e-5):
    """
    Helper function to validate kernel output against expected result.
    
    Returns:
        tuple: (is_valid, max_error, mean_error)
    """
    actual_output = kernel_func(*inputs)
    
    # Check for NaN/Inf
    has_nan = torch.isnan(actual_output).any()
    has_inf = torch.isinf(actual_output).any()
    
    if has_nan:
        print("ERROR: Output contains NaN!")
        return False, float('inf'), float('inf')
    
    if has_inf:
        print("ERROR: Output contains Inf!")
        return False, float('inf'), float('inf')
    
    # Check shape
    if actual_output.shape != expected_output.shape:
        print(f"ERROR: Shape mismatch! Expected {expected_output.shape}, got {actual_output.shape}")
        return False, float('inf'), float('inf')
    
    # Check values
    max_error = (actual_output - expected_output).abs().max().item()
    mean_error = (actual_output - expected_output).abs().mean().item()
    
    is_valid = torch.allclose(actual_output, expected_output, atol=atol, rtol=rtol)
    
    if not is_valid:
        print(f"WARNING: Values don't match!")
        print(f"  Max error: {max_error}")
        print(f"  Mean error: {mean_error}")
        
        # Find worst mismatch
        diff = (actual_output - expected_output).abs()
        worst_idx = diff.argmax().item()
        print(f"  Worst mismatch at index {worst_idx}:")
        print(f"    Expected: {expected_output.flatten()[worst_idx]}")
        print(f"    Actual:   {actual_output.flatten()[worst_idx]}")
    
    return is_valid, max_error, mean_error


# ============================================================================
# DEBUGGING TECHNIQUE 3: Boundary Testing
# ============================================================================

@triton.jit
def boundary_test_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that demonstrates boundary handling.
    
    COMMON BOUNDARY ISSUES:
    1. Missing mask on load
    2. Missing mask on store
    3. Incorrect 'other' value for masked loads
    4. Off-by-one in boundary calculation
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # CORRECT: Use mask for boundary handling
    mask = offsets < n_elements
    
    # Load with mask and safe 'other' value
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    output = x * 2.0
    
    # Store with mask
    tl.store(output_ptr + offsets, output, mask=mask)


def test_boundary_conditions():
    """
    Test kernel with various boundary conditions.
    """
    print("Testing Boundary Conditions")
    print("=" * 50)
    
    BLOCK_SIZE = 128
    
    # Test cases: sizes that don't divide evenly by BLOCK_SIZE
    test_sizes = [
        1,              # Single element
        127,            # Just under block size
        128,            // Exactly block size
        129,            // Just over block size
        255,            // Two blocks - 1
        256,            // Exactly two blocks
        1000,           // Irregular size
    ]
    
    for n in test_sizes:
        x = torch.randn(n, device="cuda")
        output = torch.empty(n, device="cuda")
        
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        boundary_test_kernel[grid](x, output, n, BLOCK_SIZE)
        
        expected = x * 2.0
        is_valid = torch.allclose(output, expected, rtol=1e-5)
        
        status = "✓" if is_valid else "✗"
        print(f"  {status} n={n:<5} num_blocks={grid[0]}")
        
        if not is_valid:
            print(f"      Max error: {(output - expected).abs().max().item()}")


# ============================================================================
# DEBUGGING TECHNIQUE 4: Common Error Patterns
# ============================================================================

def demonstrate_common_errors():
    """
    Demonstrate and explain common Triton errors.
    """
    print("\nCommon Error Patterns")
    print("=" * 50)
    
    # Error 1: Missing mask
    print("\n1. Missing mask on load/store:")
    print("   PROBLEM: Accessing out-of-bounds memory")
    print("   FIX: Always use mask for boundary elements")
    print("   Example: tl.load(ptr + offsets, mask=mask, other=0.0)")
    
    # Error 2: Wrong dtype
    print("\n2. dtype mismatch:")
    print("   PROBLEM: Kernel expects different dtype than provided")
    print("   FIX: Ensure input tensors match kernel expectations")
    print("   Example: x = x.to(torch.float16)")
    
    # Error 3: Incorrect grid size
    print("\n3. Incorrect grid size:")
    print("   PROBLEM: Not covering all elements")
    print("   FIX: Use triton.cdiv for grid calculation")
    print("   Example: grid = (triton.cdiv(n, BLOCK_SIZE),)")
    
    # Error 4: Missing constexpr
    print("\n4. Missing tl.constexpr:")
    print("   PROBLEM: Compile-time constants not marked")
    print("   FIX: Add tl.constexpr to block size parameters")
    print("   Example: BLOCK_SIZE: tl.constexpr")
    
    # Error 5: Wrong program_id axis
    print("\n5. Wrong program_id axis:")
    print("   PROBLEM: Using wrong axis for grid dimension")
    print("   FIX: Match axis to grid dimension")
    print("   Example: tl.program_id(axis=0) for 1D grid")


def debug_vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vector addition with debugging enabled.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 256
    
    output = torch.empty(n_elements, device="cuda")
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Enable debug mode
    print(f"Launching kernel with grid={grid}, BLOCK_SIZE={BLOCK_SIZE}")
    print(f"Total elements: {n_elements}")
    
    vector_add_debug[grid](x, y, output, n_elements, BLOCK_SIZE)
    
    return output


if __name__ == "__main__":
    print("Basic Debugging Techniques Module")
    print("=" * 60)
    
    # Test boundary conditions
    test_boundary_conditions()
    
    # Demonstrate common errors
    demonstrate_common_errors()
    
    # Test debug kernel
    print("\n" + "=" * 60)
    print("Testing Debug Vector Add")
    
    n = 1000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    
    output = debug_vector_add(x, y)
    expected = x + y
    
    is_valid, max_err, mean_err = validate_kernel_output(
        lambda: debug_vector_add(x, y),
        (),
        expected
    )
    
    print(f"\nValidation result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    print(f"Max error: {max_err}")
    print(f"Mean error: {mean_err}")
    
    print("\n" + "=" * 60)
    print("DEBUGGING CHECKLIST:")
    print("□ Check for NaN/Inf in output")
    print("□ Verify output shape matches expected")
    print("□ Test boundary conditions")
    print("□ Use tl.device_print for runtime values")
    print("□ Use tl.static_print for compile-time values")
    print("□ Validate against reference implementation")
    print("□ Check mask usage on loads and stores")
    print("□ Verify grid size covers all elements")
