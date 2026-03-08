"""
Module 07 — Predication
Exercise 01 — Predicated Copy

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Manual predication
        if (idx < bound) {
            dst[idx] = src[idx];
        }
  DSL:  # Predicated copy with boolean tensor
        pred = (indices < bound)
        cute.copy(atom, src, dst, pred=pred)
  Key:  Predicates are boolean tensors that control which elements are copied.

WHAT YOU'RE BUILDING:
  A predicated copy that only copies elements within bounds. This is the
  foundation for handling irregular shapes without expensive boundary checks
  in the critical path.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create predicate tensors for conditional copying
  - Use the pred= keyword in cute.copy()
  - Handle boundary conditions efficiently

REQUIRED READING:
  - CUTLASS predication docs: https://nvidia.github.io/cutlass-dsl/cute/predication.html
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the performance cost of predication vs non-predicated copy?
# Your answer:

# Q2: When is predication necessary?
# Your answer:

# Q3: How do you create a predicate tensor in CuTe DSL?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Irregular shape: 100 elements, but tile size is 32
TOTAL_ELEMENTS = 100
TILE_SIZE = 32
NUM_TILES = (TOTAL_ELEMENTS + TILE_SIZE - 1) // TILE_SIZE  # = 4


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_predicated_copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    pred: cute.Tensor,
    results: cute.Tensor,
):
    """
    Predicated copy: only copy elements where pred is True.
    
    FILL IN [MEDIUM]: Use predicated copy to handle irregular shapes.
    
    HINT: cute.copy(atom, src, dst, pred=pred[i]) for element-wise predication
    """
    # --- Step 1: Create copy atom ---
    # TODO: copy_atom = cute.Copy_atom(cute.UniversalCopy, cutlass.float32)
    
    # --- Step 2: Copy with predication ---
    # TODO: for i in range(TILE_SIZE):
    #           if pred[i]:  # Or use vectorized predicated copy
    #               cute.copy(copy_atom, src[i], dst[i])
    
    # --- Step 3: Count copied elements ---
    # Store count in results[0]
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify predicated copy.
    """
    
    # Create source tensor
    src_torch = torch.arange(TILE_SIZE, dtype=torch.float32, device='cuda')
    src_cute = from_dlpack(src_torch)
    
    # Destination
    dst_torch = torch.zeros(TILE_SIZE, dtype=torch.float32, device='cuda')
    dst_cute = from_dlpack(dst_torch)
    
    # Predicate: only copy first 4 elements (simulating irregular boundary)
    valid_elements = 4
    pred_torch = torch.arange(TILE_SIZE, device='cuda') < valid_elements
    pred_cute = from_dlpack(pred_torch)
    
    # Results
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_predicated_copy[1, 1](src_cute, dst_cute, pred_cute, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    dst_cpu = dst_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 07 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  Predicated Copy Configuration:")
    print(f"    Tile size: {TILE_SIZE}")
    print(f"    Valid elements: {valid_elements}")
    print(f"\n  Results:")
    print(f"    Elements copied: {results_cpu[0]:.0f}")
    print(f"    dst[0:4]: {dst_cpu[0:4]}")
    print(f"    dst[4:8]: {dst_cpu[4:8]} (should be zeros)")
    
    passed = (
        results_cpu[0] == valid_elements and
        all(dst_cpu[i] == i for i in range(valid_elements)) and
        all(dst_cpu[i] == 0 for i in range(valid_elements, TILE_SIZE))
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: What is the overhead of predicated vs non-predicated copy?
# C2: When can you avoid predication?
# C3: How does FlashAttention use predication for causal masking?
# C4: Can predication be vectorized?

if __name__ == "__main__":
    run()
