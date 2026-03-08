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
# Your answer: Small overhead for predicate evaluation, but avoids branches.
#              Modern GPUs handle predication efficiently.

# Q2: When is predication necessary?
# Your answer: When data size isn't tile-aligned, or for conditional ops
#              like causal masking.

# Q3: How do you create a predicate tensor in CuTe DSL?
# Your answer: Boolean tensor or comparison result (e.g., arange < bound)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
TOTAL_ELEMENTS = 100
TILE_SIZE = 32
NUM_TILES = (TOTAL_ELEMENTS + TILE_SIZE - 1) // TILE_SIZE


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
    """
    # --- Step 1: Create copy atom ---
    copy_atom = cute.Copy_atom(cute.UniversalCopy, cutlass.float32)
    
    # --- Step 2: Copy with predication ---
    count = 0
    for i in range(TILE_SIZE):
        if pred[i]:
            cute.copy(copy_atom, src[i], dst[i])
            count += 1
    
    # --- Step 3: Count copied elements ---
    results[0] = float(count)
    results[1] = dst[0]
    results[2] = dst[3]
    results[3] = dst[4]  # Should be 0 (not copied)
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify predicated copy.
    """
    
    src_torch = torch.arange(TILE_SIZE, dtype=torch.float32, device='cuda')
    src_cute = from_dlpack(src_torch)
    
    dst_torch = torch.zeros(TILE_SIZE, dtype=torch.float32, device='cuda')
    dst_cute = from_dlpack(dst_torch)
    
    valid_elements = 4
    pred_torch = torch.arange(TILE_SIZE, device='cuda') < valid_elements
    pred_cute = from_dlpack(pred_torch)
    
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
        all(dst_cpu[i] == float(i) for i in range(valid_elements)) and
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
