"""
Module 08 — MMA Atom Internals
Exercise 02 — Fragment Ownership

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Which thread owns which fragment elements?
        auto fragment = thr_mma.partition_fragment(...);
        // fragment contains the elements this thread computes
  DSL:  fragment = cute.make_rmem_tensor_like(tensor)
        # Each thread creates its own fragment
  Key:  Fragment ownership determines which thread computes which output.

WHAT YOU'RE BUILDING:
  A fragment ownership analyzer that shows which thread is responsible for
  computing each element of the output matrix. This is critical for understanding
  tensor core data flow and debugging race conditions.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Analyze fragment ownership for MMA operations
  - Understand output distribution across threads
  - Debug tensor core race conditions

REQUIRED READING:
  - CUTLASS fragment docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_mma.html#fragments
  - FlashAttention-3 warp specialization: https://arxiv.org/abs/2310.03748
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: In a (2, 4) atom layout, how many threads share the output?
# Your answer:

# Q2: For a 16x16x16 MMA, how many C elements does each thread compute?
# Your answer:

# Q3: Can two threads write to the same C element?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
MMA_ATOM = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
ATOM_LAYOUT = (2, 4)
VAL_LAYOUT = (16, 16)
NUM_THREADS = ATOM_LAYOUT[0] * ATOM_LAYOUT[1]  # = 8


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_fragment_ownership(
    C: cute.Tensor,
    ownership_map: cute.Tensor,
    results: cute.Tensor,
):
    """
    Analyze fragment ownership for MMA output.
    
    FILL IN [HARD]: Map each thread to its output fragment elements.
    
    HINT: Use thr_mma.partition_C() to get the C fragment for each thread.
          Store which thread owns which elements in ownership_map.
    """
    # --- Step 1: Create TiledMMA ---
    # TODO: tiled_mma = cute.make_tiled_mma(MMA_ATOM, ATOM_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Get thread-local MMA slice ---
    # TODO: tid = cute.thread_idx()
    #       thr_mma = tiled_mma.get_slice(tid)
    
    # --- Step 3: Partition C fragment ---
    # TODO: c_fragment = thr_mma.partition_C(C)
    
    # --- Step 4: Record ownership ---
    # For each element in c_fragment, record that 'tid' owns it
    # Store in ownership_map
    
    # --- Step 5: Compute output ---
    # Each thread writes to its owned elements
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and analyze fragment ownership.
    """
    
    # Output matrix (32x64 for (2,4) atom layout with 16x16 MMA)
    M_OUT, N_OUT = 32, 64
    C_torch = torch.zeros((M_OUT, N_OUT), dtype=torch.float32, device='cuda')
    C_cute = from_dlpack(C_torch)
    
    # Ownership map: for each element, which thread owns it
    ownership_torch = torch.full((M_OUT, N_OUT), -1, dtype=torch.int32, device='cuda')
    ownership_cute = from_dlpack(ownership_torch)
    
    # Results
    results_torch = torch.zeros(8, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_fragment_ownership[1, NUM_THREADS](C_cute, ownership_cute, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    ownership_cpu = ownership_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 08 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  Fragment Ownership Analysis:")
    print(f"    Output size: ({M_OUT}, {N_OUT}) = {M_OUT * N_OUT} elements")
    print(f"    Threads: {NUM_THREADS}")
    print(f"    Elements/thread: {(M_OUT * N_OUT) // NUM_THREADS}")
    print(f"\n  Ownership Map Sample (first 8x8):")
    print(ownership_cpu[:8, :8])
    print(f"\n  Results:")
    print(f"    Unique owners: {results_cpu[0]:.0f}")
    print(f"    Unowned elements: {results_cpu[1]:.0f}")
    
    passed = results_cpu[1] == 0  # All elements should be owned
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How are output elements distributed across threads?
# C2: Can fragment ownership overlap between threads?
# C3: How does fragment ownership affect warp specialization?
# C4: What happens if two threads write to the same element?

if __name__ == "__main__":
    run()
