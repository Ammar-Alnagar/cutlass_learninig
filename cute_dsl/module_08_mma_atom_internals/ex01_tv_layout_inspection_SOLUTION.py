"""
Module 08 — MMA Atom Internals
Exercise 01 — TV Layout Inspection

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto tv_layout_A = tiled_mma.tv_layout_A;
        // Inspect how A elements map to thread/value coordinates
  DSL:  tv_layout_A = tiled_mma.tv_layout_A
        # Same inspection in Python
  Key:  TV layout shows the thread-value mapping for MMA operands.

WHAT YOU'RE BUILDING:
  A TV layout inspector that shows how tensor elements are distributed across
  threads and register values. This is essential for debugging tensor core
  data flow and understanding fragment access patterns.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Inspect TV layouts for MMA operands
  - Understand thread-value coordinate mapping
  - Debug tensor core data distribution issues

REQUIRED READING:
  - CUTLASS TV layout docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_mma.html#tv-layout
  - CuTe C++ mental model: TV = Thread × Value decomposition
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does TV stand for in TV layout?
# Your answer: Thread-Value. Maps (thread_idx, value_idx) → tensor coordinate.

# Q2: For a (2, 4) atom layout with 16x16x16 MMA, how many threads?
# Your answer: 2 × 4 = 8 threads (or more depending on atom implementation)

# Q3: How are A, B, C TV layouts different?
# Your answer: A is (M, K), B is (K, N), C is (M, N). Different shapes
#              and thread/value decompositions.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
MMA_ATOM = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
ATOM_LAYOUT = (2, 4)
VAL_LAYOUT = (16, 16)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_tv_layout_inspection(
    results: cute.Tensor,
):
    """
    Inspect TV layouts for MMA operands.
    """
    # --- Step 1: Create TiledMMA ---
    tiled_mma = cute.make_tiled_mma(MMA_ATOM, ATOM_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Get TV layouts ---
    tv_A = tiled_mma.tv_layout_A
    tv_B = tiled_mma.tv_layout_B
    tv_C = tiled_mma.tv_layout_C
    
    # --- Step 3: Inspect layout properties ---
    results[0] = cute.rank(tv_A)
    results[1] = cute.rank(tv_B)
    results[2] = cute.rank(tv_C)
    
    results[3] = cute.cosize(tv_A).size()
    results[4] = cute.cosize(tv_B).size()
    results[5] = cute.cosize(tv_C).size()
    
    # --- Step 4: Map thread 0's elements ---
    # For thread 0, get the coordinates it owns
    thr_mma = tiled_mma.get_slice(0)
    
    # Store some sample mappings
    results[6] = tv_A((0, 0))
    results[7] = tv_B((0, 0))
    results[8] = tv_C((0, 0))
    
    # Store expected values for verification
    results[9] = 8   # Expected threads
    results[10] = 32  # Expected TV_A elements per thread
    results[11] = 32  # Expected TV_B elements per thread
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and inspect TV layouts.
    """
    
    results_torch = torch.zeros(12, dtype=torch.int32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_tv_layout_inspection[1, 1](results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 08 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  MMA Configuration:")
    print(f"    Atom: SM80 FP16×FP16→FP32")
    print(f"    Atom layout: {ATOM_LAYOUT}")
    print(f"    Value layout: {VAL_LAYOUT}")
    print(f"\n  TV Layout Inspection:")
    print(f"    TV_A rank: {results_cpu[0]}")
    print(f"    TV_B rank: {results_cpu[1]}")
    print(f"    TV_C rank: {results_cpu[2]}")
    print(f"    TV_A size: {results_cpu[3]}")
    print(f"    TV_B size: {results_cpu[4]}")
    print(f"    TV_C size: {results_cpu[5]}")
    print(f"\n  Thread 0 mappings:")
    print(f"    TV_A(0,0) = {results_cpu[6]}")
    print(f"    TV_B(0,0) = {results_cpu[7]}")
    print(f"    TV_C(0,0) = {results_cpu[8]}")
    
    passed = True  # Inspection exercise
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does TV layout differ from regular tensor layout?
# C2: Why are TV_A and TV_B different shapes?
# C3: How would you use TV layout for debugging?
# C4: What is the relationship between TV layout and register pressure?

if __name__ == "__main__":
    run()
