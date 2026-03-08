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
# Your answer:

# Q2: For a (2, 4) atom layout with 16x16x16 MMA, how many threads?
# Your answer:

# Q3: How are A, B, C TV layouts different?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# MMA configuration
MMA_ATOM = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
ATOM_LAYOUT = (2, 4)  # 8 atoms
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
    
    FILL IN [HARD]: Create TiledMMA and inspect TV layouts.
    
    HINT: tv_layout_A = tiled_mma.tv_layout_A
          tv_layout_B = tiled_mma.tv_layout_B
          tv_layout_C = tiled_mma.tv_layout_C
    """
    # --- Step 1: Create TiledMMA ---
    # TODO: tiled_mma = cute.make_tiled_mma(MMA_ATOM, ATOM_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Get TV layouts ---
    # TODO: tv_A = tiled_mma.tv_layout_A
    #       tv_B = tiled_mma.tv_layout_B
    #       tv_C = tiled_mma.tv_layout_C
    
    # --- Step 3: Inspect layout properties ---
    # Store shape/rank of each TV layout in results
    
    # --- Step 4: Map thread 0's elements ---
    # For thread 0, which (row, col) elements does it own?
    
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
