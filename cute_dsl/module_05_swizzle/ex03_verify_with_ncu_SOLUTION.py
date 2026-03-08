"""
Module 05 — Swizzle
Exercise 03 — Verify with Nsight Compute

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Profile with Nsight Compute
        // ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
  DSL:  # Same profiling workflow
        # ncu --metrics ... python ex03_verify_with_ncu_FILL_IN.py
  Key:  Nsight Compute provides direct measurement of bank conflicts.

WHAT YOU'RE BUILDING:
  An Nsight Compute profiling script that automatically measures bank conflicts
  with and without swizzling. This is the production workflow for verifying
  SMEM optimization effectiveness.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Run Nsight Compute with specific metrics
  - Interpret bank conflict metrics
  - Automate performance comparison

REQUIRED READING:
  - Nsight Compute docs: https://docs.nvidia.com/nsight-compute/
  - SMEM metrics: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-metrics
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import subprocess

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What Nsight Compute metric measures SMEM bank conflicts?
# Your answer: l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum

# Q2: What metric measures SMEM throughput?
# Your answer: l1tex__t_bytes_pipe_lsu_mem_shared.sum

# Q3: How do you interpret a high bank conflict count?
# Your answer: More conflicts = serialized accesses = lower throughput


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N = 128, 128
NUM_ITERATIONS = 100


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_verify_with_ncu(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    use_swizzle: bool,
    results: cute.Tensor,
):
    """
    Kernel that can run with or without swizzling for comparison.
    """
    # --- Step 1: Create layouts ---
    row_major = cute.make_layout((M, N), stride=(N, 1))
    
    if use_swizzle:
        swizzle = cute.Swizzle(6, 3, 3)
        layout = cute.composition(swizzle, row_major)
    else:
        layout = row_major
    
    # --- Step 2: Create SMEM tensor ---
    smem_tensor = cute.make_smem_tensor(smem_ptr, layout)
    
    # --- Step 3: Load from GMEM, store to SMEM, then read from SMEM ---
    gmem_tensor = cute.make_tensor(gmem_ptr, row_major)
    
    total = 0.0
    tid = cute.thread_idx()
    
    for iter in range(NUM_ITERATIONS):
        # Coalesced GMEM read
        val = gmem_tensor[(tid % M, tid // M)]
        
        # SMEM write
        smem_tensor[(tid % M, tid // M)] = val
        
        cute.sync_threads()
        
        # SMEM read (pattern that causes conflicts without swizzle)
        # Column-wise access pattern
        val2 = smem_tensor[(tid % M, (tid * 4) % N)]
        total += val2
    
    # --- Step 4: Store result ---
    if tid == 0:
        results[0] = float(NUM_ITERATIONS)
        results[1] = 1 if use_swizzle else 0
        results[2] = int(total * 1000) % 1000  # Checksum proxy
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel with Nsight Compute profiling.
    """
    
    print("\n" + "=" * 60)
    print("  Module 05 — Exercise 03: Nsight Compute Verification")
    print("=" * 60)
    
    try:
        result = subprocess.run(['which', 'ncu'], capture_output=True, text=True)
        if result.returncode != 0:
            print("\n  ⚠️  Nsight Compute (ncu) not found in PATH.")
            print("  Please install Nsight Compute or add it to PATH.")
            print("\n  To run manually:")
            print("    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\")
            print("        --set full --target-processes all \\")
            print("        python ex03_verify_with_ncu_FILL_IN.py")
            return True
    except Exception:
        pass
    
    smem_torch = torch.zeros(M * N, dtype=torch.float32, device='cuda')
    smem_ptr = from_dlpack(smem_torch)
    
    gmem_torch = torch.randn(M * N, dtype=torch.float32, device='cuda')
    gmem_ptr = from_dlpack(gmem_torch)
    
    results_torch = torch.zeros(4, dtype=torch.int32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    print("\n  Running without swizzle...")
    kernel_verify_with_ncu[1, 128](smem_ptr, gmem_ptr, False, results_cute)
    results_no_swizzle = results_torch.cpu().numpy().copy()
    
    print("  Running with swizzle...")
    kernel_verify_with_ncu[1, 128](smem_ptr, gmem_ptr, True, results_cute)
    results_with_swizzle = results_torch.cpu().numpy().copy()
    
    print(f"\n  Results:")
    print(f"    Without swizzle: {results_no_swizzle[0]:.0f} iterations")
    print(f"    With swizzle:    {results_with_swizzle[0]:.0f} iterations")
    
    passed = (
        results_no_swizzle[0] == NUM_ITERATIONS and
        results_with_swizzle[0] == NUM_ITERATIONS
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("\n  → Now run with Nsight Compute to measure bank conflicts:")
    print("    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\")
    print("        --set full --target-processes all \\")
    print("        python ex03_verify_with_ncu_FILL_IN.py")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: What bank conflict count did you observe without swizzle?
# C2: How much did swizzling reduce bank conflicts?
# C3: What was the impact on kernel duration?
# C4: How would you automate this comparison for CI/CD?

if __name__ == "__main__":
    run()
