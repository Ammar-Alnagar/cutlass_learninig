"""
Module 06 — Pipeline
Exercise 02 — Async Copy Overlap

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Async copy with cp.async.cg.shared.global
        cp.async.cg.shared.global [smem], [gmem], size, pred;
        cp.async.commit_group;
        cp.async.wait_group 0;
  DSL:  # Same pattern with CuTe
        cute.copy.async(smem, gmem, size)
        cute.fence()
        cute.async_wait()
  Key:  Async copies run in hardware while compute proceeds.

WHAT YOU'RE BUILDING:
  An async copy pipeline that uses CUDA's async copy instructions to overlap
  GMEM→SMEM transfers with compute. This is the Ampere+ pattern for hiding
  memory latency.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Use async copy instructions
  - Implement commit/wait synchronization
  - Measure async copy overlap effectiveness

REQUIRED READING:
  - Async copy docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async
  - CUTLASS async pipeline: https://nvidia.github.io/cutlass-dsl/cute/pipeline.html#async-pipeline
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What GPU architecture introduced async copy (cp.async)?
# Your answer:

# Q2: How many async copy groups can be in flight simultaneously?
# Your answer:

# Q3: What is the purpose of cp.async.commit_group?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
TILE_SIZE = 256
NUM_STAGES = 3  # Triple-buffer for async
NUM_ITERATIONS = 4


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_async_copy_overlap(
    gmem: cute.Tensor,
    smem_ptrs: cute.Pointer,
    results: cute.Tensor,
):
    """
    Async copy pipeline with commit/wait synchronization.
    
    FILL IN [HARD]: Implement async copy with proper synchronization.
    
    HINT: Use cute.fence() for cp.async.commit_group
          Use cute.async_wait() for cp.async.wait_group
    """
    # --- Step 1: Create SMEM buffers ---
    # TODO: smem = [cute.make_smem_tensor(...) for _ in range(NUM_STAGES)]
    
    # --- Step 2: Issue async copies ---
    # TODO: for each stage:
    #           cute.copy.async(gmem, smem[stage])
    #           cute.fence()  # commit
    
    # --- Step 3: Wait for copies and compute ---
    # TODO: for iter in range(NUM_ITERATIONS):
    #           cute.async_wait()  # wait for current stage
    #           compute(smem[iter % NUM_STAGES])
    #           issue next async copy
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify async copy pipeline.
    
    NCU PROFILING COMMAND:
    ncu --metrics smsp__inst_executed_pipe_cp.sum,\
                l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
        --set full --target-processes all \
        python ex02_async_copy_overlap_FILL_IN.py
    """
    
    gmem_torch = torch.arange(NUM_ITERATIONS * TILE_SIZE, dtype=torch.float32, device='cuda')
    gmem_cute = from_dlpack(gmem_torch)
    
    smem_torch = [torch.zeros(TILE_SIZE, dtype=torch.float32, device='cuda') 
                  for _ in range(NUM_STAGES)]
    smem_ptrs = [from_dlpack(t) for t in smem_torch]
    
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_async_copy_overlap[1, 128](gmem_cute, smem_ptrs, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 06 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  Async Copy Configuration:")
    print(f"    Tile size: {TILE_SIZE}")
    print(f"    Stages: {NUM_STAGES}")
    print(f"    Iterations: {NUM_ITERATIONS}")
    print(f"\n  Results:")
    print(f"    Iterations completed: {results_cpu[0]:.0f}")
    
    passed = results_cpu[0] == NUM_ITERATIONS
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does async copy differ from regular copy?
# C2: What is the maximum number of async copy groups?
# C3: In FlashAttention, where would async copies help?
# C4: How do you debug async copy synchronization issues?

if __name__ == "__main__":
    run()
