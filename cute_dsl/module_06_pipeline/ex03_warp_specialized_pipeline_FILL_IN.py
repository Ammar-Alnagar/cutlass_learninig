"""
Module 06 — Pipeline
Exercise 03 — Warp-Specialized Pipeline

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Warp specialization: DMA warps load, MMA warps compute
        if (warp_group == DMA_WARPS) {
            // Load data from GMEM to SMEM
        } else if (warp_group == MMA_WARPS) {
            // Wait for data, then compute
        }
  DSL:  # Same pattern in Python
        if warp_group == DMA_WARPS:
            # Load
        else:
            # Compute
  Key:  Warp specialization dedicates warps to specific roles for max overlap.

WHAT YOU'RE BUILDING:
  A warp-specialized pipeline where DMA warps handle data movement and MMA
  warps handle compute. This is the FlashAttention-3 pattern that achieves
  1.5× speedup over FA2 by fully overlapping load and compute.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Partition warps into DMA and MMA groups
  - Implement producer/consumer synchronization
  - Understand FA3's warp specialization pattern

REQUIRED READING:
  - FlashAttention-3 paper: https://arxiv.org/abs/2310.03748 (Section 3.3 on warp specialization)
  - CUTLASS warp specialization: https://nvidia.github.io/cutlass-dsl/cute/pipeline.html#warp-specialization
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: In FA3's warp specialization, what is the typical DMA:MMA warp ratio?
# Your answer:

# Q2: How do DMA and MMA warps synchronize?
# Your answer:

# Q3: What is the advantage of warp specialization over single-warp pipelines?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Warp configuration (128 threads = 4 warps)
TOTAL_THREADS = 128
DMA_WARPS = 1  # First warp (threads 0-31) does DMA
MMA_WARPS = 3  # Remaining warps (threads 32-127) do MMA

DMA_THREAD_START = 0
DMA_THREAD_END = 32
MMA_THREAD_START = 32
MMA_THREAD_END = 128

TILE_SIZE = 256
NUM_ITERATIONS = 4


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_warp_specialized_pipeline(
    gmem: cute.Tensor,
    smem_ptrs: cute.Pointer,
    results: cute.Tensor,
):
    """
    Warp-specialized pipeline: DMA warps load, MMA warps compute.
    
    FILL IN [HARD]: Implement warp-specialized producer/consumer pattern.
    
    HINT: Use thread_idx to determine warp role.
          DMA warps (0-31) load from GMEM to SMEM.
          MMA warps (32-127) wait and compute.
          Use barriers for synchronization.
    """
    # --- Step 1: Determine warp role ---
    # TODO: tid = cute.thread_idx()
    #       is_dma = (tid >= DMA_THREAD_START and tid < DMA_THREAD_END)
    #       is_mma = (tid >= MMA_THREAD_START and tid < MMA_THREAD_END)
    
    # --- Step 2: Create SMEM buffers ---
    # TODO: smem = [cute.make_smem_tensor(...) for _ in range(NUM_STAGES)]
    
    # --- Step 3: Create barrier for DMA→MMA sync ---
    # TODO: barrier = cute.Barrier(MMA_WARPS * 32)
    
    # --- Step 4: Mainloop ---
    # TODO: for iter in range(NUM_ITERATIONS):
    #           if is_dma:
    #               # Load from GMEM to SMEM[iter % NUM_STAGES]
    #               # Signal barrier when done
    #           if is_mma:
    #               # Wait on barrier
    #               # Compute on SMEM[iter % NUM_STAGES]
    
    # --- Step 5: Store results ---
    # DMA warp stores load count, MMA warp stores compute sum
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify warp-specialized pipeline.
    
    NCU PROFILING COMMAND:
    ncu --metrics smsp__inst_executed_pipe_tensor.sum,\
                l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
                gpu__time_duration.sum \
        --set full --target-processes all \
        python ex03_warp_specialized_pipeline_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - smsp__inst_executed_pipe_tensor.sum: Tensor core utilization (MMA warps)
    - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum: GMEM throughput (DMA warps)
    - gpu__time_duration.sum: Kernel duration (should be lower with overlap)
    """
    
    gmem_torch = torch.arange(NUM_ITERATIONS * TILE_SIZE, dtype=torch.float32, device='cuda')
    gmem_cute = from_dlpack(gmem_torch)
    
    smem_torch = [torch.zeros(TILE_SIZE, dtype=torch.float32, device='cuda') 
                  for _ in range(3)]  # Triple-buffer
    smem_ptrs = [from_dlpack(t) for t in smem_torch]
    
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch with 128 threads (4 warps)
    kernel_warp_specialized_pipeline[1, TOTAL_THREADS](gmem_cute, smem_ptrs, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 06 — Exercise 03 Results")
    print("=" * 60)
    print(f"\n  Warp-Specialized Configuration:")
    print(f"    Total threads: {TOTAL_THREADS} ({TOTAL_THREADS // 32} warps)")
    print(f"    DMA warps: {DMA_WARPS} (threads {DMA_THREAD_START}-{DMA_THREAD_END-1})")
    print(f"    MMA warps: {MMA_WARPS} (threads {MMA_THREAD_START}-{MMA_THREAD_END-1})")
    print(f"    Iterations: {NUM_ITERATIONS}")
    print(f"\n  Results:")
    print(f"    DMA load count:  {results_cpu[0]:.0f}")
    print(f"    MMA compute sum: {results_cpu[1]:.1f}")
    print(f"    Expected loads:  {NUM_ITERATIONS}")
    
    passed = results_cpu[0] == NUM_ITERATIONS
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("\n  → This is the core pattern from FlashAttention-3!")
    print("     FA3 achieves 1.5× speedup via warp specialization.")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does warp specialization improve over single-warp pipelines?
# C2: What is the optimal DMA:MMA warp ratio for GEMM?
# C3: In FA3, how many warps are typically assigned to each role?
# C4: What are the challenges of debugging warp-specialized kernels?

if __name__ == "__main__":
    run()
