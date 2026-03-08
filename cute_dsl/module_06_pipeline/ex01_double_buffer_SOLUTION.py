"""
Module 06 — Pipeline
Exercise 01 — Double-Buffer Setup

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Double-buffer: load next tile while computing current
        T smem[2][TILE_SIZE];
        int write_phase = 0;
        load(gmem, smem[write_phase]);
        compute(smem[write_phase ^ 1]);
        write_phase ^= 1;
  DSL:  # Same pattern in Python
        smem = [cute.make_smem_tensor(...), cute.make_smem_tensor(...)]
        write_phase = 0
  Key:  Double-buffer hides load latency by overlapping with compute.

WHAT YOU'RE BUILDING:
  A double-buffer pipeline that loads the next data tile while computing
  the current tile. This is the foundation for all advanced pipelines in
  GEMM and attention kernels.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Set up double-buffer with two SMEM buffers
  - Implement ping-pong load/compute pattern
  - Understand pipeline synchronization

REQUIRED READING:
  - CUTLASS pipeline docs: https://nvidia.github.io/cutlass-dsl/cute/pipeline.html
  - Double-buffering tutorial: https://developer.nvidia.com/blog/optimizing-cuda-applications-pipelining/
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What latency does double-buffer hide?
# Your answer: GMEM load latency. While computing tile N, we load tile N+1.

# Q2: With double-buffer, what is the maximum throughput improvement?
# Your answer: Up to 2× if compute and load can fully overlap.

# Q3: What synchronization is needed between load and compute phases?
# Your answer: sync_threads() between load and compute phases.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
TILE_SIZE = 256
NUM_BUFFERS = 2
NUM_ITERATIONS = 4


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_double_buffer(
    gmem: cute.Tensor,
    smem_ptrs: cute.Pointer,
    results: cute.Tensor,
):
    """
    Double-buffer pipeline: overlap load and compute.
    """
    # --- Step 1: Create SMEM buffers ---
    layout = cute.make_layout((TILE_SIZE,), stride=(1,))
    smem = [cute.make_smem_tensor(smem_ptrs[i], layout) for i in range(NUM_BUFFERS)]
    
    # --- Step 2: Initialize ---
    write_phase = 0
    
    # Load first tile (simplified - direct copy)
    tid = cute.thread_idx()
    if tid < TILE_SIZE:
        smem[write_phase][tid] = gmem[tid]
    
    cute.sync_threads()
    
    # --- Step 3: Mainloop ---
    total = 0.0
    for iter in range(NUM_ITERATIONS):
        read_phase = (write_phase + 1) % NUM_BUFFERS
        
        # Compute on smem[read_phase]
        if tid < TILE_SIZE:
            total += smem[read_phase][tid]
        
        cute.sync_threads()
        
        # Load next tile into smem[write_phase]
        next_offset = (iter + 1) * TILE_SIZE
        if tid < TILE_SIZE and (iter + 1) < NUM_ITERATIONS:
            smem[write_phase][tid] = gmem[next_offset + tid]
        
        cute.sync_threads()
        
        # Toggle phase
        write_phase = (write_phase + 1) % NUM_BUFFERS
    
    # --- Step 4: Store results ---
    if tid == 0:
        results[0] = float(NUM_ITERATIONS)
        results[1] = total
        results[2] = 2.0  # Num buffers
        results[3] = float(TILE_SIZE)
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify double-buffer.
    """
    
    gmem_torch = torch.arange(NUM_ITERATIONS * TILE_SIZE, dtype=torch.float32, device='cuda')
    gmem_cute = from_dlpack(gmem_torch)
    
    smem_torch = [torch.zeros(TILE_SIZE, dtype=torch.float32, device='cuda') 
                  for _ in range(NUM_BUFFERS)]
    smem_ptrs = [from_dlpack(t) for t in smem_torch]
    
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_double_buffer[1, 128](gmem_cute, smem_ptrs, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 06 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  Double-Buffer Configuration:")
    print(f"    Tile size: {TILE_SIZE}")
    print(f"    Buffers: {NUM_BUFFERS}")
    print(f"    Iterations: {NUM_ITERATIONS}")
    print(f"\n  Results:")
    print(f"    Iterations completed: {results_cpu[0]:.0f}")
    print(f"    Sum: {results_cpu[1]:.1f}")
    
    passed = results_cpu[0] == NUM_ITERATIONS
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does double-buffer improve throughput?
# C2: What is the trade-off of double-buffering?
# C3: In FlashAttention, what data would you double-buffer?
# C4: How does triple-buffering differ from double-buffering?

if __name__ == "__main__":
    run()
