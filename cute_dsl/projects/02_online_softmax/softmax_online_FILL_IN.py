"""
Project 02 — Online Softmax

ALGORITHM: Numerically Stable One-Pass Softmax

From "Online Normalizer Calculation for Softmax" (Piozzi et al.)

Pseudocode:
  max_val = -inf
  sum_exp = 0
  
  for x in input:
      new_max = max(max_val, x)
      sum_exp = sum_exp * exp(max_val - new_max) + exp(x - new_max)
      max_val = new_max
  
  softmax[i] = exp(x[i] - max_val) / sum_exp

TARGET: >85% memory bandwidth utilization
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time
import math

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN = 4096  # Sequence length
NUM_HEADS = 32  # Number of attention heads
HEAD_DIM = 128  # Head dimension


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_online_softmax(
    logits: cute.Tensor,
    output: cute.Tensor,
    seq_len: int,
):
    """
    Numerically stable one-pass online softmax.
    
    FILL IN [MEDIUM]: Implement online softmax algorithm.
    
    HINT:
      1. First pass: find max and compute sum of exp
      2. Second pass: compute softmax = exp(x - max) / sum
      3. Use thread parallelism across sequence dimension
    """
    # --- Step 1: Get thread and block indices ---
    # TODO: tid = cute.thread_idx()
    #       block_idx = cute.block_idx()
    
    # --- Step 2: Compute element range for this thread ---
    # TODO: elements_per_thread = (seq_len + NUM_THREADS - 1) // NUM_THREADS
    #       start = tid * elements_per_thread
    #       end = min(start + elements_per_thread, seq_len)
    
    # --- Step 3: First pass - find max ---
    # TODO: local_max = -inf
    #       for i in range(start, end):
    #           local_max = max(local_max, logits[i])
    
    # --- Step 4: Block-wide reduction for global max ---
    # TODO: Use shared memory for reduction
    
    # --- Step 5: Second pass - compute exp and sum ---
    # TODO: local_sum = 0.0
    #       for i in range(start, end):
    #           local_sum += exp(logits[i] - global_max)
    
    # --- Step 6: Block-wide reduction for total sum ---
    
    # --- Step 7: Third pass - compute softmax ---
    # TODO: for i in range(start, end):
    #           output[i] = exp(logits[i] - global_max) / total_sum
    
    pass


# ─────────────────────────────────────────────
# NAIVE BASELINE (for comparison)
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_softmax_naive(
    logits: cute.Tensor,
    output: cute.Tensor,
    seq_len: int,
):
    """
    Naive two-pass softmax (numerically unstable for large values).
    """
    tid = cute.thread_idx()
    elements_per_thread = (seq_len + 127) // 128
    start = tid * elements_per_thread
    end = min(start + elements_per_thread, seq_len)
    
    # Pass 1: find max
    local_max = -1e30
    for i in range(start, end):
        local_max = max(local_max, logits[i])
    
    # Simple reduction (thread 0 collects)
    if tid == 0:
        global_max = local_max
    else:
        global_max = local_max
    
    # Pass 2: compute exp and sum
    local_sum = 0.0
    for i in range(start, end):
        local_sum = local_sum + math.exp(logits[i] - global_max)
    
    # Pass 3: normalize
    for i in range(start, end):
        output[i] = math.exp(logits[i] - global_max) / local_sum
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """Run the softmax kernel and measure performance."""
    
    print("\n" + "=" * 60)
    print("  Project 02 — Online Softmax")
    print("=" * 60)
    
    # Create input with potentially large values (tests numerical stability)
    torch.manual_seed(42)
    logits_torch = torch.randn(SEQ_LEN, dtype=torch.float32, device='cuda') * 10
    output_torch = torch.zeros(SEQ_LEN, dtype=torch.float32, device='cuda')
    
    # Reference: PyTorch softmax
    ref_torch = torch.softmax(logits_torch, dim=0).cpu().numpy()
    
    logits_cute = from_dlpack(logits_torch)
    output_cute = from_dlpack(output_torch)
    
    NUM_THREADS = 128
    NUM_BLOCKS = 1
    
    # Warmup
    print(f"\n  Sequence length: {SEQ_LEN}")
    print("  Warming up...")
    kernel_online_softmax[NUM_BLOCKS, NUM_THREADS](logits_cute, output_cute, SEQ_LEN)
    torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 100
    print(f"  Running {num_runs} iterations...")
    
    start = time.perf_counter()
    for _ in range(num_runs):
        kernel_online_softmax[NUM_BLOCKS, NUM_THREADS](logits_cute, output_cute, SEQ_LEN)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / num_runs) * 1000
    
    # Memory bandwidth
    bytes_transferred = 2 * SEQ_LEN * 4  # Read + write, float32
    bandwidth_gbs = (bytes_transferred / avg_time_ms / 1e6)
    
    # Peak bandwidth (A100: ~1555 GB/s)
    peak_bandwidth = 1555.0
    bw_utilization = (bandwidth_gbs / peak_bandwidth) * 100
    
    print(f"\n  Results:")
    print(f"    Average time: {avg_time_ms:.4f} ms")
    print(f"    Memory bandwidth: {bandwidth_gbs:.1f} GB/s")
    print(f"    Peak bandwidth (A100): {peak_bandwidth:.0f} GB/s")
    print(f"    BW utilization: {bw_utilization:.1f}%")
    
    # Verify correctness
    output_cpu = output_torch.cpu().numpy()
    max_diff = abs(output_cpu - ref_torch).max()
    
    # Check sum to 1
    sum_check = abs(output_cpu.sum() - 1.0)
    
    print(f"\n  Correctness:")
    print(f"    Max difference from PyTorch: {max_diff:.8f}")
    print(f"    Sum of output: {output_cpu.sum():.8f} (should be 1.0)")
    
    passed = max_diff < 1e-5 and sum_check < 1e-5
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"  Target: >85% BW utilization")
    print(f"  Status: {'✓ ACHIEVED' if bw_utilization >= 85 else '✗ BELOW TARGET'}")
    print("=" * 60 + "\n")
    
    return passed


if __name__ == "__main__":
    run()
