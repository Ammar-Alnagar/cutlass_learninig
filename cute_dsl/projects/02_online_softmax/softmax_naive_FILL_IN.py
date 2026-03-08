"""
Project 02 — Naive Softmax (Baseline)

Two-pass naive softmax implementation for comparison.
Numerically unstable for large input values.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time
import math

SEQ_LEN = 4096


@cutlass.jit
def kernel_softmax_naive(
    logits: cute.Tensor,
    output: cute.Tensor,
    seq_len: int,
):
    """Naive two-pass softmax."""
    tid = cute.thread_idx()
    NUM_THREADS = 128
    
    elements_per_thread = (seq_len + NUM_THREADS - 1) // NUM_THREADS
    start = tid * elements_per_thread
    end = min(start + elements_per_thread, seq_len)
    
    # Pass 1: find max
    local_max = -1e30
    for i in range(start, end):
        local_max = max(local_max, logits[i])
    
    global_max = local_max
    
    # Pass 2: compute sum of exp
    local_sum = 0.0
    for i in range(start, end):
        local_sum = local_sum + math.exp(logits[i] - global_max)
    
    total_sum = local_sum
    if total_sum < 1e-10:
        total_sum = 1e-10
    
    # Pass 3: normalize
    for i in range(start, end):
        output[i] = math.exp(logits[i] - global_max) / total_sum
    
    pass


def run():
    print("\n" + "=" * 60)
    print("  Project 02 — Naive Softmax (Baseline)")
    print("=" * 60)
    
    torch.manual_seed(42)
    logits_torch = torch.randn(SEQ_LEN, dtype=torch.float32, device='cuda')
    output_torch = torch.zeros(SEQ_LEN, dtype=torch.float32, device='cuda')
    
    ref_torch = torch.softmax(logits_torch, dim=0).cpu().numpy()
    
    logits_cute = from_dlpack(logits_torch)
    output_cute = from_dlpack(output_torch)
    
    kernel_softmax_naive[1, 128](logits_cute, output_cute, SEQ_LEN)
    torch.cuda.synchronize()
    
    output_cpu = output_torch.cpu().numpy()
    max_diff = abs(output_cpu - ref_torch).max()
    
    print(f"\n  Max difference from PyTorch: {max_diff:.8f}")
    print(f"  Sum of output: {output_cpu.sum():.8f}")
    
    passed = max_diff < 1e-5
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


if __name__ == "__main__":
    run()
