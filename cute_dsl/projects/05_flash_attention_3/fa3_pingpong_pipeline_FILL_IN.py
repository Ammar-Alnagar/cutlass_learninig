"""
Project 05 — FlashAttention-3 Ping-Pong Pipeline

ALGORITHM: Ping-pong pipeline with double-buffering
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time
import math

SEQ_Q = 1024
SEQ_K = 1024
NUM_HEADS = 8
HEAD_DIM = 64
NUM_BUFFERS = 2  # Double-buffer


@cutlass.jit
def kernel_fa3_pingpong(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    scale: float,
):
    """
    FlashAttention-3 with ping-pong pipeline.
    
    FILL IN [HARD]: Implement double-buffered pipeline.
    
    HINT:
      1. Use two SMEM buffers for Q, K, V
      2. Load next tile while computing current
      3. Ping-pong between buffers
    """
    # --- Step 1: Create double-buffer ---
    # TODO: smem_q = [cute.make_smem_tensor(...) for _ in range(NUM_BUFFERS)]
    #       smem_k = [cute.make_smem_tensor(...) for _ in range(NUM_BUFFERS)]
    #       smem_v = [cute.make_smem_tensor(...) for _ in range(NUM_BUFFERS)]
    
    # --- Step 2: Ping-pong mainloop ---
    # TODO: write_phase = 0
    #       for k_tile in range(seq_k // BLOCK_K):
    #           # Load into smem[write_phase]
    #           # Compute on smem[write_phase ^ 1]
    #           # write_phase ^= 1
    
    pass


def run():
    print("\n" + "=" * 60)
    print("  Project 05 — FlashAttention-3 (Ping-Pong Pipeline)")
    print("=" * 60)
    
    print(f"\n  Configuration:")
    print(f"    Q/K seq: {SEQ_Q}/{SEQ_K}, Heads: {NUM_HEADS}, Dim: {HEAD_DIM}")
    print(f"    Double-buffer: {NUM_BUFFERS} buffers")
    
    torch.manual_seed(42)
    
    Q = torch.randn((NUM_HEADS, SEQ_Q, HEAD_DIM), dtype=torch.float16, device='cuda')
    K = torch.randn((NUM_HEADS, SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    V = torch.randn((NUM_HEADS, SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    Q_cute = from_dlpack(Q)
    K_cute = from_dlpack(K)
    V_cute = from_dlpack(V)
    O_cute = from_dlpack(torch.zeros_like(Q))
    
    kernel_fa3_pingpong[NUM_HEADS, 128](
        Q_cute, K_cute, V_cute, O_cute,
        SEQ_Q, SEQ_K, HEAD_DIM, scale
    )
    torch.cuda.synchronize()
    
    print("\n  ✓ Kernel launched (placeholder)")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    run()
