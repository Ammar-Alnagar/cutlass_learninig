"""
Project 04 — FlashAttention-2 Decode

Single query decode with KV cache.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time
import math

SEQ_Q = 1  # Decode: single query
SEQ_K = 1024  # KV cache length
NUM_HEADS = 8
HEAD_DIM = 64


@cutlass.jit
def kernel_fa2_decode(
    Q: cute.Tensor,
    K_cache: cute.Tensor,
    V_cache: cute.Tensor,
    O: cute.Tensor,
    seq_k: int,
    head_dim: int,
    scale: float,
):
    """
    FlashAttention-2 decode kernel (single query).
    
    FILL IN [MEDIUM]: Implement single-query attention with KV cache.
    
    HINT: Simpler than prefill - only one Q row to process.
    """
    pass


def run():
    print("\n" + "=" * 60)
    print("  Project 04 — FlashAttention-2 Decode")
    print("=" * 60)
    
    print(f"\n  Configuration:")
    print(f"    Q seq: {SEQ_Q}, KV cache: {SEQ_K}, Heads: {NUM_HEADS}, Dim: {HEAD_DIM}")
    
    torch.manual_seed(42)
    
    Q = torch.randn((NUM_HEADS, SEQ_Q, HEAD_DIM), dtype=torch.float16, device='cuda')
    K_cache = torch.randn((NUM_HEADS, SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    V_cache = torch.randn((NUM_HEADS, SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    Q_cute = from_dlpack(Q)
    K_cute = from_dlpack(K_cache)
    V_cute = from_dlpack(V_cache)
    O_cute = from_dlpack(torch.zeros_like(Q))
    
    kernel_fa2_decode[NUM_HEADS, 128](
        Q_cute, K_cute, V_cute, O_cute,
        SEQ_K, HEAD_DIM, scale
    )
    torch.cuda.synchronize()
    
    print("\n  ✓ Kernel launched (placeholder)")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    run()
