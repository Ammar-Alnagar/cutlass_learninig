"""
Project 04 — FlashAttention-2 Prefill

ALGORITHM: Tiled attention with causal masking and online softmax
Paper: https://arxiv.org/abs/2307.08691
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
BLOCK_Q = 64
BLOCK_K = 64


@cutlass.jit
def kernel_fa2_prefill(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    scale: float,
    causal: bool,
):
    """
    FlashAttention-2 prefill kernel.
    
    FILL IN [HARD]: Implement tiled FA2 with causal masking.
    
    HINT:
      1. Tile over Q (rows) and K/V (columns)
      2. Online softmax: track max and sum across tiles
      3. Causal mask: set S[i,j] = -inf for j > i
    """
    # --- Step 1: Compute tile indices ---
    # TODO: tile_q = cute.block_idx() % (seq_q // BLOCK_Q)
    #       tile_k = cute.block_idx() // (seq_q // BLOCK_Q)
    
    # --- Step 2: Create MMA for QK^T and PV ---
    # TODO: mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
    
    # --- Step 3: Initialize online softmax state ---
    # TODO: m_i = -inf (max so far)
    #       d_i = 0 (sum so far)
    #       O_acc = 0 (output accumulator)
    
    # --- Step 4: Mainloop over K tiles ---
    # TODO: for k_tile in range(seq_k // BLOCK_K):
    #           # Load Q, K, V tiles
    #           # QK^T
    #           # Apply causal mask
    #           # Online softmax update
    #           # PV accumulation
    
    pass


def run():
    print("\n" + "=" * 60)
    print("  Project 04 — FlashAttention-2 Prefill")
    print("=" * 60)
    
    print(f"\n  Configuration:")
    print(f"    Q/K seq: {SEQ_Q}/{SEQ_K}, Heads: {NUM_HEADS}, Dim: {HEAD_DIM}")
    print(f"    Block sizes: Q={BLOCK_Q}, K={BLOCK_K}")
    
    torch.manual_seed(42)
    
    Q = torch.randn((NUM_HEADS, SEQ_Q, HEAD_DIM), dtype=torch.float16, device='cuda')
    K = torch.randn((NUM_HEADS, SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    V = torch.randn((NUM_HEADS, SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Reference: PyTorch attention
    scores = torch.einsum('hqk,hkd->hqk', Q.float(), K.float()) * scale
    # Causal mask
    mask = torch.triu(torch.ones(SEQ_Q, SEQ_K), diagonal=1).bool().to('cuda')
    scores[:, mask] = float('-inf')
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('hqk,hkd->hqk', attn, V.float())
    
    Q_cute = from_dlpack(Q)
    K_cute = from_dlpack(K)
    V_cute = from_dlpack(V)
    O_cute = from_dlpack(torch.zeros_like(ref_out))
    
    num_blocks = (SEQ_Q // BLOCK_Q) * (SEQ_K // BLOCK_K)
    
    kernel_fa2_prefill[num_blocks, 128](
        Q_cute, K_cute, V_cute, O_cute,
        SEQ_Q, SEQ_K, HEAD_DIM, scale, True
    )
    torch.cuda.synchronize()
    
    print("\n  ✓ Kernel launched (placeholder implementation)")
    print("  → Implement the tiled FA2 mainloop with online softmax")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    run()
