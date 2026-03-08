"""
Project 03 — Multi-Head Attention
Unfused Baseline Implementation

ALGORITHM: Standard MHA with separate GEMM operations
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time
import math

# Configuration
BATCH_SIZE = 32
SEQ_LEN = 512
NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM


@cutlass.jit
def kernel_mha_unfused(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    S: cute.Tensor,  # Attention scores
    O: cute.Tensor,  # Output
    seq_len: int,
    num_heads: int,
    head_dim: int,
):
    """
    Unfused multi-head attention baseline.
    
    FILL IN [MEDIUM]: Implement standard MHA with separate operations.
    
    HINT:
      1. QK^T matmul for attention scores
      2. Softmax normalization
      3. PV matmul for output
    """
    # --- Step 1: Compute QK^T for attention scores ---
    # TODO: for each head:
    #           S[head] = Q[head] @ K[head].T / sqrt(head_dim)
    
    # --- Step 2: Softmax over attention scores ---
    # TODO: for each head:
    #           P[head] = softmax(S[head])
    
    # --- Step 3: Compute PV for output ---
    # TODO: for each head:
    #           O[head] = P[head] @ V[head]
    
    pass


def run():
    print("\n" + "=" * 60)
    print("  Project 03 — Multi-Head Attention (Unfused Baseline)")
    print("=" * 60)
    
    print(f"\n  Configuration:")
    print(f"    Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}, Heads: {NUM_HEADS}, Dim: {HEAD_DIM}")
    
    torch.manual_seed(42)
    
    # Create Q, K, V
    Q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=torch.float16, device='cuda')
    K = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=torch.float16, device='cuda')
    V = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    # Reference: PyTorch attention
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * scale
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bhqk,bkhd->bqhd', attn, V)
    
    # Reshape for kernel
    Q_reshaped = Q.reshape(BATCH_SIZE * NUM_HEADS, SEQ_LEN, HEAD_DIM)
    K_reshaped = K.reshape(BATCH_SIZE * NUM_HEADS, SEQ_LEN, HEAD_DIM)
    V_reshaped = V.reshape(BATCH_SIZE * NUM_HEADS, SEQ_LEN, HEAD_DIM)
    
    Q_cute = from_dlpack(Q_reshaped.contiguous())
    K_cute = from_dlpack(K_reshaped.contiguous())
    V_cute = from_dlpack(V_reshaped.contiguous())
    
    S_cute = from_dlpack(torch.zeros(BATCH_SIZE * NUM_HEADS, SEQ_LEN, SEQ_LEN, dtype=torch.float32, device='cuda'))
    O_cute = from_dlpack(torch.zeros(BATCH_SIZE * NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32, device='cuda'))
    
    # Launch kernel
    kernel_mha_unfused[BATCH_SIZE * NUM_HEADS, 128](
        Q_cute, K_cute, V_cute, S_cute, O_cute,
        SEQ_LEN, NUM_HEADS, HEAD_DIM
    )
    torch.cuda.synchronize()
    
    print("\n  ✓ Kernel launched (placeholder implementation)")
    print("  → Implement the QK^T + softmax + PV operations in the kernel")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    run()
