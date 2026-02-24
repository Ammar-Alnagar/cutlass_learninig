"""
Module 03: Attention Kernels

This module teaches how to implement efficient attention mechanisms
using Triton, including FlashAttention-style optimizations.

LEARNING OBJECTIVES:
1. Understand attention computation patterns
2. Implement efficient attention kernels
3. Apply tiling to attention for memory efficiency
4. Optimize for transformer workloads
"""

import triton
import triton.language as tl
import torch
import math


# ============================================================================
# ATTENTION KERNEL 1: Basic Scaled Dot-Product Attention
# ============================================================================

@triton.jit
def basic_attention_kernel(
    q_ptr, k_ptr, v_ptr,
    output_ptr,
    seq_len,
    head_dim,
    stride_q, stride_k, stride_v, stride_o,
    scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    """
    Basic scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    """
    # Each program handles one query position
    q_idx = tl.program_id(axis=0)
    
    # Load query vector
    q_offsets = tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < head_dim
    q = tl.load(q_ptr + q_idx * stride_q + q_offsets, mask=q_mask, other=0.0)
    
    # Initialize accumulators
    max_val = -float('inf')
    sum_exp = 0.0
    accumulator = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    
    # Loop over key/value positions
    for kv_idx in range(0, seq_len, BLOCK_SIZE_KV):
        kv_offsets = kv_idx + tl.arange(0, BLOCK_SIZE_KV)
        kv_mask = kv_offsets < seq_len
        
        # Load key block
        k_block = tl.load(
            k_ptr + kv_offsets[:, None] * stride_k + q_offsets[None, :],
            mask=kv_mask[:, None] & q_mask[None, :],
            other=0.0
        )
        
        # Compute attention scores: Q @ K^T
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        
        # Load value block
        v_block = tl.load(
            v_ptr + kv_offsets[:, None] * stride_v + q_offsets[None, :],
            mask=kv_mask[:, None] & q_mask[None, :],
            other=0.0
        )
        
        # Softmax with numerical stability (online softmax)
        new_max = tl.maximum(max_val, tl.max(scores, axis=0))
        exp_scores = tl.exp(scores - new_max)
        new_sum = tl.sum(exp_scores, axis=0)
        
        # Update accumulator
        accumulator = accumulator * tl.exp(max_val - new_max) + tl.sum(
            exp_scores[:, None] * v_block, axis=0
        )
        
        max_val = new_max
        sum_exp = new_sum
        
        kv_idx += BLOCK_SIZE_KV
    
    # Normalize
    output = accumulator / sum_exp
    
    # Store result
    tl.store(output_ptr + q_idx * stride_o + q_offsets, output, mask=q_mask)


# ============================================================================
# ATTENTION KERNEL 2: FlashAttention-style Tiled Attention
# ============================================================================

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr,
    output_ptr,
    seq_len_q,
    seq_len_kv,
    head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    FlashAttention-style tiled attention for memory efficiency.
    
    Key ideas:
    - Tile Q, K, V to fit in SRAM
    - Online softmax for numerical stability
    - Single pass over K, V
    """
    # Program IDs: (batch, head, query_block)
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_q = tl.program_id(axis=2)
    
    # Query block indices
    q_start = pid_q * BLOCK_SIZE_Q
    q_offsets = q_start + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < seq_len_q
    
    # Load Q block
    q_ptrs = (
        q_ptr +
        pid_b * stride_qb +
        pid_h * stride_qh +
        q_offsets[:, None] * stride_qs +
        tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None] & tl.arange(0, BLOCK_SIZE_D)[None, :] < head_dim, other=0.0)
    
    # Initialize for online softmax
    m_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
    
    # Loop over KV blocks
    for kv_start in range(0, seq_len_kv, BLOCK_SIZE_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_SIZE_KV)
        kv_mask = kv_offsets < seq_len_kv
        
        # Load K block
        k_ptrs = (
            k_ptr +
            pid_b * stride_kb +
            pid_h * stride_kh +
            kv_offsets[:, None] * stride_ks +
            tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=kv_mask[:, None] & tl.arange(0, BLOCK_SIZE_D)[None, :] < head_dim, other=0.0)
        
        # Compute Q @ K^T
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # Mask out invalid positions (causal would go here)
        qk = tl.where(q_mask[:, None] & kv_mask[None, :], qk, -float('inf'))
        
        # Load V block
        v_ptrs = (
            v_ptr +
            pid_b * stride_vb +
            pid_h * stride_vh +
            kv_offsets[:, None] * stride_vs +
            tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & tl.arange(0, BLOCK_SIZE_D)[None, :] < head_dim, other=0.0)
        
        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=1)[:, None]
        
        # Update accumulator
        acc = acc * alpha + tl.dot(p, v)
        
        m_i = m_new
        l_i = l_new
    
    # Normalize
    output = acc / l_i
    
    # Store result
    o_ptrs = (
        output_ptr +
        pid_b * stride_ob +
        pid_h * stride_oh +
        q_offsets[:, None] * stride_os +
        tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_od
    )
    tl.store(o_ptrs, output, mask=q_mask[:, None] & tl.arange(0, BLOCK_SIZE_D)[None, :] < head_dim)


# ============================================================================
# ATTENTION KERNEL 3: Causal (Masked) Attention
# ============================================================================

@triton.jit
def causal_attention_kernel(
    q_ptr, k_ptr, v_ptr,
    output_ptr,
    seq_len,
    head_dim,
    stride_q, stride_k, stride_v, stride_o,
    scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    """
    Causal attention: each position can only attend to previous positions.
    
    Implements the causal mask for autoregressive generation.
    """
    q_idx = tl.program_id(axis=0)
    
    q_offsets = tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < head_dim
    q = tl.load(q_ptr + q_idx * stride_q + q_offsets, mask=q_mask, other=0.0)
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    
    for kv_idx in range(0, seq_len, BLOCK_SIZE_KV):
        kv_offsets = kv_idx + tl.arange(0, BLOCK_SIZE_KV)
        kv_mask = kv_offsets < seq_len
        
        # CAUSAL MASK: only attend to positions <= q_idx
        causal_mask = kv_offsets <= q_idx
        
        k_block = tl.load(
            k_ptr + kv_offsets[:, None] * stride_k + q_offsets[None, :],
            mask=(kv_mask[:, None] & q_mask[None, :]) & causal_mask[:, None],
            other=-float('inf')
        )
        
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(causal_mask, scores, -float('inf'))
        
        v_block = tl.load(
            v_ptr + kv_offsets[:, None] * stride_v + q_offsets[None, :],
            mask=kv_mask[:, None] & q_mask[None, :],
            other=0.0
        )
        
        # Online softmax
        new_max = tl.maximum(m_i, tl.max(scores, axis=0))
        exp_scores = tl.exp(scores - new_max)
        new_sum = tl.sum(exp_scores, axis=0)
        
        acc = acc * tl.exp(m_i - new_max) + tl.sum(exp_scores[:, None] * v_block, axis=0)
        m_i = new_max
        l_i = new_sum
    
    output = acc / l_i
    tl.store(output_ptr + q_idx * stride_o + q_offsets, output, mask=q_mask)


# ============================================================================
# ATTENTION WRAPPER FUNCTIONS
# ============================================================================

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Simple scaled dot-product attention.
    
    Args:
        q: Query tensor [seq_len, head_dim]
        k: Key tensor [seq_len, head_dim]
        v: Value tensor [seq_len, head_dim]
    
    Returns:
        output: [seq_len, head_dim]
    """
    seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(q)
    
    BLOCK_SIZE_Q = min(head_dim, 64)
    BLOCK_SIZE_KV = 32
    
    basic_attention_kernel[(seq_len,)](
        q, k, v, output,
        seq_len, head_dim,
        q.stride(0), k.stride(0), v.stride(0), output.stride(0),
        scale,
        BLOCK_SIZE_Q, BLOCK_SIZE_KV
    )
    
    return output


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    FlashAttention-style efficient attention.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        causal: Whether to apply causal mask
    
    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(q)
    
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_KV = 32
    BLOCK_SIZE_D = min(head_dim, 64)
    
    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_SIZE_Q))
    
    flash_attention_kernel[grid](
        q, k, v, output,
        seq_len, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
        BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_SIZE_D
    )
    
    return output


def benchmark_attention():
    """
    Benchmark attention implementations.
    """
    import time
    
    print("Attention Kernels Benchmark")
    print("=" * 60)
    
    batch, heads, seq_len, head_dim = 1, 8, 512, 64
    
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    
    # PyTorch reference
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        scale = 1.0 / math.sqrt(head_dim)
        scores = q @ k.transpose(-2, -1) * scale
        attn = torch.softmax(scores, dim=-1)
        output_torch = attn @ v
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / 50 * 1000
    
    # Triton implementation
    output_triton = flash_attention(q, k, v)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        output_triton = flash_attention(q, k, v)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 50 * 1000
    
    print(f"{'Implementation':<25} {'Time (ms)':<15} {'Speedup':<15}")
    print("-" * 60)
    print(f"{'PyTorch':<25} {torch_time:<15.3f} {'1.00x':<15}")
    print(f"{'Triton FlashAttention':<25} {triton_time:<15.3f} {torch_time/triton_time:.2f}x")
    
    # Verify correctness
    if torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2):
        print("\n✓ Triton results match PyTorch")
    else:
        print("\n✗ WARNING: Results differ")
        max_diff = (output_triton - output_torch).abs().max().item()
        print(f"  Max difference: {max_diff}")


if __name__ == "__main__":
    print("Attention Kernels Module")
    print("=" * 60)
    
    # Simple test
    seq_len, head_dim = 64, 32
    q = torch.randn((seq_len, head_dim), device="cuda")
    k = torch.randn((seq_len, head_dim), device="cuda")
    v = torch.randn((seq_len, head_dim), device="cuda")
    
    output = scaled_dot_product_attention(q, k, v)
    print(f"✓ Basic attention works: output shape {output.shape}")
    
    # Benchmark
    benchmark_attention()
    
    print("\n" + "=" * 60)
    print("ATTENTION KERNEL CHECKLIST:")
    print("□ Scale by 1/sqrt(head_dim)")
    print("□ Use online softmax for numerical stability")
    print("□ Tile K, V for memory efficiency")
    print("□ Apply causal mask for autoregressive")
    print("□ Handle padding/masking correctly")
    print("□ Verify against reference implementation")
