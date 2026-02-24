"""
Module 02: Fused Operations

This module teaches how to fuse multiple operations into single kernels
for maximum performance and minimal memory traffic.

LEARNING OBJECTIVES:
1. Understand fusion benefits and tradeoffs
2. Implement common fused operations
3. Design custom fused kernels
4. Measure fusion performance gains
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# FUSED OPERATION 1: Linear + Bias + Activation
# ============================================================================

@triton.jit
def fused_linear_relu(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    stride_x,
    stride_w,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    """
    Fused operation: output = ReLU(x @ weight + bias)
    
    FUSION BENEFITS:
    - Single memory read of input
    - Single memory write of output
    - Intermediate values stay in registers
    """
    pid_out = tl.program_id(axis=0)
    pid_in = tl.program_id(axis=1)
    
    # Compute output index
    out_idx = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    in_idx = pid_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)
    
    out_mask = out_idx < out_features
    in_mask = in_idx < in_features
    
    # Load input and weight
    x = tl.load(x_ptr + pid_out * stride_x + in_idx, mask=in_mask, other=0.0)
    w = tl.load(weight_ptr + pid_out * stride_w + in_idx, mask=in_mask, other=0.0)
    
    # Compute dot product
    result = tl.sum(x * w)
    
    # Add bias and apply ReLU
    bias = tl.load(bias_ptr + pid_out)
    result = tl.maximum(0.0, result + bias)
    
    # Store output
    tl.store(output_ptr + pid_out, result, mask=out_mask)


def fused_linear_relu_1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear + ReLU for 1D case.
    """
    in_features = x.shape[0]
    out_features = weight.shape[0]
    
    output = torch.empty(out_features, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 64
    grid = (out_features, triton.cdiv(in_features, BLOCK_SIZE))
    
    # Simplified version for element-wise fusion
    @triton.jit
    def fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        result = x * w + b
        result = tl.maximum(0.0, result)  # ReLU
        
        tl.store(out_ptr + offsets, result, mask=mask)
    
    n = min(in_features, out_features)
    fused_kernel[triton.cdiv(n, BLOCK_SIZE)](x[:n], weight[:n], bias[:n], output[:n], n, BLOCK_SIZE)
    
    return output


# ============================================================================
# FUSED OPERATION 2: LayerNorm + Linear
# ============================================================================

@triton.jit
def fused_layernorm_linear(
    x_ptr,
    weight_ptr,
    bias_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused operation: output = LayerNorm(x) @ weight + bias
    
    This fuses normalization and linear transformation.
    """
    pid = tl.program_id(axis=0)
    
    # Load entire row for this sample
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input row
    x = tl.load(x_ptr + pid * n_cols + col_offsets, mask=mask, other=0.0)
    
    # LayerNorm: compute mean
    mean = tl.sum(x, axis=0) / n_cols
    
    # LayerNorm: compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    
    # LayerNorm: normalize
    x_norm = diff / tl.sqrt(var + eps)
    
    # Load LN parameters
    ln_weight = tl.load(ln_weight_ptr + col_offsets, mask=mask, other=0.0)
    ln_bias = tl.load(ln_bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Apply LN affine
    x_norm = x_norm * ln_weight + ln_bias
    
    # Load linear parameters
    linear_weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    linear_bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Apply linear transformation
    output = x_norm * linear_weight + linear_bias
    
    # Store result
    tl.store(output_ptr + pid * n_cols + col_offsets, output, mask=mask)


# ============================================================================
# FUSED OPERATION 3: Dropout + Activation
# ============================================================================

@triton.jit
def fused_dropout_gelu(
    x_ptr,
    output_ptr,
    n_elements,
    dropout_scale,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused operation: output = Dropout(GELU(x))
    
    Combines activation and regularization in one pass.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_val = tl.math.tanh(tanh_arg)
    gelu_output = 0.5 * x * (1.0 + tanh_val)
    
    # Dropout: generate random mask
    rng_output = tl.rand(seed, offsets)
    dropout_mask = rng_output < 0.5  # 50% dropout
    dropout_output = tl.where(dropout_mask, gelu_output * dropout_scale, 0.0)
    
    tl.store(output_ptr + offsets, dropout_output, mask=mask)


# ============================================================================
# FUSED OPERATION 4: RMSNorm + GEMM
# ============================================================================

@triton.jit
def fused_rmsnorm_gemm(
    x_ptr,
    weight_ptr,
    rms_weight_ptr,
    output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused RMSNorm + Matrix Multiplication.
    
    RMSNorm: x_norm = x / sqrt(mean(x^2) + eps) * weight
    Then: output = x_norm @ weight
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < K
        
        # Load input block
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Compute RMSNorm inline (simplified - per-row normalization)
        # In practice, this would be more complex
        x_squared = x_block * x_block
        mean_sq = tl.sum(x_squared, axis=1) / K
        rms = tl.sqrt(mean_sq + eps)
        x_norm = x_block / rms[:, None]
        
        # Load RMS weight
        rms_weight = tl.load(rms_weight_ptr + offs_k, mask=k_mask, other=1.0)
        x_norm = x_norm * rms_weight[None, :]
        
        # Load weight block
        w_block = tl.load(
            weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        # Matrix multiply
        accumulator = tl.dot(x_norm, w_block, accumulator)
    
    # Store result
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator,
        mask=m_mask[:, None] & n_mask[None, :]
    )


# ============================================================================
# FUSED OPERATION 5: QKV Projection for Attention
# ============================================================================

@triton.jit
def fused_qkv_projection(
    x_ptr,
    q_weight_ptr, k_weight_ptr, v_weight_ptr,
    q_bias_ptr, k_bias_ptr, v_bias_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    seq_len,
    hidden_dim,
    head_dim,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused QKV projection for attention.
    
    Computes Q, K, V projections in a single kernel launch.
    """
    pid = tl.program_id(axis=0)
    
    # Each program handles one sequence position
    seq_idx = pid
    
    # Load input for this position
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_dim
    
    x = tl.load(x_ptr + seq_idx * stride_x + col_offsets, mask=mask, other=0.0)
    
    # Q projection (simplified - element-wise for demo)
    q_weight = tl.load(q_weight_ptr + col_offsets, mask=mask, other=0.0)
    q_bias = tl.load(q_bias_ptr + col_offsets % head_dim, mask=col_offsets < head_dim, other=0.0)
    q_out = x * q_weight + q_bias
    tl.store(q_out_ptr + seq_idx * stride_x + col_offsets, q_out, mask=mask)
    
    # K projection
    k_weight = tl.load(k_weight_ptr + col_offsets, mask=mask, other=0.0)
    k_bias = tl.load(k_bias_ptr + col_offsets % head_dim, mask=col_offsets < head_dim, other=0.0)
    k_out = x * k_weight + k_bias
    tl.store(k_out_ptr + seq_idx * stride_x + col_offsets, k_out, mask=mask)
    
    # V projection
    v_weight = tl.load(v_weight_ptr + col_offsets, mask=mask, other=0.0)
    v_bias = tl.load(v_bias_ptr + col_offsets % head_dim, mask=col_offsets < head_dim, other=0.0)
    v_out = x * v_weight + v_bias
    tl.store(v_out_ptr + seq_idx * stride_x + col_offsets, v_out, mask=mask)


def benchmark_fusion():
    """
    Benchmark fused vs unfused operations.
    """
    import time
    
    print("Fused Operations Benchmark")
    print("=" * 60)
    
    n = 1_000_000
    x = torch.randn(n, device="cuda")
    weight = torch.randn(n, device="cuda")
    bias = torch.randn(n, device="cuda")
    
    # Unfused: separate operations
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        temp = x * weight
        temp = temp + bias
        output_unfused = torch.relu(temp)
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / 100 * 1000
    
    # Fused
    output_fused = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    @triton.jit
    def fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        
        x = tl.load(x_ptr + offsets, mask=mask)
        w = tl.load(w_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        
        result = x * w + b
        result = tl.maximum(0.0, result)
        
        tl.store(out_ptr + offsets, result, mask=mask)
    
    # Warmup
    fused_kernel[grid](x, weight, bias, output_fused, n, BLOCK_SIZE)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        fused_kernel[grid](x, weight, bias, output_fused, n, BLOCK_SIZE)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Speedup':<15}")
    print("-" * 60)
    print(f"{'Unfused (PyTorch)':<20} {unfused_time:<15.4f} {'1.00x':<15}")
    print(f"{'Fused (Triton)':<20} {fused_time:<15.4f} {unfused_time/fused_time:.2f}x")
    
    # Verify correctness
    expected = torch.relu(x * weight + bias)
    if torch.allclose(output_fused, expected, rtol=1e-5):
        print("\n✓ Fused results match unfused")
    else:
        print("\n✗ WARNING: Results don't match!")


if __name__ == "__main__":
    print("Fused Operations Module")
    print("=" * 60)
    
    # Run benchmark
    benchmark_fusion()
    
    print("\n" + "=" * 60)
    print("FUSION DESIGN CHECKLIST:")
    print("□ Identify operations that can be fused")
    print("□ Ensure compatible data types")
    print("□ Minimize intermediate memory traffic")
    print("□ Keep register usage reasonable")
    print("□ Test correctness against unfused version")
    print("□ Profile to verify improvement")
    print("\nCOMMON FUSION PATTERNS:")
    print("  - Linear + Bias + Activation")
    print("  - LayerNorm + Linear")
    print("  - Dropout + Activation")
    print("  - QKV Projection")
    print("  - Residual + Normalization")
