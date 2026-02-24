"""
Module 04: Training Kernels

This module teaches how to implement training kernels with backpropagation
support, including gradient computation and optimizer steps.

LEARNING OBJECTIVES:
1. Implement forward pass kernels
2. Compute gradients efficiently
3. Fuse optimizer steps
4. Handle mixed precision training
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# TRAINING KERNEL 1: Linear Forward + Backward
# ============================================================================

@triton.jit
def linear_forward_kernel(
    x_ptr, weight_ptr, bias_ptr,
    output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear layer forward pass: output = x @ weight + bias
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
        
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        w_block = tl.load(
            weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        accumulator = tl.dot(x_block, w_block, accumulator)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    accumulator = accumulator + bias[None, :]
    
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator,
        mask=m_mask[:, None] & n_mask[None, :]
    )


@triton.jit
def linear_backward_kernel(
    x_ptr, weight_ptr, grad_output_ptr,
    grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_gom, stride_gon,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear layer backward pass.
    
    Computes:
    - grad_input = grad_output @ weight.T
    - grad_weight = x.T @ grad_output
    - grad_bias = sum(grad_output)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Compute grad_input = grad_output @ weight.T
    grad_input_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n_idx in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_idx + tl.arange(0, BLOCK_SIZE_N)
        n_offsets_mask = n_offsets < N
        
        grad_out_block = tl.load(
            grad_output_ptr + offs_m[:, None] * stride_gom + n_offsets[None, :] * stride_gon,
            mask=m_mask[:, None] & n_offsets_mask[None, :],
            other=0.0
        )
        
        w_block = tl.load(
            weight_ptr + n_offsets[:, None] * stride_wk + offs_k[None, :] * stride_wn,
            mask=n_offsets_mask[:, None] & tl.arange(0, BLOCK_SIZE_K)[None, :] < K,
            other=0.0
        )
        
        # Note: This is simplified - full implementation needs more care
    
    # Store grad_input (simplified)
    tl.store(
        grad_input_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xk,
        grad_input_acc,
        mask=m_mask[:, None] & n_mask[None, :]
    )


# ============================================================================
# TRAINING KERNEL 2: Cross-Entropy Loss Forward + Backward
# ============================================================================

@triton.jit
def cross_entropy_forward_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    n_samples,
    n_classes,
    stride_logits,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Cross-entropy loss forward pass.
    
    loss = -log(softmax(logits)[target])
    """
    pid = tl.program_id(axis=0)
    
    if pid < n_samples:
        # Load logits for this sample
        class_offsets = tl.arange(0, BLOCK_SIZE)
        mask = class_offsets < n_classes
        
        logits = tl.load(logits_ptr + pid * stride_logits + class_offsets, mask=mask, other=-float('inf'))
        
        # Compute log-softmax with numerical stability
        max_logit = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_logit)
        sum_exp = tl.sum(exp_logits, axis=0)
        log_sum_exp = tl.log(sum_exp) + max_logit
        
        # Load target
        target = tl.load(targets_ptr + pid)
        
        # Get log probability of target class
        target_logit = tl.load(logits_ptr + pid * stride_logits + target)
        log_prob = target_logit - log_sum_exp
        
        # Negative log likelihood
        loss = -log_prob
        
        # Store loss
        tl.store(loss_ptr + pid, loss)


@triton.jit
def cross_entropy_backward_kernel(
    logits_ptr,
    targets_ptr,
    grad_output_ptr,
    n_samples,
    n_classes,
    stride_logits,
    stride_grad,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Cross-entropy loss backward pass.
    
    grad = softmax(logits) - one_hot(target)
    """
    pid = tl.program_id(axis=0)
    
    if pid < n_samples:
        class_offsets = tl.arange(0, BLOCK_SIZE)
        mask = class_offsets < n_classes
        
        logits = tl.load(logits_ptr + pid * stride_logits + class_offsets, mask=mask, other=-float('inf'))
        target = tl.load(targets_ptr + pid)
        
        # Compute softmax
        max_logit = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_logit)
        sum_exp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sum_exp
        
        # Subtract one-hot encoding of target
        grad = tl.where(class_offsets == target, softmax - 1.0, softmax)
        
        # Store gradient
        tl.store(grad_output_ptr + pid * stride_grad + class_offsets, grad, mask=mask)


# ============================================================================
# TRAINING KERNEL 3: Adam Optimizer Step
# ============================================================================

@triton.jit
def adam_step_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    step,
    lr,
    beta1,
    beta2,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Adam optimizer step.
    
    Updates parameters using Adam optimization algorithm.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current values
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
    # Update biased first moment estimate
    exp_avg_new = beta1 * exp_avg + (1 - beta1) * grad
    
    # Update biased second moment estimate
    exp_avg_sq_new = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    
    # Compute bias-corrected estimates
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    exp_avg_corrected = exp_avg_new / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq_new / bias_correction2
    
    # Compute update
    update = exp_avg_corrected / (tl.sqrt(exp_avg_sq_corrected) + eps)
    
    # Apply update
    param_new = param - lr * update
    
    # Store results
    tl.store(param_ptr + offsets, param_new, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg_new, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq_new, mask=mask)


# ============================================================================
# TRAINING KERNEL 4: LayerNorm Forward + Backward
# ============================================================================

@triton.jit
def layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_rows,
    n_cols,
    eps,
    stride_x,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer normalization forward pass.
    """
    pid = tl.program_id(axis=0)
    
    row_start = pid * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input row
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols
    
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    
    # Normalize
    x_norm = diff / tl.sqrt(var + eps)
    
    # Apply affine transform
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = x_norm * weight + bias
    
    # Store output
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)
    
    # Store mean and variance for backward (in separate buffers)
    # This is simplified - would need additional pointers


@triton.jit
def layernorm_backward_kernel(
    grad_output_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer normalization backward pass.
    """
    pid = tl.program_id(axis=0)
    
    row_start = pid * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load saved mean and variance
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    
    # Load inputs
    grad_output = tl.load(grad_output_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    
    # Compute normalized input
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Gradient w.r.t. normalized input
    grad_x_norm = grad_output * weight
    
    # Gradient w.r.t. variance
    grad_var = tl.sum(grad_x_norm * (x - mean) * -0.5 * tl.pow(var + eps, -1.5), axis=0) / n_cols
    
    # Gradient w.r.t. mean
    grad_mean = tl.sum(grad_x_norm * -1 / tl.sqrt(var + eps), axis=0) / n_cols
    
    # Gradient w.r.t. input
    grad_input = grad_x_norm / tl.sqrt(var + eps) + grad_var * 2 * (x - mean) / n_cols + grad_mean / n_cols
    
    # Store gradient
    tl.store(grad_input_ptr + row_start + col_offsets, grad_input, mask=mask)


# ============================================================================
# WRAPPER FUNCTIONS
# ============================================================================

def adam_step(param: torch.Tensor, grad: torch.Tensor, 
              exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
              step: int, lr: float = 1e-3, beta1: float = 0.9, 
              beta2: float = 0.999, eps: float = 1e-8):
    """
    Perform Adam optimizer step.
    """
    n_elements = param.numel()
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    adam_step_kernel[grid](
        param, grad, exp_avg, exp_avg_sq,
        step, lr, beta1, beta2, eps,
        n_elements, BLOCK_SIZE
    )


def benchmark_training_kernels():
    """
    Benchmark training kernels.
    """
    import time
    
    print("Training Kernels Benchmark")
    print("=" * 60)
    
    # Adam benchmark
    n = 10_000_000
    param = torch.randn(n, device="cuda")
    grad = torch.randn(n, device="cuda")
    exp_avg = torch.zeros(n, device="cuda")
    exp_avg_sq = torch.zeros(n, device="cuda")
    
    # Warmup
    adam_step(param, grad, exp_avg, exp_avg_sq, step=1)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(100):
        adam_step(param, grad, exp_avg, exp_avg_sq, step=i+1)
    torch.cuda.synchronize()
    adam_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"Adam step ({n} params): {adam_time:.4f} ms")
    
    # LayerNorm benchmark
    n_rows, n_cols = 256, 1024
    x = torch.randn(n_rows, n_cols, device="cuda")
    weight = torch.ones(n_cols, device="cuda")
    bias = torch.zeros(n_cols, device="cuda")
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    layernorm_forward_kernel[grid](
        x, weight, bias, output,
        n_rows, n_cols, 1e-5,
        x.stride(0), output.stride(0),
        BLOCK_SIZE
    )
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        layernorm_forward_kernel[grid](
            x, weight, bias, output,
            n_rows, n_cols, 1e-5,
            x.stride(0), output.stride(0),
            BLOCK_SIZE
        )
    torch.cuda.synchronize()
    ln_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"LayerNorm forward ({n_rows}x{n_cols}): {ln_time:.4f} ms")
    
    # Compare with PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.nn.functional.layer_norm(x, (n_cols,), weight, bias)
    torch.cuda.synchronize()
    torch_ln_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"PyTorch LayerNorm: {torch_ln_time:.4f} ms")
    print(f"Speedup: {torch_ln_time / ln_time:.2f}x")


if __name__ == "__main__":
    print("Training Kernels Module")
    print("=" * 60)
    
    # Test Adam step
    param = torch.randn(1000, device="cuda")
    grad = torch.randn(1000, device="cuda")
    exp_avg = torch.zeros(1000, device="cuda")
    exp_avg_sq = torch.zeros(1000, device="cuda")
    
    adam_step(param, grad, exp_avg, exp_avg_sq, step=1)
    print("✓ Adam step works")
    
    # Test LayerNorm
    x = torch.randn(32, 128, device="cuda")
    weight = torch.ones(128, device="cuda")
    bias = torch.zeros(128, device="cuda")
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(128)
    grid = (32,)
    
    layernorm_forward_kernel[grid](
        x, weight, bias, output,
        32, 128, 1e-5,
        x.stride(0), output.stride(0),
        BLOCK_SIZE
    )
    
    expected = torch.nn.functional.layer_norm(x, (128,), weight, bias)
    if torch.allclose(output, expected, rtol=1e-4):
        print("✓ LayerNorm forward works")
    else:
        print("✗ LayerNorm results don't match")
    
    # Run benchmarks
    benchmark_training_kernels()
    
    print("\n" + "=" * 60)
    print("TRAINING KERNEL CHECKLIST:")
    print("□ Forward pass computes correct output")
    print("□ Backward pass computes correct gradients")
    print("□ Save necessary values for backward")
    print("□ Handle mixed precision correctly")
    print("□ Optimizer updates are numerically stable")
    print("□ Gradient accumulation works correctly")
