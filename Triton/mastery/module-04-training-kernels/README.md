# Module 04: Training Kernels

## Learning Objectives
1. Implement forward pass kernels
2. Compute gradients efficiently
3. Fuse optimizer steps
4. Handle mixed precision training

## Training Pipeline Components

### 1. Forward Pass
```
Input → Layer 1 → Activation → Layer 2 → ... → Output → Loss
```

### 2. Backward Pass
```
Loss → Grad Output → Grad Layer 2 → ... → Grad Input
```

### 3. Optimizer Step
```
Gradient → Update Moments → Compute Step → Update Params
```

## Forward Pass Kernels

### Linear Forward
```python
@triton.jit
def linear_forward(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K, ...):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Matrix multiply
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        x_block = tl.load(...)
        w_block = tl.load(...)
        acc = tl.dot(x_block, w_block, acc)
    
    # Add bias
    bias = tl.load(b_ptr + offsets)
    acc = acc + bias
    
    tl.store(out_ptr + offsets, acc)
```

### LayerNorm Forward
```python
@triton.jit
def layernorm_forward(x_ptr, w_ptr, b_ptr, out_ptr, n_cols, eps, ...):
    pid = tl.program_id(0)  # One program per row
    
    # Load row
    x = tl.load(x_ptr + row_offsets)
    
    # Compute statistics
    mean = tl.sum(x) / n_cols
    var = tl.sum((x - mean)**2) / n_cols
    
    # Normalize and transform
    x_norm = (x - mean) / sqrt(var + eps)
    out = x_norm * w + b
    
    # Save for backward
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)
```

## Backward Pass Kernels

### Gradient Computation Rules
```
y = f(x)
dy/dx = f'(x)

Chain rule:
L = g(y), y = f(x)
dL/dx = dL/dy * dy/dx
```

### LayerNorm Backward
```python
@triton.jit
def layernorm_backward(grad_out, x, w, mean, var, ...):
    # Load saved statistics
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    
    # Normalized input
    x_norm = (x - mean) / sqrt(var + eps)
    
    # Gradient through affine transform
    grad_x_norm = grad_out * w
    
    # Gradient through normalization
    grad_var = sum(grad_x_norm * (x - mean) * -0.5 * (var+eps)^-1.5) / N
    grad_mean = sum(grad_x_norm * -1/sqrt(var+eps)) / N
    
    grad_input = grad_x_norm / sqrt(var+eps) + 
                 grad_var * 2 * (x - mean) / N + 
                 grad_mean / N
```

## Optimizer Kernels

### Adam Optimizer
```python
@triton.jit
def adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps):
    # Update moments
    exp_avg_new = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq_new = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    
    # Bias correction
    bc1 = 1 - beta1^step
    bc2 = 1 - beta2^step
    
    # Compute update
    update = (exp_avg_new / bc1) / (sqrt(exp_avg_sq_new / bc2) + eps)
    
    # Apply update
    param_new = param - lr * update
```

### SGD with Momentum
```python
@triton.jit
def sgd_momentum_step(param, grad, momentum_buf, lr, momentum):
    momentum_buf_new = momentum * momentum_buf + grad
    param_new = param - lr * momentum_buf_new
```

## Mixed Precision Training

### AMP Pattern
```python
# Forward in FP16
x_fp16 = x.to(torch.float16)
output = kernel_fp16(x_fp16)

# Loss in FP32
loss = compute_loss(output.float(), target)

# Backward with scaling
scaler.scale(loss).backward()

# Optimizer step
scaler.step(optimizer)
scaler.update()
```

### Gradient Scaling
```python
@triton.jit
def scale_gradient(grad_ptr, scale, n, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    grad = tl.load(grad_ptr + offsets)
    scaled_grad = grad * scale
    
    tl.store(grad_ptr + offsets, scaled_grad)
```

## Exercises

### Exercise 1: ReLU Backward
Implement ReLU backward kernel.

### Exercise 2: Dropout Backward
Implement dropout backward with saved mask.

### Exercise 3: SGD Optimizer
Implement SGD with momentum kernel.

## Best Practices

1. **Save intermediates** - Store values needed for backward
2. **Use FP32 accumulation** - Avoid precision loss
3. **Handle numerical stability** - Clip gradients if needed
4. **Fuse when possible** - Reduce memory traffic
5. **Profile memory usage** - Watch for OOM

## Memory Management

### Activation Checkpointing
```python
# Save memory by recomputing forward
with torch.utils.checkpoint.checkpoint():
    output = model(input)
```

### Gradient Accumulation
```python
# Accumulate gradients over multiple batches
for micro_batch in micro_batches:
    loss = forward(micro_batch)
    loss.backward()  # Accumulates gradients

optimizer.step()  # Single step with accumulated grads
```

## Next Steps
After mastering training kernels, move to Module 05: Production-Ready for deployment considerations.
