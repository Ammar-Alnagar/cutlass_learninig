# Module 03: Attention Kernels

## Learning Objectives
1. Understand attention computation patterns
2. Implement efficient attention kernels
3. Apply tiling to attention for memory efficiency
4. Optimize for transformer workloads

## Attention Background

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

### Computational Complexity
- **Time**: O(n² × d) for sequence length n, head dimension d
- **Memory**: O(n²) for attention matrix

## FlashAttention Concept

### Problem
Standard attention materializes the n×n attention matrix, which:
- Uses O(n²) memory
- Causes multiple HBM accesses
- Limits sequence length

### Solution: Tiling
FlashAttention tiles the computation to fit in SRAM:
- Process Q, K, V in blocks
- Never materialize full attention matrix
- O(n) memory complexity

### Online Softmax
For numerical stability with tiling:
```python
# Online softmax algorithm
max_val = -inf
sum_exp = 0
accumulator = 0

for block in blocks:
    scores = Q @ K_block.T
    new_max = max(max_val, max(scores))
    exp_scores = exp(scores - new_max)
    accumulator = accumulator * exp(max_val - new_max) + exp_scores @ V_block
    max_val = new_max
    sum_exp = sum_exp * exp(max_val - new_max) + sum(exp_scores)

output = accumulator / sum_exp
```

## Kernel Implementations

### Basic Attention
```python
@triton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, seq_len, head_dim, ...):
    q_idx = tl.program_id(0)
    
    # Load query
    q = tl.load(q_ptr + q_idx * stride + offsets)
    
    # Loop over keys/values
    for kv_idx in range(0, seq_len, BLOCK_SIZE):
        k_block = tl.load(k_ptr + kv_offsets)
        v_block = tl.load(v_ptr + kv_offsets)
        
        scores = tl.sum(q * k_block) * scale
        
        # Online softmax update
        ...
```

### Causal Attention
```python
# Add causal mask
causal_mask = kv_offsets <= q_idx
scores = tl.where(causal_mask, scores, -float('inf'))
```

### Multi-Head Attention
```python
# Program grid: (batch, heads, query_blocks)
pid_b = tl.program_id(0)
pid_h = tl.program_id(1)
pid_q = tl.program_id(2)
```

## Block Size Selection

| Parameter | Typical Value | Considerations |
|-----------|---------------|----------------|
| BLOCK_SIZE_Q | 32-64 | Query block size |
| BLOCK_SIZE_KV | 32-64 | KV block size |
| BLOCK_SIZE_D | 32-64 | Head dimension tile |

## Memory Efficiency

### Standard Attention
```
HBM accesses: O(n² × d)
SRAM usage: O(n² + n × d)
```

### FlashAttention
```
HBM accesses: O(n × d)
SRAM usage: O(BLOCK_SIZE² + BLOCK_SIZE × d)
```

## Exercises

### Exercise 1: Implement Basic Attention
Create a simple attention kernel without tiling.

### Exercise 2: Add Causal Masking
Modify attention to only attend to previous positions.

### Exercise 3: Multi-Head Support
Extend to handle batch and multiple heads.

## Performance Tips

1. **Choose block sizes carefully** - Balance SRAM usage
2. **Use online softmax** - Numerical stability
3. **Minimize HBM accesses** - Tile K, V effectively
4. **Consider causal pattern** - Triangular computation
5. **Profile memory traffic** - Use Nsight Compute

## Common Patterns

### QKV Projection Fusion
```python
# Fuse Q, K, V projections
qkv = fused_qkv_projection(x, w_qkv, b_qkv)
q, k, v = split(qkv)
```

### RoPE Integration
```python
# Apply rotary embeddings inline
q = apply_rope(q, positions)
k = apply_rope(k, positions)
```

### ALiBi Integration
```python
# Add positional bias to scores
scores = scores + alibi_bias
```

## Next Steps
After mastering attention kernels, move to Module 04: Training Kernels for backpropagation support.
