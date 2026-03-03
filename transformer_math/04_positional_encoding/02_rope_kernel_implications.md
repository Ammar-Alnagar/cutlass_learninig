# RoPE Kernel Implications

## What This Is

RoPE must be fused with QK computation for efficiency. The rotation is applied per-dimension pair, requiring $d_h$ to be divisible by 2.

## Why A Kernel Engineer Needs This

**You will implement fused RoPE in your FlashAttention kernel.** The rotation is applied during the QK^T tile computation, not as a separate kernel.

## The Math

### Head Dimension Divisibility

**RoPE requirement:** $d_h$ must be divisible by 2.

**Reason:** RoPE operates on 2D planes (dimension pairs). Each pair $(2i, 2i+1)$ is rotated together.

**LLaMA-3:** $d_h = 128$ (divisible by 2, and by 64 for tensor cores).

### Partial RoPE

Some models apply RoPE to only a subset of dimensions:

**LLaMA-3 8B:**
- RoPE dimensions: $d_{\text{rope}} = 128$ (all dimensions)
- Non-RoPE dimensions: 0

**Some models:**
- RoPE dimensions: $d_{\text{rope}} < d_h$
- Non-RoPE dimensions: $d_h - d_{\text{rope}}$

**Kernel handles both:**
```cuda
for (int i = 0; i < d_rope / 2; ++i) {
    // Apply RoPE to dimension pair (2i, 2i+1)
}
for (int i = d_rope / 2; i < d_h / 2; ++i) {
    // No rotation (identity)
}
```

## The Kernel Implication

### Fused RoPE in FlashAttention

```cuda
__global__ void flash_attention_with_rope(Q, K, V, O, cos_table, sin_table) {
    // Load Q, K tiles
    float Q_tile[B_r][d_h];
    float K_tile[B_c][d_h];
    
    // Apply RoPE on-the-fly during QK^T computation
    for (int i = 0; i < B_r; ++i) {
        int q_pos = tile_i * B_r + i;
        for (int j = 0; j < B_c; ++j) {
            int k_pos = tile_j * B_c + j;
            
            // Compute QK^T with RoPE
            float score = 0;
            for (int dim = 0; dim < d_h / 2; ++dim) {
                // Load cos/sin for this position and frequency
                float cos_q = cos_table[q_pos][dim];
                float sin_q = sin_table[q_pos][dim];
                float cos_k = cos_table[k_pos][dim];
                float sin_k = sin_table[k_pos][dim];
                
                // Rotate Q
                float q_2i = Q_tile[i][2*dim];
                float q_2i_plus_1 = Q_tile[i][2*dim + 1];
                float q_rot_2i = q_2i * cos_q - q_2i_plus_1 * sin_q;
                float q_rot_2i_plus_1 = q_2i * sin_q + q_2i_plus_1 * cos_q;
                
                // Rotate K
                float k_2i = K_tile[j][2*dim];
                float k_2i_plus_1 = K_tile[j][2*dim + 1];
                float k_rot_2i = k_2i * cos_k - k_2i_plus_1 * sin_k;
                float k_rot_2i_plus_1 = k_2i * sin_k + k_2i_plus_1 * cos_k;
                
                // Dot product
                score += q_rot_2i * k_rot_2i + q_rot_2i_plus_1 * k_rot_2i_plus_1;
            }
            scores[i][j] = score / sqrt(d_h);
        }
    }
}
```

### CuTe RoPE Example

```cpp
// CuTe fused RoPE + QK
struct FusedRoPEQK {
    template <typename QTile, typename KTile, typename CosSinTable>
    __device__ static auto compute(QTile const& Q, KTile const& K, 
                                    CosSinTable const& table,
                                    int q_pos, int k_pos) {
        float score = 0;
        
        // Unroll over dimension pairs
        #pragma unroll
        for (int i = 0; i < d_h / 2; ++i) {
            float cos_q = table.cos(q_pos, i);
            float sin_q = table.sin(q_pos, i);
            float cos_k = table.cos(k_pos, i);
            float sin_k = table.sin(k_pos, i);
            
            // Rotate and accumulate
            score += rope_dot(Q(i), K(i), cos_q, sin_q, cos_k, sin_k);
        }
        
        return score;
    }
};
```

## Numbers That Matter

| Operation | Separate RoPE | Fused RoPE | Speedup |
|-----------|---------------|------------|---------|
| HBM reads | 2× Q (read, read) | 1× Q (read) | 2x |
| HBM writes | 1× Q_rot | 0 | ∞ |
| Latency | 2 kernel launches | 1 kernel launch | 2x |

## Common Interview Questions

**Q1: Why must d_h be divisible by 2 for RoPE?**

<details>
<summary>Answer</summary>

RoPE operates on 2D planes. Each dimension pair $(2i, 2i+1)$ is rotated together using a 2×2 rotation matrix.

If $d_h$ is odd, the last dimension has no pair and cannot be rotated.

LLaMA-3 uses $d_h = 128$ (divisible by 2, and by 64 for tensor core efficiency).
</details>

**Q2: What is the memory overhead of RoPE cos/sin tables?**

<details>
<summary>Answer</summary>

Cos/sin table size:
- Positions: $S_{\text{max}}$ (e.g., 8192)
- Frequencies: $d_h / 2$ (e.g., 64)
- Elements: $S_{\text{max}} \times d_h / 2 \times 2$ (cos + sin)

For LLaMA-3 8B ($S_{\text{max}} = 8192, d_h = 128$):
- Elements: $8192 \times 64 \times 2 = 1,048,576$
- Bytes (FP32): 4 MB

This is small and fits in L2 cache.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 04.1 (RoPE math)

**Next:** Run `rope.py` to see RoPE rotation visually.
