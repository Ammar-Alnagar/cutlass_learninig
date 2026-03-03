# RoPE Math: Rotary Position Embeddings

## What This Is

RoPE encodes token positions by rotating query and key vectors in the complex plane. The rotation angle depends on the token position and the dimension index.

**The key property:** The dot product $Q_m \cdot K_n$ depends on the relative position $m - n$, not absolute positions.

## Why A Kernel Engineer Needs This

**You will implement fused RoPE kernels that rotate Q, K before attention.** The rotation is applied per-dimension and must be fused with the QK projection or QK computation for efficiency.

**Interview relevance:** NVIDIA interviewers ask: "How does RoPE work? Why does it encode relative positions?"

## The Math

### RoPE Rotation Formula

For a 2D vector $(x, y)$ at position $m$, RoPE applies a rotation by angle $m \cdot \theta$:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

$$x' = x \cos(m\theta) - y \sin(m\theta)$$
$$y' = x \sin(m\theta) + y \cos(m\theta)$$

### Frequency Definition

**RoPE frequencies:**
$$\theta_i = 10000^{-2i / d_h} \quad \text{for } i = 0, 1, \ldots, d_h/2 - 1$$

Where $d_h$ is the head dimension.

**Worked example (d_h = 4):**
- $\theta_0 = 10000^0 = 1.0$
- $\theta_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01$

### Complex Number Formulation

RoPE can be written elegantly using complex numbers:

$$z = x + iy$$
$$\text{RoPE}(z, m) = z \cdot e^{im\theta}$$

where $e^{im\theta} = \cos(m\theta) + i\sin(m\theta)$ (Euler's formula).

### RoPE for Attention

**Query at position $m$:**
$$Q_m^{\text{rot}} = Q_m \cdot e^{im\theta}$$

**Key at position $n$:**
$$K_n^{\text{rot}} = K_n \cdot e^{in\theta}$$

**Attention score with RoPE:**
$$Q_m^{\text{rot}} \cdot (K_n^{\text{rot}})^* = (Q_m \cdot e^{im\theta}) \cdot (K_n \cdot e^{in\theta})^*$$
$$= Q_m \cdot K_n^* \cdot e^{i(m-n)\theta}$$

where $^*$ denotes complex conjugate.

**Key insight:** The score depends on $m - n$ (relative position), not $m$ or $n$ individually.

### Why Relative Positions Fall Out

**Without RoPE:**
$$Q_m \cdot K_n = \text{similarity}(Q_m, K_n)$$

No position information.

**With RoPE:**
$$Q_m^{\text{rot}} \cdot (K_n^{\text{rot}})^* = Q_m \cdot K_n^* \cdot e^{i(m-n)\theta}$$

The term $e^{i(m-n)\theta}$ encodes the relative position $m - n$.

**For different dimensions:**
Each dimension pair $(2i, 2i+1)$ has a different frequency $\theta_i$, so the model learns to attend at different relative position scales.

## Shapes and Sizes

| Tensor | Shape | RoPE applied |
|--------|-------|--------------|
| Q | $[B, H, S, d_h]$ | Rotate each $(Q_{:, :, m, 2i}, Q_{:, :, m, 2i+1})$ by $m \cdot \theta_i$ |
| K | $[B, H, S, d_h]$ | Rotate each $(K_{:, :, n, 2i}, K_{:, :, n, 2i+1})$ by $n \cdot \theta_i$ |
| V | $[B, H, S, d_h]$ | No rotation |

## The Kernel Implication

### Fused RoPE Kernel

**Separate RoPE (inefficient):**
```cuda
// Kernel 1: Apply RoPE
for each position m:
    for each dimension pair (2i, 2i+1):
        Q_rot[m, 2i] = Q[m, 2i] * cos(m*θ_i) - Q[m, 2i+1] * sin(m*θ_i)
        Q_rot[m, 2i+1] = Q[m, 2i] * sin(m*θ_i) + Q[m, 2i+1] * cos(m*θ_i)

// Kernel 2: Attention with Q_rot
```

**Fused RoPE (efficient):**
```cuda
// Single kernel: RoPE + QK computation
for each position m, n:
    score = 0
    for each dimension pair (2i, 2i+1):
        // Apply RoPE on-the-fly
        q_rot_2i = Q[m, 2i] * cos(m*θ_i) - Q[m, 2i+1] * sin(m*θ_i)
        q_rot_2i_plus_1 = Q[m, 2i] * sin(m*θ_i) + Q[m, 2i+1] * cos(m*θ_i)
        
        // Dot product with K (also rotated)
        k_rot_2i = K[n, 2i] * cos(n*θ_i) - K[n, 2i+1] * sin(n*θ_i)
        k_rot_2i_plus_1 = K[n, 2i] * sin(n*θ_i) + K[n, 2i+1] * cos(n*θ_i)
        
        score += q_rot_2i * k_rot_2i + q_rot_2i_plus_1 * k_rot_2i_plus_1
```

### Precomputing Cos/Sin

**Frequency table:**
```cpp
// Precompute cos/sin for all positions and frequencies
float cos_table[S_max][d_h/2];
float sin_table[S_max][d_h/2];

for (int m = 0; m < S_max; ++m)
    for (int i = 0; i < d_h/2; ++i) {
        float theta = m * pow(10000, -2.0f * i / d_h);
        cos_table[m][i] = cosf(theta);
        sin_table[m][i] = sinf(theta);
    }
```

**Kernel uses table lookup:**
```cuda
float cos_val = cos_table[position][dim_pair];
float sin_val = sin_table[position][dim_pair];
```

## Numbers That Matter

| Model | d_h | Frequencies | Min θ | Max θ |
|-------|-----|-------------|-------|-------|
| LLaMA-3 8B | 128 | 64 | $10000^{-126/128} \approx 0.0001$ | $10000^0 = 1.0$ |
| LLaMA-3 70B | 128 | 64 | Same | Same |

**Frequency range:** Lower dimensions rotate faster (large θ), higher dimensions rotate slower (small θ).

## Common Interview Questions

**Q1: Why does RoPE encode relative positions?**

<details>
<summary>Answer</summary>

With RoPE, the attention score is:
$Q_m^{\text{rot}} \cdot (K_n^{\text{rot}})^* = Q_m \cdot K_n^* \cdot e^{i(m-n)\theta}$

The term $e^{i(m-n)\theta}$ depends only on $m - n$ (relative position), not on $m$ or $n$ individually.

This is why RoPE naturally encodes relative positions: the rotation angles cancel out except for the difference.
</details>

**Q2: How are RoPE frequencies computed?**

<details>
<summary>Answer</summary>

$\theta_i = 10000^{-2i / d_h}$ for $i = 0, 1, \ldots, d_h/2 - 1$.

For $d_h = 128$:
- $\theta_0 = 10000^0 = 1.0$ (fastest rotation)
- $\theta_{63} = 10000^{-126/128} \approx 0.0001$ (slowest rotation)

Lower dimensions rotate faster, higher dimensions rotate slower.
</details>

**Q3: Why must RoPE be fused with QK computation?**

<details>
<summary>Answer</summary>

Separate RoPE kernel would:
1. Read Q from HBM
2. Apply rotation
3. Write Q_rot to HBM
4. Read Q_rot for attention

Fused RoPE:
1. Read Q from HBM
2. Apply rotation on-the-fly during QK computation
3. Never write intermediate Q_rot

Fused RoPE saves HBM traffic and reduces latency.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01 (attention)

**What this unlocks:**
- Module 05 (FlashAttention): RoPE is applied before the tile loop

**Next:** `02_rope_kernel_implications.md` — kernel implementation details.
