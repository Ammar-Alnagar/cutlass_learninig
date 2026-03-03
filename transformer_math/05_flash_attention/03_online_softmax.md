# Online Softmax: The Rescaling Formula

## What This Is

Online softmax computes softmax in a streaming fashion, without requiring all scores upfront. It maintains a running maximum $m_i$ and running sum $l_i$, and **rescales** the output accumulator when a new maximum is found.

**This is the hardest math in this directory.** Understanding the rescaling formula is essential for implementing FlashAttention correctly.

## Why A Kernel Engineer Needs This

**This is the exact formula you implement inside your FlashAttention tile loop.** When you process key/value tiles one at a time, each tile has different scores. The softmax must be normalized across **all** tiles, not just the current tile.

**Interview relevance:** Cerebras interviewers ask: "Derive the online softmax rescaling formula. Why do you need to rescale the output accumulator?"

## The Math

### Standard Softmax (Batch Mode)

Given scores $s \in \mathbb{R}^S$ for one query attending to $S$ keys:

$$p_i = \frac{\exp(s_i)}{\sum_{j=1}^S \exp(s_j)}$$

**Problem:** Requires all $s_1, \ldots, s_S$ upfront. For FlashAttention, scores come tile-by-tile.

### Online Softmax (Streaming Mode)

**Goal:** Compute softmax as scores arrive in chunks (tiles).

**Key insight:** Softmax can be written in terms of the maximum score:

$$p_i = \frac{\exp(s_i)}{\sum_{j=1}^S \exp(s_j)} = \frac{\exp(s_i - m)}{\sum_{j=1}^S \exp(s_j - m)}$$

where $m = \max_j s_j$.

**Why subtract $m$?** Numerical stability. $\exp(s_i - m) \leq 1$, avoiding overflow.

### Running Maximum and Sum

**State per query:**
- $m_i$: Running maximum of scores seen so far
- $l_i$: Running sum of $\exp(s_j - m_i)$ for all scores seen so far

**Initialization:**
$$m_i^{(0)} = -\infty, \quad l_i^{(0)} = 0$$

### Processing Tile $t$

**Input:**
- Previous state: $m_i^{(t-1)}, l_i^{(t-1)}$
- New scores for this tile: $s^{(t)} = [s_1^{(t)}, \ldots, s_{B_c}^{(t)}]$

**Step 1: Compute tile maximum**
$$m^{(t)}_{\text{tile}} = \max_j s_j^{(t)}$$

**Step 2: Update running maximum**
$$m_i^{(t)} = \max\left(m_i^{(t-1)}, m^{(t)}_{\text{tile}}\right)$$

**Step 3: Compute rescaling factor**

When the maximum changes from $m_i^{(t-1)}$ to $m_i^{(t)}$, all previous exponentials must be rescaled:

$$\exp(s_j - m_i^{(t)}) = \exp(s_j - m_i^{(t-1)} + m_i^{(t-1)} - m_i^{(t)}) = \exp(s_j - m_i^{(t-1)}) \cdot \exp(m_i^{(t-1)} - m_i^{(t)})$$

**Rescaling factor:**
$$\alpha = \exp(m_i^{(t-1)} - m_i^{(t)})$$

**Step 4: Update running sum**

The new running sum is:
$$l_i^{(t)} = \alpha \cdot l_i^{(t-1)} + \sum_{j=1}^{B_c} \exp(s_j^{(t)} - m_i^{(t)})$$

**Derivation:**
$$l_i^{(t)} = \sum_{\text{all scores}} \exp(s - m_i^{(t)})$$
$$= \sum_{\text{old scores}} \exp(s - m_i^{(t)}) + \sum_{\text{new scores}} \exp(s - m_i^{(t)})$$
$$= \sum_{\text{old scores}} \exp(s - m_i^{(t-1)}) \cdot \exp(m_i^{(t-1)} - m_i^{(t)}) + \sum_{\text{new scores}} \exp(s - m_i^{(t)})$$
$$= \alpha \cdot \underbrace{\sum_{\text{old scores}} \exp(s - m_i^{(t-1)})}_{l_i^{(t-1)}} + \sum_{\text{new scores}} \exp(s - m_i^{(t)})$$
$$= \alpha \cdot l_i^{(t-1)} + \sum_{j=1}^{B_c} \exp(s_j^{(t)} - m_i^{(t)})$$

### Output Accumulator Rescaling

**The output accumulator** $O_i$ stores the weighted sum of values:
$$O_i = \sum_{j=1}^{S} p_j V_j = \sum_{j=1}^{S} \frac{\exp(s_j - m_i)}{l_i} V_j$$

**When the maximum changes,** the output accumulator must be rescaled:

**Before tile $t$:**
$$O_i^{(t-1)} = \sum_{\text{old scores}} \frac{\exp(s - m_i^{(t-1)})}{l_i^{(t-1)}} V$$

**After tile $t$ (before normalization):**
$$O_i^{(t)} = \alpha \cdot O_i^{(t-1)} + \sum_{j=1}^{B_c} \exp(s_j^{(t)} - m_i^{(t)}) V_j^{(t)}$$

**Derivation:**

The unnormalized output (before dividing by $l_i$) is:
$$\tilde{O}_i = \sum_{j=1}^{S} \exp(s_j - m_i) V_j$$

When $m_i$ changes from $m_i^{(t-1)}$ to $m_i^{(t)}$:
$$\tilde{O}_i^{(t)} = \sum_{\text{old}} \exp(s - m_i^{(t)}) V + \sum_{\text{new}} \exp(s - m_i^{(t)}) V$$
$$= \sum_{\text{old}} \exp(s - m_i^{(t-1)}) \cdot \exp(m_i^{(t-1)} - m_i^{(t)}) V + \sum_{\text{new}} \exp(s - m_i^{(t)}) V$$
$$= \alpha \cdot \underbrace{\sum_{\text{old}} \exp(s - m_i^{(t-1)}) V}_{\tilde{O}_i^{(t-1)}} + \sum_{\text{new}} \exp(s - m_i^{(t)}) V$$
$$= \alpha \cdot \tilde{O}_i^{(t-1)} + \sum_{\text{new}} \exp(s - m_i^{(t)}) V$$

**Final normalization:**
$$O_i = \frac{\tilde{O}_i^{(T)}}{l_i^{(T)}}$$

where $T$ is the number of tiles.

### Complete Online Softmax Algorithm

**Per query, for each tile $t = 1, \ldots, T$:**

```
Input: Previous state (m_i, l_i, O_acc), new scores s^t, new values V^t

1. m_tile = max(s^t)                    # Tile maximum
2. m_new = max(m_i, m_tile)             # Updated running maximum
3. alpha = exp(m_i - m_new)             # Rescaling factor
4. l_new = alpha * l_i + sum(exp(s^t - m_new))  # Updated running sum
5. O_acc = alpha * O_acc + exp(s^t - m_new) @ V^t  # Rescale and update output
6. m_i = m_new, l_i = l_new             # Update state
```

**After all tiles:**
```
O = O_acc / l_i                          # Final normalization
```

## Worked Numerical Example

**Setup:**
- One query attending to 4 keys (S=4)
- 2 tiles, 2 keys per tile ($B_c = 2$)
- Scores: $s = [2.0, 1.0, 4.0, 0.5]$
- Values: $V = [V_1, V_2, V_3, V_4]$ (scalars for simplicity)

**Tile 1: $s^{(1)} = [2.0, 1.0]$, $V^{(1)} = [V_1, V_2]$**

1. $m_{\text{tile}} = \max(2.0, 1.0) = 2.0$
2. $m^{(1)} = \max(-\infty, 2.0) = 2.0$
3. $\alpha = \exp(-\infty - 2.0) = 0$ (no previous state to rescale)
4. $l^{(1)} = 0 \cdot 0 + \exp(2.0 - 2.0) + \exp(1.0 - 2.0) = 1 + 0.368 = 1.368$
5. $\tilde{O}^{(1)} = 0 \cdot 0 + \exp(2.0 - 2.0) V_1 + \exp(1.0 - 2.0) V_2 = V_1 + 0.368 V_2$

**State after tile 1:** $m^{(1)} = 2.0$, $l^{(1)} = 1.368$, $\tilde{O}^{(1)} = V_1 + 0.368 V_2$

**Tile 2: $s^{(2)} = [4.0, 0.5]$, $V^{(2)} = [V_3, V_4]$**

1. $m_{\text{tile}} = \max(4.0, 0.5) = 4.0$
2. $m^{(2)} = \max(2.0, 4.0) = 4.0$
3. $\alpha = \exp(2.0 - 4.0) = \exp(-2.0) = 0.135$
4. $l^{(2)} = 0.135 \cdot 1.368 + \exp(4.0 - 4.0) + \exp(0.5 - 4.0)$
   $= 0.185 + 1 + 0.030 = 1.215$
5. $\tilde{O}^{(2)} = 0.135 \cdot (V_1 + 0.368 V_2) + \exp(4.0 - 4.0) V_3 + \exp(0.5 - 4.0) V_4$
   $= 0.135 V_1 + 0.050 V_2 + V_3 + 0.030 V_4$

**Final normalization:**
$$O = \frac{\tilde{O}^{(2)}}{l^{(2)}} = \frac{0.135 V_1 + 0.050 V_2 + V_3 + 0.030 V_4}{1.215}$$
$$= 0.111 V_1 + 0.041 V_2 + 0.823 V_3 + 0.025 V_4$$

**Verify against batch softmax:**

Batch softmax of $s = [2.0, 1.0, 4.0, 0.5]$:
- $m = \max(s) = 4.0$
- $\exp(s - m) = [\exp(-2), \exp(-3), \exp(0), \exp(-3.5)] = [0.135, 0.050, 1, 0.030]$
- $l = 0.135 + 0.050 + 1 + 0.030 = 1.215$
- $p = [0.111, 0.041, 0.823, 0.025]$

**Matches!** Online softmax produces the exact same result as batch softmax.

## Shapes and Sizes

| Variable | Shape | Description |
|----------|-------|-------------|
| $m_i$ | $[B, H, S_q]$ | Running maximum per query |
| $l_i$ | $[B, H, S_q]$ | Running sum per query |
| $O_{\text{acc}}$ | $[B, H, S_q, d_h]$ | Output accumulator per query |
| $s^{(t)}$ | $[B, H, S_q, B_c]$ | Score tile |
| $V^{(t)}$ | $[B, H, B_c, d_h]$ | Value tile |

## The Kernel Implication

### CUDA Implementation

```cuda
__device__ void online_softmax_update(
    float& m_i, float& l_i, float O_acc[d_h],
    const float s_tile[B_c], const float V_tile[B_c][d_h]
) {
    // Step 1: Tile maximum
    float m_tile = -INFINITY;
    for (int j = 0; j < B_c; ++j)
        m_tile = fmaxf(m_tile, s_tile[j]);
    
    // Step 2: Updated running maximum
    float m_new = fmaxf(m_i, m_tile);
    
    // Step 3: Rescaling factor
    float alpha = expf(m_i - m_new);
    
    // Step 4: Updated running sum
    float l_new = alpha * l_i;
    for (int j = 0; j < B_c; ++j)
        l_new += expf(s_tile[j] - m_new);
    
    // Step 5: Rescale and update output accumulator
    for (int k = 0; k < d_h; ++k) {
        O_acc[k] *= alpha;
        for (int j = 0; j < B_c; ++j)
            O_acc[k] += expf(s_tile[j] - m_new) * V_tile[j][k];
    }
    
    // Step 6: Update state
    m_i = m_new;
    l_i = l_new;
}
```

### Register Pressure

**Per thread (for $B_r = 1, B_c = 64, d_h = 128$):**
- $m_i, l_i$: 2 floats (8 bytes)
- $O_{\text{acc}}$: 128 floats (512 bytes) — one per head dimension
- $s_{\text{tile}}$: 64 floats (256 bytes)
- $V_{\text{tile}}$: Shared memory, not registers

**Total registers:** ~160 floats = 640 bytes per thread.

With 1024 threads per SM, this is 640 KB — more than available. **Solution:** Use fewer threads per block or spill to local memory.

## Numbers That Matter

| Operation | FLOPs per tile | Memory ops per tile |
|-----------|----------------|---------------------|
| Tile max | $B_c$ comparisons | 0 |
| Rescaling | $d_h$ multiplies | $d_h$ reads + $d_h$ writes |
| Sum update | $B_c$ exp + $B_c$ adds | 0 |
| Output update | $B_c \cdot d_h$ FMA | $B_c \cdot d_h$ reads (V_tile) |

**Dominant cost:** Output update ($O(B_c \cdot d_h)$ FLOPs) — same as matmul.

## Common Interview Questions

**Q1: Derive the online softmax rescaling formula. Why is $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$?**

<details>
<summary>Answer</summary>

When the maximum changes from $m_{\text{old}}$ to $m_{\text{new}}$, all previous exponentials must be rescaled:

$\exp(s - m_{\text{new}}) = \exp(s - m_{\text{old}} + m_{\text{old}} - m_{\text{new}})$
$= \exp(s - m_{\text{old}}) \cdot \exp(m_{\text{old}} - m_{\text{new}})$

So $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ is the factor that converts $\exp(s - m_{\text{old}})$ to $\exp(s - m_{\text{new}})$.

This ensures the running sum and output accumulator are correctly normalized with respect to the new maximum.
</details>

**Q2: What happens if the maximum doesn't change when processing a new tile?**

<details>
<summary>Answer</summary>

If $m_{\text{new}} = m_{\text{old}}$, then $\alpha = \exp(0) = 1$.

The rescaling becomes a no-op:
- $l_{\text{new}} = 1 \cdot l_{\text{old}} + \sum \exp(s^{\text{new}} - m_{\text{old}})$
- $O_{\text{acc}}^{\text{new}} = 1 \cdot O_{\text{acc}}^{\text{old}} + \exp(s^{\text{new}} - m_{\text{old}}) V^{\text{new}}$

Only the new tile's contributions are added; no rescaling of old values is needed.
</details>

**Q3: Why must the output accumulator be rescaled, not just the running sum?**

<details>
<summary>Answer</summary>

The output accumulator stores the unnormalized weighted sum:
$\tilde{O} = \sum \exp(s - m) V$

When $m$ changes, the weights $\exp(s - m)$ change for **all** scores, including those already processed. The rescaling factor $\alpha$ converts the old weights to the new weights:

$\exp(s - m_{\text{new}}) = \alpha \cdot \exp(s - m_{\text{old}})$

Without rescaling $O_{\text{acc}}$, the final output would be incorrect (old values would have wrong weights).
</details>

## Connection To Other Concepts

**Prerequisites:** Module 05.2 (tiling) — you need to understand why scores come tile-by-tile.

**What this unlocks:**
- Module 05.4 (FA2 Improvements): Optimizations to the online softmax loop.
- Module 10 (Arithmetic Intensity): The non-matmul FLOPs of online softmax.

**Next:** `04_fa2_improvements.md` — how FlashAttention v2 reduces overhead.
