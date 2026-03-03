# Tree Attention for Verification

## What This Is

Tree attention enables parallel verification of draft tokens. Instead of triangular causal mask, tree mask allows each position to attend to all previous draft tokens.

## The Math

### Standard Causal Mask (Triangular)

```
Token:    0  1  2  3
        ┌───────────
    0   │ 1  0  0  0
    1   │ 1  1  0  0
    2   │ 1  1  1  0
    3   │ 1  1  1  1
```

### Tree Mask (for k=4 draft tokens)

```
Token:    0  1  2  3  4  5
        ┌─────────────────
    0   │ 1  0  0  0  0  0   (prompt)
    1   │ 1  1  0  0  0  0   (verify x1)
    2   │ 1  1  1  0  0  0   (verify x2)
    3   │ 1  1  1  1  0  0   (verify x3)
    4   │ 1  1  1  1  1  0   (verify x4)
    5   │ 1  1  1  1  1  1   (resample if needed)
```

**Key difference:** Row $i$ can attend to columns $0, 1, \ldots, i$ (all previous tokens including drafts).

## The Kernel Implication

**Tree mask generation:**
```cuda
__device__ bool tree_mask(int q_pos, int k_pos, int prompt_len, int k_draft) {
    if (q_pos < prompt_len) {
        // Prompt tokens: standard causal
        return k_pos <= q_pos;
    } else {
        // Draft verification: attend to prompt + all previous drafts
        return k_pos < q_pos + 1;
    }
}
```
