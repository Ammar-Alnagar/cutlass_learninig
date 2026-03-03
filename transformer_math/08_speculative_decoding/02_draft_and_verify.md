# Draft and Verify Algorithm

## What This Is

Speculative decoding alternates between:
1. **Draft:** Small model generates $k$ tokens autoregressively
2. **Verify:** Large model evaluates all $k$ tokens in parallel, accepts/rejects each

## The Math

### Acceptance Sampling

**For each draft token $x_i$:**
- Draft probability: $q(x_i | x_{<i})$
- Target probability: $p(x_i | x_{<i})$

**Accept with probability:**
$$\min\left(1, \frac{p(x_i | x_{<i})}{q(x_i | x_{<i})}\right)$$

**If rejected:** Sample from $\text{normalize}(\max(0, p - q))$ as the new token.

### Expected Accepted Tokens

**For $k$ draft tokens with acceptance rate $\alpha$:**

$$E[\text{accepted}] = \sum_{i=1}^{k} \alpha^i = \alpha \cdot \frac{1 - \alpha^k}{1 - \alpha}$$

Including the final verified token:
$$E[\text{total}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

**Worked example ($\alpha = 0.7, k = 4$):**
$$E[\text{total}] = \frac{1 - 0.7^5}{1 - 0.7} = \frac{0.832}{0.3} \approx 2.77$$

## The Kernel Implication

**Tree attention for parallel verification:**
```
Draft tokens: [x1, x2, x3, x4]

Verify in parallel:
  Position 1: context = [prompt], verify x1
  Position 2: context = [prompt, x1], verify x2
  Position 3: context = [prompt, x1, x2], verify x3
  Position 4: context = [prompt, x1, x2, x3], verify x4
```

**Non-triangular mask:** Each position attends to all previous draft tokens.
