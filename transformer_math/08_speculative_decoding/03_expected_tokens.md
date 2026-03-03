# Expected Accepted Tokens Derivation

## The Math

### Geometric Distribution

**Each draft token is accepted with probability $\alpha$.**

**Probability of accepting exactly $i$ tokens:**
$$P(\text{accept } i) = \alpha^i (1 - \alpha)$$

(Accept first $i$, reject $(i+1)$-th)

### Expected Value

$$E[\text{accepted}] = \sum_{i=0}^{k} i \cdot P(\text{accept } i)$$

But it's easier to compute:
$$E[\text{accepted}] = \sum_{i=1}^{k} P(\text{accept at least } i)$$

$$P(\text{accept at least } i) = \alpha^i$$

$$E[\text{accepted}] = \sum_{i=1}^{k} \alpha^i = \alpha + \alpha^2 + \cdots + \alpha^k$$

### Geometric Series

$$\sum_{i=1}^{k} \alpha^i = \alpha \cdot \frac{1 - \alpha^k}{1 - \alpha}$$

**Including the final token (always accepted after verification):**

$$E[\text{total}] = \alpha \cdot \frac{1 - \alpha^k}{1 - \alpha} + (1 - \alpha^k)$$

$$= \frac{\alpha(1 - \alpha^k) + (1 - \alpha)(1 - \alpha^k)}{1 - \alpha}$$

$$= \frac{(1 - \alpha^k)(\alpha + 1 - \alpha)}{1 - \alpha} = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

### Worked Examples

| α | k | E[accepted] | Speedup |
|---|---|-------------|---------|
| 0.5 | 4 | 1.94 | 1.94x |
| 0.6 | 4 | 2.21 | 2.21x |
| 0.7 | 4 | 2.77 | 2.77x |
| 0.8 | 4 | 3.36 | 3.36x |
| 0.9 | 4 | 4.10 | 4.10x |
