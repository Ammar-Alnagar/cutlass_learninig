# MoE Architecture

## What This Is

Mixture of Experts (MoE) replaces the dense FFN with multiple expert FFNs. Each token is routed to top-k experts (typically k=1 or k=2).

**Benefit:** Model capacity scales with number of experts, but compute scales with k (not total experts).

## Why A Kernel Engineer Needs This

**MoE requires all-to-all communication for expert parallelism.** You will implement token dispatch/combine kernels and handle load balancing.

## The Math

### MoE Layer Structure

**Standard transformer:**
- Attention → FFN (dense)

**MoE transformer:**
- Attention → Router → Expert FFNs (sparse) → Combine

### Router Function

**Router computes expert probabilities:**
$$g(x) = \text{softmax}(W_r x)$$

where $W_r \in \mathbb{R}^{E \times d}$, $E$ = number of experts.

**Top-k selection:**
$$\text{Experts}(x) = \text{top-}k(g(x))$$

**Gating weights (normalized over selected experts):**
$$w_i = \frac{g(x)_i}{\sum_{j \in \text{Experts}(x)} g(x)_j}$$

### MoE Output

$$\text{MoE}(x) = \sum_{i \in \text{Experts}(x)} w_i \cdot \text{Expert}_i(x)$$

## Numbers That Matter

| Model | Experts | k | Capacity | Active Params |
|-------|---------|---|----------|---------------|
| Mixtral 8x7B | 8 | 2 | 47B | 13B |
| Grok-1 | 64 | 2 | 314B | 39B |
| LLaMA-3 405B | (dense) | N/A | 405B | 405B |
