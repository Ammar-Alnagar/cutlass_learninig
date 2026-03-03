# The Underutilization Problem

## What This Is

Autoregressive decode generates one token at a time, severely underutilizing the GPU. Each token requires loading all weights from HBM but does minimal compute.

## Why A Kernel Engineer Needs This

**Speculative decoding is the primary technique to improve decode throughput.** You will implement the draft-and-verify kernel that processes multiple tokens in parallel.

## The Math

### Decode Underutilization

**LLaMA-3 8B decode (per token):**
- FLOPs: ~0.1 GF (attention + MLP)
- Memory: ~16 GB (load all weights)
- Time: ~0.1 ms (memory-bound)

**H100 utilization:**
$$\text{Utilization} = \frac{0.1 \times 10^9 \text{ FLOPs}}{0.1 \times 10^{-3} \text{ s} \times 989 \times 10^{12} \text{ FLOPs/s}} \approx 0.001\%$$

**This is why decode is the bottleneck in LLM serving.**

### Speculative Decoding Idea

**Draft phase:** Small model generates $k$ tokens sequentially (fast).

**Verify phase:** Large model verifies all $k$ tokens in parallel (same time as 1 token).

**Speedup:** If acceptance rate is $\alpha$, expected accepted tokens per verify = $\frac{1 - \alpha^{k+1}}{1 - \alpha}$.

For $\alpha = 0.7$, $k = 4$:
$$E[\text{accepted}] = \frac{1 - 0.7^5}{1 - 0.7} = \frac{1 - 0.168}{0.3} \approx 2.77$$

**Effective speedup:** 2.77x tokens per large model forward pass.
