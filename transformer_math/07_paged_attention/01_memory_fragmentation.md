# Memory Fragmentation Problem

## What This Is

Naive KV cache allocation pre-allocates contiguous memory for maximum sequence length. This wastes 60-80% of memory because most sequences are shorter than max.

## Why A Kernel Engineer Needs This

**You will implement PagedAttention that allocates KV cache in fixed-size blocks.** This requires non-contiguous memory access patterns and block table management.

## The Math

### Naive Allocation Waste

**Pre-allocate for max length:**
```
KV_cache[B, H_kv, S_max, d_h]  // S_max = 8192
```

**Actual usage (average sequence S_avg = 2048):**
$$\text{Utilization} = \frac{S_{\text{avg}}}{S_{\text{max}}} = \frac{2048}{8192} = 25\%$$

**Waste:** 75% of allocated memory is unused.

### PagedAttention Allocation

**Allocate blocks on-demand:**
- Block size: B tokens (e.g., B = 16)
- Blocks per sequence: $\lceil S / B \rceil$
- Total blocks: $\sum_{i=1}^{B} \lceil S_i / B \rceil$

**Utilization:**
$$\text{Utilization} \approx 1 - \frac{B}{2 \cdot S_{\text{avg}}}$$

For B = 16, S_avg = 2048:
$$\text{Utilization} \approx 1 - \frac{16}{2 \cdot 2048} = 99.6\%$$

## Numbers That Matter

| Allocation | S_max | S_avg | Utilization | Waste |
|------------|-------|-------|-------------|-------|
| Naive | 8192 | 2048 | 25% | 75% |
| Naive | 4096 | 1024 | 25% | 75% |
| PagedAttention | N/A | 2048 | 99.6% | 0.4% |
