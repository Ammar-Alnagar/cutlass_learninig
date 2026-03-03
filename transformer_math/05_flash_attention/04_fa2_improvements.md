# FlashAttention v2 Improvements

## What This Is

FlashAttention v2 improves upon v1 with better work partitioning, reduced non-matmul FLOPs, and improved parallelism. The key changes are:
1. Parallelize over sequence dimension (not just heads)
2. Reduce thread block overhead
3. Better memory access patterns

**Result:** 2x speedup over FlashAttention v1 on A100.

## Why A Kernel Engineer Needs This

**FlashAttention v2 is the production algorithm.** v1 is pedagogically simpler but v2 is what you'll implement in production. Understanding the differences helps you choose the right parallelization strategy.

**Interview relevance:** NVIDIA interviewers ask: "What's the difference between FlashAttention v1 and v2? Why is v2 faster?"

## The Math

### FlashAttention v1: Parallelize Over Heads

**Grid structure:**
- `gridDim.x = num_query_tiles`
- `gridDim.y = num_heads`
- `gridDim.z = 1`

**Each thread block handles:** One query tile for one head.

**Problem:** Limited parallelism. For LLaMA-3 8B (H=32), only 32 SMs can be utilized for the head dimension. The rest are idle.

### FlashAttention v2: Parallelize Over Sequence

**Grid structure:**
- `gridDim.x = num_query_tiles * num_heads / num_sm_groups`
- Partition sequence into chunks, assign to SM groups

**Each thread block handles:** Multiple query tiles across multiple heads.

**Benefit:** Better load balancing. All SMs can be utilized regardless of head count.

### Work Partitioning

**FlashAttention v1:**
```
for each head h:
    for each query tile i:
        block(h, i) processes Q[h,i], all K[:], V[:]
```

**FlashAttention v2:**
```
for each SM group g:
    for each query tile i in group's range:
        for each head h in group's range:
            block(g, i, h) processes Q[h,i], all K[:], V[:]
```

**Key insight:** v2 decouples the parallelism from the head count.

### Non-Matmul FLOP Reduction

**FlashAttention v1 overhead:**
- Online softmax: $O(B_c)$ exponentials, $O(B_c)$ additions per tile
- Rescaling: $O(d_h)$ multiplies per tile
- Total non-matmul: $O(B_c + d_h)$ per tile

**FlashAttention v2 optimizations:**
1. **Fused rescaling:** Combine rescaling with output accumulation
2. **Reduced exponentials:** Compute exp only when necessary
3. **Better register usage:** Fewer spills to local memory

**Non-matmul FLOPs (v2):**
$$\text{FLOPs}_{\text{non-matmul}} \approx 0.5 \times \text{FLOPs}_{\text{non-matmul, v1}}$$

### Memory Access Optimization

**FlashAttention v1:**
- Q tile loaded once per block
- K, V tiles loaded $T_c$ times per block (once per inner iteration)
- Potential bank conflicts in shared memory

**FlashAttention v2:**
- Q tile loaded once per block (same)
- K, V tiles loaded with better coalescing
- Shared memory layout optimized for tensor cores

**Effective bandwidth (v2):** ~10% higher than v1 due to better coalescing.

## Shapes and Sizes

| Parameter | FlashAttention v1 | FlashAttention v2 |
|-----------|-------------------|-------------------|
| Grid X | `num_query_tiles` | `num_query_tiles * num_heads` |
| Grid Y | `num_heads` | 1 |
| Grid Z | 1 | 1 |
| Blocks total | `num_query_tiles * num_heads` | `num_query_tiles * num_heads` |
| SM utilization | Limited by H | Full (all SMs) |

## The Kernel Implication

### FlashAttention v2 Kernel Structure

```cuda
// FlashAttention v2: parallelize over sequence
__global__ void flash_attention_v2(Q, K, V, O) {
    // Each block handles a subset of (head, query_tile) pairs
    int block_idx = blockIdx.x;
    int total_tiles = num_query_tiles * num_heads;
    
    // Stride over (head, query_tile) pairs
    for (int idx = block_idx; idx < total_tiles; idx += gridDim.x) {
        int head_idx = idx / num_query_tiles;
        int tile_i = idx % num_query_tiles;
        
        // Process this (head, query_tile) pair
        process_tile(Q, K, V, O, head_idx, tile_i);
    }
}

__device__ void process_tile(...) {
    // Same tile processing as v1
    // Load Q tile, loop over K/V tiles, online softmax, write O
}
```

**Key difference:** The outer loop strides over work, allowing each block to handle multiple (head, query_tile) pairs. This ensures all SMs are utilized.

### Performance Comparison

| Metric | FA v1 | FA v2 | Improvement |
|--------|-------|-------|-------------|
| SM utilization | H / num_SM | 100% | 2-4x |
| Non-matmul FLOPs | 1.0x | 0.5x | 2x |
| Memory coalescing | Baseline | +10% | 1.1x |
| **Overall speedup** | 1.0x | 2.0-2.5x | 2x |

## Numbers That Matter

**LLaMA-3 8B on A100 (80 SMs):**

| Metric | FA v1 | FA v2 |
|--------|-------|-------|
| Active SMs | 32 (limited by H=32) | 80 (all SMs) |
| Utilization | 40% | 100% |
| Time (prefill, S=4096) | ~1.0 ms | ~0.4 ms |
| Throughput | 1.0x | 2.5x |

**LLaMA-3 70B on A100:**

| Metric | FA v1 | FA v2 |
|--------|-------|-------|
| Active SMs | 64 (limited by H=64) | 80 (all SMs) |
| Utilization | 80% | 100% |
| Time (prefill, S=4096) | ~2.0 ms | ~1.0 ms |
| Throughput | 1.0x | 2.0x |

**Note:** v2 matters more for models with fewer heads (like LLaMA-3 8B with H=32).

## Common Interview Questions

**Q1: What is the main difference between FlashAttention v1 and v2?**

<details>
<summary>Answer</summary>

FlashAttention v1 parallelizes over (head, query_tile) pairs, limiting SM utilization to the number of heads. For LLaMA-3 8B (H=32), only 32 of 80 SMs are used on A100.

FlashAttention v2 parallelizes over the sequence dimension, decoupling parallelism from head count. All SMs can be utilized regardless of head count.

Result: v2 achieves 2x speedup on A100 for LLaMA-3 8B.
</details>

**Q2: Why does FlashAttention v2 have better SM utilization?**

<details>
<summary>Answer</summary>

FA v1 assigns one block per (head, query_tile) pair. With H heads, at most H blocks can run in parallel (one per head). If H < num_SM, some SMs are idle.

FA v2 uses a grid-stride loop: each block processes multiple (head, query_tile) pairs. The total number of blocks is independent of H, so all SMs can be utilized.

Example: A100 has 80 SMs. LLaMA-3 8B has H=32.
- FA v1: 32 blocks max → 32 SMs used (40%)
- FA v2: 80+ blocks → 80 SMs used (100%)
</details>

**Q3: What non-matmul optimizations does FlashAttention v2 make?**

<details>
<summary>Answer</summary>

1. Fused rescaling: Combine the online softmax rescaling with output accumulation, reducing intermediate storage.

2. Reduced exponentials: Only compute exp when the maximum changes, not for every tile.

3. Better register usage: Reorganize register allocation to reduce spills to local memory.

These reduce non-matmul FLOPs by ~50%, improving overall throughput since attention is not purely matmul-bound.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 05.3 (online softmax) — you need the rescaling formula.

**What this unlocks:**
- Module 10 (Arithmetic Intensity): Complete FLOP analysis including non-matmul overhead.

**Next:** `flash_attention.py` — complete tile-by-tile implementation.
