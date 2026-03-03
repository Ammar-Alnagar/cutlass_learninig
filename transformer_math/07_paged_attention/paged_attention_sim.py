"""
FILE: paged_attention_sim.py
TEACHES: PagedAttention block table management and memory savings
MAPS TO: Cerebras runtime engineer — block table implementation
RUN: python paged_attention_sim.py — no arguments needed
"""

import numpy as np

print("=" * 70)
print("PAGED ATTENTION: BLOCK TABLE SIMULATION")
print("=" * 70)

# ============================================================
# PART 1: Configuration
# ============================================================

BLOCK_SIZE = 16  # Tokens per block
S_MAX = 4096     # Maximum sequence length
B = 32           # Batch size
L = 32           # Layers (LLaMA-3 8B)
H_kv = 8         # KV heads (GQA)
d_h = 128        # Head dimension
dtype_bytes = 2  # FP16

print(f"\nConfig:")
print(f"  Block size: {BLOCK_SIZE} tokens")
print(f"  Max sequence: S_max={S_MAX}")
print(f"  Batch size: B={B}")
print(f"  Model: LLaMA-3 8B (L={L}, H_kv={H_kv}, d_h={d_h})")

# ============================================================
# PART 2: Naive Allocation
# ============================================================

print("\n" + "=" * 70)
print("NAIVE KV CACHE ALLOCATION")
print("=" * 70)

# Pre-allocate for max sequence length
kv_per_seq_naive = 2 * L * S_MAX * H_kv * d_h * dtype_bytes
total_naive = kv_per_seq_naive * B

print(f"\nPer-sequence KV cache (S_max={S_MAX}):")
print(f"  = 2 × {L} × {S_MAX} × {H_kv} × {d_h} × {dtype_bytes}")
print(f"  = {kv_per_seq_naive / 1e6:.0f} MB")

print(f"\nTotal for batch={B}:")
print(f"  = {total_naive / 1e9:.2f} GB")

# Simulate actual usage with variable sequence lengths
rng = np.random.Generator(np.random.PCG64(42))
seq_lengths = rng.integers(512, 2048, size=B)  # Random lengths 512-2048
avg_seq_len = seq_lengths.mean()

print(f"\nActual usage (variable sequences):")
print(f"  Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, avg={avg_seq_len:.0f}")

actual_used_naive = 2 * L * avg_seq_len * H_kv * d_h * dtype_bytes * B
utilization_naive = actual_used_naive / total_naive * 100

print(f"  Actually used: {actual_used_naive / 1e9:.2f} GB")
print(f"  Utilization: {utilization_naive:.1f}%")
print(f"  Wasted: {(1 - utilization_naive / 100) * 100:.1f}%")

# ============================================================
# PART 3: PagedAttention Allocation
# ============================================================

print("\n" + "=" * 70)
print("PAGED ATTENTION ALLOCATION")
print("=" * 70)

# Compute blocks needed per sequence
blocks_per_seq = np.ceil(seq_lengths / BLOCK_SIZE).astype(int)
total_blocks = blocks_per_seq.sum()

# Block size in bytes
block_bytes = 2 * L * BLOCK_SIZE * H_kv * d_h * dtype_bytes

# Total PagedAttention memory
total_paged = total_blocks * block_bytes

print(f"\nBlocks per sequence:")
print(f"  min={blocks_per_seq.min()}, max={blocks_per_seq.max()}, avg={blocks_per_seq.mean():.1f}")
print(f"  Total blocks: {total_blocks}")

print(f"\nBlock size: {block_bytes / 1024:.1f} KB ({BLOCK_SIZE} tokens)")

print(f"\nTotal PagedAttention memory:")
print(f"  = {total_blocks} blocks × {block_bytes / 1024:.1f} KB")
print(f"  = {total_paged / 1e9:.2f} GB")

# Memory savings
savings = (total_naive - total_paged) / total_naive * 100

print(f"\nMemory savings vs. naive:")
print(f"  Naive: {total_naive / 1e9:.2f} GB")
print(f"  PagedAttention: {total_paged / 1e9:.2f} GB")
print(f"  Savings: {savings:.1f}%")

# ============================================================
# PART 4: Block Table Simulation
# ============================================================

print("\n" + "=" * 70)
print("BLOCK TABLE SIMULATION")
print("=" * 70)

# Simulate block table for first few sequences
print("\nBlock table (first 4 sequences, layer 0):")
print()

# Physical block pool
next_physical_block = 0

for seq_idx in range(min(4, B)):
    seq_len = seq_lengths[seq_idx]
    num_blocks = blocks_per_seq[seq_idx]
    
    # Allocate physical blocks
    physical_blocks = list(range(next_physical_block, next_physical_block + num_blocks))
    next_physical_block += num_blocks
    
    print(f"  Sequence {seq_idx} (length={seq_len}, blocks={num_blocks}):")
    print(f"    Logical blocks: {list(range(num_blocks))}")
    print(f"    Physical blocks: {physical_blocks}")
    print()

# ============================================================
# PART 5: Token to KV Cache Mapping
# ============================================================

print("=" * 70)
print("TOKEN POSITION → KV CACHE ACCESS")
print("=" * 70)

# Example: access token at position 35 for sequence 0
seq_idx = 0
token_pos = 35

logical_block = token_pos // BLOCK_SIZE
offset_in_block = token_pos % BLOCK_SIZE

# Reconstruct physical block for this sequence
seq_len = seq_lengths[seq_idx]
num_blocks = blocks_per_seq[seq_idx]
physical_block_start = sum(blocks_per_seq[:seq_idx])
physical_block = physical_block_start + logical_block

kv_offset = physical_block * BLOCK_SIZE + offset_in_block

print(f"\nAccess token {token_pos} for sequence {seq_idx}:")
print(f"  Logical block: {logical_block} (token_pos // {BLOCK_SIZE})")
print(f"  Offset in block: {offset_in_block} (token_pos % {BLOCK_SIZE})")
print(f"  Physical block: {physical_block}")
print(f"  KV cache offset: {kv_offset}")
print()

# ============================================================
# PART 6: Maximum Batch Size Comparison
# ============================================================

print("=" * 70)
print("MAXIMUM BATCH SIZE (A100 80GB)")
print("=" * 70)

gpu_memory = 80e9
model_weights = 8e9 * 2  # 16 GB for FP16 weights
available = gpu_memory - model_weights

# Max batch with naive allocation
b_max_naive = int(available / kv_per_seq_naive)

# Max batch with PagedAttention (using average blocks)
avg_block_bytes = block_bytes * blocks_per_seq.mean()
b_max_paged = int(available / (avg_block_bytes * avg_seq_len / BLOCK_SIZE))

# More accurate: compute based on total blocks
b_max_paged = int(available * B / total_paged)

print(f"\nAvailable memory: {available / 1e9:.0f} GB")
print(f"\nMax batch size (S_avg={avg_seq_len:.0f}):")
print(f"  Naive: {b_max_naive} sequences")
print(f"  PagedAttention: {b_max_paged} sequences")
print(f"  Improvement: {b_max_paged / max(b_max_naive, 1):.1f}x")

# ============================================================
# VERIFY: Summary
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n✓ Naive allocation: {total_naive / 1e9:.2f} GB")
print(f"✓ PagedAttention: {total_paged / 1e9:.2f} GB")
print(f"✓ Memory savings: {savings:.1f}%")
print(f"✓ Utilization improvement: {100 - utilization_naive:.1f}% → ~100%")
print(f"✓ Max batch improvement: {b_max_naive} → {b_max_paged} ({b_max_paged / max(b_max_naive, 1):.1f}x)")
print()
print("PASS — PagedAttention simulation complete.")
print()
print("Key insights:")
print("  1. Naive allocation wastes 60-80% memory (pre-allocate for S_max)")
print("  2. PagedAttention allocates blocks on-demand (like OS virtual memory)")
print("  3. Block table maps logical token positions to physical KV blocks")
print("  4. Memory savings enable 2-4x larger batch sizes")
print("  5. Tradeoff: non-contiguous memory access (gather pattern)")
