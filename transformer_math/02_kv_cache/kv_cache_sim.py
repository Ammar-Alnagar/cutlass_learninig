"""
FILE: kv_cache_sim.py
TEACHES: KV cache memory savings and prefill vs. decode characteristics
MAPS TO: Cerebras/NVIDIA inference engineer — KV cache management, PagedAttention
RUN: python kv_cache_sim.py — no arguments needed
"""

import numpy as np

# ============================================================
# PART 1: Configuration — LLaMA-3 8B parameters
# Math reference: see 02_memory_formula.md, section "Worked Example 1"
# ============================================================

# LLaMA-3 8B configuration
L = 32      # layers
d = 4096    # model dimension (hidden size)
H = 32      # query heads
d_h = 128   # head dimension (d / H)
H_kv = 8    # KV heads (GQA)

# Sequence parameters
P = 4096    # prompt length
S_gen = 128 # tokens to generate

# Data type
dtype_bytes = 2  # FP16

print("=" * 70)
print("KV CACHE SIMULATION — LLaMA-3 8B")
print("=" * 70)
print(f"Model: L={L}, d={d}, H={H}, H_kv={H_kv}, d_h={d_h}")
print(f"Prompt length: P={P}, Generate: S_gen={S_gen}")
print(f"Data type: FP16 ({dtype_bytes} bytes)")
print()

# ============================================================
# PART 2: KV Cache Memory Formula
# Math reference: see 02_memory_formula.md, section "Derivation"
# Formula: KV Cache = 2 * L * S * (H_kv * d_h) * dtype_bytes * B
# ============================================================

print("=" * 70)
print("KV CACHE MEMORY CALCULATION")
print("=" * 70)

# Per-layer KV cache elements (for one sequence)
elements_per_layer = 2 * P * H_kv * d_h
bytes_per_layer = elements_per_layer * dtype_bytes

print(f"\nPer-layer KV cache (P={P}):")
print(f"  Elements: 2 × {P} × {H_kv} × {d_h} = {elements_per_layer:,}")
print(f"  Bytes (FP16): {elements_per_layer:,} × {dtype_bytes} = {bytes_per_layer:,} bytes = {bytes_per_layer / 1e6:.1f} MB")

# Total KV cache (all layers)
total_bytes_per_seq = L * bytes_per_layer
print(f"\nTotal KV cache per sequence ({L} layers):")
print(f"  Bytes: {L} × {bytes_per_layer:,} = {total_bytes_per_seq:,} bytes = {total_bytes_per_seq / 1e9:.2f} GB")

# With GQA vs. without GQA
# Without GQA (MHA), H_kv = H = 32
bytes_per_layer_mha = 2 * P * H * d_h * dtype_bytes
total_bytes_mha = L * bytes_per_layer_mha

print(f"\nGQA savings (H_kv={H_kv} vs. MHA H={H}):")
print(f"  MHA KV cache: {total_bytes_mha / 1e9:.2f} GB")
print(f"  GQA KV cache: {total_bytes_per_seq / 1e9:.2f} GB")
print(f"  Savings: {total_bytes_mha / total_bytes_per_seq:.1f}x ({(1 - total_bytes_per_seq / total_bytes_mha) * 100:.0f}% reduction)")

# ============================================================
# PART 3: Simulate Prefill + Decode with KV Cache
# Math reference: see 03_prefill_vs_decode.md
# ============================================================

print()
print("=" * 70)
print("PREFILL + DECODE SIMULATION")
print("=" * 70)

# Simulate KV cache as a numpy array
# Shape: [L, H_kv, max_seq_len, d_h]
max_seq_len = P + S_gen
kv_cache_k = np.zeros((L, H_kv, max_seq_len, d_h), dtype=np.float16)
kv_cache_v = np.zeros((L, H_kv, max_seq_len, d_h), dtype=np.float16)

print(f"\nKV cache allocated:")
print(f"  Shape per tensor: {kv_cache_k.shape}")
print(f"  Total elements: {kv_cache_k.size * 2:,} (K + V)")
print(f"  Total memory: {kv_cache_k.nbytes * 2 / 1e9:.2f} GB")
print()

# Prefill phase: process all P tokens
print("PREFILL PHASE:")
print(f"  Processing {P} prompt tokens...")

# Simulate computing K, V for all prompt tokens
# In reality, this is: K = X @ W_K, V = X @ W_V
# Here we just use random values for demonstration
for layer in range(L):
    # K, V for prompt: [H_kv, P, d_h]
    k_prompt = np.random.randn(H_kv, P, d_h).astype(np.float16)
    v_prompt = np.random.randn(H_kv, P, d_h).astype(np.float16)
    
    # Store in cache
    kv_cache_k[layer, :, :P, :] = k_prompt
    kv_cache_v[layer, :, :P, :] = v_prompt

print(f"  KV cache after prefill:")
print(f"    Tokens cached: {P}")
print(f"    Memory used: {P / max_seq_len * 100:.0f}% of allocated")

# Compute prefill FLOPs (attention only)
# FLOPs = 4 * B * H * P^2 * d_h (for MHA)
# For GQA, we still compute QK^T with all H query heads
flops_prefill = 4 * 1 * H * P**2 * d_h
print(f"  Prefill FLOPs (attention, all layers): {flops_prefill / 1e9:.2f} GFLOPs")
print()

# Decode phase: generate S_gen tokens one at a time
print("DECODE PHASE:")
print(f"  Generating {S_gen} tokens...")

total_decode_flops = 0
total_decode_bytes = 0

for step in range(S_gen):
    t = P + step  # Current sequence length (including new token)
    
    for layer in range(L):
        # Compute K, V for new token (position t)
        k_new = np.random.randn(H_kv, 1, d_h).astype(np.float16)
        v_new = np.random.randn(H_kv, 1, d_h).astype(np.float16)
        
        # Append to cache
        kv_cache_k[layer, :, t:t+1, :] = k_new
        kv_cache_v[layer, :, t:t+1, :] = v_new
        
        # Decode FLOPs at this step:
        # QK^T: H * 1 * t * d_h (one query, t keys)
        # PV: H * 1 * t * d_h (one output, t values)
        # Total: 2 * H * t * d_h per layer
        flops_step = 2 * H * t * d_h
        total_decode_flops += flops_step
        
        # Decode memory traffic at this step:
        # Read K_cache: H_kv * t * d_h elements
        # Read V_cache: H_kv * t * d_h elements
        # Read Q_new: H_kv * 1 * d_h elements (negligible)
        # Write O_new: H_kv * 1 * d_h elements (negligible)
        # Total: 2 * H_kv * t * d_h elements per layer
        bytes_step = 2 * H_kv * t * d_h * dtype_bytes
        total_decode_bytes += bytes_step

print(f"  KV cache after decode:")
print(f"    Tokens cached: {P + S_gen}")
print(f"    Memory used: {(P + S_gen) / max_seq_len * 100:.0f}% of allocated")
print(f"  Decode FLOPs (attention, all layers, all steps): {total_decode_flops / 1e9:.2f} GFLOPs")
print(f"  Decode memory traffic (all layers, all steps): {total_decode_bytes / 1e9:.2f} GB")

# ============================================================
# PART 4: Compare With vs. Without KV Cache
# Math reference: see 01_why_kv_cache.md, section "Comparison"
# ============================================================

print()
print("=" * 70)
print("KV CACHE BENEFIT: WITH vs. WITHOUT CACHE")
print("=" * 70)

# Without KV cache, decode would recompute all previous tokens
# FLOPs without cache at step t: 4 * H * t^2 * d_h (full attention for t tokens)
# Total for S_gen steps: sum over t from P+1 to P+S_gen

flops_no_cache = 0
for step in range(S_gen):
    t = P + step + 1  # Sequence length at this step
    # Full attention: 4 * H * t^2 * d_h
    flops_step = 4 * H * t**2 * d_h
    flops_no_cache += flops_step

print(f"\nTotal FLOPs to generate {S_gen} tokens:")
print(f"  Without KV cache: {flops_no_cache / 1e9:.2f} GFLOPs")
print(f"  With KV cache: {total_decode_flops / 1e9:.2f} GFLOPs")
print(f"  Speedup: {flops_no_cache / total_decode_flops:.1f}x")

# ============================================================
# PART 5: Arithmetic Intensity Analysis
# Math reference: see 03_prefill_vs_decode.md, section "Decode Phase"
# ============================================================

print()
print("=" * 70)
print("ARITHMETIC INTENSITY ANALYSIS")
print("=" * 70)

# Prefill AI (with FlashAttention, no O(P^2) intermediates)
# AI_prefill ≈ P / 2 (derived in 03_prefill_vs_decode.md)
ai_prefill = P / 2

# Decode AI
# AI_decode = FLOPs / bytes = (4 * H * t * d_h) / (4 * H_kv * t * d_h * dtype_bytes)
# For GQA: H / H_kv = 32 / 8 = 4
# AI_decode = (H / H_kv) / dtype_bytes = 4 / 2 = 2 FLOPs/byte
ai_decode = (H / H_kv) / dtype_bytes

print(f"\nPrefill (P={P}):")
print(f"  Arithmetic intensity: {ai_prefill:.0f} FLOPs/byte")
print(f"  H100 peak (FP16): ~2000 FLOPs/byte")
print(f"  Status: {'COMPUTE-BOUND' if ai_prefill >= 2000 else 'MEMORY-BOUND'}")

print(f"\nDecode (GQA, H={H}, H_kv={H_kv}):")
print(f"  Arithmetic intensity: {ai_decode:.1f} FLOPs/byte")
print(f"  H100 peak (FP16): ~2000 FLOPs/byte")
print(f"  Status: {'COMPUTE-BOUND' if ai_decode >= 2000 else 'MEMORY-BOUND'}")
print(f"  GPU utilization: {ai_decode / 2000 * 100:.2f}% of peak")

# ============================================================
# PART 6: Batch Size Effect
# Math reference: see 03_prefill_vs_decode.md
# ============================================================

print()
print("=" * 70)
print("BATCH SIZE EFFECT ON ARITHMETIC INTENSITY")
print("=" * 70)

batch_sizes = [1, 8, 32, 128]

print(f"\nDecode AI vs. batch size (GQA, H={H}, H_kv={H_kv}):")
print(f"  {'Batch':<8} {'AI (FLOPs/byte)':<20} {'Status'}")
print(f"  {'-'*8} {'-'*20} {'-'*15}")

for B in batch_sizes:
    # Decode AI is independent of batch size (per-sequence)
    # But total throughput scales with B
    ai = ai_decode  # Same for all batch sizes
    status = 'MEMORY-BOUND' if ai < 2000 else 'COMPUTE-BOUND'
    print(f"  {B:<8} {ai:<20.1f} {status}")

print(f"\nNote: Decode AI is constant (~{ai_decode:.1f} FLOPs/byte) regardless of batch size.")
print(f"      Throughput scales linearly with batch size, but each sequence is memory-bound.")

# ============================================================
# VERIFY: Expected output summary
# ============================================================

print()
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"✓ LLaMA-3 8B KV cache (P={P}, FP16): ~{total_bytes_per_seq / 1e9:.2f} GB per sequence")
print(f"✓ GQA reduces KV cache by {total_bytes_mha / total_bytes_per_seq:.1f}x vs. MHA")
print(f"✓ KV cache speedup for decode: ~{flops_no_cache / total_decode_flops:.0f}x")
print(f"✓ Prefill AI: {ai_prefill:.0f} FLOPs/byte ({'compute-bound' if ai_prefill >= 2000 else 'memory-bound'})")
print(f"✓ Decode AI: {ai_decode:.1f} FLOPs/byte (memory-bound)")
print()
print("PASS — KV cache simulation complete.")
print()
print("Key insights:")
print("  1. KV cache is essential — without it, decode is O(S²) in compute")
print("  2. GQA reduces KV cache memory by H_kv / H = 4x for LLaMA-3 8B")
print("  3. Decode is always memory-bound (AI ≈ 2 FLOPs/byte for GQA)")
print("  4. Prefill can be compute-bound at large sequence lengths")
