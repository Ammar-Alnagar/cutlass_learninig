"""
FILE: flash_attention.py
TEACHES: FlashAttention tile-by-tile computation with online softmax
MAPS TO: NVIDIA inference kernel engineer — FlashAttention CuTe kernel implementation
RUN: python flash_attention.py — no arguments needed
"""

import numpy as np

# ============================================================
# PART 1: Setup — tiny shapes for inspection
# Math reference: see 02_tiling_insight.md, section "Tiling Parameters"
# ============================================================

# Use tiny defaults for easy inspection
B = 1   # batch size
H = 2   # heads (small for clarity)
S = 8   # sequence length (tiny for inspection)
d_h = 4 # head dimension (tiny for inspection)

# Tile sizes (matching the math in 02_tiling_insight.md)
B_r = 2  # Query tile size (rows)
B_c = 3  # Key/Value tile size (columns)

print("=" * 70)
print("FLASH ATTENTION — TILE-BY-TILE IMPLEMENTATION")
print("=" * 70)
print(f"Config: B={B}, H={H}, S={S}, d_h={d_h}")
print(f"Tile sizes: B_r={B_r} (query rows), B_c={B_c} (key/val cols)")
print()

# Number of tiles
T_r = (S + B_r - 1) // B_r  # ceil(S / B_r)
T_c = (S + B_c - 1) // B_c  # ceil(S / B_c)
print(f"Number of tiles: T_r={T_r} (query), T_c={T_c} (key/value)")
print()

# ============================================================
# PART 2: Create Q, K, V tensors
# Math reference: see 01_scaled_dot_product.md
# ============================================================

rng = np.random.Generator(np.random.PCG64(42))
Q = rng.standard_normal((B, H, S, d_h)).astype(np.float32)
K = rng.standard_normal((B, H, S, d_h)).astype(np.float32)
V = rng.standard_normal((B, H, S, d_h)).astype(np.float32)

print("Q, K, V shapes:")
print(f"  Q: {Q.shape}")
print(f"  K: {K.shape}")
print(f"  V: {V.shape}")
print()

# ============================================================
# PART 3: Reference Implementation (Naive Attention)
# Math reference: see 01_scaled_dot_product.md, section "Step 6"
# This gives us the expected output to compare against
# ============================================================

print("=" * 70)
print("REFERENCE: NAIVE ATTENTION (for verification)")
print("=" * 70)

# Compute attention scores
# scores = Q @ K^T / sqrt(d_h)
# shapes: [B,H,S,d_h] @ [B,H,d_h,S] -> [B,H,S,S]
scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_h)
print(f"Scores shape (Q @ K^T): {scores.shape}")

# Apply causal mask (upper triangular = -inf)
mask = np.zeros((S, S), dtype=np.float32)
for i in range(S):
    for j in range(S):
        if j > i:
            mask[i, j] = -1e9
scores_masked = scores + mask
print(f"Masked scores shape: {scores_masked.shape}")

# Softmax
scores_max = np.max(scores_masked, axis=-1, keepdims=True)
scores_shifted = scores_masked - scores_max
scores_exp = np.exp(scores_shifted)
scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
P = scores_exp / scores_sum
print(f"Attention probs P shape: {P.shape}")

# Verify rows sum to 1
row_sums = np.sum(P, axis=-1)
print(f"Row sums (should be 1.0): min={row_sums.min():.6f}, max={row_sums.max():.6f}")

# Weighted sum of values
# output = P @ V
# shapes: [B,H,S,S] @ [B,H,S,d_h] -> [B,H,S,d_h]
output_ref = np.matmul(P, V)
print(f"Output shape (P @ V): {output_ref.shape}")
print()

# ============================================================
# PART 4: FlashAttention — Tile-by-Tile Computation
# Math reference: see 02_tiling_insight.md, section "Naive vs. FlashAttention Loop"
# and 03_online_softmax.md for the rescaling formula
# ============================================================

print("=" * 70)
print("FLASH ATTENTION: TILE-BY-TILE COMPUTATION")
print("=" * 70)

# Output tensor
output_fa = np.zeros((B, H, S, d_h), dtype=np.float32)

# Process each head independently
for b in range(B):
    for h in range(H):
        print(f"\n--- Head ({b},{h}) ---")
        
        # Get Q, K, V for this head
        Q_h = Q[b, h]  # [S, d_h]
        K_h = K[b, h]  # [S, d_h]
        V_h = V[b, h]  # [S, d_h]
        
        # Process each query tile
        for tile_i in range(T_r):
            q_start = tile_i * B_r
            q_end = min((tile_i + 1) * B_r, S)
            q_len = q_end - q_start
            
            # Load Q tile
            Q_tile = Q_h[q_start:q_end]  # [q_len, d_h]
            print(f"  Query tile {tile_i}: Q_tile shape {Q_tile.shape}, positions [{q_start}:{q_end}]")
            
            # Initialize online softmax state (per query in the tile)
            # m_i: running maximum, l_i: running sum, O_acc: output accumulator
            m_i = np.full(q_len, -np.inf, dtype=np.float32)  # [q_len]
            l_i = np.zeros(q_len, dtype=np.float32)           # [q_len]
            O_acc = np.zeros((q_len, d_h), dtype=np.float32)  # [q_len, d_h]
            
            print(f"    Initial state: m_i={m_i}, l_i={l_i}")
            
            # Process each key/value tile
            for tile_j in range(T_c):
                k_start = tile_j * B_c
                k_end = min((tile_j + 1) * B_c, S)
                k_len = k_end - k_start
                
                # Load K, V tiles
                K_tile = K_h[k_start:k_end]  # [k_len, d_h]
                V_tile = V_h[k_start:k_end]  # [k_len, d_h]
                
                # Compute QK^T tile
                # scores_tile = Q_tile @ K_tile^T / sqrt(d_h)
                # shapes: [q_len, d_h] @ [d_h, k_len] -> [q_len, k_len]
                scores_tile = np.matmul(Q_tile, K_tile.T) / np.sqrt(d_h)
                
                # Apply causal mask within this tile
                # Query position q attends to key position k only if k <= q
                for qi in range(q_len):
                    for ki in range(k_len):
                        global_q = q_start + qi
                        global_k = k_start + ki
                        if global_k > global_q:
                            scores_tile[qi, ki] = -1e9
                
                print(f"    Key/Value tile {tile_j}: K_tile shape {K_tile.shape}, positions [{k_start}:{k_end}]")
                print(f"      Scores tile shape: {scores_tile.shape}")
                print(f"      Scores tile (first query row): {scores_tile[0, :]}")
                
                # ========== ONLINE SOFTMAX UPDATE ==========
                # Math reference: see 03_online_softmax.md, section "Processing Tile t"
                
                # Step 1: Tile maximum (per query row)
                m_tile = np.max(scores_tile, axis=1)  # [q_len]
                print(f"      m_tile (max per row): {m_tile}")
                
                # Step 2: Updated running maximum
                m_new = np.maximum(m_i, m_tile)  # [q_len]
                print(f"      m_new = max(m_i, m_tile): {m_new}")
                
                # Step 3: Rescaling factor
                # alpha = exp(m_i - m_new)
                alpha = np.exp(m_i - m_new)  # [q_len]
                print(f"      alpha = exp(m_i - m_new): {alpha}")
                
                # Step 4: Updated running sum
                # l_new = alpha * l_i + sum(exp(scores_tile - m_new), axis=1)
                # Need to broadcast m_new for the subtraction
                exp_scores = np.exp(scores_tile - m_new[:, np.newaxis])  # [q_len, k_len]
                l_new = alpha * l_i + np.sum(exp_scores, axis=1)  # [q_len]
                print(f"      exp(scores - m_new): {exp_scores[0, :]}")
                print(f"      l_new = alpha*l_i + sum(exp): {l_new}")
                
                # Step 5: Rescale and update output accumulator
                # O_acc = alpha[:, None] * O_acc + exp_scores @ V_tile
                O_acc = alpha[:, np.newaxis] * O_acc + np.matmul(exp_scores, V_tile)  # [q_len, d_h]
                print(f"      O_acc updated (first row): {O_acc[0, :]}")
                
                # Step 6: Update state
                m_i = m_new
                l_i = l_new
            
            # Final normalization: O = O_acc / l_i
            O_normalized = O_acc / l_i[:, np.newaxis]  # [q_len, d_h]
            print(f"    Final l_i: {l_i}")
            print(f"    O_normalized (first row): {O_normalized[0, :]}")
            
            # Write output tile
            output_fa[b, h, q_start:q_end] = O_normalized

print()

# ============================================================
# PART 5: Verification
# Compare FlashAttention output with naive reference
# ============================================================

print("=" * 70)
print("VERIFICATION: FlashAttention vs. Naive")
print("=" * 70)

# Compute max absolute difference
max_diff = np.max(np.abs(output_fa - output_ref))
print(f"Max absolute difference: {max_diff:.10f}")

# Compute mean absolute difference
mean_diff = np.mean(np.abs(output_fa - output_ref))
print(f"Mean absolute difference: {mean_diff:.10f}")

# Check if they match (within numerical tolerance)
tolerance = 1e-5
if max_diff < tolerance:
    print(f"\n✓ PASS: FlashAttention matches naive attention (max diff {max_diff:.2e} < {tolerance})")
else:
    print(f"\n✗ FAIL: FlashAttention does not match (max diff {max_diff:.2e} >= {tolerance})")

print()

# ============================================================
# PART 6: HBM Traffic Analysis
# Math reference: see 01_the_io_problem.md
# ============================================================

print("=" * 70)
print("HBM TRAFFIC ANALYSIS")
print("=" * 70)

# Naive attention HBM traffic
# Bytes_naive = 8 * B * H * S * (d_h + S)
bytes_naive = 8 * B * H * S * (d_h + S)
print(f"\nNaive attention HBM traffic:")
print(f"  Formula: 8 × B × H × S × (d_h + S)")
print(f"  = 8 × {B} × {H} × {S} × ({d_h} + {S})")
print(f"  = {bytes_naive:,} bytes")

# FlashAttention HBM traffic
# Bytes_FA = 8 * B * H * S * d_h
bytes_fa = 8 * B * H * S * d_h
print(f"\nFlashAttention HBM traffic:")
print(f"  Formula: 8 × B × H × S × d_h")
print(f"  = 8 × {B} × {H} × {S} × {d_h}")
print(f"  = {bytes_fa:,} bytes")

# Speedup
speedup = bytes_naive / bytes_fa
print(f"\nHBM traffic speedup:")
print(f"  = {bytes_naive} / {bytes_fa} = {speedup:.2f}x")
print(f"  = (d_h + S) / d_h = ({d_h} + {S}) / {d_h} = {(d_h + S) / d_h:.2f}x")

# ============================================================
# PART 7: Scaling Test — LLaMA-3 Sizes
# Math reference: see 01_the_io_problem.md, section "Numbers That Matter"
# ============================================================

print()
print("=" * 70)
print("SCALING TEST — LLaMA-3 8B Configuration")
print("=" * 70)

S_llama = 4096
H_llama = 32
d_h_llama = 128
B_llama = 1

bytes_naive_llama = 8 * B_llama * H_llama * S_llama * (d_h_llama + S_llama)
bytes_fa_llama = 8 * B_llama * H_llama * S_llama * d_h_llama
speedup_llama = bytes_naive_llama / bytes_fa_llama

print(f"\nLLaMA-3 8B: B={B_llama}, H={H_llama}, S={S_llama}, d_h={d_h_llama}")
print(f"  Naive HBM traffic: {bytes_naive_llama / 1e9:.2f} GB")
print(f"  FlashAttention HBM traffic: {bytes_fa_llama / 1e9:.2f} GB")
print(f"  HBM speedup: {speedup_llama:.1f}x")

# Arithmetic intensity
flops_attention = 4 * B_llama * H_llama * S_llama**2 * d_h_llama
ai_naive = flops_attention / bytes_naive_llama
ai_fa = flops_attention / bytes_fa_llama

print(f"\nArithmetic intensity:")
print(f"  Naive: {ai_naive:.1f} FLOPs/byte")
print(f"  FlashAttention: {ai_fa:.0f} FLOPs/byte")
print(f"  H100 peak (FP16): ~2000 FLOPs/byte")
print(f"  Naive status: {'COMPUTE-BOUND' if ai_naive >= 2000 else 'MEMORY-BOUND'}")
print(f"  FlashAttention status: {'COMPUTE-BOUND' if ai_fa >= 2000 else 'MEMORY-BOUND'}")

# ============================================================
# VERIFY: Expected output summary
# ============================================================

print()
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"✓ Q, K, V shapes: ({B}, {H}, {S}, {d_h})")
print(f"✓ Tile sizes: B_r={B_r}, B_c={B_c}")
print(f"✓ Number of tiles: T_r={T_r}, T_c={T_c}")
print(f"✓ FlashAttention output matches naive: max_diff={max_diff:.2e}")
print(f"✓ HBM traffic reduction: {speedup:.2f}x ({bytes_naive/1e3:.1f} KB → {bytes_fa/1e3:.1f} KB)")
print(f"✓ LLaMA-3 8B HBM speedup: {speedup_llama:.1f}x")
print(f"✓ LLaMA-3 8B AI improvement: {ai_naive:.1f} → {ai_fa:.0f} FLOPs/byte")
print()

if max_diff < tolerance:
    print("PASS — FlashAttention tile-by-tile implementation verified.")
else:
    print("FAIL — FlashAttention output does not match naive reference.")

print()
print("Key insights:")
print("  1. FlashAttention computes attention tile-by-tile in SRAM")
print("  2. Online softmax maintains running max (m_i) and sum (l_i)")
print("  3. Output accumulator is rescaled when max changes: O_acc *= alpha")
print("  4. Final normalization: O = O_acc / l_i")
print("  5. HBM traffic reduced from O(S²) to O(S) — exact same output!")
