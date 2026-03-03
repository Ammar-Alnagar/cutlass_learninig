"""
FILE: attention_forward.py
TEACHES: Complete scaled dot-product attention forward pass with all intermediate shapes
MAPS TO: NVIDIA inference kernel engineer — FlashAttention QK^T and PV matmuls
RUN: python attention_forward.py — no arguments needed
"""

import numpy as np

# ============================================================
# PART 1: Setup — tiny shapes for inspection
# Math reference: see 01_scaled_dot_product.md, section "Input Shapes"
# ============================================================

# Use tiny defaults for easy inspection
B = 1   # batch size
H = 2   # heads (small for clarity)
S = 4   # sequence length (tiny for inspection)
d_h = 8 # head dimension (tiny for inspection)

print("=" * 70)
print("ATTENTION FORWARD PASS — SHAPE INSPECTION")
print("=" * 70)
print(f"Config: B={B}, H={H}, S={S}, d_h={d_h}")
print()

# ============================================================
# PART 2: Create Q, K, V tensors
# Math reference: see 01_scaled_dot_product.md, section "Step 1: Input Shapes"
# ============================================================

# Create random Q, K, V with proper shapes
# In practice, these come from QKV projections: Q = X @ W_Q
rng = np.random.Generator(np.random.PCG64(42))
Q = rng.standard_normal((B, H, S, d_h)).astype(np.float32)
K = rng.standard_normal((B, H, S, d_h)).astype(np.float32)
V = rng.standard_normal((B, H, S, d_h)).astype(np.float32)

print("Q, K, V creation:")
print(f"  Q shape: {Q.shape}")  # Expected: (1, 2, 4, 8)
print(f"  K shape: {K.shape}")  # Expected: (1, 2, 4, 8)
print(f"  V shape: {V.shape}")  # Expected: (1, 2, 4, 8)
assert Q.shape == (B, H, S, d_h), f"Q shape mismatch: {Q.shape}"
assert K.shape == (B, H, S, d_h), f"K shape mismatch: {K.shape}"
assert V.shape == (B, H, S, d_h), f"V shape mismatch: {V.shape}"
print("  ✓ Shape assertions passed")
print()

# ============================================================
# PART 3: QK^T Matmul — compute attention scores
# Math reference: see 01_scaled_dot_product.md, section "Step 2: Query-Key Similarity"
# Formula: scores = Q @ K^T / sqrt(d_h)
# Shapes: [B,H,S,d_h] @ [B,H,d_h,S] -> [B,H,S,S]
# ============================================================

print("QK^T Matmul:")
print(f"  Q shape: {Q.shape}")
print(f"  K^T shape: {K.transpose(0, 1, 3, 2).shape}")  # [B,H,d_h,S]

# Transpose K to get K^T: swap last two dimensions
# K: [B, H, S, d_h] -> K^T: [B, H, d_h, S]
K_T = K.transpose(0, 1, 3, 2)
print(f"  K^T shape after transpose: {K_T.shape}")
assert K_T.shape == (B, H, d_h, S), f"K^T shape mismatch: {K_T.shape}"

# Compute raw scores: Q @ K^T
# [B,H,S,d_h] @ [B,H,d_h,S] -> [B,H,S,S]
scores_raw = np.matmul(Q, K_T)
print(f"  scores_raw shape (Q @ K^T): {scores_raw.shape}")
assert scores_raw.shape == (B, H, S, S), f"scores_raw shape mismatch: {scores_raw.shape}"

# Scale by sqrt(d_h) to prevent softmax saturation
# See 01_scaled_dot_product.md for why sqrt(d_h) is necessary
d_k = d_h  # d_k is the same as d_h in self-attention
scale = np.sqrt(d_k)
scores = scores_raw / scale
print(f"  scores shape (scaled by sqrt({d_k})={scale:.3f}): {scores.shape}")
print()

# Print actual values for inspection (first head, first batch)
print("  Sample scores (batch=0, head=0):")
print("  " + str(scores[0, 0]))
print()

# ============================================================
# PART 4: Causal Masking (optional, for autoregressive)
# Math reference: see 02_causal_masking.md
# Formula: M[i,j] = 0 if j <= i, else -infinity
# ============================================================

print("Causal Masking:")
# Create causal mask: upper triangular is masked (set to -inf)
# Token i can only attend to tokens j <= i
# Use a large negative number instead of -inf to avoid NaN issues
mask = np.zeros((S, S), dtype=np.float32)
for i in range(S):
    for j in range(S):
        if j > i:
            mask[i, j] = -1e9  # Large negative instead of -inf
print(f"  Causal mask shape: {mask.shape}")  # [S, S]
print(f"  Causal mask (S={S}x{S}):")
print("  " + str(mask))

# Apply mask to scores (broadcasts over batch and heads)
scores_masked = scores + mask
print(f"  scores_masked shape: {scores_masked.shape}")
assert scores_masked.shape == (B, H, S, S), f"scores_masked shape mismatch: {scores_masked.shape}"

print("  Sample masked scores (batch=0, head=0):")
print("  " + str(scores_masked[0, 0]))
print()

# ============================================================
# PART 5: Softmax Normalization
# Math reference: see 01_scaled_dot_product.md, section "Step 4: Softmax"
# Formula: P[i,j] = exp(s[i,j]) / sum_k(exp(s[i,k]))
# Applied along last dimension (over keys)
# ============================================================

print("Softmax:")
# Softmax along last dimension (axis=-1, over keys)
# For numerical stability, subtract max before exp
scores_max = np.max(scores_masked, axis=-1, keepdims=True)
scores_shifted = scores_masked - scores_max
scores_exp = np.exp(scores_shifted)
scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
P = scores_exp / scores_sum

print(f"  P (attention probs) shape: {P.shape}")
assert P.shape == (B, H, S, S), f"P shape mismatch: {P.shape}"

# Verify each row sums to 1
row_sums = np.sum(P, axis=-1)
print(f"  Row sums (should be 1.0): {row_sums[0, 0]}")
assert np.allclose(row_sums, 1.0, atol=1e-6), "Softmax rows do not sum to 1"

print("  Sample attention probabilities (batch=0, head=0):")
print("  " + str(P[0, 0]))
print()

# ============================================================
# PART 6: PV Matmul — Weighted Sum of Values
# Math reference: see 01_scaled_dot_product.md, section "Step 5: Weighted Sum"
# Formula: output = P @ V
# Shapes: [B,H,S,S] @ [B,H,S,d_h] -> [B,H,S,d_h]
# ============================================================

print("PV Matmul:")
print(f"  P shape: {P.shape}")
print(f"  V shape: {V.shape}")

# Weighted sum of values
# [B,H,S,S] @ [B,H,S,d_h] -> [B,H,S,d_h]
output = np.matmul(P, V)
print(f"  output shape (P @ V): {output.shape}")
assert output.shape == (B, H, S, d_h), f"output shape mismatch: {output.shape}"

print(f"  Output tensor stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
print()

# ============================================================
# PART 7: Scaling Test — realistic sizes
# Math reference: see 04_flop_and_memory_analysis.md
# This demonstrates O(S^2) memory scaling
# ============================================================

print("=" * 70)
print("SCALING TEST — Realistic LLaMA-3 Sizes")
print("=" * 70)

# LLaMA-3 8B configuration
S_realistic = 4096
H_realistic = 32
d_h_realistic = 128
B_realistic = 1

print(f"\nLLaMA-3 8B config: B={B_realistic}, H={H_realistic}, S={S_realistic}, d_h={d_h_realistic}")

# Compute intermediate tensor sizes
scores_elements = B_realistic * H_realistic * S_realistic * S_realistic
scores_bytes_fp16 = scores_elements * 2  # FP16 = 2 bytes
scores_bytes_fp32 = scores_elements * 4  # FP32 = 4 bytes

print(f"\nQK^T intermediate tensor:")
print(f"  Elements: {scores_elements:,} = {B_realistic} × {H_realistic} × {S_realistic} × {S_realistic}")
print(f"  Memory (FP16): {scores_bytes_fp16 / 1e9:.2f} GB")
print(f"  Memory (FP32): {scores_bytes_fp32 / 1e9:.2f} GB")

# Compute FLOPs
flops_qk = 2 * B_realistic * H_realistic * S_realistic * S_realistic * d_h_realistic
flops_pv = 2 * B_realistic * H_realistic * S_realistic * S_realistic * d_h_realistic
flops_total = flops_qk + flops_pv

print(f"\nFLOP count (attention only, excluding projections):")
print(f"  QK^T matmul: {flops_qk / 1e9:.2f} GFLOPs")
print(f"  PV matmul: {flops_pv / 1e9:.2f} GFLOPs")
print(f"  Total: {flops_total / 1e9:.2f} GFLOPs")

# Arithmetic intensity (naive implementation)
bytes_read = 6 * B_realistic * H_realistic * S_realistic * d_h_realistic * 2  # Q, K, V
bytes_write = 2 * B_realistic * H_realistic * S_realistic * S_realistic * 2 + B_realistic * H_realistic * S_realistic * d_h_realistic * 2  # scores, P, output
bytes_total = bytes_read + bytes_write
ai = flops_total / bytes_total

print(f"\nArithmetic intensity (naive):")
print(f"  Bytes read: {bytes_read / 1e6:.2f} MB")
print(f"  Bytes written: {bytes_write / 1e6:.2f} MB")
print(f"  Total HBM traffic: {bytes_total / 1e9:.2f} GB")
print(f"  AI = {flops_total / 1e9:.2f} GFLOPs / {bytes_total / 1e9:.2f} GB = {ai:.1f} FLOPs/byte")
print(f"  H100 peak (FP16): ~2000 FLOPs/byte")
print(f"  Status: {'COMPUTE-BOUND' if ai > 2000 else 'MEMORY-BOUND'}")

print()

# ============================================================
# VERIFY: Expected output summary
# ============================================================

print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"✓ Q, K, V shapes: ({B}, {H}, {S}, {d_h})")
print(f"✓ QK^T scores shape: ({B}, {H}, {S}, {S})")
print(f"✓ Attention probs P shape: ({B}, {H}, {S}, {S})")
print(f"✓ Output shape: ({B}, {H}, {S}, {d_h})")
print(f"✓ Softmax rows sum to 1.0")
print(f"✓ LLaMA-3 8B QK^T intermediate: ~{scores_bytes_fp16 / 1e9:.1f} GB (FP16)")
print(f"✓ LLaMA-3 8B attention FLOPs: ~{flops_total / 1e9:.0f} GFLOPs")
print(f"✓ Arithmetic intensity: {ai:.1f} FLOPs/byte (MEMORY-BOUND at B=1)")
print()
print("PASS — All shape assertions and computations verified.")
print()
print("Key insight: The O(S²) intermediate (QK^T scores) is the memory wall.")
print("FlashAttention avoids materializing this in HBM by computing tile-by-tile in SRAM.")
