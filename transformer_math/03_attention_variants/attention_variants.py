"""
FILE: attention_variants.py
TEACHES: MHA vs. MQA vs. GQA forward pass with KV cache comparison
MAPS TO: NVIDIA inference engineer — GQA kernel implementation with stride-0 layout
RUN: python attention_variants.py — no arguments needed
"""

import numpy as np

# ============================================================
# PART 1: Configuration
# Math reference: see 02_gqa.md, section "GQA Head Structure"
# ============================================================

# Use tiny defaults for easy inspection
B = 1   # batch size
S = 8   # sequence length
d_h = 4 # head dimension

# MHA configuration
H_mha = 4  # query heads = KV heads

# GQA configuration
H_q_gqa = 4   # query heads
H_kv_gqa = 2  # KV heads (GQA: H_kv < H_q)
G_gqa = H_q_gqa // H_kv_gqa  # group size = 2

# MQA configuration
H_mqa = 4  # query heads
H_kv_mqa = 1  # single KV head

print("=" * 70)
print("ATTENTION VARIANTS: MHA vs. GQA vs. MQA")
print("=" * 70)
print(f"Config: B={B}, S={S}, d_h={d_h}")
print(f"  MHA: H={H_mha}")
print(f"  GQA: H_q={H_q_gqa}, H_kv={H_kv_gqa}, G={G_gqa}")
print(f"  MQA: H_q={H_mqa}, H_kv={H_kv_mqa}")
print()

# ============================================================
# PART 2: Create Q, K, V tensors for each variant
# Math reference: see 01_mqa.md and 02_gqa.md for shapes
# ============================================================

rng = np.random.Generator(np.random.PCG64(42))

# MHA: Q, K, V all have H heads
Q_mha = rng.standard_normal((B, H_mha, S, d_h)).astype(np.float32)
K_mha = rng.standard_normal((B, H_mha, S, d_h)).astype(np.float32)
V_mha = rng.standard_normal((B, H_mha, S, d_h)).astype(np.float32)

print("MHA shapes:")
print(f"  Q: {Q_mha.shape}, K: {K_mha.shape}, V: {V_mha.shape}")
print()

# GQA: Q has H_q heads, K/V have H_kv heads
Q_gqa = rng.standard_normal((B, H_q_gqa, S, d_h)).astype(np.float32)
K_gqa = rng.standard_normal((B, H_kv_gqa, S, d_h)).astype(np.float32)
V_gqa = rng.standard_normal((B, H_kv_gqa, S, d_h)).astype(np.float32)

print("GQA shapes:")
print(f"  Q: {Q_gqa.shape}")
print(f"  K: {K_gqa.shape} (H_kv={H_kv_gqa}, broadcast to H_q={H_q_gqa})")
print(f"  V: {V_gqa.shape}")
print()

# MQA: Q has H heads, K/V have 1 head
Q_mqa = rng.standard_normal((B, H_mqa, S, d_h)).astype(np.float32)
K_mqa = rng.standard_normal((B, 1, S, d_h)).astype(np.float32)
V_mqa = rng.standard_normal((B, 1, S, d_h)).astype(np.float32)

print("MQA shapes:")
print(f"  Q: {Q_mqa.shape}")
print(f"  K: {K_mqa.shape} (single KV head, broadcast to H={H_mqa})")
print(f"  V: {V_mqa.shape}")
print()

# ============================================================
# PART 3: Attention Forward Pass for Each Variant
# Math reference: see 01_scaled_dot_product.md
# ============================================================

def scaled_dot_product_attention(Q, K, V, causal=True):
    """
    Scaled dot-product attention with optional broadcasting.
    
    Handles:
    - MHA: Q, K, V all have same H
    - GQA: K, V have fewer heads (broadcast)
    - MQA: K, V have 1 head (broadcast)
    """
    B, H_q, S_q, d_h = Q.shape
    H_kv = K.shape[1]
    S_k = K.shape[2]
    
    # Compute QK^T with broadcasting
    scores = np.zeros((B, H_q, S_q, S_k), dtype=np.float32)
    
    for b in range(B):
        for h_q in range(H_q):
            if H_kv == 1:
                # MQA: all query heads use the same KV head
                h_kv = 0
            elif H_kv < H_q:
                # GQA: query head h_q uses KV head h_q // G
                h_kv = h_q // (H_q // H_kv)
            else:
                # MHA: one-to-one mapping
                h_kv = h_q
            
            # Compute scores for this query head
            # Q[b, h_q]: [S_q, d_h], K[b, h_kv]: [S_k, d_h]
            # scores: [S_q, S_k]
            scores_hq = np.matmul(Q[b, h_q], K[b, h_kv].T) / np.sqrt(d_h)
            scores[b, h_q, :, :] = scores_hq
    
    # Apply causal mask
    if causal:
        mask = np.zeros((S_q, S_k), dtype=np.float32)
        for i in range(S_q):
            for j in range(S_k):
                if j > i:
                    mask[i, j] = -1e9
        scores = scores + mask  # Broadcasts over B, H
    
    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_shifted = scores - scores_max
    scores_exp = np.exp(scores_shifted)
    scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
    P = scores_exp / scores_sum
    
    # Weighted sum of values
    O = np.zeros((B, H_q, S_q, d_h), dtype=np.float32)
    
    for b in range(B):
        for h_q in range(H_q):
            if H_kv == 1:
                h_kv = 0
            elif H_kv < H_q:
                h_kv = h_q // (H_q // H_kv)
            else:
                h_kv = h_q
            
            O[b, h_q] = np.matmul(P[b, h_q], V[b, h_kv])
    
    return O, P

# ============================================================
# PART 4: Compute Attention for Each Variant
# ============================================================

print("=" * 70)
print("ATTENTION FORWARD PASS")
print("=" * 70)

# MHA attention
O_mha, P_mha = scaled_dot_product_attention(Q_mha, K_mha, V_mha)
print(f"\nMHA output shape: {O_mha.shape}")
print(f"  Attention probs shape: {P_mha.shape}")
print(f"  Output stats: min={O_mha.min():.4f}, max={O_mha.max():.4f}, mean={O_mha.mean():.4f}")

# GQA attention
O_gqa, P_gqa = scaled_dot_product_attention(Q_gqa, K_gqa, V_gqa)
print(f"\nGQA output shape: {O_gqa.shape}")
print(f"  Attention probs shape: {P_gqa.shape}")
print(f"  Output stats: min={O_gqa.min():.4f}, max={O_gqa.max():.4f}, mean={O_gqa.mean():.4f}")

# MQA attention
O_mqa, P_mqa = scaled_dot_product_attention(Q_mqa, K_mqa, V_mqa)
print(f"\nMQA output shape: {O_mqa.shape}")
print(f"  Attention probs shape: {P_mqa.shape}")
print(f"  Output stats: min={O_mqa.min():.4f}, max={O_mqa.max():.4f}, mean={O_mqa.mean():.4f}")

# ============================================================
# PART 5: KV Cache Memory Comparison
# Math reference: see 02_memory_formula.md
# ============================================================

print()
print("=" * 70)
print("KV CACHE MEMORY COMPARISON")
print("=" * 70)

dtype_bytes = 2  # FP16
L = 32  # layers (LLaMA-3 8B)
S_llama = 4096

def kv_cache_bytes(L, S, H_kv, d_h, B, dtype_bytes):
    """Compute KV cache memory in bytes."""
    return 2 * L * S * H_kv * d_h * dtype_bytes * B

# LLaMA-3 8B configurations
print("\nLLaMA-3 8B (L=32, S=4096, d_h=128, FP16):")
print()

# MHA
H_mha_llama = 32
kv_mha = kv_cache_bytes(L, S_llama, H_mha_llama, 128, 1, dtype_bytes)
print(f"  MHA (H_kv={H_mha_llama}): {kv_mha / 1e9:.2f} GB per sequence")

# GQA
H_q_gqa_llama = 32
H_kv_gqa_llama = 8
kv_gqa = kv_cache_bytes(L, S_llama, H_kv_gqa_llama, 128, 1, dtype_bytes)
print(f"  GQA (H_kv={H_kv_gqa_llama}): {kv_gqa / 1e9:.2f} GB per sequence")
print(f"    Reduction vs. MHA: {kv_mha / kv_gqa:.1f}x")

# MQA
kv_mqa = kv_cache_bytes(L, S_llama, 1, 128, 1, dtype_bytes)
print(f"  MQA (H_kv=1): {kv_mqa / 1e9:.2f} GB per sequence")
print(f"    Reduction vs. MHA: {kv_mha / kv_mqa:.1f}x")

# ============================================================
# PART 6: GQA Head Mapping Verification
# Math reference: see 02_gqa.md, section "Query-to-KV Head Mapping"
# ============================================================

print()
print("=" * 70)
print("GQA HEAD MAPPING VERIFICATION")
print("=" * 70)

print(f"\nLLaMA-3 8B GQA: H_q={H_q_gqa_llama}, H_kv={H_kv_gqa_llama}, G={H_q_gqa_llama // H_kv_gqa_llama}")
print()
print("  Query Head → KV Head mapping:")

for h_q in range(H_q_gqa_llama):
    h_kv = h_q // (H_q_gqa_llama // H_kv_gqa_llama)
    print(f"    h_q={h_q:2d} → h_kv={h_kv}")

# Verify grouping
print()
print("  Group verification:")
for h_kv in range(H_kv_gqa_llama):
    start_hq = h_kv * (H_q_gqa_llama // H_kv_gqa_llama)
    end_hq = start_hq + (H_q_gqa_llama // H_kv_gqa_llama)
    h_q_range = list(range(start_hq, end_hq))
    print(f"    KV head {h_kv}: query heads {h_q_range}")

# ============================================================
# PART 7: Stride-0 Layout Explanation
# Math reference: see 02_gqa.md, section "Stride-0 Layout for GQA"
# ============================================================

print()
print("=" * 70)
print("STRIDE-0 LAYOUT FOR GQA (CuTe)")
print("=" * 70)

print("""
In CuTe, GQA is implemented using stride-0 for the KV head dimension:

  // Physical KV cache: [B, H_kv, S, d_h]
  // Logical access: [B, H_q, S, d_h] with stride-0 broadcast
  
  auto kv_layout = make_layout(
      make_shape(B, H_q, S, d_h),           // Logical shape
      make_stride(kv_stride_b, 0,           // stride-0 for head!
                  kv_stride_s, kv_stride_d)
  );

Stride-0 means: Query heads in the same group read the same KV data.

Example (LLaMA-3 8B, G=4):
  Query heads 0,1,2,3 → all read KV head 0 (same memory location)
  Query heads 4,5,6,7 → all read KV head 1 (same memory location)
  
No data duplication. No explicit broadcast. Hardware handles it.
""")

# ============================================================
# VERIFY: Summary
# ============================================================

print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n✓ MHA shapes: Q={Q_mha.shape}, K={K_mha.shape}, V={V_mha.shape}")
print(f"✓ GQA shapes: Q={Q_gqa.shape}, K={K_gqa.shape}, V={V_gqa.shape}")
print(f"✓ MQA shapes: Q={Q_mqa.shape}, K={K_mqa.shape}, V={V_mqa.shape}")
print(f"✓ MHA output: {O_mha.shape}")
print(f"✓ GQA output: {O_gqa.shape}")
print(f"✓ MQA output: {O_mqa.shape}")
print()
print(f"KV Cache (LLaMA-3 8B, S=4096, FP16):")
print(f"  MHA: {kv_mha / 1e9:.2f} GB")
print(f"  GQA: {kv_gqa / 1e9:.2f} GB ({kv_mha / kv_gqa:.1f}x reduction)")
print(f"  MQA: {kv_mqa / 1e9:.2f} GB ({kv_mha / kv_mqa:.1f}x reduction)")
print()
print("PASS — Attention variants comparison complete.")
print()
print("Key insights:")
print("  1. GQA uses fewer KV heads than query heads (H_kv < H_q)")
print("  2. Query heads are grouped: each group shares one KV head")
print("  3. GQA KV cache reduction = H_q / H_kv (4x for LLaMA-3 8B)")
print("  4. Stride-0 layout in CuTe enables efficient GQA without duplication")
print("  5. MQA has maximum reduction (H_x) but may have accuracy loss")
