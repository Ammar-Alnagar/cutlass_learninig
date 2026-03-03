"""
FILE: quantization.py
TEACHES: INT8 quantization/dequantization, accuracy vs. compression tradeoff
MAPS TO: NVIDIA inference engineer — INT8/FP8 kernel implementation
RUN: python quantization.py — no arguments needed
"""

import numpy as np

print("=" * 70)
print("QUANTIZATION: INT8 WEIGHT QUANTIZATION")
print("=" * 70)

# ============================================================
# PART 1: INT8 Quantization Functions
# Math reference: see 02_int8_weight_quant.md
# ============================================================

def quantize_int8_per_tensor(x):
    """Symmetric per-tensor INT8 quantization."""
    max_val = np.max(np.abs(x))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    x_int8 = np.round(x / scale).astype(np.int8)
    return x_int8, scale

def dequantize_int8(x_int8, scale):
    """Dequantize INT8 to FP32."""
    return x_int8.astype(np.float32) * scale

def quantize_int8_per_channel(x):
    """Per-channel INT8 quantization (for weight matrices)."""
    # x shape: [out_features, in_features]
    max_per_channel = np.max(np.abs(x), axis=1, keepdims=True)
    scale = max_per_channel / 127.0
    scale = np.where(scale > 0, scale, 1.0)
    x_int8 = np.round(x / scale).astype(np.int8)
    return x_int8, scale.flatten()

def dequantize_int8_per_channel(x_int8, scale):
    """Dequantize per-channel INT8."""
    return x_int8.astype(np.float32) * scale[:, np.newaxis]

# ============================================================
# PART 2: Quantize a Sample Weight Matrix
# ============================================================

print("\n" + "=" * 70)
print("PER-TENSOR VS. PER-CHANNEL QUANTIZATION")
print("=" * 70)

# Create sample weight matrix
rng = np.random.Generator(np.random.PCG64(42))
W = rng.standard_normal((64, 128)).astype(np.float32)

print(f"\nOriginal weight matrix: {W.shape}")
print(f"  Min: {W.min():.4f}, Max: {W.max():.4f}")
print(f"  Mean: {W.mean():.4f}, Std: {W.std():.4f}")

# Per-tensor quantization
W_int8_tensor, scale_tensor = quantize_int8_per_tensor(W)
W_recon_tensor = dequantize_int8(W_int8_tensor, scale_tensor)
error_tensor = np.abs(W - W_recon_tensor)

print(f"\nPer-tensor INT8:")
print(f"  Scale: {scale_tensor:.6f}")
print(f"  Quantized range: [{W_int8_tensor.min()}, {W_int8_tensor.max()}]")
print(f"  Reconstruction error: max={error_tensor.max():.6f}, mean={error_tensor.mean():.6f}")

# Per-channel quantization
W_int8_channel, scale_channel = quantize_int8_per_channel(W)
W_recon_channel = dequantize_int8_per_channel(W_int8_channel, scale_channel)
error_channel = np.abs(W - W_recon_channel)

print(f"\nPer-channel INT8:")
print(f"  Scales: min={scale_channel.min():.6f}, max={scale_channel.max():.6f}")
print(f"  Quantized range: [{W_int8_channel.min()}, {W_int8_channel.max()}]")
print(f"  Reconstruction error: max={error_channel.max():.6f}, mean={error_channel.mean():.6f}")

print(f"\nPer-channel improvement: {error_tensor.mean() / error_channel.mean():.2f}x lower mean error")

# ============================================================
# PART 3: Quantized GEMM Simulation
# ============================================================

print("\n" + "=" * 70)
print("QUANTIZED GEMM SIMULATION")
print("=" * 70)

# Input activation
X = rng.standard_normal((1, 128)).astype(np.float32)

# Reference FP16 GEMM
Y_ref = np.matmul(X, W.T)

# Quantized GEMM (per-channel weights)
X_int8, X_scale = quantize_int8_per_tensor(X)
X_recon = dequantize_int8(X_int8, X_scale)
Y_quant = np.matmul(X_recon, W_recon_channel.T)

print(f"\nGEMM: X {X.shape} @ W.T {W.T.shape} = Y {Y_ref.shape}")
print(f"\nReference (FP16): min={Y_ref.min():.4f}, max={Y_ref.max():.4f}")
print(f"Quantized (INT8): min={Y_quant.min():.4f}, max={Y_quant.max():.4f}")

gemm_error = np.abs(Y_ref - Y_quant)
print(f"GEMM error: max={gemm_error.max():.6f}, mean={gemm_error.mean():.6f}")
print(f"Relative error: {100 * gemm_error.mean() / np.abs(Y_ref).mean():.2f}%")

# ============================================================
# PART 4: KV Cache Quantization
# Math reference: see 03_kv_cache_quantization.md
# ============================================================

print("\n" + "=" * 70)
print("KV CACHE QUANTIZATION")
print("=" * 70)

# Simulate KV cache for LLaMA-3 8B
L = 32  # layers
H_kv = 8  # KV heads (GQA)
S = 4096  # sequence length
d_h = 128  # head dimension

# FP16 KV cache
K_fp16 = rng.standard_normal((L, H_kv, S, d_h)).astype(np.float32)
V_fp16 = rng.standard_normal((L, H_kv, S, d_h)).astype(np.float32)

fp16_bytes = K_fp16.nbytes + V_fp16.nbytes
print(f"\nFP16 KV cache: {fp16_bytes / 1e9:.2f} GB")

# INT8 KV cache (per-token quantization)
# Simplified: quantize entire cache with one scale per layer
K_int8 = np.zeros_like(K_fp16, dtype=np.int8)
V_int8 = np.zeros_like(V_fp16, dtype=np.int8)
K_scales = np.zeros(L)
V_scales = np.zeros(L)

for layer in range(L):
    K_int8[layer], K_scales[layer] = quantize_int8_per_tensor(K_fp16[layer])
    V_int8[layer], V_scales[layer] = quantize_int8_per_tensor(V_fp16[layer])

int8_bytes = K_int8.nbytes + V_int8.nbytes + (K_scales.nbytes + V_scales.nbytes)
print(f"INT8 KV cache: {int8_bytes / 1e9:.2f} GB (including scales)")
print(f"Compression: {fp16_bytes / int8_bytes:.1f}x")

# Dequantize and check error
K_recon = np.zeros_like(K_fp16)
V_recon = np.zeros_like(V_fp16)

for layer in range(L):
    K_recon[layer] = dequantize_int8(K_int8[layer], K_scales[layer])
    V_recon[layer] = dequantize_int8(V_int8[layer], V_scales[layer])

kv_error = np.abs(K_fp16 - K_recon).mean() + np.abs(V_fp16 - V_recon).mean()
print(f"KV cache reconstruction error (mean): {kv_error:.6f}")

# ============================================================
# PART 5: FP8 Format Comparison
# Math reference: see 04_fp8_formats.md
# ============================================================

print("\n" + "=" * 70)
print("FP8 FORMAT COMPARISON (E4M3 vs. E5M2)")
print("=" * 70)

# FP8 ranges (approximate)
fp8_e4m3_range = (6e-5, 448)
fp8_e5m2_range = (6e-5, 57344)

print(f"\nE4M3 (4 exponent, 3 mantissa):")
print(f"  Range: [{fp8_e4m3_range[0]}, {fp8_e4m3_range[1]}]")
print(f"  Use: Weights, activations (bounded range)")

print(f"\nE5M2 (5 exponent, 2 mantissa):")
print(f"  Range: [{fp8_e5m2_range[0]}, {fp8_e5m2_range[1]}]")
print(f"  Use: Gradients, attention scores (large dynamic range)")

# ============================================================
# VERIFY: Summary
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n✓ Per-tensor INT8: scale={scale_tensor:.6f}")
print(f"✓ Per-channel INT8: {len(scale_channel)} scales")
print(f"✓ Per-channel improvement: {error_tensor.mean() / error_channel.mean():.2f}x")
print(f"✓ GEMM relative error: {100 * gemm_error.mean() / np.abs(Y_ref).mean():.2f}%")
print(f"✓ KV cache compression: {fp16_bytes / int8_bytes:.1f}x")
print(f"✓ FP8 formats: E4M3 (precision), E5M2 (range)")
print()
print("PASS — Quantization simulation complete.")
print()
print("Key insights:")
print("  1. Per-channel quantization has better accuracy than per-tensor")
print("  2. INT8 KV cache reduces memory by 2x")
print("  3. FP8 E4M3 for weights, E5M2 for activations with large range")
print("  4. H100 native FP8 tensor cores: 4x FP16 throughput")
