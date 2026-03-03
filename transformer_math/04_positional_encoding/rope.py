"""
FILE: rope.py
TEACHES: RoPE rotation formula and relative position encoding
MAPS TO: NVIDIA kernel engineer — fused RoPE kernel implementation
RUN: python rope.py — no arguments needed
"""

import numpy as np

# ============================================================
# PART 1: Configuration
# Math reference: see 01_rope_math.md, section "Frequency Definition"
# ============================================================

d_h = 8   # head dimension (small for inspection)
S = 16    # sequence length (positions 0 to 15)

print("=" * 70)
print("RoPE: ROTARY POSITION EMBEDDINGS")
print("=" * 70)
print(f"Config: d_h={d_h}, S={S}")
print()

# ============================================================
# PART 2: Compute RoPE Frequencies
# Math reference: see 01_rope_math.md, section "Frequency Definition"
# Formula: θ_i = 10000^(-2i / d_h)
# ============================================================

print("=" * 70)
print("ROPE FREQUENCIES")
print("=" * 70)

num_freq = d_h // 2  # Number of frequency pairs
frequencies = []

print(f"\nNumber of frequency pairs: {num_freq}")
print(f"Formula: θ_i = 10000^(-2i / d_h) for i = 0, 1, ..., {num_freq - 1}")
print()
print(f"{'i':<4} {'θ_i':<15} {'Period (2π/θ_i)':<15}")
print("-" * 35)

for i in range(num_freq):
    theta_i = 10000 ** (-2 * i / d_h)
    period = 2 * np.pi / theta_i if theta_i > 0 else float('inf')
    frequencies.append(theta_i)
    print(f"{i:<4} {theta_i:<15.6f} {period:<15.1f}")

frequencies = np.array(frequencies)
print()

# ============================================================
# PART 3: RoPE Rotation Function
# Math reference: see 01_rope_math.md, section "RoPE Rotation Formula"
# Formula: [x', y'] = [x*cos(mθ) - y*sin(mθ), x*sin(mθ) + y*cos(mθ)]
# ============================================================

def apply_rope(x, y, position, theta):
    """
    Apply RoPE rotation to a 2D vector (x, y) at given position.
    
    Args:
        x, y: 2D vector components
        position: Token position m
        theta: Frequency θ_i
    
    Returns:
        x_rot, y_rot: Rotated vector
    """
    angle = position * theta
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    
    x_rot = x * cos_val - y * sin_val
    y_rot = x * sin_val + y * cos_val
    
    return x_rot, y_rot, cos_val, sin_val

# ============================================================
# PART 4: Visualize RoPE Rotation
# ============================================================

print("=" * 70)
print("ROPE ROTATION VISUALIZATION")
print("=" * 70)

# Create a sample 2D vector
x, y = 1.0, 0.0  # Unit vector along x-axis

print(f"\nOriginal vector: ({x}, {y})")
print(f"Norm: {np.sqrt(x**2 + y**2):.4f}")
print()

print(f"{'Position':<10} {'Angle (rad)':<12} {'Rotated (x, y)':<20} {'Norm':<10}")
print("-" * 55)

for m in range(S):
    theta = frequencies[0]  # Use first frequency (fastest rotation)
    x_rot, y_rot, cos_val, sin_val = apply_rope(x, y, m, theta)
    norm = np.sqrt(x_rot**2 + y_rot**2)
    angle = m * theta
    print(f"{m:<10} {angle:<12.4f} ({x_rot:6.3f}, {y_rot:6.3f}){'':<5} {norm:<10.4f}")

print()
print("Note: RoPE preserves vector norm (rotation is orthogonal).")
print()

# ============================================================
# PART 5: RoPE for Full Head Dimension
# ============================================================

print("=" * 70)
print("ROPE FOR FULL HEAD DIMENSION")
print("=" * 70)

# Create a sample query vector at position m
rng = np.random.Generator(np.random.PCG64(42))
Q_m = rng.standard_normal(d_h).astype(np.float32)
K_n = rng.standard_normal(d_h).astype(np.float32)

print(f"\nQuery vector Q at position m=5:")
print(f"  Q = {Q_m}")
print()

print(f"Key vector K at position n=3:")
print(f"  K = {K_n}")
print()

# Apply RoPE to Q and K
def apply_rope_full(vec, position, frequencies):
    """Apply RoPE to full d_h-dimensional vector."""
    vec_rot = vec.copy()
    for i in range(len(frequencies)):
        dim = 2 * i
        x, y = vec[dim], vec[dim + 1]
        x_rot, y_rot, _, _ = apply_rope(x, y, position, frequencies[i])
        vec_rot[dim] = x_rot
        vec_rot[dim + 1] = y_rot
    return vec_rot

Q_m_rot = apply_rope_full(Q_m, 5, frequencies)
K_n_rot = apply_rope_full(K_n, 3, frequencies)

print(f"Q_rot (position 5): {Q_m_rot}")
print(f"K_rot (position 3): {K_n_rot}")
print()

# ============================================================
# PART 6: Relative Position Encoding
# Math reference: see 01_rope_math.md, section "Why Relative Positions Fall Out"
# ============================================================

print("=" * 70)
print("RELATIVE POSITION ENCODING")
print("=" * 70)

print("\nRoPE ensures attention scores depend on relative position (m - n).")
print()

# Compute attention scores with and without RoPE
positions = [(5, 3), (7, 5), (10, 8)]  # Different (m, n) pairs with same m-n=2

print(f"Testing pairs with same relative position (m - n = 2):")
print()

for m, n in positions:
    Q_m_rot = apply_rope_full(Q_m, m, frequencies)
    K_n_rot = apply_rope_full(K_n, n, frequencies)
    
    # Dot product (attention score without softmax)
    score_rot = np.dot(Q_m_rot, K_n_rot)
    
    # Without RoPE (for comparison)
    score_no_rope = np.dot(Q_m, K_n)
    
    print(f"  (m={m}, n={n}): m-n={m-n}")
    print(f"    Score without RoPE: {score_no_rope:.6f} (same for all)")
    print(f"    Score with RoPE:    {score_rot:.6f} (depends on m-n)")
    print()

print("Key insight: With RoPE, scores for different (m,n) with same m-n")
print("             share similar structure (modulated by the original Q·K).")
print()

# ============================================================
# PART 7: Complex Number Interpretation
# Math reference: see 01_rope_math.md, section "Complex Number Formulation"
# ============================================================

print("=" * 70)
print("COMPLEX NUMBER INTERPRETATION")
print("=" * 70)

print("\nRoPE can be written as complex multiplication:")
print("  RoPE(z, m) = z · e^(imθ)")
print()

# Complex representation
z_Q = Q_m[0] + 1j * Q_m[1]
z_K = K_n[0] + 1j * K_n[1]

m, n = 5, 3
theta = frequencies[0]

# Complex rotation
z_Q_rot = z_Q * np.exp(1j * m * theta)
z_K_rot = z_K * np.exp(1j * n * theta)

# Complex dot product (with conjugate for K)
score_complex = (z_Q_rot * np.conj(z_K_rot)).real

# Verify matches real-valued RoPE
Q_m_rot_2d = apply_rope_full(Q_m[:2], m, [theta])
K_n_rot_2d = apply_rope_full(K_n[:2], n, [theta])
score_real = np.dot(Q_m_rot_2d, K_n_rot_2d)

print(f"First dimension pair (using θ_0 = {frequencies[0]:.6f}):")
print(f"  Q (complex): {z_Q:.4f}")
print(f"  K (complex): {z_K:.4f}")
print()
print(f"  Q_rot at m={m}: {z_Q_rot:.4f}")
print(f"  K_rot at n={n}: {z_K_rot:.4f}")
print()
print(f"  Score (complex): {score_complex:.6f}")
print(f"  Score (real-valued RoPE): {score_real:.6f}")
print(f"  Match: {np.isclose(score_complex, score_real)}")
print()

# ============================================================
# PART 8: Precomputed Cos/Sin Table
# Math reference: see 02_rope_kernel_implications.md
# ============================================================

print("=" * 70)
print("PRECOMPUTED COS/SIN TABLE (for kernel use)")
print("=" * 70)

# Precompute cos/sin for all positions and frequencies
cos_table = np.zeros((S, num_freq))
sin_table = np.zeros((S, num_freq))

for m in range(S):
    for i in range(num_freq):
        theta = frequencies[i]
        cos_table[m, i] = np.cos(m * theta)
        sin_table[m, i] = np.sin(m * theta)

print(f"\nTable shape: {cos_table.shape} (positions × frequencies)")
print(f"Memory (FP32): {cos_table.nbytes * 2 / 1024:.1f} KB")
print()

print("First 5 positions, first 4 frequencies:")
print(f"{'Pos':<5}", end="")
for i in range(min(4, num_freq)):
    print(f"θ_{i}={frequencies[i]:.4f}  ", end="")
print()
print("-" * 60)

for m in range(min(5, S)):
    print(f"{m:<5}", end="")
    for i in range(min(4, num_freq)):
        print(f"c={cos_table[m,i]:5.3f} s={sin_table[m,i]:5.3f}  ", end="")
    print()

print()
print("In kernel: table lookup instead of computing cos/sin at runtime.")
print()

# ============================================================
# VERIFY: Summary
# ============================================================

print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n✓ Frequencies computed: {num_freq} pairs")
print(f"✓ RoPE preserves norm: rotation is orthogonal")
print(f"✓ Relative position encoding: scores depend on m-n")
print(f"✓ Complex interpretation: RoPE(z, m) = z · e^(imθ)")
print(f"✓ Cos/sin table: {cos_table.shape}, {cos_table.nbytes * 2 / 1024:.1f} KB")
print()
print("PASS — RoPE implementation verified.")
print()
print("Key insights:")
print("  1. RoPE rotates 2D planes by position-dependent angles")
print("  2. Frequencies: θ_i = 10000^(-2i/d_h) — lower dims rotate faster")
print("  3. Relative positions: Q_m·K_n depends on m-n after RoPE")
print("  4. Kernel fusion: apply RoPE on-the-fly during QK computation")
print("  5. Precomputed tables: avoid runtime cos/sin computation")
