# Module 04: Positional Encoding — RoPE

**One concept:** Rotary Position Embeddings (RoPE) encode relative positions via rotation in the complex plane. RoPE must be fused with QK computation for efficiency.

**Job mapping:** NVIDIA kernel engineer — you will implement fused RoPE kernels that rotate Q, K before attention.

---

## Files in This Module

1. **01_rope_math.md** — Rotation formula, why relative positions fall out naturally.

2. **02_rope_kernel_implications.md** — Why RoPE must be fused, head_dim divisibility.

3. **rope.py** — RoPE implementation with visual explanation of rotation.

---

## What You Must Be Able To Do After This Module

1. Compute RoPE rotation for any position and frequency

2. Explain why RoPE gives relative position encoding

3. Implement fused RoPE in a CUDA kernel

---

**Next:** `01_rope_math.md` — the rotation formula
