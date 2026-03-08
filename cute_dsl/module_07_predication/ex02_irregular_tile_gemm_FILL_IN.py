"""
Module 07 — Predication
Exercise 02 — Irregular Tile GEMM

CONCEPT BRIDGE (C++ → DSL):
  C++:  // GEMM with irregular M dimension
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                if (m < actual_M && n < actual_N) {
                    C[m,n] = dot(A[m,:], B[:,n]);
                }
            }
        }
  DSL:  # Same with predicated tiled GEMM
        pred = (m < actual_M) & (n < actual_N)
        cute.gemm(tiled_mma, C, A, B, C, pred=pred)
  Key:  Predicated GEMM handles non-tile-aligned dimensions.

WHAT YOU'RE BUILDING:
  A GEMM kernel that handles irregular M and N dimensions using predication.
  This is essential for production kernels where matrix sizes are rarely
  perfect multiples of tile sizes.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Implement predicated GEMM for irregular shapes
  - Create 2D predicate tensors
  - Handle boundary tiles correctly

REQUIRED READING:
  - CUTLASS GEMM tutorial: https://nvidia.github.io/cutlass-dsl/tutorials/gemm.html
  - FlashAttention irregular shapes: https://arxiv.org/abs/2307.08691
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: For M=100, N=100, K=128 with tile size 32, how many tiles in each dimension?
# Your answer:

# Q2: Which tiles need predication?
# Your answer:

# Q3: What is the computational waste without predication?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Irregular matrix dimensions
M, N, K = 100, 100, 128

# Tile size
TILE_M, TILE_N, TILE_K = 32, 32, 32

# Number of tiles (ceiling division)
NUM_TILES_M = (M + TILE_M - 1) // TILE_M  # = 4
NUM_TILES_N = (N + TILE_N - 1) // TILE_N  # = 4
NUM_TILES_K = (K + TILE_K - 1) // TILE_K  # = 4


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_irregular_tile_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    actual_M: int,
    actual_N: int,
    results: cute.Tensor,
):
    """
    GEMM with predication for irregular tile boundaries.
    
    FILL IN [HARD]: Implement predicated GEMM that handles non-aligned dimensions.
    
    HINT: Create predicate tensor: pred[i,j] = (i < actual_M) & (j < actual_N)
          Pass pred to cute.gemm() or use conditional execution.
    """
    # --- Step 1: Create MMA atom and TiledMMA ---
    # TODO: mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
    #       tiled_mma = cute.make_tiled_mma(mma_atom, (2, 4), (16, 16))
    
    # --- Step 2: Create predicate tensor ---
    # TODO: pred = cute.make_tensor(...)
    #       for i in range(TILE_M):
    #           for j in range(TILE_N):
    #               pred[i,j] = (i < actual_M) and (j < actual_N)
    
    # --- Step 3: Tiled GEMM mainloop ---
    # TODO: for k_block in range(NUM_TILES_K):
    #           # Load A and B tiles with predication
    #           # MMA with predication
    #           cute.gemm(tiled_mma, accum, a_frag, b_frag, accum, pred=pred)
    
    # --- Step 4: Store results ---
    # Store C[0,0], C[actual_M-1, actual_N-1], and element count
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify irregular tile GEMM.
    """
    
    # Create matrices
    torch.manual_seed(42)
    A_torch = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B_torch = torch.randn((K, N), dtype=torch.float16, device='cuda')
    C_torch = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    # Reference
    C_ref = torch.matmul(A_torch.float(), B_torch.float()).cpu().numpy()
    
    A_cute = from_dlpack(A_torch)
    B_cute = from_dlpack(B_torch)
    C_cute = from_dlpack(C_torch)
    
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_irregular_tile_gemm[1, 32](A_cute, B_cute, C_cute, M, N, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    C_cpu = C_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 07 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  Irregular GEMM Configuration:")
    print(f"    A: ({M}, {K})")
    print(f"    B: ({K}, {N})")
    print(f"    C: ({M}, {N})")
    print(f"    Tile size: ({TILE_M}, {TILE_N}, {TILE_K})")
    print(f"\n  Results:")
    print(f"    C[0,0]:      {C_cpu[0, 0]:.4f} (ref: {C_ref[0, 0]:.4f})")
    print(f"    C[{M-1},{N-1}]: {C_cpu[M-1, N-1]:.4f} (ref: {C_ref[M-1, N-1]:.4f})")
    print(f"    Valid elems: {results_cpu[4]:.0f}")
    
    # Verify (allow some numerical difference)
    max_diff = abs(C_cpu[:M, :N] - C_ref[:M, :N]).max()
    passed = max_diff < 0.5
    
    print(f"\n  Max difference: {max_diff:.6f}")
    print(f"  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How much computation is wasted without predication?
# C2: What is the performance cost of predication?
# C3: In FlashAttention, when do you need predication?
# C4: How would you handle 3D batched GEMM with irregular shapes?

if __name__ == "__main__":
    run()
