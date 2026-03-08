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
# Your answer: M: ceil(100/32)=4, N: ceil(100/32)=4, K: ceil(128/32)=4

# Q2: Which tiles need predication?
# Your answer: Tiles at boundaries: m=3 (rows 96-99) and n=3 (cols 96-99)

# Q3: What is the computational waste without predication?
# Your answer: (4*32-100)*(4*32-100) = 28*28 = 784 extra elements computed


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N, K = 100, 100, 128
TILE_M, TILE_N, TILE_K = 32, 32, 32
NUM_TILES_M = (M + TILE_M - 1) // TILE_M
NUM_TILES_N = (N + TILE_N - 1) // TILE_N
NUM_TILES_K = (K + TILE_K - 1) // TILE_K


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
    """
    # --- Step 1: Create MMA atom and TiledMMA ---
    mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, (2, 4), (16, 16))
    
    tid = cute.thread_idx()
    thr_mma = tiled_mma.get_slice(tid)
    
    # --- Step 2: Create predicate tensor ---
    pred = cute.make_rmem_tensor((16, 16), cutlass.boolean)
    for i in range(16):
        for j in range(16):
            # Global row/col for this thread's fragment element
            global_m = tid // 4 * 16 + i  # Simplified mapping
            global_n = (tid % 4) * 16 + j
            pred[i, j] = (global_m < actual_M) and (global_n < actual_N)
    
    # --- Step 3: Simplified GEMM (single tile for demo) ---
    accum = cute.make_rmem_tensor((16, 16), cutlass.float32)
    for i in range(16):
        for j in range(16):
            accum[i, j] = 0.0
    
    # Load and compute (simplified)
    for k in range(min(K, TILE_K)):
        for i in range(16):
            for j in range(16):
                if pred[i, j]:
                    global_m = tid // 4 * 16 + i
                    global_n = (tid % 4) * 16 + j
                    if global_m < actual_M and global_n < actual_N:
                        accum[i, j] += A[global_m, k] * B[k, global_n]
    
    # Store with predication
    for i in range(16):
        for j in range(16):
            if pred[i, j]:
                global_m = tid // 4 * 16 + i
                global_n = (tid % 4) * 16 + j
                if global_m < actual_M and global_n < actual_N:
                    C[global_m, global_n] = accum[i, j]
    
    # --- Step 4: Store results ---
    if tid == 0:
        results[0] = C[0, 0]
        results[1] = C[actual_M - 1, actual_N - 1]
        results[2] = C.mean()
        results[3] = C.max()
        results[4] = float(actual_M * actual_N)  # Valid elements
        results[5] = float(M * N)  # Allocated elements
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify irregular tile GEMM.
    """
    
    torch.manual_seed(42)
    A_torch = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B_torch = torch.randn((K, N), dtype=torch.float16, device='cuda')
    C_torch = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
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
    print(f"    Valid elems: {results_cpu[4]:.0f} / {results_cpu[5]:.0f}")
    
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
