"""
Project 01 — Tiled GEMM
Ampere (SM80) Implementation with Double-Buffering

ALGORITHM HEADER:
  GEMM: C = A @ B where A[M,K], B[K,N], C[M,N]
  
  Pseudocode:
    for m_tile in range(ceil(M / BLOCK_M)):
        for n_tile in range(ceil(N / BLOCK_N)):
            accum = zeros(BLOCK_M, BLOCK_N)
            for k_tile in range(ceil(K / BLOCK_K)):
                # Double-buffer: load next while computing current
                load(A_tile, B_tile)
                accum += A_tile @ B_tile
            store(C_tile, accum)

  Tiling Diagram:
    ┌─────────────────────────────────┐
    │            C (M x N)            │
    │  ┌───────┬───────┬───────┐      │
    │  │Tile(0,│Tile(0,│ ...   │      │
    │  │  0)   │  1)   │       │      │
    │  ├───────┼───────┼───────┤      │
    │  │Tile(1,│Tile(1,│ ...   │      │
    │  │  0)   │  1)   │       │      │
    │  └───────┴───────┴───────┘      │
    └─────────────────────────────────┘

JOB RELEVANCE:
  - NVIDIA DL Software Engineer: GEMM is core interview topic
  - Cerebras Performance Engineer: Wafer-scale matrix multiply
  - vLLM/TensorRT-LLM: Foundation for all linear layers

TARGET: >75% of theoretical TFLOPS (A100: 234+ TFLOPS)
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Matrix dimensions (adjust for your GPU)
M, N, K = 4096, 4096, 4096

# Block sizes (tuning parameters)
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

# Thread configuration (128 threads = 4 warps per block)
NUM_THREADS = 128


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_gemm_ampere(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    alpha: float,
    beta: float,
):
    """
    Tiled GEMM for Ampere (SM80) with double-buffering.
    
    FILL IN [HARD]: Implement the complete tiled GEMM mainloop.
    
    HINT: 
      1. Compute tile coordinates from block/thread indices
      2. Create TiledMMA for FP16×FP16→FP32
      3. Implement double-buffered mainloop
      4. Store results with alpha/beta scaling
    """
    # --- Step 1: Compute tile indices ---
    # TODO: tile_m = cute.block_idx() % (M // BLOCK_M)
    #       tile_n = cute.block_idx() // (M // BLOCK_M)
    #       tid = cute.thread_idx()
    
    # --- Step 2: Create MMA atom and TiledMMA ---
    # TODO: mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
    #       tiled_mma = cute.make_tiled_mma(mma_atom, (2, 4), (16, 16))
    
    # --- Step 3: Create double-buffer ---
    # TODO: smem_a = [cute.make_smem_tensor(...) for _ in range(2)]
    #       smem_b = [cute.make_smem_tensor(...) for _ in range(2)]
    
    # --- Step 4: Mainloop over K dimension ---
    # TODO: for k_tile in range(K // BLOCK_K):
    #           # Double-buffer: load next tile while computing current
    #           # MMA: accum += A_tile @ B_tile
    #           pass
    
    # --- Step 5: Store result ---
    # TODO: C_tile = alpha * accum + beta * C_tile
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the GEMM kernel and measure performance.
    """
    print("\n" + "=" * 60)
    print("  Project 01 — Tiled GEMM (Ampere SM80)")
    print("=" * 60)
    
    # Create input matrices
    print(f"\n  Matrix sizes: A[{M},{K}], B[{K},{N}], C[{M},{N}]")
    print(f"  Block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    
    torch.manual_seed(42)
    A_torch = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B_torch = torch.randn((K, N), dtype=torch.float16, device='cuda')
    C_torch = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    # Reference result
    C_ref = torch.matmul(A_torch.float(), B_torch.float()).cpu().numpy()
    
    A_cute = from_dlpack(A_torch)
    B_cute = from_dlpack(B_torch)
    C_cute = from_dlpack(C_torch)
    
    # Warmup
    print("\n  Warming up...")
    kernel_gemm_ampere[64, NUM_THREADS](A_cute, B_cute, C_cute, 1.0, 0.0)
    torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 10
    print(f"  Running {num_runs} iterations...")
    
    start = time.perf_counter()
    for _ in range(num_runs):
        kernel_gemm_ampere[64, NUM_THREADS](A_cute, B_cute, C_cute, 1.0, 0.0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate performance
    avg_time_ms = (elapsed / num_runs) * 1000
    flops = 2 * M * N * K  # 2 for multiply-add
    tflops = flops / (avg_time_ms * 1e-3) / 1e12
    
    # Theoretical peak for A100
    peak_tflops = 312.0  # FP16 tensor core
    efficiency = (tflops / peak_tflops) * 100
    
    print(f"\n  Results:")
    print(f"    Average time: {avg_time_ms:.3f} ms")
    print(f"    Achieved: {tflops:.1f} TFLOPS")
    print(f"    Peak (A100): {peak_tflops:.0f} TFLOPS")
    print(f"    Efficiency: {efficiency:.1f}%")
    
    # Verify correctness (sample check)
    C_cpu = C_torch.cpu().numpy()
    max_diff = abs(C_cpu - C_ref).max()
    print(f"\n  Correctness:")
    print(f"    Max difference from reference: {max_diff:.6f}")
    
    passed = max_diff < 1.0  # Allow some numerical difference
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


if __name__ == "__main__":
    run()
