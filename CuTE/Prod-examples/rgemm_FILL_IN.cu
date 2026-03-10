/***************************************************************************************************
 * FILL-IN EXERCISE: rgemm.cu — Register-Staged GEMM (gmem → rmem → smem pipeline)
 *
 * PREREQUISITE: Complete sgemm_FILL_IN.cu first.
 *
 * NEW CONCEPT vs sgemm.cu:
 *   sgemm copies gmem → smem DIRECTLY (blocking).
 *   This file adds a REGISTER STAGING BUFFER (tArA / tBrB):
 *
 *     gmem → rmem  (copy_a into tArA — can overlap with smem compute)
 *     rmem → smem  (copy tArA → tAsA — fast, same CTA)
 *     smem → compute (gemm on tCsA)
 *
 *   This staging pattern is the conceptual predecessor to cp.async pipelining.
 *   Key difference: TiledMMA now uses partition_A/B directly on smem (no ldmatrix),
 *   and also exposes make_fragment_A/B for register-level pipelining of the k-block loop.
 *
 * THIS FILE COVERS TWO KERNELS:
 *   Kernel 1 (simple):   gmem→rmem prefetch, then rmem→smem, then gemm on smem
 *   Kernel 2 (pipelined): full 3-level pipeline: gmem→rmem overlaps smem→reg→compute
 *
 * BUILD:
 *   nvcc -std=c++17 -O3 -arch=sm_70 rgemm_FILL_IN.cu \
 *        -I /path/to/cutlass/include -o rgemm && ./rgemm
 *
 * PREDICT BEFORE RUNNING:
 *   - In Kernel 2, at k_block == K_BLOCK_MAX-1, why do we sync BEFORE copy(tArA, tAsA)?
 *   - What does make_fragment_A(tCsA) return? A smem pointer or a register buffer?
 *   - Why is K_BLOCK_MAX derived from tCrA and not from tCsA?
 ***************************************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


// ═══════════════════════════════════════════════════════════════════
// KERNEL 1: Simple register-staged GEMM
// Pipeline: prefetch k=0 into regs, then for each k-tile:
//   sync → rmem→smem → sync → prefetch k+1 into rmem → gemm on smem
// ═══════════════════════════════════════════════════════════════════
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device_simple(ProblemShape shape_MNK, CtaTiler cta_tiler,
                   TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                   TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                   TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
                   Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions (given — read as documentation)
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

  // ─────────────────────────────────────────────────────────────────
  // SECTION 1 — GLOBAL + CTA TILES (identical to sgemm, already known)
  // ─────────────────────────────────────────────────────────────────

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  // ─────────────────────────────────────────────────────────────────
  // SECTION 2 — TILEDCOPY PARTITIONING (new vs sgemm: now 3 tensors)
  //
  // NEW: tArA / tBrB — register staging buffers!
  // make_fragment_like(tAsA) allocates a register tensor with the
  // SAME shape and layout as the smem destination.  No smem pointer.
  //
  // Data flow:
  //   gmem  →  tArA  (register)  →  tAsA  (smem)
  //   tAgA         tArA                tAsA
  // ─────────────────────────────────────────────────────────────────

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                // (CPY, CPY_M, CPY_K, k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                // (CPY, CPY_M, CPY_K)

  // HINT: make_fragment_like(tAsA) — allocates registers matching smem shape
  //       This is the staging buffer: gmem reads land here first.
  Tensor tArA = /* <<< FILL IN >>> */;                     // (CPY, CPY_M, CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);

  // HINT: same pattern for B
  Tensor tBrB = /* <<< FILL IN >>> */;

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));

  // SECTION 3 — PREFETCH k_tile=0 into registers
  // HINT: copy(copy_a, tAgA(_,_,_,0), tArA)
  //       Slice the k=0 tile from global and copy into register buffer.
  /* <<< FILL IN: prefetch A tile 0 into tArA >>> */
  /* <<< FILL IN: prefetch B tile 0 into tBrB >>> */

  // ─────────────────────────────────────────────────────────────────
  // SECTION 4 — TILEDMMA PARTITIONING
  //
  // NEW vs sgemm's local_partition: now use thr_mma.partition_A/B(smem)
  // These create smem-backed views in the MMA's thread distribution.
  // No register fragments yet — gemm() reads directly from smem here.
  // ─────────────────────────────────────────────────────────────────

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);

  // HINT: thr_mma.partition_A(sA)  → (MMA, MMA_M, MMA_K) — smem view
  Tensor tCsA = /* <<< FILL IN >>> */;

  // HINT: thr_mma.partition_B(sB)  → (MMA, MMA_N, MMA_K) — smem view
  Tensor tCsB = /* <<< FILL IN >>> */;

  // HINT: thr_mma.partition_C(gC)  → (MMA, MMA_M, MMA_N) — gmem output view
  Tensor tCgC = /* <<< FILL IN >>> */;

  // HINT: thr_mma.make_fragment_C(tCgC) — register accumulator
  Tensor tCrC = /* <<< FILL IN >>> */;

  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));

  /* <<< FILL IN: clear(tCrC) >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 5 — SIMPLE K LOOP (rmem-staged, no register k-blocking)
  //
  // For each k-tile:
  //   1. sync (wait for all threads done consuming previous smem)
  //   2. copy rmem → smem  (tArA → tAsA)
  //   3. sync (wait for smem writes visible to all)
  //   4. prefetch NEXT k-tile into rmem  (non-blocking)
  //   5. compute gemm on smem (blocks until done, but rmem load overlaps)
  // ─────────────────────────────────────────────────────────────────

  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // HINT: __syncthreads() — wait for previous iteration's smem consumers
    /* <<< FILL IN >>> */;

    // HINT: copy(tArA, tAsA)  — move registers → smem (no copy_a atom needed)
    /* <<< FILL IN: rmem A → smem >>> */;
    /* <<< FILL IN: rmem B → smem >>> */;

    // HINT: __syncthreads() — wait for smem writes to be visible
    /* <<< FILL IN >>> */;

    // HINT: prefetch next tile into registers
    //       k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    /* <<< FILL IN: copy(copy_a, tAgA(_,_,_,k_tile_next), tArA) >>> */;
    /* <<< FILL IN: same for B >>> */;

    // HINT: gemm(mma, tCsA, tCsB, tCrC)  — compute on smem directly
    /* <<< FILL IN >>> */;
  }

  // HINT: axpby(alpha, tCrC, beta, tCgC)
  /* <<< FILL IN: epilogue >>> */;
}


// ═══════════════════════════════════════════════════════════════════
// KERNEL 2: Pipelined register GEMM (k-block level register pipeline)
// Adds a second pipeline level: smem → registers (k-block loop)
// ═══════════════════════════════════════════════════════════════════
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // (Preconditions omitted — same as above)
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

  // Global + CTA tiles (identical)
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  // TiledCopy + register staging (same as Kernel 1)
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  Tensor tArA = make_fragment_like(tAsA);

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  Tensor tBrB = make_fragment_like(tBsB);

  // Prefetch k=0 into registers (same as Kernel 1)
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  // ─────────────────────────────────────────────────────────────────
  // SECTION 6 — REGISTER FRAGMENTS FOR K-BLOCK PIPELINING (NEW!)
  //
  // make_fragment_A(tCsA) / make_fragment_B(tCsB):
  //   Allocates register tensors that MATCH the MMA atom's A/B operand layout.
  //   Shape mirrors tCsA / tCsB respectively.
  //   This is a REGISTER buffer for one k-block's worth of A/B data.
  //
  // This enables: while computing k_block N, load k_block N+1 from smem.
  // ─────────────────────────────────────────────────────────────────

  // HINT: thr_mma.make_fragment_A(tCsA)  → register buffer, shape = shape(tCsA)
  Tensor tCrA = /* <<< FILL IN >>> */;   // (MMA, MMA_M, MMA_K)

  // HINT: thr_mma.make_fragment_B(tCsB)
  Tensor tCrB = /* <<< FILL IN >>> */;   // (MMA, MMA_N, MMA_K)

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);

  CUTE_STATIC_ASSERT_V(  shape(tCrA) ==   shape(tCsA));
  CUTE_STATIC_ASSERT_V(  shape(tCrB) ==   shape(tCsB));
  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));

  clear(tCrC);

  // ─────────────────────────────────────────────────────────────────
  // SECTION 7 — BOOTSTRAP: rmem→smem + initial smem→regs for k_block=0
  //
  // Before the main loop: commit register staging buffer → smem,
  // then prefetch k_block=0 from smem into register fragments.
  // ─────────────────────────────────────────────────────────────────

  // HINT: copy(tArA, tAsA) — flush register staging buffer to smem
  /* <<< FILL IN: tArA → tAsA >>> */;
  /* <<< FILL IN: tBrB → tBsB >>> */;
  __syncthreads();

  // HINT: copy(tCsA(_,_,0), tCrA(_,_,0))  — smem k_block=0 → register fragment
  /* <<< FILL IN: load A k_block=0 into register fragment >>> */;
  /* <<< FILL IN: load B k_block=0 into register fragment >>> */;

  auto K_TILE_MAX  = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);   // NOTE: from tCrA (register), not tCsA

  // ─────────────────────────────────────────────────────────────────
  // SECTION 8 — PIPELINED MAIN LOOP (k_tile × k_block)
  //
  // Three levels of pipelining happening simultaneously:
  //   Level 1 (k_tile):   gmem → register staging buffer (tArA)
  //   Level 2 (k_block):  smem → register fragments (tCrA)
  //   Level 3 (compute):  register fragments → accumulator (tCrC)
  //
  // At k_block == K_BLOCK_MAX-1 (last block): flush staging → smem
  // At k_block == 0:              prefetch next gmem tile into staging
  // Always:                       prefetch next k_block smem → regs
  //                               compute current k_block
  // ─────────────────────────────────────────────────────────────────

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      // ── At LAST k_block: flush new gmem data (in regs) to smem ──
      if (k_block == K_BLOCK_MAX - 1)
      {
        // HINT: sync, then copy tArA→tAsA, tBrB→tBsB, then sync again
        /* <<< FILL IN: __syncthreads() >>> */;
        /* <<< FILL IN: copy(tArA, tAsA) >>> */;
        /* <<< FILL IN: copy(tBrB, tBsB) >>> */;
        /* <<< FILL IN: __syncthreads() >>> */;
      }

      // ── Prefetch NEXT k_block from smem → register fragments ──
      // HINT: k_block_next = (k_block + 1) % K_BLOCK_MAX
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;

      // HINT: copy(tCsA(_,_,k_block_next), tCrA(_,_,k_block_next))
      /* <<< FILL IN: smem → register fragment for A k_block_next >>> */;
      /* <<< FILL IN: smem → register fragment for B k_block_next >>> */;

      // ── At FIRST k_block: prefetch next gmem tile into registers ──
      if (k_block == 0)
      {
        // HINT: k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;

        // HINT: copy(copy_a, tAgA(_,_,_,k_tile_next), tArA)
        /* <<< FILL IN: gmem → register staging buffer >>> */;
        /* <<< FILL IN: same for B >>> */;
      }

      // ── Compute mma on current k_block ──
      // HINT: gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC)
      /* <<< FILL IN >>> */;

    } // k_block
  } // k_tile

  // HINT: axpby(alpha, tCrC, beta, tCgC)
  /* <<< FILL IN: epilogue >>> */;
}


// ═══════════════════════════════════════════════════════════════════
// HOST SIDE (given — focus is on the kernel)
// ═══════════════════════════════════════════════════════════════════
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_nt(int m, int n, int k, Alpha alpha,
        TA const* A, int ldA, TB const* B, int ldB,
        Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto prob_shape = make_shape(int(m), int(n), int(k));
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<8>{};
  auto cta_tiler = make_shape(bM,bN,bK);
  auto sA = make_layout(make_shape(bM,bK));
  auto sB = make_layout(make_shape(bN,bK));
  auto sC = make_layout(make_shape(bM,bN));
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>,TA>{},
                                    Layout<Shape<_32,_8>>{}, Layout<Shape<_4,_1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>,TB>{},
                                    Layout<Shape<_32,_8>>{}, Layout<Shape<_4,_1>>{});
  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{}, Layout<Shape<_16,_16,_1>>{});
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(int(m),bM)), size(ceil_div(int(n),bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler, A, dA, sA, copyA, B, dB, sB, copyB, C, dC, sC, mmaC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_tn(int m, int n, int k, Alpha alpha,
        TA const* A, int ldA, TB const* B, int ldB,
        Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto prob_shape = make_shape(int(m), int(n), int(k));
  auto dA=make_stride(ldA,Int<1>{}); auto dB=make_stride(ldB,Int<1>{});
  auto dC=make_stride(Int<1>{},ldC);
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<8>{};
  auto cta_tiler=make_shape(bM,bN,bK);
  auto sA=make_layout(make_shape(bM,bK), make_stride(Int<1>{},bM+Int<1>{}));
  auto sB=make_layout(make_shape(bN,bK), make_stride(Int<1>{},bN+Int<1>{}));
  auto sC=make_layout(make_shape(bM,bN));
  TiledCopy copyA=make_tiled_copy(Copy_Atom<UniversalCopy<TA>,TA>{},
                                  Layout<Shape<_32,_8>,Stride<_8,_1>>{},Layout<Shape<_1,_1>>{});
  TiledCopy copyB=make_tiled_copy(Copy_Atom<UniversalCopy<TB>,TB>{},
                                  Layout<Shape<_32,_8>,Stride<_8,_1>>{},Layout<Shape<_1,_1>>{});
  TiledMMA mmaC=make_tiled_mma(UniversalFMA<TC,TA,TB>{},Layout<Shape<_16,_16,_1>>{});
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(int(m),bM)),size(ceil_div(int(n),bN)));
  gemm_device<<<dimGrid,dimBlock,0,stream>>>
      (prob_shape,cta_tiler,A,dA,sA,copyA,B,dB,sB,copyB,C,dC,sC,mmaC,alpha,beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char tA, char tB, int m, int n, int k, Alpha alpha,
          TA const* A, int ldA, TB const* B, int ldB, Beta beta,
          TC* C, int ldC, cudaStream_t stream = 0)
{
  if (tA=='N' && tB=='T') return gemm_nt(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC,stream);
  if (tA=='T' && tB=='N') return gemm_tn(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC,stream);
  assert(false && "Not implemented");
}

int main(int argc, char** argv)
{
  int m=5120; if(argc>=2) sscanf(argv[1],"%d",&m);
  int n=5120; if(argc>=3) sscanf(argv[2],"%d",&n);
  int k=4096; if(argc>=4) sscanf(argv[3],"%d",&k);
  char tA='N'; if(argc>=5) sscanf(argv[4],"%c",&tA);
  char tB='T'; if(argc>=6) sscanf(argv[5],"%c",&tB);
  using T=float; T alpha=1.f, beta=0.f;
  printf("M=%d N=%d K=%d  C=A^%c B^%c\n",m,n,k,tA,tB);
  cute::device_init(0);
  thrust::host_vector<T> hA(m*k),hB(n*k),hC(m*n);
  for(auto& v:hA) v=T(2*(rand()/double(RAND_MAX))-1);
  for(auto& v:hB) v=T(2*(rand()/double(RAND_MAX))-1);
  for(auto& v:hC) v=T(-1);
  thrust::device_vector<T> dA=hA,dB=hB,dC=hC;
  double gf=2.0*m*n*k*1e-9;
  const int iters=100;
  cute::GPU_Clock timer;
  int ldA=(tA=='N')?m:k, ldB=(tB=='N')?k:n, ldC=m;
  dC=hC;
  gemm(tA,tB,m,n,k,alpha,dA.data().get(),ldA,dB.data().get(),ldB,beta,dC.data().get(),ldC);
  CUTE_CHECK_LAST();
  timer.start();
  for(int i=0;i<iters;++i)
    gemm(tA,tB,m,n,k,alpha,dA.data().get(),ldA,dB.data().get(),ldB,beta,dC.data().get(),ldC);
  double t=timer.seconds()/iters; CUTE_CHECK_LAST();
  printf("CUTE_RGEMM:  [%6.1f] GFlop/s  (%6.4f ms)\n",gf/t,t*1000);

  // ─────────────────────────────────────────────────────────────────
  // CHECKPOINT QUESTIONS:
  //
  // Q1: In Kernel 1, there are TWO __syncthreads() per k-tile iteration.
  //     What race condition does each one prevent?
  //     First sync:  ________________________________________________
  //     Second sync: ________________________________________________
  //
  // Q2: In Kernel 2, K_BLOCK_MAX = size<2>(tCrA).
  //     What is this dimension? How does it relate to bK?
  //     Answer: ______________________________________________________
  //
  // Q3: Why does Kernel 2 prefetch the NEXT gmem tile at k_block==0
  //     rather than k_block==K_BLOCK_MAX-1?
  //     (Hint: consider where the register staging buffer tArA is read)
  //     Answer: ______________________________________________________
  //
  // Q4: What is the difference between make_fragment_like(tAsA)
  //     and make_fragment_A(tCsA)?
  //     When would you use each?
  //     Answer: ______________________________________________________
  // ─────────────────────────────────────────────────────────────────
  return 0;
}
