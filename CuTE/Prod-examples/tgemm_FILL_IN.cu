/***************************************************************************************************
 * FILL-IN EXERCISE: tgemm.cu — Production TiledMMA GEMM with cp.async + ldmatrix + Swizzle
 *
 * PREREQUISITE: Complete sgemm_FILL_IN.cu first.
 *
 * OBJECTIVE: Master the three-object production pattern:
 *   (1) TiledCopy  (gmem→smem via cp.async, 128-bit vectorized)
 *   (2) TiledMMA   (smem→regs→Tensor Cores via SM80_16x8x16)
 *   (3) S2R retiling (ldmatrix: smem→register fragments)
 *   Plus: Swizzle for conflict-free smem, 3-stage pipeline
 *
 * RULES:
 *   - Fill in every  // <<< FILL IN >>>  block
 *   - Think about the shape at every step — write it in the comment
 *   - Run with #if 1 on the debug print blocks first, check shapes,
 *     then switch to the full pipeline
 *
 * BUILD (SM80+ required — Ampere or newer):
 *   nvcc -std=c++17 -O3 -arch=sm_80 tgemm_FILL_IN.cu \
 *        -I /path/to/cutlass/include \
 *        -o tgemm && ./tgemm
 *
 * PREDICT BEFORE RUNNING:
 *   - How many half_t values does each thread load per cp.async call?
 *   - What is the shape of tCrC for the SM80_16x8x16 atom with 2×2 tiling?
 *   - Why does bK jump from 8 (sgemm) to 64 here?
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
// SECTION 1 — SHARED STORAGE STRUCT
//
// Production kernels use dynamic shared memory so the CUDA driver
// can set the smem size at launch.  We wrap both A and B smem into
// a single struct so we can compute sizeof() for the launch config.
//
// ArrayEngine<T, N> is a CuTe utility: a fixed-size array of N Ts
// cosize_v<Layout> is the number of elements the layout spans.
// ═══════════════════════════════════════════════════════════════════

template <class ElementA, class ElementB,
          class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
  // HINT: cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>>
  /* <<< FILL IN: declare member A >>> */
  /* <<< FILL IN: declare member B >>> */
};


// ═══════════════════════════════════════════════════════════════════
// SECTION 2 — KERNEL SIGNATURE
//
// Compared to sgemm, we now pass:
//   TiledCopyA / TiledCopyB  — the vectorized gmem→smem copy objects
//   S2RAtomA  / S2RAtomB    — the ldmatrix atoms for smem→regs
//   TiledMma                — the MMA object (Tensor Core dispatch)
// ═══════════════════════════════════════════════════════════════════

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
// HINT: launch_bounds uses size(TiledMma{}) — the total thread count
//       that TiledMma requires.
__launch_bounds__(/* <<< FILL IN >>> */decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Static assertions — already filled in for reference
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
  // SECTION 3 — GLOBAL + CTA-LEVEL TENSORS
  // Identical pattern to sgemm — already familiar.
  // ─────────────────────────────────────────────────────────────────

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // ─────────────────────────────────────────────────────────────────
  // SECTION 4 — DYNAMIC SHARED MEMORY
  //
  // We use `extern __shared__` + reinterpret_cast instead of static
  // arrays so CUDA can configure smem size at kernel launch.
  // ─────────────────────────────────────────────────────────────────

  extern __shared__ char shared_memory[];
  using SmemStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);

  // HINT: make_tensor(make_smem_ptr(smem.A.begin()), sA_layout)
  //       sA has shape (BLK_M, BLK_K, PIPE=3)
  Tensor sA = /* <<< FILL IN >>> */;
  Tensor sB = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 5 — TILEDCOPY PARTITIONING  (gmem→smem, new API)
  //
  // OLD (sgemm):  local_partition(gA, tA, threadIdx.x)
  // NEW (tgemm):  copy_a.get_slice(threadIdx.x).partition_S(gA)
  //
  // get_slice() extracts this thread's portion of the TiledCopy.
  // partition_S() = partition Source  (gmem side)
  // partition_D() = partition Destination (smem side)
  //
  // The leading CPY dimension = atom vector width (8 half_t = 128 bits)
  // ─────────────────────────────────────────────────────────────────

  // HINT: ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  /* <<< FILL IN: declare thr_copy_a >>> */

  // HINT: thr_copy_a.partition_S(gA)  → shape (CPY, CPY_M, CPY_K, k_tiles)
  Tensor tAgA = /* <<< FILL IN >>> */;

  // HINT: thr_copy_a.partition_D(sA)  → shape (CPY, CPY_M, CPY_K, PIPE)
  Tensor tAsA = /* <<< FILL IN >>> */;

  // HINT: same for B
  /* <<< FILL IN: thr_copy_b, tBgB, tBsB >>> */

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));

  // ─────────────────────────────────────────────────────────────────
  // SECTION 6 — PREFETCH SETUP
  //
  // With a 3-stage pipeline (bP=3) we pre-fill 2 smem buffers before
  // the main loop so computation is never stalled waiting for data.
  //
  // K_PIPE_MAX  = size<3>(tAsA) = bP = 3
  // k_tile_count = total number of K tiles to process
  // k_tile_next  = index of next gmem tile to load
  // ─────────────────────────────────────────────────────────────────

  auto K_PIPE_MAX  = size<3>(tAsA);      // 3 pipeline stages
  int  k_tile_count = size<3>(tAgA);     // total K tiles
  int  k_tile_next  = 0;                 // next gmem tile to fetch

  // HINT: Loop k_pipe from 0 to K_PIPE_MAX-2 (fill all but last pipe)
  //       Inside: copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe))
  //               copy(copy_b, ...)
  //               cp_async_fence()
  //               decrement k_tile_count; if still >0 increment k_tile_next
  /* <<< FILL IN: prefetch loop >>> */

  // ─────────────────────────────────────────────────────────────────
  // SECTION 7 — TILEDMMA PARTITIONING  (hardware register fragments)
  //
  // mma.get_slice() analogous to copy_a.get_slice()
  // partition_C()         → this thread's view of the C output tile
  // partition_fragment_A  → allocates A register fragment (no smem pointer yet)
  // partition_fragment_B  → allocates B register fragment
  // make_fragment_C       → allocates C accumulator registers
  //
  // IMPORTANT: partition_fragment_A/B take a 2D smem slice (sA(_,_,0))
  //            to infer the shape, but do NOT create a pointer to smem.
  //            They return register tensors with hardware-dictated layout.
  // ─────────────────────────────────────────────────────────────────

  // HINT: ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  /* <<< FILL IN: declare thr_mma >>> */

  // HINT: thr_mma.partition_C(gC)  → (MMA, MMA_M, MMA_N)
  Tensor tCgC = /* <<< FILL IN >>> */;

  // HINT: thr_mma.partition_fragment_A(sA(_,_,0))  → (MMA, MMA_M, MMA_K)
  Tensor tCrA = /* <<< FILL IN >>> */;

  // HINT: thr_mma.partition_fragment_B(sB(_,_,0))  → (MMA, MMA_N, MMA_K)
  Tensor tCrB = /* <<< FILL IN >>> */;

  // HINT: thr_mma.make_fragment_C(tCgC)  → (MMA, MMA_M, MMA_N)
  Tensor tCrC = /* <<< FILL IN >>> */;

  CUTE_STATIC_ASSERT_V((shape(tCrC) == take<0,3>(shape(tCgC))));
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));

  // HINT: clear(tCrC)
  /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 8 — S2R (SMEM→REGISTER) RETILING
  //
  // This is the trickiest section.  Read carefully.
  //
  // Problem: TiledMMA dictates how registers are laid out for mma.sync.
  //          ldmatrix dictates how registers are loaded from smem.
  //          These two layouts MUST agree — but they're defined independently.
  //
  // Solution: make_tiled_copy_A(s2r_atom, mma) creates a TiledCopy whose
  //           thread-value layout is DERIVED FROM the MMA's thread layout.
  //           This guarantees compatibility.
  //
  //           retile_D(tCrA) reinterprets tCrA (MMA view) as the destination
  //           for the copy (ldmatrix view).  ZERO BYTES MOVED — it's a view.
  //
  // The naming convention:
  //   tX = "copy-X" tensors (X because they're in copy-space, not mma-space)
  //   tXsA = source in smem,  tXrA = destination in registers
  // ─────────────────────────────────────────────────────────────────

  // HINT: make_tiled_copy_A(s2r_atom_a, mma)
  TiledCopy s2r_copy_a = /* <<< FILL IN >>> */;

  // HINT: s2r_copy_a.get_slice(threadIdx.x)
  ThrCopy s2r_thr_copy_a = /* <<< FILL IN >>> */;

  // HINT: s2r_thr_copy_a.partition_S(sA)  → (CPY, MMA_M, MMA_K, PIPE)
  Tensor tXsA = /* <<< FILL IN >>> */;

  // HINT: s2r_thr_copy_a.retile_D(tCrA)  → (CPY, MMA_M, MMA_K)
  //       This is NOT a copy — it's a zero-copy reinterpretation of registers.
  Tensor tXrA = /* <<< FILL IN >>> */;

  // HINT: same for B (use s2r_atom_b, make_tiled_copy_B)
  /* <<< FILL IN: s2r_copy_b, s2r_thr_copy_b, tXsB, tXrB >>> */

  // ─────────────────────────────────────────────────────────────────
  // SECTION 9 — PIPELINE STATE VARIABLES
  //
  // We maintain two indices cycling through 0..K_PIPE_MAX-1:
  //   smem_pipe_read  — which smem slot to consume this iteration
  //   smem_pipe_write — which smem slot to write new gmem data into
  //
  // K_BLOCK_MAX — number of MMA k-blocks per smem tile
  //               = size<2>(tCrA)  (the MMA_K dimension)
  //   Each k_block issues one ldmatrix + one mma.sync.
  // ─────────────────────────────────────────────────────────────────

  int smem_pipe_read  = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slices — views into tXsA/tXsB at smem_pipe_read
  // HINT: tXsA(_,_,_,smem_pipe_read)  — slice the PIPE dimension
  Tensor tXsA_p = /* <<< FILL IN >>> */;
  Tensor tXsB_p = /* <<< FILL IN >>> */;

  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // ─────────────────────────────────────────────────────────────────
  // SECTION 10 — REGISTER PREFETCH (first k_block of first smem tile)
  //
  // Before the main loop starts we preload k_block=0 into registers
  // so the first mma.sync in the loop has data ready immediately.
  // ─────────────────────────────────────────────────────────────────

  if (K_BLOCK_MAX > 1) {
    // HINT: Wait for K_PIPE_MAX-2 outstanding cp.async groups to complete
    //       (i.e., at least the first prefetched tile is in smem)
    /* <<< FILL IN: cp_async_wait >>> */;
    __syncthreads();

    // HINT: copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}))
    //       This issues ldmatrix for k_block=0
    /* <<< FILL IN: prefetch A k_block=0 to registers >>> */;
    /* <<< FILL IN: prefetch B k_block=0 to registers >>> */;
  }

  // ─────────────────────────────────────────────────────────────────
  // SECTION 11 — PIPELINED MAIN LOOP
  //
  // The outer while loop runs until all K tiles (plus drain) are done.
  // The inner for loop iterates over MMA k-blocks within one smem tile.
  //
  // Inside the inner loop, three things happen CONCURRENTLY:
  //   A. gmem→smem copy (cp.async) fired once per k_tile (at k_block==0)
  //   B. smem→regs copy (ldmatrix) for k_block+1 (always, prefetch)
  //   C. mma.sync on k_block (always, compute)
  //
  // The interleaving:
  //   iteration k_block=0: fire gmem→smem, load regs[1], compute regs[0]
  //   iteration k_block=1: load regs[2], compute regs[1]
  //   ...
  //   iteration k_block=K_BLOCK_MAX-1: wait smem, load regs[0], compute regs[last]
  // ─────────────────────────────────────────────────────────────────

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX - 1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      // ── At last k_block: advance smem read pointer, wait for next tile ──
      if (k_block == K_BLOCK_MAX - 1)
      {
        // HINT: Re-slice tXsA_p and tXsB_p to smem_pipe_read
        /* <<< FILL IN: tXsA_p = tXsA(_,_,_,smem_pipe_read) >>> */
        /* <<< FILL IN: tXsB_p = tXsB(_,_,_,smem_pipe_read) >>> */

        // HINT: Wait for K_PIPE_MAX-2 outstanding groups (drain one)
        /* <<< FILL IN: cp_async_wait >>> */;
        __syncthreads();
      }

      // ── Prefetch next k_block's data from smem into registers ──
      // HINT: k_block_next = (k_block + 1) % K_BLOCK_MAX  (static modulo)
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;

      // HINT: copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next))
      /* <<< FILL IN: ldmatrix A for k_block_next >>> */;
      /* <<< FILL IN: ldmatrix B for k_block_next >>> */;

      // ── At first k_block: fire gmem→smem async copy for next tile ──
      if (k_block == 0)
      {
        // HINT: copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write))
        /* <<< FILL IN: issue cp.async for A >>> */;
        /* <<< FILL IN: issue cp.async for B >>> */;
        // HINT: cp_async_fence() to label this group
        /* <<< FILL IN >>> */;

        // Advance tile counters and pipeline ring
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // HINT: smem_pipe_write = smem_pipe_read
        //       smem_pipe_read  = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1
        /* <<< FILL IN: advance pipe indices >>> */
      }

      // ── Compute: mma.sync on current k_block ──
      // HINT: gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC)
      //       This dispatches to mma.sync.aligned PTX — NOT a loop.
      /* <<< FILL IN: mma.sync dispatch >>> */;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // SECTION 12 — EPILOGUE
  // Same as sgemm: scale and write accumulator back to global memory.
  // ─────────────────────────────────────────────────────────────────

  // HINT: axpby(alpha, tCrC, beta, tCgC)
  /* <<< FILL IN >>> */;
}


// ═══════════════════════════════════════════════════════════════════
// HOST SIDE — gemm_tn (half_t, Tensor Cores)
// This is the PRODUCTION path: SM80_16x8x16 + ldmatrix + swizzle + cp.async
// ═══════════════════════════════════════════════════════════════════
template <class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t* C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // TN strides: A and B are row-major (transposed)
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<3>{};  // 3-stage pipeline

  // ─────────────────────────────────────────────────────────────────
  // SECTION 13 — SWIZZLED SMEM LAYOUT
  //
  // Swizzle<B,M,S> with <3,3,3>:
  //   B=3 → 8-element XOR mask
  //   M=3 → 8-element middle offset
  //   S=3 → 8-element shift
  //
  // This remaps smem addresses so that ldmatrix instructions from a
  // warp land on different smem banks, avoiding 8-way bank conflicts.
  //
  // The inner Layout defines the base tile geometry (8×64 in k-major
  // format) that the swizzle operates on.
  //
  // tile_to_shape(atom, target_shape) repeats the atom to fill
  // target_shape (bM×bK×bP = 128×64×3).
  // ─────────────────────────────────────────────────────────────────

  // HINT: composition(Swizzle<3,3,3>{},
  //                   Layout<Shape <_8,Shape <_8, _8>>,
  //                          Stride<_8,Stride<_1,_64>>>{})
  auto swizzle_atom = /* <<< FILL IN >>> */;

  // HINT: tile_to_shape(swizzle_atom, make_shape(bM,bK,bP))
  auto sA = /* <<< FILL IN >>> */;
  auto sB = /* <<< FILL IN >>> */;
  auto sC = make_layout(make_shape(bM, bN));

  // ─────────────────────────────────────────────────────────────────
  // SECTION 14 — TILEDCOPY FOR GMEM→SMEM (128-bit cp.async)
  //
  // Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>
  //   → issues cp.async.ca.shared.global.128  (16 bytes per call)
  //   → bypasses L1, writes directly to smem
  //
  // Thread layout (16,8) k-major: 16 threads in M/N, 8 in K, K walks fastest
  // Value layout  (1,8):  each thread covers 1 row × 8 half_t = 128 bits
  // ─────────────────────────────────────────────────────────────────

  // HINT: make_tiled_copy(
  //           Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
  //           Layout<Shape<_16,_8>,Stride<_8,_1>>{},   // k-major threads
  //           Layout<Shape< _1,_8>>{})                 // 8 values per thread
  TiledCopy copyA = /* <<< FILL IN >>> */;
  TiledCopy copyB = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 15 — TILEDMMA  (SM80 Tensor Core atom)
  //
  // make_tiled_mma(atom, warp_layout, tiler)
  //
  // SM80_16x8x16_F16F16F16F16_TN:
  //   One warp (32 threads) computes 16×8×16 in FP16
  //   Maps to PTX: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  //
  // Layout<Shape<_2,_2>>: tile 2×2 MMA atoms across warps
  //   → effective warp tile = 32×16×16
  //
  // Tile<_32,_32,_16>: total per-CTA tile from this TiledMMA
  // ─────────────────────────────────────────────────────────────────

  // HINT: make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
  //                      Layout<Shape<_2,_2>>{},
  //                      Tile<_32,_32,_16>{})
  TiledMMA mmaC = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 16 — S2R ATOM  (ldmatrix: smem→register fragments)
  //
  // SM75_U32x4_LDSM_N:
  //   Warp cooperative: each warp loads 4× 8×8 matrices of uint32
  //   (= 4× 8×8 pairs of FP16 = 256 FP16 values per warp)
  //   Maps to PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16
  //
  // This produces registers in EXACTLY the layout mma.sync expects.
  // Comment/uncomment alternatives to observe shape changes with
  // the debug print block in the kernel.
  // ─────────────────────────────────────────────────────────────────

  //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_A;   // x1: loads 1 matrix fragment
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_A;   // x2: loads 2 matrix fragments
  // HINT: Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
  /* <<< FILL IN: declare s2r_atom_A >>> */
  /* <<< FILL IN: declare s2r_atom_B >>> */

  // ─────────────────────────────────────────────────────────────────
  // SECTION 17 — LAUNCH CONFIG
  //
  // smem_size = sizeof(SharedStorage<half_t, half_t, sA_type, sB_type>)
  // dimBlock  = size(mmaC)   — TiledMMA knows how many threads it needs
  // dimGrid   = ceil(M/bM) × ceil(N/bN)
  //
  // Two cudaFuncSetAttribute calls:
  //   MaxDynamicSharedMemorySize  → tell CUDA how much smem we need
  //   PreferredSharedMemoryCarveout = 100 → use 100% of available smem
  // ─────────────────────────────────────────────────────────────────

  // HINT: sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>)
  int smem_size = /* <<< FILL IN >>> */;

  // HINT: dim3 dimBlock(size(mmaC))
  dim3 dimBlock = /* <<< FILL IN >>> */;
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  // Capture the kernel function pointer for attribute setting
  auto kernel_fptr = gemm_device<
    decltype(prob_shape), decltype(cta_tiler),
    cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
    cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
    cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
    decltype(alpha), decltype(beta)>;

  // HINT: cudaFuncSetAttribute(kernel_fptr,
  //           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
  /* <<< FILL IN: set max dynamic smem >>> */;

  // HINT: cudaFuncSetAttribute(kernel_fptr,
  //           cudaFuncAttributePreferredSharedMemoryCarveout, 100)
  /* <<< FILL IN: set smem carveout to 100% >>> */;

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, s2r_atom_A,
       B, dB, sB, copyB, s2r_atom_B,
       C, dC, sC, mmaC,
       alpha, beta);
}


// ═══════════════════════════════════════════════════════════════════
// SCALAR FALLBACK PATHS (float, NT/TN) — already complete
// These use UniversalFMA (scalar) + AutoVectorizingCopy (no ldmatrix)
// They exercise the same kernel template but degrade to sgemm behavior
// ═══════════════════════════════════════════════════════════════════
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha, TA const* A, int ldA,
        TB const* B, int ldB, Beta beta,
        TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto M=int(m); auto N=int(n); auto K=int(k);
  auto prob_shape = make_shape(M,N,K);
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<8>{}; auto bP=Int<3>{};
  auto cta_tiler = make_shape(bM,bN,bK);
  auto sA = make_layout(make_shape(bM,bK,bP));
  auto sB = make_layout(make_shape(bN,bK,bP));
  auto sC = make_layout(make_shape(bM,bN));
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,TA>{},
                                    Layout<Shape<_32,_8>>{}, Layout<Shape<_4,_1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,TB>{},
                                    Layout<Shape<_32,_8>>{}, Layout<Shape<_4,_1>>{});
  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{}, Layout<Shape<_16,_16,_1>>{});
  int smem_size = int(sizeof(SharedStorage<TA,TB,decltype(sA),decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M,bM)), size(ceil_div(N,bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, Copy_Atom<AutoVectorizingCopy,TA>{},
       B, dB, sB, copyB, Copy_Atom<AutoVectorizingCopy,TB>{},
       C, dC, sC, mmaC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_tn_generic(int m, int n, int k,
                Alpha alpha, TA const* A, int ldA,
                TB const* B, int ldB, Beta beta,
                TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto M=int(m); auto N=int(n); auto K=int(k);
  auto prob_shape = make_shape(M,N,K);
  auto dA=make_stride(ldA,Int<1>{}); auto dB=make_stride(ldB,Int<1>{});
  auto dC=make_stride(Int<1>{},ldC);
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<8>{}; auto bP=Int<3>{};
  auto cta_tiler=make_shape(bM,bN,bK);
  auto sA_atom=make_layout(make_shape(bM,bK),make_stride(Int<1>{},bM+Int<1>{}));
  auto sA=tile_to_shape(sA_atom,make_shape(bM,bK,bP));
  auto sB=tile_to_shape(sA_atom,make_shape(bN,bK,bP));
  auto sC=make_layout(make_shape(bM,bN));
  TiledCopy copyA=make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>,TA>{},
                                  Layout<Shape<_32,_8>,Stride<_8,_1>>{},Layout<Shape<_1,_1>>{});
  TiledCopy copyB=make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>,TB>{},
                                  Layout<Shape<_32,_8>,Stride<_8,_1>>{},Layout<Shape<_1,_1>>{});
  TiledMMA mmaC=make_tiled_mma(UniversalFMA<TC,TA,TB>{},Layout<Shape<_16,_16,_1>>{});
  int smem_size=int(sizeof(SharedStorage<TA,TB,decltype(sA),decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M,bM)),size(ceil_div(N,bN)));
  gemm_device<<<dimGrid,dimBlock,smem_size,stream>>>
      (prob_shape,cta_tiler,
       A,dA,sA,copyA,Copy_Atom<AutoVectorizingCopy,TA>{},
       B,dB,sB,copyB,Copy_Atom<AutoVectorizingCopy,TB>{},
       C,dC,sC,mmaC,alpha,beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha, TA const* A, int ldA,
     TB const* B, int ldB, Beta beta,
     TC* C, int ldC, cudaStream_t stream = 0)
{
  if (transA=='N' && transB=='T')
    return gemm_nt(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC,stream);
  else if (transA=='T' && transB=='N')
    return gemm_tn_generic(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC,stream);
  assert(false && "Not implemented");
}


// ═══════════════════════════════════════════════════════════════════
// MAIN — uses half_t TN path (production Tensor Core path)
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
  cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) return -1;
  if (props.major < 8) {
    printf("This example requires SM80 (Ampere) or newer.\n");
    return 0;
  }
  printf("Device: %s (SM%d%d, %d SMs)\n",
         props.name, props.major, props.minor, props.multiProcessorCount);

  int m=5120; if(argc>=2) sscanf(argv[1],"%d",&m);
  int n=5120; if(argc>=3) sscanf(argv[2],"%d",&n);
  int k=4096; if(argc>=4) sscanf(argv[3],"%d",&k);

  // Force TN + half_t for the production path
  char transA='T', transB='N';
  using TA=cute::half_t; using TB=cute::half_t;
  using TC=cute::half_t; using TI=cute::half_t;
  TI alpha=static_cast<TI>(1.0f), beta=static_cast<TI>(0.0f);

  printf("M=%d N=%d K=%d  half_t TN (Tensor Core path)\n", m, n, k);

  thrust::host_vector<TA> h_A(m*k), h_B(n*k), h_C(m*n);
  for(auto& v:h_A) v=static_cast<TA>(2*(rand()/double(RAND_MAX))-1);
  for(auto& v:h_B) v=static_cast<TB>(2*(rand()/double(RAND_MAX))-1);
  for(auto& v:h_C) v=static_cast<TC>(-1);

  thrust::device_vector<TA> d_A=h_A;
  thrust::device_vector<TB> d_B=h_B;
  thrust::device_vector<TC> d_C=h_C;

  double gflops = 2.0*m*n*k*1e-9;
  const int timing_iterations = 100;
  cute::GPU_Clock timer;

  int ldA=k, ldB=n, ldC=m;  // TN: A is (M,K) row-major → ldA=K

  d_C=h_C;
  gemm_tn(m,n,k,alpha,d_A.data().get(),ldA,d_B.data().get(),ldB,
          beta,d_C.data().get(),ldC);
  CUTE_CHECK_LAST();

  timer.start();
  for(int i=0;i<timing_iterations;++i)
    gemm_tn(m,n,k,alpha,d_A.data().get(),ldA,d_B.data().get(),ldB,
            beta,d_C.data().get(),ldC);
  double t = timer.seconds()/timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_TGEMM:  [%6.1f] GFlop/s  (%6.4f ms)\n", gflops/t, t*1000);

  // ─────────────────────────────────────────────────────────────────
  // CHECKPOINT QUESTIONS (write answers in comments here):
  //
  // Q1: retile_D(tCrA) — does this move any bytes?
  //     What two things does it reconcile?
  //     Answer: _______________________________________________
  //
  // Q2: Why is bK=64 here but bK=8 in sgemm?
  //     What hardware constraint forces the larger K-tile for ldmatrix?
  //     Answer: _______________________________________________
  //
  // Q3: The Swizzle<3,3,3> XOR formula is: bank(i,j) = j XOR (i >> M).
  //     With M=3: for rows 0 and 8, what happens to the bank conflict
  //     that would otherwise occur at column 0 with naive k-major layout?
  //     Answer: _______________________________________________
  //
  // Q4: If you changed SM75_U32x4_LDSM_N to SM75_U32x2_LDSM_N,
  //     how would the shape of tXrA change?
  //     Would the mma.sync still work? Why or why not?
  //     Answer: _______________________________________________
  //
  // Q5: In the pipeline loop, why does the gmem→smem copy fire at
  //     k_block==0 and NOT at k_block==K_BLOCK_MAX-1?
  //     Answer: _______________________________________________
  // ─────────────────────────────────────────────────────────────────
  return 0;
}
