/***************************************************************************************************
 * FILL-IN EXERCISE: sgemm.cu — Scalar GEMM with CuTe (Tutorial Level)
 *
 * OBJECTIVE: Understand CuTe's layout algebra and two-partition pattern
 *   (1) local_tile  : slice the full global tensor into CTA-sized tiles
 *   (2) local_partition (copy)   : divide a tile across threads for loading
 *   (3) local_partition (compute): divide smem across threads for the GEMM
 *
 * RULES:
 *   - Fill in every section marked  // <<< FILL IN >>>
 *   - Read the comment ABOVE each blank — it tells you exactly what to write
 *   - Do NOT look at the reference until you have tried yourself first
 *   - Compile after EACH section to catch errors early
 *
 * BUILD (SM89 Ada / RTX 4090):
 *   nvcc -std=c++17 -O3 -arch=sm_89 sgemm_FILL_IN.cu \
 *        -I /path/to/cutlass/include \
 *        -o sgemm && ./sgemm
 *
 * PREDICT BEFORE RUNNING:
 *   - How many floats does each thread own in tCrC (the accumulator)?
 *   - How many global loads does each thread issue per k-tile?
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
// SECTION 1 — KERNEL SIGNATURE
// The kernel is templated on ALL layout types so the same device
// function works for NT and TN transposes without branching.
// ═══════════════════════════════════════════════════════════════════
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
// HINT: __launch_bounds__ takes the *number of threads per block*.
//       Use the CuTe utility that computes the size of a static layout.
__launch_bounds__(/* <<< FILL IN: size(CThreadLayout{}) as a constexpr value >>> */
                  decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 2 — STATIC ASSERTIONS (already filled in for reference)
  // These fire at compile time and document the contracts this kernel
  // requires.  Read them — they are documentation.
  // ─────────────────────────────────────────────────────────────────
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);
  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});
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
  // SECTION 3 — FULL GLOBAL TENSORS
  // Wrap raw pointers in CuTe Tensor objects that know their shape
  // and stride.  These represent the ENTIRE M×K, N×K, M×N matrices.
  //
  // Pattern:  make_tensor(make_gmem_ptr(ptr), shape, stride)
  // select<i,j>(shape_MNK) picks two of the three (M,N,K) dimensions.
  // ─────────────────────────────────────────────────────────────────

  // HINT: A has shape (M, K).  Pick dimensions 0 and 2 from shape_MNK.
  Tensor mA = /* <<< FILL IN >>> */;   // (M, K)

  // HINT: B has shape (N, K).  Pick dimensions 1 and 2.
  Tensor mB = /* <<< FILL IN >>> */;   // (N, K)

  // HINT: C has shape (M, N).  Pick dimensions 0 and 1.
  Tensor mC = /* <<< FILL IN >>> */;   // (M, N)

  // ─────────────────────────────────────────────────────────────────
  // SECTION 4 — CTA-LEVEL TILES  (local_tile)
  //
  // local_tile(tensor, tiler, coord, Step<...>{}) slices out the
  // tile owned by this CTA.  It is a zero-copy view.
  //
  // Step<_1, X, _1>  means: participate in dims 0 and 2, skip dim 1.
  // The underscore in the output shape comes from the "_ " wildcard
  // that keeps the K-loop index as a trailing dimension.
  // ─────────────────────────────────────────────────────────────────

  // CTA coordinate: (block_m, block_n, all_k)
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);   // already given

  // HINT: gA participates in M (dim0) and K (dim2), skips N (dim1).
  //       Result shape: (BLK_M, BLK_K, k_tiles)
  Tensor gA = /* <<< FILL IN >>> */;   // (BLK_M, BLK_K, k)

  // HINT: gB participates in N (dim1) and K (dim2), skips M (dim0).
  //       Result shape: (BLK_N, BLK_K, k_tiles)
  Tensor gB = /* <<< FILL IN >>> */;   // (BLK_N, BLK_K, k)

  // HINT: gC participates in M (dim0) and N (dim1), skips K (dim2).
  //       Result shape: (BLK_M, BLK_N)  — no k-loop dimension
  Tensor gC = /* <<< FILL IN >>> */;   // (BLK_M, BLK_N)

  // ─────────────────────────────────────────────────────────────────
  // SECTION 5 — SHARED MEMORY TENSORS
  // Allocate static smem arrays and wrap them as CuTe Tensors.
  // cosize_v<Layout> gives the number of elements needed.
  // ─────────────────────────────────────────────────────────────────

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];

  // HINT: Use make_smem_ptr(smemA) and sA_layout.
  Tensor sA = /* <<< FILL IN >>> */;   // (BLK_M, BLK_K)
  Tensor sB = /* <<< FILL IN >>> */;   // (BLK_N, BLK_K)

  // ─────────────────────────────────────────────────────────────────
  // SECTION 6 — THREAD PARTITIONING FOR COPY  (local_partition)
  //
  // local_partition(tensor, thread_layout, thread_id)
  // Divides `tensor` among threads using a raked (interleaved) pattern.
  // Each thread gets a strided sub-tensor.
  //
  // tAgA: this thread's slice of gA in global memory  → used as source
  // tAsA: this thread's slice of sA in shared memory  → used as dest
  // ─────────────────────────────────────────────────────────────────

  // HINT: Partition gA over tA for copy. Use threadIdx.x.
  //       Result shape: (THR_M, THR_K, k_tiles)
  Tensor tAgA = /* <<< FILL IN >>> */;

  // HINT: Partition sA over tA for copy. Same thread layout and id.
  //       Result shape: (THR_M, THR_K)
  Tensor tAsA = /* <<< FILL IN >>> */;

  // HINT: Same for B.
  Tensor tBgB = /* <<< FILL IN >>> */;
  Tensor tBsB = /* <<< FILL IN >>> */;

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));

  // ─────────────────────────────────────────────────────────────────
  // SECTION 7 — THREAD PARTITIONING FOR COMPUTE
  //
  // For the GEMM we use tC (the C thread layout) with Step projections:
  //   Step<_1, X>   → select only the M dimension of tC
  //   Step< X,_1>   → select only the N dimension of tC
  //   Step<_1,_1>   → select both M and N
  //
  // This lets us reuse a single 2D thread layout for three different
  // partitioning operations on A, B, and C respectively.
  // ─────────────────────────────────────────────────────────────────

  // HINT: Partition sA by the M-rows of tC.
  //       Extra arg: Step<_1, X>{}   (participate in M, skip N)
  //       Result shape: (THR_M, BLK_K)
  Tensor tCsA = /* <<< FILL IN >>> */;

  // HINT: Partition sB by the N-cols of tC.
  //       Extra arg: Step< X,_1>{}   (skip M, participate in N)
  //       Result shape: (THR_N, BLK_K)
  Tensor tCsB = /* <<< FILL IN >>> */;

  // HINT: Partition gC by both dimensions of tC.
  //       Extra arg: Step<_1,_1>{}
  //       Result shape: (THR_M, THR_N)
  Tensor tCgC = /* <<< FILL IN >>> */;

  // HINT: Allocate register accumulator with the same shape as tCgC.
  //       Use make_tensor_like(tCgC).
  Tensor tCrC = /* <<< FILL IN >>> */;

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));

  // Zero out the accumulator
  // HINT: CuTe utility function that zeroes any tensor: clear(t)
  /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 8 — MAIN K LOOP
  //
  // For each k-tile:
  //   1. Copy the tile from gmem → smem  (tAgA → tAsA)
  //   2. Fence + wait to ensure all cp instructions are visible
  //   3. Sync threads so all smem writes are visible to all threads
  //   4. Compute partial GEMM on smem  (tCsA, tCsB → tCrC)
  //   5. Sync again before next iteration overwrites smem
  // ─────────────────────────────────────────────────────────────────

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // HINT: copy(src, dst)  — slice the k_tile from tAgA with (_,_,k_tile)
    /* <<< FILL IN: copy A tile gmem→smem >>> */
    /* <<< FILL IN: copy B tile gmem→smem >>> */

    // HINT: cp_async_fence() marks the end of async copy group
    /* <<< FILL IN >>> */;
    // HINT: cp_async_wait<0>() waits for all pending async copies
    /* <<< FILL IN >>> */;
    // HINT: standard CUDA thread barrier
    /* <<< FILL IN >>> */;

    // HINT: gemm(A_smem_fragment, B_smem_fragment, C_register_accum)
    //       This expands to the scalar triple loop at compile time.
    /* <<< FILL IN >>> */;

    // HINT: must sync before next iteration writes to smem
    /* <<< FILL IN >>> */;
  }

  // ─────────────────────────────────────────────────────────────────
  // SECTION 9 — EPILOGUE
  // Write accumulator back to global memory with scaling:
  //   C[i] = alpha * tCrC[i] + beta * tCgC[i]
  // ─────────────────────────────────────────────────────────────────

  // HINT: axpby(alpha, src_reg, beta, dst_global)
  /* <<< FILL IN >>> */;
}


// ═══════════════════════════════════════════════════════════════════
// HOST SIDE — gemm_nt  (A column-major, B row-major)
// ═══════════════════════════════════════════════════════════════════
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha, TA const* A, int ldA,
        TB const* B, int ldB, Beta beta,
        TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // ─────────────────────────────────────────────────────────────────
  // SECTION 10 — STRIDES FOR NT
  // NT means A is Not-transposed (col-major) and B is Transposed.
  // Col-major: stride-1 in the row dimension, ldX in the col dim.
  // ─────────────────────────────────────────────────────────────────

  // HINT: A (M×K) col-major: (dM=1, dK=ldA)
  auto dA = /* <<< FILL IN >>> */;

  // HINT: B (N×K) col-major (stored as N×K): (dN=1, dK=ldB)
  auto dB = /* <<< FILL IN >>> */;

  // HINT: C (M×N) col-major: (dM=1, dN=ldC)
  auto dC = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 11 — TILE SIZES AND SMEM LAYOUTS (NT)
  // ─────────────────────────────────────────────────────────────────

  // HINT: BLK_M=128, BLK_N=128, BLK_K=8
  auto bM = /* <<< FILL IN >>> */;
  auto bN = /* <<< FILL IN >>> */;
  auto bK = /* <<< FILL IN >>> */;
  auto cta_tiler = make_shape(bM, bN, bK);

  // HINT: smem layouts — no stride given → column-major by default
  //       sA: shape (bM, bK), sB: shape (bN, bK), sC: shape (bM, bN)
  auto sA = /* <<< FILL IN >>> */;
  auto sB = /* <<< FILL IN >>> */;
  auto sC = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION 12 — THREAD LAYOUTS (NT)
  // tA and tB tile the copy; tC tiles the compute.
  // NT uses column-major thread layouts (default, no LayoutRight).
  // ─────────────────────────────────────────────────────────────────

  // HINT: tA: (32, 8) col-major  (32 threads in M, 8 in K)
  auto tA = /* <<< FILL IN >>> */;
  // HINT: tB: (32, 8) col-major
  auto tB = /* <<< FILL IN >>> */;
  // HINT: tC: (16, 16) col-major  → 256 threads total
  auto tC = /* <<< FILL IN >>> */;

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}


// ═══════════════════════════════════════════════════════════════════
// HOST SIDE — gemm_tn  (A row-major, B col-major)
// ═══════════════════════════════════════════════════════════════════
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha, TA const* A, int ldA,
        TB const* B, int ldB, Beta beta,
        TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // ─────────────────────────────────────────────────────────────────
  // SECTION 13 — STRIDES FOR TN
  // TN: A is Transposed (row-major), B is Not-transposed.
  // Row-major: stride-1 in the column (K) dimension, ldX in the row.
  // ─────────────────────────────────────────────────────────────────

  // HINT: A (M×K) row-major: (dM=ldA, dK=1)
  auto dA = /* <<< FILL IN >>> */;
  // HINT: B (N×K) row-major: (dN=ldB, dK=1)
  auto dB = /* <<< FILL IN >>> */;
  // HINT: C (M×N) col-major: (dM=1, dN=ldC)  — same as NT
  auto dC = /* <<< FILL IN >>> */;

  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  // ─────────────────────────────────────────────────────────────────
  // SECTION 14 — SMEM LAYOUTS FOR TN (padded k-major)
  // TN needs k-major smem so that threads walking in K-order hit
  // consecutive smem addresses.  Use LayoutRight{} for row-major.
  // ─────────────────────────────────────────────────────────────────

  // HINT: sA shape (bM,bK), LayoutRight{} → k-major (row-major)
  auto sA = /* <<< FILL IN >>> */;
  // HINT: sB shape (bN,bK), LayoutRight{}
  auto sB = /* <<< FILL IN >>> */;
  auto sC = make_layout(make_shape(bM, bN));

  // ─────────────────────────────────────────────────────────────────
  // SECTION 15 — THREAD LAYOUTS FOR TN (k-major threads)
  // Threads must also walk K fastest to match the k-major smem.
  // ─────────────────────────────────────────────────────────────────

  // HINT: tA (32,8) LayoutRight{} → k-major thread walk
  auto tA = /* <<< FILL IN >>> */;
  // HINT: tB (32,8) LayoutRight{}
  auto tB = /* <<< FILL IN >>> */;
  // HINT: tC (16,16) col-major — same as NT (C is always col-major)
  auto tC = /* <<< FILL IN >>> */;

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}


// ═══════════════════════════════════════════════════════════════════
// DISPATCH + MAIN (already complete — do not modify)
// ═══════════════════════════════════════════════════════════════════
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha, TA const* A, int ldA,
     TB const* B, int ldB, Beta beta,
     TC* C, int ldC, cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T')
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N')
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  assert(false && "Not implemented");
}

int main(int argc, char** argv)
{
  int m = 5120; if (argc >= 2) sscanf(argv[1], "%d", &m);
  int n = 5120; if (argc >= 3) sscanf(argv[2], "%d", &n);
  int k = 4096; if (argc >= 4) sscanf(argv[3], "%d", &k);
  char transA = 'N'; if (argc >= 5) sscanf(argv[4], "%c", &transA);
  char transB = 'T'; if (argc >= 6) sscanf(argv[5], "%c", &transB);

  using TA = float; using TB = float; using TC = float; using TI = float;
  TI alpha = 1.0f, beta = 0.0f;

  printf("M=%d N=%d K=%d  C = A^%c B^%c\n", m, n, k, transA, transB);
  cute::device_init(0);

  thrust::host_vector<TA> h_A(m*k), h_B(n*k), h_C(m*n);
  for (auto& v : h_A) v = static_cast<TA>(2*(rand()/double(RAND_MAX))-1);
  for (auto& v : h_B) v = static_cast<TB>(2*(rand()/double(RAND_MAX))-1);
  for (auto& v : h_C) v = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A=h_A; thrust::device_vector<TB> d_B=h_B;
  thrust::device_vector<TC> d_C=h_C;

  double gflops = 2.0*m*n*k*1e-9;
  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = (transA=='N') ? m : k;
  int ldB = (transB=='N') ? k : n;
  int ldC = m;

  d_C = h_C;
  gemm(transA, transB, m, n, k, alpha,
       d_A.data().get(), ldA, d_B.data().get(), ldB, beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();

  timer.start();
  for (int i = 0; i < timing_iterations; ++i)
    gemm(transA, transB, m, n, k, alpha,
         d_A.data().get(), ldA, d_B.data().get(), ldB, beta,
         d_C.data().get(), ldC);
  double t = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:  [%6.1f] GFlop/s  (%6.4f ms)\n", gflops/t, t*1000);

  // ─────────────────────────────────────────────────────────────────
  // CHECKPOINT QUESTIONS (answer in comments before submitting):
  //
  // Q1: With bM=128, bN=128, bK=8 and tC=(16,16):
  //     What is size<0>(tCrC) and size<1>(tCrC)?
  //     How many register floats does each thread hold?
  //
  // Q2: Why does gemm_tn use LayoutRight for smem but gemm_nt does not?
  //     What memory access problem would occur if you used col-major
  //     smem in the TN case?
  //
  // Q3: The copy loop does:  copy(tAgA(_,_,k_tile), tAsA)
  //     How is this different from:  tAsA = tAgA(_,_,k_tile) ?
  //     (Hint: one moves data, one creates a view)
  // ─────────────────────────────────────────────────────────────────
  return 0;
}
