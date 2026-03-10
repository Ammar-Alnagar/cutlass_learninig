/***************************************************************************************************
 * FILL-IN EXERCISE: tiled_copy.cu + predicated_copy.cu
 *
 * This file contains TWO exercises in one:
 *
 * PART A — tiled_copy (vectorized global memory copy)
 *   Covers: tiled_divide, copy_kernel (local_partition), TiledCopy, copy_kernel_vectorized
 *   This is the PURE COPY primitive — no GEMM, no smem compute.
 *   Everything you fill in here maps directly to Module 03 TiledCopy exercises.
 *
 * PART B — predicated_copy (copy with boundary guards)
 *   Covers: make_identity_tensor, lazy::transform, elem_less, copy_if
 *   NEW: tensors whose shapes are NOT divisible by the block size.
 *   This is Module 07 (Predication) material — essential for real FlashAttention
 *   where sequence length doesn't divide tile size.
 *
 * BUILD:
 *   nvcc -std=c++17 -O3 -arch=sm_89 tiled_predicated_copy_FILL_IN.cu \
 *        -I /path/to/cutlass/include -o copy_ex && ./copy_ex
 *
 * PREDICT BEFORE RUNNING:
 *   - With tensor_shape=(256,512) and block_shape=(128,64):
 *     What are the grid dimensions? How many blocks total?
 *   - In the vectorized kernel: what is size<0>(thr_tile_S)?
 *     (the CPY/CopyOp dimension — hint: val_layout = (4,1))
 *   - For tensor_shape=(528,300), what blocks need predication?
 *     (528 % 128 = ?, 300 % 64 = ?)
 ***************************************************************************************************/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


// ═══════════════════════════════════════════════════════════════════════
// ██████████████  PART A: TILED COPY  ██████████████████████████████████
// ═══════════════════════════════════════════════════════════════════════


// ───────────────────────────────────────────────────────────────────────
// KERNEL A1: Scalar copy using local_partition
//
// The tensors S and D have ALREADY been tiled by the host via tiled_divide:
//   Shape: ((BlockShape_M, BlockShape_N), num_tiles_m, num_tiles_n)
//
// Each threadblock handles one tile: S(_,blockIdx.x,blockIdx.y)
// Within the tile, threads use local_partition to divide work.
// ───────────────────────────────────────────────────────────────────────
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
  using namespace cute;

  // ─────────────────────────────────────────────────────────────────
  // SECTION A1.1 — SLICE THIS BLOCK'S TILE
  //
  // S has shape ((M, N), m', n').  This block owns tile (blockIdx.x, blockIdx.y).
  // Use (make_coord(_,_), blockIdx.x, blockIdx.y) to slice it.
  // Result shape: (BlockShape_M, BlockShape_N) — the 2D tile for this block.
  // ─────────────────────────────────────────────────────────────────

  // HINT: S(make_coord(_,_), blockIdx.x, blockIdx.y)
  Tensor tile_S = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)

  // ─────────────────────────────────────────────────────────────────
  // SECTION A1.2 — PARTITION AMONG THREADS  (local_partition)
  //
  // Divide the tile among threads using ThreadLayout.
  // Each thread gets a (ThrValM, ThrValN) slice.
  // Pattern: local_partition(tensor, ThreadLayout{}, threadIdx.x)
  // ─────────────────────────────────────────────────────────────────

  // HINT: local_partition(tile_S, ThreadLayout{}, threadIdx.x)
  Tensor thr_tile_S = /* <<< FILL IN >>> */;   // (ThrValM, ThrValN)
  Tensor thr_tile_D = /* <<< FILL IN >>> */;   // (ThrValM, ThrValN)

  // ─────────────────────────────────────────────────────────────────
  // SECTION A1.3 — REGISTER FRAGMENT + COPY
  //
  // Allocate a register-backed tensor with the same shape.
  // Then: gmem → rmem → gmem  (two copy calls).
  // ─────────────────────────────────────────────────────────────────

  // HINT: make_tensor_like(thr_tile_S) — register tensor, same shape
  //       Note: make_tensor_like vs make_fragment_like:
  //         make_tensor_like  → tries to match src layout (may be smem/gmem)
  //         make_fragment_like → always makes a register tensor (first mode = atom)
  //       For scalar copies, make_tensor_like is fine.
  Tensor fragment = /* <<< FILL IN >>> */;     // (ThrValM, ThrValN)

  // HINT: copy(thr_tile_S, fragment)   — no TiledCopy atom: scalar copy
  /* <<< FILL IN: gmem → register >>> */;
  /* <<< FILL IN: register → gmem >>> */;
}


// ───────────────────────────────────────────────────────────────────────
// KERNEL A2: Vectorized copy using TiledCopy
//
// Same structure as A1 but uses a TiledCopy object that issues
// vector load/store instructions (128-bit reads = 4× float per thread).
//
// Key difference: use tiled_copy.get_thread_slice() and partition_S/D,
// then copy(tiled_copy, src, dst) to invoke the vector atom.
// ───────────────────────────────────────────────────────────────────────
template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
  using namespace cute;

  // ─────────────────────────────────────────────────────────────────
  // SECTION A2.1 — SLICE THIS BLOCK'S TILE (identical to A1.1)
  // ─────────────────────────────────────────────────────────────────

  Tensor tile_S = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)

  // ─────────────────────────────────────────────────────────────────
  // SECTION A2.2 — TILEDCOPY PARTITION (NEW vs A1)
  //
  // Instead of local_partition, use the TiledCopy API:
  //   tiled_copy.get_thread_slice(threadIdx.x)  → ThrCopy for this thread
  //   thr_copy.partition_S(tile_S)  → source partition  (CopyOp, CopyM, CopyN)
  //   thr_copy.partition_D(tile_D)  → dest partition    (CopyOp, CopyM, CopyN)
  //
  // The CopyOp leading dimension = atom width (4 floats = 128 bits here).
  // ─────────────────────────────────────────────────────────────────

  // HINT: ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  /* <<< FILL IN: declare thr_copy >>> */;

  // HINT: thr_copy.partition_S(tile_S)
  Tensor thr_tile_S = /* <<< FILL IN >>> */;   // (CopyOp, CopyM, CopyN)
  Tensor thr_tile_D = /* <<< FILL IN >>> */;   // (CopyOp, CopyM, CopyN)

  // ─────────────────────────────────────────────────────────────────
  // SECTION A2.3 — REGISTER FRAGMENT + VECTORIZED COPY
  //
  // make_fragment_like(thr_tile_D) — register tensor where first mode
  // is the "instruction-local" mode (the CopyOp atom dimension).
  // Must use make_fragment_like (not make_tensor_like) here because
  // the first mode is the atom dimension, not a spatial dimension.
  //
  // copy(tiled_copy, src, dst) — invokes the vector atom.
  // ─────────────────────────────────────────────────────────────────

  // HINT: make_fragment_like(thr_tile_D)
  Tensor fragment = /* <<< FILL IN >>> */;     // (CopyOp, CopyM, CopyN)

  // HINT: copy(tiled_copy, thr_tile_S, fragment)  — vector load from gmem
  /* <<< FILL IN: vectorized gmem → register >>> */;
  // HINT: copy(tiled_copy, fragment, thr_tile_D)  — vector store to gmem
  /* <<< FILL IN: vectorized register → gmem >>> */;
}


// ═══════════════════════════════════════════════════════════════════════
// ██████████████  PART B: PREDICATED COPY  █████████████████████████████
// ═══════════════════════════════════════════════════════════════════════
//
// Problem: tensor_shape=(528,300), block_shape=(128,64)
//   528 % 128 = 16  → last row-block is PARTIAL (only 16 valid rows)
//   300 % 64  = 44  → last col-block is PARTIAL (only 44 valid cols)
//
// Solution: predicate tensor P marks which elements are in-bounds.
// copy_if(P, S, D) only copies where P is true.
//
// The three new CuTe utilities:
//   make_identity_tensor(shape)  → tensor of (row, col) coordinate tuples
//   lazy::transform(C, fn)       → lazy elementwise transform (no allocation)
//   elem_less(coord, shape)      → true iff coord is in-bounds for shape
//   copy_if(pred, src, dst)      → predicated copy


// ───────────────────────────────────────────────────────────────────────
// KERNEL B1: Scalar predicated copy
// ───────────────────────────────────────────────────────────────────────
template <class TensorS, class TensorD, class BlockShape, class ThreadLayout>
__global__ void copy_if_kernel(TensorS S, TensorD D, BlockShape block_shape, ThreadLayout)
{
  using namespace cute;

  // ─────────────────────────────────────────────────────────────────
  // SECTION B1.1 — BUILD COORDINATE + PREDICATE TENSORS
  //
  // Step 1: make_identity_tensor(shape_S)
  //   Creates a tensor where element (i,j) holds the tuple (i,j).
  //   Think of it as a coordinate grid over the FULL tensor shape.
  //
  // Step 2: lazy::transform(C, lambda)
  //   Applies a function elementwise, lazily (no memory allocation).
  //   The lambda receives a coordinate tuple and returns bool.
  //
  // Step 3: elem_less(coord, shape)
  //   Returns true iff ALL dimensions of coord < corresponding shape dim.
  //   This is the in-bounds predicate.
  // ─────────────────────────────────────────────────────────────────

  auto shape_S = shape(S);

  // HINT: make_identity_tensor(shape_S)  → coordinate tensor
  Tensor C = /* <<< FILL IN >>> */;

  // HINT: cute::lazy::transform(C, [&](auto c){ return elem_less(c, shape_S); })
  Tensor P = /* <<< FILL IN >>> */;   // lazy predicate tensor

  // ─────────────────────────────────────────────────────────────────
  // SECTION B1.2 — TILE ALL THREE TENSORS (S, D, P)
  //
  // Unlike tiled_divide (which requires evenly-divisible shapes),
  // local_tile works on non-divisible shapes — it just over-tiles
  // and relies on predication to guard the out-of-bounds accesses.
  //
  // block_coord = (blockIdx.x, blockIdx.y) — this block's tile position
  // ─────────────────────────────────────────────────────────────────

  auto block_coord = make_coord(blockIdx.x, blockIdx.y);

  // HINT: local_tile(S, block_shape, block_coord) — slice this block's region
  Tensor tile_S = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)

  // HINT: local_tile(P, block_shape, block_coord) — same tiling on predicate
  Tensor tile_P = /* <<< FILL IN >>> */;   // (BlockShape_M, BlockShape_N)

  // ─────────────────────────────────────────────────────────────────
  // SECTION B1.3 — THREAD PARTITION ON ALL THREE
  // ─────────────────────────────────────────────────────────────────

  // HINT: local_partition(tile_S, ThreadLayout{}, threadIdx.x)
  Tensor thr_tile_S = /* <<< FILL IN >>> */;
  Tensor thr_tile_D = /* <<< FILL IN >>> */;
  Tensor thr_tile_P = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION B1.4 — PREDICATED COPY
  //
  // copy_if(predicate_tensor, src, dst)
  // Only copies elements where predicate is true.
  // No intermediate register tensor needed here (direct gmem→gmem).
  // ─────────────────────────────────────────────────────────────────

  // HINT: copy_if(thr_tile_P, thr_tile_S, thr_tile_D)
  /* <<< FILL IN >>> */;
}


// ───────────────────────────────────────────────────────────────────────
// KERNEL B2: Vectorized predicated copy
//
// Combines TiledCopy vectorization with predication.
// Key: partition_S applied to BOTH the data tile AND the predicate tile
// so they have the same (CopyOp, CopyM, CopyN) shape.
// ───────────────────────────────────────────────────────────────────────
template <class TensorS, class TensorD, class BlockShape, class Tiled_Copy>
__global__ void copy_if_kernel_vectorized(TensorS S, TensorD D, BlockShape block_shape, Tiled_Copy tiled_copy)
{
  using namespace cute;

  auto shape_S = shape(S);

  // HINT: same coordinate/predicate construction as B1.1
  Tensor C = /* <<< FILL IN >>> */;
  Tensor P = /* <<< FILL IN >>> */;

  auto block_coord = make_coord(blockIdx.x, blockIdx.y);

  // HINT: tile S, D, and P
  Tensor tile_S = /* <<< FILL IN >>> */;
  Tensor tile_D = /* <<< FILL IN >>> */;
  Tensor tile_P = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION B2.1 — TILEDCOPY PARTITION ON S, D, AND P
  //
  // IMPORTANT: partition_S is applied to BOTH tile_S AND tile_P.
  // The predicate tensor P must be partitioned the same way as S
  // so that thr_tile_P[i] guards thr_tile_S[i] correctly.
  //
  // NOTE: partition_S is used for P (not partition_D) because P
  //       describes the SOURCE elements we're allowed to read.
  // ─────────────────────────────────────────────────────────────────

  // HINT: tiled_copy.get_thread_slice(threadIdx.x)
  ThrCopy thr_copy = /* <<< FILL IN >>> */;

  // HINT: thr_copy.partition_S(tile_S)
  Tensor thr_tile_S = /* <<< FILL IN >>> */;   // (CopyOp, CopyM, CopyN)
  // HINT: thr_copy.partition_D(tile_D)
  Tensor thr_tile_D = /* <<< FILL IN >>> */;   // (CopyOp, CopyM, CopyN)
  // HINT: thr_copy.partition_S(tile_P)  — NOTE: partition_S, not partition_D!
  Tensor thr_tile_P = /* <<< FILL IN >>> */;   // (CopyOp, CopyM, CopyN)

  // ─────────────────────────────────────────────────────────────────
  // SECTION B2.2 — TWO-STAGE PREDICATED COPY (gmem→rmem→gmem)
  //
  // Direct copy_if(tiled_copy, P, S, D) works but does gmem→gmem.
  // Better: stage through registers to allow compiler optimizations.
  //
  // make_fragment_like(thr_tile_S) → register buffer
  // copy_if(tiled_copy, P, S, frag) → guarded load to registers
  // copy_if(tiled_copy, P, frag, D) → guarded store from registers
  // ─────────────────────────────────────────────────────────────────

  // HINT: make_fragment_like(thr_tile_S)
  Tensor frag = /* <<< FILL IN >>> */;

  // HINT: copy_if(tiled_copy, thr_tile_P, thr_tile_S, frag)
  /* <<< FILL IN: guarded load gmem → register >>> */;
  // HINT: copy_if(tiled_copy, thr_tile_P, frag, thr_tile_D)
  /* <<< FILL IN: guarded store register → gmem >>> */;
}


// ═══════════════════════════════════════════════════════════════════════
// PART A MAIN — Tiled Copy
// ═══════════════════════════════════════════════════════════════════════
void run_tiled_copy()
{
  using namespace cute;
  using Element = float;

  auto tensor_shape = make_shape(256, 512);
  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));
  for(size_t i=0;i<h_S.size();++i){ h_S[i]=Element(i); h_D[i]=Element{}; }
  thrust::device_vector<Element> d_S=h_S, d_D=h_D;

  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                                make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                                make_layout(tensor_shape));

  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if (!evenly_divides(tensor_shape, block_shape)) {
    printf("PART A: tensor shape must be divisible by block shape\n"); return;
  }

  // ─────────────────────────────────────────────────────────────────
  // SECTION A3 — HOST: TILED DIVIDE
  //
  // tiled_divide(tensor, block_shape) reshapes a (m,n) tensor into
  // ((BlockM, BlockN), m', n') where m' = m/BlockM, n' = n/BlockN.
  // This is used to pass pre-tiled tensors to the kernel so blockIdx
  // can index directly into the tile dimension.
  // ─────────────────────────────────────────────────────────────────

  // HINT: tiled_divide(tensor_S, block_shape)  → ((M,N), m', n')
  Tensor tiled_S = /* <<< FILL IN >>> */;
  Tensor tiled_D = /* <<< FILL IN >>> */;

  // ─────────────────────────────────────────────────────────────────
  // SECTION A4 — HOST: TILEDCOPY CONFIG
  //
  // UniversalCopy<uint_byte_t<N>>: copy atom that issues N-byte loads
  //   N = sizeof(Element) * size(val_layout) = 4 * 4 = 16 bytes = 128 bits
  // ─────────────────────────────────────────────────────────────────

  // Thread layout: 32 threads in M, 8 in N (256 threads total)
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  // Value layout: each thread loads 4×1 floats per atom invocation
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  // HINT: using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;
  /* <<< FILL IN: define CopyOp type >>> */

  // HINT: using Atom = Copy_Atom<CopyOp, Element>;
  /* <<< FILL IN: define Atom type >>> */

  // HINT: make_tiled_copy(Atom{}, thr_layout, val_layout)
  TiledCopy tiled_copy = /* <<< FILL IN >>> */;

  // Grid: one block per tile
  // HINT: size<1>(tiled_D) = m' tiles in M, size<2>(tiled_D) = n' tiles in N
  dim3 gridDim(/* <<< FILL IN: size<1> >>> */, /* <<< FILL IN: size<2> >>> */);
  dim3 blockDim(size(thr_layout));

  copy_kernel_vectorized<<<gridDim, blockDim>>>(tiled_S, tiled_D, tiled_copy);
  CUTE_CHECK_LAST();
  cudaDeviceSynchronize();

  h_D = d_D;
  int errs=0;
  for(size_t i=0;i<h_D.size();++i) if(h_S[i]!=h_D[i]){ printf("PART A ERROR at %zu\n",i); if(++errs>=5) break; }
  if(!errs) printf("PART A: Success (vectorized copy, tensor %dx%d, block %dx%d)\n",
                   size<0>(tensor_shape), size<1>(tensor_shape),
                   size<0>(block_shape), size<1>(block_shape));
}


// ═══════════════════════════════════════════════════════════════════════
// PART B MAIN — Predicated Copy
// ═══════════════════════════════════════════════════════════════════════
void run_predicated_copy()
{
  using namespace cute;
  using Element = float;

  // NOTE: 528 % 128 = 16 (not divisible), 300 % 64 = 44 (not divisible)
  // This is intentionally non-divisible to exercise predication.
  auto tensor_shape = make_shape(528, 300);
  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));
  for(size_t i=0;i<h_S.size();++i){ h_S[i]=Element(i); h_D[i]=Element{}; }
  thrust::device_vector<Element> d_S=h_S, d_D=h_D;
  thrust::device_vector<Element> d_Zero=h_D;  // all zeros for reset

  Tensor tensor_S = make_tensor(make_gmem_ptr(d_S.data().get()), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(d_D.data().get()), make_layout(tensor_shape));

  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  // ─────────────────────────────────────────────────────────────────
  // SECTION B3 — HOST: GRID SIZE FOR NON-DIVISIBLE SHAPES
  //
  // We can't use tiled_divide (requires even division).
  // Instead, use ceil_div to over-tile and rely on predication.
  //
  // Grid: ceil(528/128) × ceil(300/64) = 5 × 5 = 25 blocks
  // Some blocks will have partial tiles — predication guards them.
  // ─────────────────────────────────────────────────────────────────

  // For predicated copy, tile D is used to get grid dims
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));

  // Grid covers ALL tiles including partial ones
  dim3 gridDim(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));
  dim3 blockDim(size(thr_layout));

  // ── B3.1: Run scalar predicated copy (Kernel B1) ──
  copy_if_kernel<<<gridDim, blockDim>>>(tensor_S, tensor_D, block_shape, thr_layout);
  CUTE_CHECK_LAST();
  cudaDeviceSynchronize();
  h_D = d_D;
  int errs=0;
  for(size_t i=0;i<h_D.size();++i) if(h_S[i]!=h_D[i]){ printf("PART B (scalar) ERROR at %zu\n",i); if(++errs>=5) break; }
  if(!errs) printf("PART B (scalar):     Success (non-divisible %dx%d)\n",
                   size<0>(tensor_shape), size<1>(tensor_shape));

  // Reset D
  thrust::copy(d_Zero.begin(), d_Zero.end(), d_D.begin());

  // ── B3.2: Run vectorized predicated copy (Kernel B2) ──
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));
  using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;
  using Atom = Copy_Atom<CopyOp, Element>;
  TiledCopy tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);

  copy_if_kernel_vectorized<<<gridDim, blockDim>>>(tensor_S, tensor_D, block_shape, tiled_copy);
  CUTE_CHECK_LAST();
  cudaDeviceSynchronize();
  h_D = d_D;
  errs=0;
  for(size_t i=0;i<h_D.size();++i) if(h_S[i]!=h_D[i]){ printf("PART B (vector) ERROR at %zu\n",i); if(++errs>=5) break; }
  if(!errs) printf("PART B (vectorized): Success (non-divisible %dx%d)\n",
                   size<0>(tensor_shape), size<1>(tensor_shape));
}

int main()
{
  run_tiled_copy();
  run_predicated_copy();

  // ─────────────────────────────────────────────────────────────────
  // CHECKPOINT QUESTIONS:
  //
  // Q1: In copy_kernel (A1), why is make_tensor_like used instead of
  //     make_fragment_like? When does the distinction matter?
  //     Answer: ______________________________________________________
  //
  // Q2: In copy_kernel_vectorized (A2), what is size<0>(thr_tile_S)?
  //     (the CopyOp dimension)  How does it relate to val_layout=(4,1)?
  //     Answer: ______________________________________________________
  //
  // Q3: tiled_divide REQUIRES evenly_divides(tensor_shape, block_shape).
  //     Part B uses local_tile instead.  What does local_tile do
  //     differently when the shapes don't divide evenly?
  //     Answer: ______________________________________________________
  //
  // Q4: In copy_if_kernel_vectorized (B2), thr_tile_P uses partition_S
  //     (not partition_D).  Why?  What would happen if you used partition_D?
  //     Answer: ______________________________________________________
  //
  // Q5: In Part B, tensor_shape=(528,300) with block_shape=(128,64).
  //     Draw the grid: which block indices have ONLY out-of-bounds elements
  //     in BOTH dimensions simultaneously?
  //     (blocks needing predication in M: blockIdx.x == ?)
  //     (blocks needing predication in N: blockIdx.y == ?)
  //     Answer: ______________________________________________________
  //
  // Q6: How does the lazy in lazy::transform(C, fn) help?
  //     What would happen if it materialized the predicate tensor eagerly?
  //     Answer: ______________________________________________________
  // ─────────────────────────────────────────────────────────────────
  return 0;
}
