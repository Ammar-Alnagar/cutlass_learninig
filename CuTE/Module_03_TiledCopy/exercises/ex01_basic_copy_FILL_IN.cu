/*
 * EXERCISE: Basic TiledCopy - Fill in the Gaps
 *
 * WHAT THIS TEACHES:
 *   - Construct TiledCopy with make_tiled_copy
 *   - Understand Copy_Atom (what size transfer per thread)
 *   - Execute copy() to move data from gmem to smem
 *
 * MENTAL MODEL:
 *   make_tiled_copy(Copy_Atom, thread_layout, value_layout) creates a copy
 * operator
 *   - Copy_Atom:      the hardware instruction + dtype (what moves, how wide)
 *   - thread_layout:  how threads map to tile positions (who owns what)
 *   - value_layout:   how many elements one thread moves per Copy_Atom call
 *   copy(tiled_copy, src, dst) fires the Copy_Atom across all threads
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: Basic TiledCopy gmem -> smem
// ============================================================================
__global__ void basic_copy_kernel(float *gmem_data) {

  // ── Source tensor in global memory [8, 16] = 128 floats ──────────────────
  auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
  auto gmem_layout =
      make_layout(make_shape(Int<8>{}, Int<16>{}), GenRowMajor{});
  auto gmem_tensor = make_tensor(gmem_ptr, gmem_layout);

  // ── Shared memory destination [8, 16] ────────────────────────────────────
  __shared__ float smem_static[8 * 16];
  auto smem_ptr = make_smem_ptr<float>(smem_static);
  // auto smem_tensor = ; // same layout as gmem

  // ── Init gmem on thread 0 only (avoids 128-way race) ─────────────────────
  if (threadIdx.x == 0) {
    for (int i = 0; i < 8 * 16; i++)
      gmem_data[i] = i + 1.0f; // [1, 2, 3, ..., 128]
  }
  __syncthreads();

  // ── Build TiledCopy ───────────────────────────────────────────────────────
  //   Copy_Atom<UniversalCopy<uint32_t>, float>
  //     └─ UniversalCopy<uint32_t>: plain scalar load/store, 32-bit width
  //     └─ float: the element dtype
  //   make_layout(Int<128>{}): flat 1-D thread layout, thread i → position i
  //   make_layout(Int<1>{}):   each thread moves 1 element per atom call
  //
  //   Result: 128 threads × 1 element = 128 elements per copy() call
  //   → exactly covers the 8×16 = 128-element tile, no gaps, no overlaps
  // auto tiled_copy = ;

  if (threadIdx.x == 0) {
    printf("=== TiledCopy Setup ===\n");
    printf("Source : gmem [8, 16] = 128 floats\n");
    printf("Dest   : smem [8, 16] = 128 floats\n");
    printf(
        "Threads: 128 (each copies 1 float via UniversalCopy<uint32_t>)\n\n");
  }

  // ── Execute the copy ──────────────────────────────────────────────────────
  //   Step 1: extract this thread's personal assignment from the tiling map
  // auto thr_copy = ;

  //   Step 2: apply that assignment to the actual tensors
  //           src / dst are now zero-copy views of only *this thread's*
  //           elements
  // auto src = ; // my gmem slice
  // auto dst = ; // my smem slice

  //   Step 3: fire — Copy_Atom executes on each thread's (src, dst) pair

  __syncthreads();

  // ── Verify ────────────────────────────────────────────────────────────────
  if (threadIdx.x == 0) {
    printf("=== Copy Verification ===\n");
    printf("smem[0,0]  = %.1f  (expected  1.0)\n", float(smem_tensor(0, 0)));
    printf("smem[0,15] = %.1f  (expected 16.0)\n", float(smem_tensor(0, 15)));
    printf("smem[1,0]  = %.1f  (expected 17.0)\n", float(smem_tensor(1, 0)));
    printf("smem[7,15] = %.1f  (expected 128.0)\n", float(smem_tensor(7, 15)));
  }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== Basic TiledCopy Exercise ===\n");
  printf("GPU: %s (SM %d%d)\n\n", prop.name, prop.major, prop.minor);

  constexpr int SIZE = 8 * 16;
  float *d_data;
  cudaMalloc(&d_data, SIZE * sizeof(float));
  cudaMemset(d_data, 0,
             SIZE * sizeof(float)); // zero so a silent crash is visible

  // ── Warmup ────────────────────────────────────────────────────────────────
  basic_copy_kernel<<<1, 128, SIZE * sizeof(float)>>>(d_data);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Warmup launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Warmup execution error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  // ── Timed run ─────────────────────────────────────────────────────────────
  printf("--- Kernel Output ---\n");
  nvtxRangePush("basic_copy_kernel");
  basic_copy_kernel<<<1, 128, SIZE * sizeof(float)>>>(d_data);
  nvtxRangePop();

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  // ── Host-side verification ────────────────────────────────────────────────
  float h_data[SIZE];
  cudaMemcpy(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\n=== CPU Reference ===\n");
  printf("Expected: smem[i] = gmem[i] for all i\n");
  printf("gmem initialized to [1, 2, 3, ..., 128]\n");

  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    if (h_data[i] != i + 1.0f) {
      printf("MISMATCH at index %d: got %.1f expected %.1f\n", i, h_data[i],
             i + 1.0f);
      pass = false;
      break;
    }
  }

  printf("\n[%s] Basic TiledCopy verified\n", pass ? "PASS" : "FAIL");

  cudaFree(d_data);
  return pass ? 0 : 1;
}

/*
 * CHECKPOINT ANSWERS:
 *
 * Q1: What are the three arguments to make_tiled_copy?
 *     Copy_Atom (instruction + dtype), thread_layout (who owns what position),
 *     value_layout (how many elements per thread per atom call)
 *
 * Q2: What does copy(tiled_copy, src, dst) do?
 *     Fires the Copy_Atom on each thread's partitioned (src, dst) slice.
 *     src/dst must already be partitioned via get_thread_slice + partition_S/D.
 *
 * Q3: Why use shared memory instead of reading directly from gmem in MMA?
 *     gmem latency = 400-800 cycles; smem = ~30 cycles.
 *     ldmatrix (smem->reg for MMA operands) cannot target gmem directly.
 *     Staging through smem also enables cp.async overlap in the pipeline.
 */
