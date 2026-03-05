// ============================================================
// MODULE 08 — MMA Atom Internals
// Exercise 03: Fragment Rescale — FlashAttention-2 Online Softmax
// FILL-IN VERSION — Complete the TODO sections
// ============================================================
// CONCEPT:
//   Implement FlashAttention-2 rescale: acc_o[i] *= alpha[row_of(i)]
//   for each fragment element. This is the EXACT function called
//   between KV blocks when a new max is found.
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex03_fragment_rescale_FILL_IN.cu -o ex03_fragment_rescale_FILL_IN
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <cmath>

using namespace cute;

constexpr int M_ATOM = 16;
constexpr int N_ATOM = 8;
constexpr int K_ATOM = 16;
constexpr int THREADS_PER_WARP = 32;
constexpr int ELEMENTS_PER_THREAD = (M_ATOM * N_ATOM) / THREADS_PER_WARP;

// ============================================================================
// TODO: Add device function to get row for a fragment element
// Hint: fragment[0,1] are in one row, fragment[2,3] are in another row
// ============================================================================
__device__ __host__
int get_row_for_fragment_element(int thread_id, int fragment_idx) {
  // TODO: Implement row lookup formula
  int row_group = /* TODO: thread_id / 16 */;
  int thread_in_group = /* TODO: thread_id % 16 */;
  int pair_idx = /* TODO: thread_in_group / 4 */;
  int thread_in_pair = /* TODO: thread_in_group % 4 */;
  
  int row_in_block = /* TODO: (thread_in_pair / 2) * 2 */;
  int row_offset = /* TODO: (row_group == 0) ? 0 : 4 */;
  
  int base_row = row_offset + row_in_block;
  // TODO: fragment[0,1] → base_row, fragment[2,3] → base_row + 2
  int row = /* TODO: (fragment_idx < 2) ? base_row : base_row + 2 */;
  
  return row;
}

// ============================================================================
// TODO: Add device function to rescale output fragment
// Hint: For each fragment element, lookup its row and apply scale[row]
// ============================================================================
template <int ElementsPerThread>
__device__ void rescale_output_fragment(float acc[], float scale_factors[],
                                         int thread_id) {
  // TODO: Implement rescale loop
  #pragma unroll
  for (int v = 0; v < ElementsPerThread; ++v) {
    int row = /* TODO: get_row_for_fragment_element(thread_id, v) */;
    acc[v] *= /* TODO: scale_factors[row] */;
  }
}

// ============================================================================
// TODO: Add helper to print fragment before and after rescale
// Hint: Print each fragment element with its row and applied scale
// ============================================================================


// ============================================================================
// KERNEL: Demonstrate fragment rescale
// ============================================================================
__global__ void fragment_rescale_kernel() {
  int thread_id = threadIdx.x;
  if (thread_id >= THREADS_PER_WARP) return;
  
  // Each thread has 4 accumulator elements
  float acc[ELEMENTS_PER_THREAD];
  float acc_original[ELEMENTS_PER_THREAD];
  
  // TODO: Initialize acc with thread-specific values
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    acc[v] = /* TODO: static_cast<float>(thread_id * 10 + v + 1) */;
    acc_original[v] = acc[v];
  }
  
  // Per-row scale factors (simulating exp(prev_max - new_max))
  float scale_factors[M_ATOM];
  // TODO: Initialize scale_factors with different values per row
  for (int r = 0; r < M_ATOM; ++r) {
    scale_factors[r] = /* TODO: 0.1f * static_cast<float>((r % 4) + 1) */;
  }
  
  // STEP 2: Apply rescale
  // ======================
  // TODO: Call rescale_output_fragment
  
  // STEP 3: Print results for key threads
  // ======================================
  if (thread_id == 0 || thread_id == 1 || thread_id == 16) {
    // TODO: Print fragment before and after rescale
  }
  
  // STEP 4: Verify correctness
  // ===========================
  bool pass = true;
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    float expected = acc_original[v] * scale_factors[row];
    float actual = acc[v];
    // TODO: Check if actual matches expected
  }
  
  if (pass && (thread_id == 0 || thread_id == 16)) {
    printf("Thread %d rescale verification: PASS\n", thread_id);
  }
}

// ============================================================================
// TODO: Add CPU reference function to verify rescale
// Hint: For thread 0, fragment[0]=1 at row 0, scale[0]=0.1 → expected 0.1
// ============================================================================


// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 08 — MMA Atom Internals\n";
  std::cout << "Exercise 03: Fragment Rescale (FILL-IN)\n";
  std::cout << "============================================================\n\n";
  
  // PREDICTIONS:
  // Q1: If fragment[0] = 10 and scale[row_0] = 0.5, what is result?
  //     Your answer: ___
  //
  // Q2: Do fragment[0] and fragment[1] of thread 0 use same scale?
  //     Your answer: ___
  //
  // Q3: What is scale factor formula in FlashAttention-2?
  //     Your answer: ___
  
  std::cout << "Your predictions:\n";
  std::cout << "  Q1: ___\n";
  std::cout << "  Q2: ___\n";
  std::cout << "  Q3: ___\n\n";
  
  // Run kernel
  fragment_rescale_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  
  // TODO: Call CPU reference function
  
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  FlashAttention-2 rescale operation:\n";
  std::cout << "    for each fragment element e:\n";
  std::cout << "      acc[e] *= alpha[row_of(e)]\n";
  std::cout << "\n";
  std::cout << "  Where alpha = exp(prev_max - new_max) per row\n";
  std::cout << "  This is called between KV blocks in production FA2!\n";
  std::cout << "============================================================\n";
  
  return 0;
}

/*
 * CHECKPOINT: You've completed Module 08!
 *
 * Q1: What is the purpose of rescale in FlashAttention-2?
 *     Your answer: ___
 *
 * Q2: Why can't you do acc[v] *= scale[v]?
 *     Your answer: ___
 *
 * Q3: What's the connection between thread layout and value layout?
 *     Your answer: ___
 *
 * You are now ready for the FlashAttention-2 capstone project!
 */
