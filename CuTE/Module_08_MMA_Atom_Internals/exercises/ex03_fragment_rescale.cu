// ============================================================
// MODULE 08 — MMA Atom Internals
// Exercise 03: Fragment Rescale — FlashAttention-2 Online Softmax
// ============================================================
// CONCEPT:
//   Implement the FlashAttention-2 rescale step using the value layout.
//   Operation: given an existing accumulator fragment acc_o and a
//   per-row scale factor alpha, compute acc_o[i] *= alpha[row_of(i)]
//   for each fragment element.
//   
//   This is the EXACT function that FlashAttention-2 calls between
//   KV blocks to correct the running sum when a new max is found.
//
// JOB RELEVANCE:
//   NVIDIA DL Software Engineer — "Efficient attention kernels" —
//   The FlashAttention-2 online softmax rescale requires knowing
//   exactly which registers each thread owns to apply per-row scaling.
//   This is production-level code for LLM inference.
//
// ASCII DIAGRAM:
//   FlashAttention-2 Online Softmax (simplified):
//   
//   Initialize:
//     acc_o = 0 (accumulator for output)
//     m_i = -inf (running max)
//     ell_i = 0 (running sum of exp)
//   
//   For each KV block:
//     Compute QK^T scores for this block
//     m_block = max(scores in block)
//     m_new = max(m_i, m_block)
//   
//     RESCALE STEP:
//     alpha = exp(m_i - m_new)  // Scale factor for previous accumulator
//     for each fragment element e in acc_o:
//       acc_o[e] *= alpha[row_of(e)]  // ← This is what we implement!
//   
//     Update running sum:
//     ell_i = ell_i * alpha + sum(exp(scores - m_new))
//   
//     Accumulate weighted V:
//     acc_o += exp(scores - m_new) @ V
//     m_i = m_new
//   
//   Final normalization:
//     O = acc_o / ell_i
//   
//   Key: row_of(e) comes from the VALUE LAYOUT, not fragment index!
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex03_fragment_rescale.cu -o ex03_fragment_rescale
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <iomanip>
#include <iostream>
#include <cmath>

using namespace cute;

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr int M_ATOM = 16;
constexpr int N_ATOM = 8;
constexpr int K_ATOM = 16;
constexpr int THREADS_PER_WARP = 32;
constexpr int ELEMENTS_PER_THREAD = (M_ATOM * N_ATOM) / THREADS_PER_WARP;  // = 4

// ============================================================================
// DEVICE FUNCTION: Get row for a fragment element
// ============================================================================
__device__ __host__
int get_row_for_fragment_element(int thread_id, int fragment_idx) {
  // Value layout formula for SM80_16x8x16_F16F16F16F16F16_TN
  // Returns the row index for fragment[fragment_idx] of thread[thread_id]
  
  int row_group = thread_id / 16;
  int thread_in_group = thread_id % 16;
  
  int pair_idx = thread_in_group / 4;
  int thread_in_pair = thread_in_group % 4;
  
  int row_in_block = (thread_in_pair / 2) * 2;
  int row_offset = (row_group == 0) ? 0 : 4;
  
  // fragment[0] and [1] are in row (row_offset + row_in_block)
  // fragment[2] and [3] are in row (row_offset + row_in_block + 2)
  
  int base_row = row_offset + row_in_block;
  int row = (fragment_idx < 2) ? base_row : base_row + 2;
  
  return row;
}

// ============================================================================
// DEVICE FUNCTION: Rescale output fragment (FlashAttention-2 pattern)
// ============================================================================
template <int ElementsPerThread>
__device__ void rescale_output_fragment(float acc[], float scale_factors[],
                                         int thread_id) {
  // FlashAttention-2 rescale: acc[e] *= alpha[row_of(e)]
  // scale_factors[row] contains the per-row scale factor
  
  #pragma unroll
  for (int v = 0; v < ElementsPerThread; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    acc[v] *= scale_factors[row];
  }
}

// ============================================================================
// HELPER: Print fragment before and after rescale
// ============================================================================
__host__ void print_fragment_rescale(int thread_id, float acc_before[], 
                                      float acc_after[], float scale_factors[]) {
  printf("Thread %d fragment rescale:\n", thread_id);
  printf("  Scale factors: [");
  for (int r = 0; r < M_ATOM; ++r) {
    printf("%.3f ", scale_factors[r]);
  }
  printf("]\n");
  
  printf("  Before rescale:\n");
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    printf("    fragment[%d] at row %d: %.4f\n", v, row, acc_before[v]);
  }
  
  printf("  After rescale:\n");
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    printf("    fragment[%d] at row %d: %.4f (× %.3f)\n", 
           v, row, acc_after[v], scale_factors[row]);
  }
}

// ============================================================================
// KERNEL: Demonstrate fragment rescale
// ============================================================================
__global__ void fragment_rescale_kernel() {
  // STEP 1: Setup test data
  // ========================
  // Simulate a thread's accumulator fragment before rescale
  
  int thread_id = threadIdx.x;
  if (thread_id >= THREADS_PER_WARP) return;
  
  // Each thread has 4 accumulator elements
  float acc[ELEMENTS_PER_THREAD];
  float acc_original[ELEMENTS_PER_THREAD];
  
  // Initialize with thread-specific values for verification
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    acc[v] = static_cast<float>(thread_id * 10 + v + 1);
    acc_original[v] = acc[v];
  }
  
  // Per-row scale factors (simulating exp(prev_max - new_max))
  float scale_factors[M_ATOM];
  for (int r = 0; r < M_ATOM; ++r) {
    // Use different scale factors per row for verification
    scale_factors[r] = 0.1f * static_cast<float>((r % 4) + 1);
  }
  
  // STEP 2: Apply rescale
  // ======================
  rescale_output_fragment<ELEMENTS_PER_THREAD>(acc, scale_factors, thread_id);
  
  // STEP 3: Print results for key threads
  // ======================================
  if (thread_id == 0 || thread_id == 1 || thread_id == 16) {
    print_fragment_rescale(thread_id, acc_original, acc, scale_factors);
    printf("\n");
  }
  
  // STEP 4: Verify correctness
  // ===========================
  bool pass = true;
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    float expected = acc_original[v] * scale_factors[row];
    float actual = acc[v];
    float error = fabsf(actual - expected);
    if (error > 1e-5f) {
      pass = false;
      printf("Thread %d fragment %d: expected %.6f, got %.6f\n",
             thread_id, v, expected, actual);
    }
  }
  
  if (pass && (thread_id == 0 || thread_id == 16)) {
    printf("Thread %d rescale verification: PASS ✓\n", thread_id);
  }
}

// ============================================================================
// CPU REFERENCE: Verify rescale computation
// ============================================================================
void cpu_reference_rescale() {
  std::cout << "\n=== CPU Reference: Fragment Rescale ===\n";
  
  // Verify for thread 0
  int thread_id = 0;
  float acc[ELEMENTS_PER_THREAD];
  float scale_factors[M_ATOM];
  
  // Initialize
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    acc[v] = static_cast<float>(thread_id * 10 + v + 1);
  }
  for (int r = 0; r < M_ATOM; ++r) {
    scale_factors[r] = 0.1f * static_cast<float>((r % 4) + 1);
  }
  
  std::cout << "Thread 0 before rescale:\n";
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    std::cout << "  fragment[" << v << "] at row " << row 
              << ": " << acc[v] << "\n";
  }
  
  // Apply rescale
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    acc[v] *= scale_factors[row];
  }
  
  std::cout << "Thread 0 after rescale:\n";
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = get_row_for_fragment_element(thread_id, v);
    std::cout << "  fragment[" << v << "] at row " << row 
              << ": " << acc[v] << " (× " << scale_factors[row] << ")\n";
  }
  
  // Verify specific values
  std::cout << "\nVerification:\n";
  // fragment[0] = 1, row 0, scale[0] = 0.1 → expected 0.1
  float expected_0 = 1.0f * 0.1f;
  std::cout << "  fragment[0]: expected " << expected_0 
            << ", formula: 1.0 × scale[0] = 1.0 × 0.1\n";
  
  // fragment[2] = 3, row 2, scale[2] = 0.3 → expected 0.9
  float expected_2 = 3.0f * 0.3f;
  std::cout << "  fragment[2]: expected " << expected_2 
            << ", formula: 3.0 × scale[2] = 3.0 × 0.3\n";
}

// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 08 — MMA Atom Internals\n";
  std::cout << "Exercise 03: Fragment Rescale\n";
  std::cout << "============================================================\n\n";
  
  // PREDICT BEFORE RUNNING:
  // Q1: If fragment[0] = 10 and scale[row_0] = 0.5, what is the result?
  //     Answer: 10 × 0.5 = 5.0
  //
  // Q2: Do fragment[0] and fragment[1] of thread 0 use the same scale?
  //     Answer: Yes, both are in row 0
  //
  // Q3: What is the scale factor formula in FlashAttention-2?
  //     Answer: alpha = exp(prev_max - new_max)
  
  std::cout << "Predictions:\n";
  std::cout << "  Q1: fragment[0] = 5.0 after rescale\n";
  std::cout << "  Q2: fragment[0] and fragment[1] share scale → Yes\n";
  std::cout << "  Q3: alpha = exp(prev_max - new_max)\n\n";
  
  // Run kernel
  fragment_rescale_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  
  // CPU reference
  cpu_reference_rescale();
  
  // KEY INSIGHT
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  FlashAttention-2 fragment rescale:\n";
  std::cout << "\n";
  std::cout << "  The rescale operation:\n";
  std::cout << "    for each fragment element e:\n";
  std::cout << "      acc[e] *= alpha[row_of(e)]\n";
  std::cout << "\n";
  std::cout << "  Where alpha = exp(prev_max - new_max) per row\n";
  std::cout << "  And row_of(e) comes from the VALUE LAYOUT\n";
  std::cout << "\n";
  std::cout << "  This is the EXACT function called between KV blocks\n";
  std::cout << "  in production FlashAttention-2 implementations!\n";
  std::cout << "============================================================\n";
  
  return 0;
}

/*
 * CHECKPOINT: You've completed Module 08!
 *
 * Q1: What is the purpose of the rescale operation in FlashAttention-2?
 *     Answer: When a new max is found in a KV block, the previous
 *             accumulator must be scaled by exp(prev_max - new_max)
 *             to maintain numerical stability in the online softmax.
 *
 * Q2: Why can't you simply do acc[v] *= scale[v]?
 *     Answer: Because fragment index v doesn't correspond to row v.
 *             The value layout determines which row each fragment
 *             element belongs to.
 *
 * Q3: What's the connection between thread layout and value layout?
 *     Answer: Thread layout maps thread_id → output positions.
 *             Value layout maps (thread_id, fragment_idx) → (row, col).
 *             Both are needed for correct FlashAttention-2 implementation.
 *
 * You are now ready for the FlashAttention-2 capstone project!
 */
