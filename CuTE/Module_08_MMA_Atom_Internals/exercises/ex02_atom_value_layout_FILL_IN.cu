// ============================================================
// MODULE 08 — MMA Atom Internals
// Exercise 02: Atom Value Layout — Register to Position Mapping
// FILL-IN VERSION — Complete the TODO sections
// ============================================================
// CONCEPT:
//   Value layout answers: "Which registers does thread T own,
//   and what (row, col) do they correspond to in the logical tile?"
//   For SM80_16x8x16: each thread owns 4 elements, not in sequential order.
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex02_atom_value_layout_FILL_IN.cu -o ex02_atom_value_layout_FILL_IN
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <iostream>

using namespace cute;

constexpr int M_ATOM = 16;
constexpr int N_ATOM = 8;
constexpr int K_ATOM = 16;
constexpr int THREADS_PER_WARP = 32;
constexpr int ELEMENTS_PER_THREAD = (M_ATOM * N_ATOM) / THREADS_PER_WARP;

// ============================================================================
// TODO: Add device function to get value layout for a thread
// Hint: For SM80_16x8x16, fragment indices map to (row, col) as:
//   [0] = (row_offset + row_in_block, base_col + col_in_pair)
//   [1] = (row_offset + row_in_block, base_col + col_in_pair + 2)
//   [2] = (row_offset + row_in_block + 2, base_col + col_in_pair)
//   [3] = (row_offset + row_in_block + 2, base_col + col_in_pair + 2)
// ============================================================================
__host__ __device__
void get_value_layout_for_thread(int thread_id, int positions[][2]) {
  // TODO: Implement value layout formula
  int row_group = /* TODO: thread_id / 16 */;
  int thread_in_group = /* TODO: thread_id % 16 */;
  
  int pair_idx = /* TODO: thread_in_group / 4 */;
  int thread_in_pair = /* TODO: thread_in_group % 4 */;
  
  int row_in_block = /* TODO: (thread_in_pair / 2) * 2 */;
  int col_in_pair = /* TODO: thread_in_pair % 2 */;
  
  int base_col = /* TODO: pair_idx * 2 */;
  int row_offset = /* TODO: (row_group == 0) ? 0 : 4 */;
  
  // TODO: Fill in positions array for 4 fragment elements
  positions[0][0] = /* TODO: row_offset + row_in_block */;
  positions[0][1] = /* TODO: base_col + col_in_pair */;
  
  positions[1][0] = /* TODO: row_offset + row_in_block */;
  positions[1][1] = /* TODO: base_col + col_in_pair + 2 */;
  
  positions[2][0] = /* TODO: row_offset + row_in_block + 2 */;
  positions[2][1] = /* TODO: base_col + col_in_pair */;
  
  positions[3][0] = /* TODO: row_offset + row_in_block + 2 */;
  positions[3][1] = /* TODO: base_col + col_in_pair + 2 */;
}

// ============================================================================
// TODO: Add helper to print value layout for a thread
// Hint: Print fragment[v] → (row, col) for v = 0,1,2,3
//       Then group by row to show which fragments share scale factor
// ============================================================================


// ============================================================================
// KERNEL: Explore value layout
// ============================================================================
__global__ void atom_value_layout_kernel() {
  // STEP 1: Create the MMA atom
  // ============================
  auto mma_atom = /* TODO: MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{} */;
  
  std::cout << "=== Step 1: MMA Atom Info ===\n";
  std::cout << "Output tile: " << M_ATOM << " × " << N_ATOM 
            << " = " << (M_ATOM * N_ATOM) << " elements\n";
  std::cout << "Threads: " << THREADS_PER_WARP << "\n";
  std::cout << "Elements per thread: " << ELEMENTS_PER_THREAD << "\n\n";
  
  // STEP 2: Print value layout for key threads
  // ===========================================
  std::cout << "=== Step 2: Value Layout for Key Threads ===\n\n";
  
  std::cout << "--- Thread 0 ---\n";
  // TODO: Print value layout for thread 0
  
  std::cout << "--- Thread 1 ---\n";
  // TODO: Print value layout for thread 1
  
  std::cout << "--- Thread 16 ---\n";
  // TODO: Print value layout for thread 16
  
  // STEP 3: Verify with identity GEMM concept
  // ==========================================
  std::cout << "\n=== Step 3: Identity GEMM Verification ===\n";
  printf("If A = I and B = I, then C[i,j] = 1 if i==j, else 0\n\n");
  
  printf("Expected fragment values for thread 0 (identity GEMM):\n");
  // TODO: For each fragment, print expected value (1.0 if row==col, else 0.0)
  
  // STEP 4: Connection to FlashAttention-2 rescale
  // ===============================================
  std::cout << "\n=== Step 4: FlashAttention-2 Rescale Pattern ===\n";
  printf("When a new max is found:\n");
  printf("  for each fragment element e:\n");
  printf("    acc[e] *= exp(prev_max - new_max)[row_of(e)]\n");
  printf("\n");
  printf("For thread 0 with scale factors [s0, s1, s2, ...]:\n");
  // TODO: Print which scale factor each fragment uses
}

// ============================================================================
// TODO: Add CPU reference function to verify value layout
// Hint: For thread 0, expected positions are (0,0), (0,2), (2,0), (2,2)
// ============================================================================


// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 08 — MMA Atom Internals\n";
  std::cout << "Exercise 02: Atom Value Layout (FILL-IN)\n";
  std::cout << "============================================================\n\n";
  
  // PREDICTIONS:
  // Q1: Which row does fragment[0] of thread 0 belong to?
  //     Your answer: ___
  //
  // Q2: Do fragment[0] and fragment[1] of thread 0 share the same scale?
  //     Your answer: ___
  //
  // Q3: How many unique rows does each thread participate in?
  //     Your answer: ___
  
  std::cout << "Your predictions:\n";
  std::cout << "  Q1: ___\n";
  std::cout << "  Q2: ___\n";
  std::cout << "  Q3: ___\n\n";
  
  // Run kernel
  atom_value_layout_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  
  // TODO: Call CPU reference function
  
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Value layout maps fragment index → (row, col)\n";
  std::cout << "  Elements in same row share same scale factor\n";
  std::cout << "  Next: ex03 implements the actual rescale function\n";
  std::cout << "============================================================\n";
  
  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 *
 * Q1: For thread 0, which fragment elements are in row 0?
 *     Your answer: ___
 *
 * Q2: If scale factor for row 0 is 0.5, what operation?
 *     Your answer: ___
 *
 * Q3: Why can't you iterate fragment sequentially with scale[v]?
 *     Your answer: ___
 */
