// ============================================================
// MODULE 08 — MMA Atom Internals
// Exercise 01: Atom Thread Layout — Which Thread Owns Which Position
// FILL-IN VERSION — Complete the TODO sections
// ============================================================
// CONCEPT:
//   Thread layout of SM80_16x8x16_F16F16F16F16_TN answers:
//   "Which of the 32 warp threads participates in which row/column
//   of the 16×8 output tile?"
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex01_atom_thread_layout_FILL_IN.cu -o ex01_atom_thread_layout_FILL_IN
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <iostream>

using namespace cute;

constexpr int M_ATOM = 16;
constexpr int N_ATOM = 8;
constexpr int K_ATOM = 16;

// ============================================================================
// TODO: Add helper to print thread layout grid
// Hint: For each (row, col) in 16×8 output, print which thread owns it
// ============================================================================


// ============================================================================
// KERNEL: Explore thread layout
// ============================================================================
__global__ void atom_thread_layout_kernel() {
  // STEP 1: Create the MMA atom
  // ============================
  // TODO: Create MMA_Atom for SM80_16x8x16_F16F16F16F16_TN
  auto mma_atom = /* TODO: MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{} */;
  
  std::cout << "=== Step 1: MMA Atom Info ===\n";
  std::cout << "MMA Atom: SM80_16x8x16_F16F16F16F16_TN\n";
  std::cout << "Output tile: " << M_ATOM << " × " << N_ATOM << "\n";
  std::cout << "K dimension: " << K_ATOM << "\n\n";
  
  // STEP 2: Get the thread layout for the C fragment (output)
  // ==========================================================
  // TODO: Use layoutC_TV() to get thread layout for C fragment
  auto layout_C_TV = /* TODO: mma_atom.layoutC_TV() */;
  
  std::cout << "=== Step 2: C Fragment Thread Layout ===\n";
  // TODO: Print layout_C_TV
  
  // STEP 3: Print the thread layout grid
  // =====================================
  std::cout << "=== Step 3: Thread Layout Grid ===\n";
  // TODO: Print 16×8 grid with thread IDs at each position
  
  // STEP 4: Verify specific thread positions
  // =========================================
  std::cout << "=== Step 4: Thread Position Verification ===\n";
  
  printf("Thread 0 owns positions:\n");
  // TODO: Print positions for thread 0
  // Hint: (0,0), (0,2), (2,0), (2,2)
  
  printf("\nThread 1 owns positions:\n");
  // TODO: Print positions for thread 1
  // Hint: (0,1), (0,3), (2,1), (2,3)
  
  // STEP 5: Key insight for FlashAttention-2
  printf("\n=== Step 5: FlashAttention-2 Connection ===\n");
  printf("In FlashAttention-2, online softmax rescale requires:\n");
  printf("  for each fragment element e:\n");
  printf("    acc[e] *= scale[row_of(e)]\n");
  printf("\n");
  printf("You must use value layout to find row_of(fragment[i])!\n");
}

// ============================================================================
// TODO: Add CPU reference function to verify thread layout pattern
// Hint: Thread ID formula for SM80_16x8x16:
//   row_group = row / 4, row_in_group = row % 4
//   col_pair = col / 2, col_in_pair = col % 2
//   thread_id = (row_group % 2) * 16 + (row_in_group / 2) * 2 + col_in_pair + col_pair * 4
// ============================================================================


// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 08 — MMA Atom Internals\n";
  std::cout << "Exercise 01: Atom Thread Layout (FILL-IN)\n";
  std::cout << "============================================================\n\n";
  
  // PREDICTIONS:
  // Q1: How many elements does each thread own in 16×8 tile?
  //     Your answer: ___
  //
  // Q2: Which thread owns position (0, 0)?
  //     Your answer: ___
  //
  // Q3: Which thread owns position (4, 0)?
  //     Your answer: ___
  
  std::cout << "Your predictions:\n";
  std::cout << "  Q1: ___\n";
  std::cout << "  Q2: ___\n";
  std::cout << "  Q3: ___\n\n";
  
  // Run kernel
  atom_thread_layout_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  
  // TODO: Call CPU reference function
  
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Thread layout is NOT row-major!\n";
  std::cout << "  Why: Avoids bank conflicts, matches Tensor Core structure\n";
  std::cout << "  Next: ex02 explores value layout (register → position)\n";
  std::cout << "============================================================\n";
  
  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 *
 * Q1: How many elements does each thread own in SM80_16x8x16?
 *     Your answer: ___
 *
 * Q2: Which rows does thread 0 participate in?
 *     Your answer: ___
 *
 * Q3: Why isn't the thread layout row-major?
 *     Your answer: ___
 */
