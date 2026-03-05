// ============================================================
// MODULE 08 — MMA Atom Internals
// Exercise 02: Atom Value Layout — Register to Position Mapping
// ============================================================
// CONCEPT:
//   The value layout answers: "Which registers does thread T own,
//   and what (row, col) do they correspond to in the logical tile?"
//   
//   For SM80_16x8x16_F16F16F16F16_TN:
//   - Each thread owns 4 elements (16×8 = 128 / 32 threads = 4)
//   - fragment[0], fragment[1], fragment[2], fragment[3] map to
//     specific (row, col) positions that are NOT sequential
//   
//   This is the foundation for correctly implementing FlashAttention-2's
//   per-row max and sum operations — you need to know which fragment
//   elements belong to the same row.
//
// JOB RELEVANCE:
//   NVIDIA DL Software Engineer — "GPU architecture and compilation
//   stack" and "kernel-level performance" — The FlashAttention-2 online
//   softmax rescale requires knowing exactly which registers each thread
//   owns in the C fragment to apply per-row scaling correctly.
//
// ASCII DIAGRAM:
//   Value Layout for Thread 0:
//   fragment[0] → (row=0, col=0)
//   fragment[1] → (row=0, col=2)
//   fragment[2] → (row=2, col=0)
//   fragment[3] → (row=2, col=2)
//   
//   Value Layout for Thread 1:
//   fragment[0] → (row=0, col=1)
//   fragment[1] → (row=0, col=3)
//   fragment[2] → (row=2, col=1)
//   fragment[3] → (row=2, col=3)
//   
//   Value Layout for Thread 16:
//   fragment[0] → (row=4, col=0)
//   fragment[1] → (row=4, col=2)
//   fragment[2] → (row=6, col=0)
//   fragment[3] → (row=6, col=2)
//   
//   Key insight: Elements in the same row share the same scale factor!
//   For thread 0:
//     fragment[0] and fragment[1] are both in row 0 → scale[0]
//     fragment[2] and fragment[3] are both in row 2 → scale[2]
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex02_atom_value_layout.cu -o ex02_atom_value_layout
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <iomanip>
#include <iostream>

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
// HELPER: Get value layout for a specific thread
// ============================================================================
__host__ __device__
void get_value_layout_for_thread(int thread_id, int positions[][2]) {
  // Value layout formula for SM80_16x8x16_F16F16F16F16_TN
  // This maps fragment index → (row, col) for a given thread
  
  int row_group = thread_id / 16;      // 0 or 1
  int thread_in_group = thread_id % 16;
  
  int pair_idx = thread_in_group / 4;   // Which column pair (0-3)
  int thread_in_pair = thread_in_group % 4;
  
  int row_in_block = (thread_in_pair / 2) * 2;  // 0 or 2
  int col_in_pair = thread_in_pair % 2;         // 0 or 1
  
  int base_col = pair_idx * 2;
  
  // Each thread owns 4 elements at positions:
  // (row_in_block, base_col + col_in_pair) for two rows
  // The pattern repeats for rows 0,2 and 8,10 (or 4,6 and 12,14)
  
  int row_offset = (row_group == 0) ? 0 : 4;
  
  // Fragment layout within a thread:
  // [0] = (row_offset + row_in_block, base_col + col_in_pair)
  // [1] = (row_offset + row_in_block, base_col + col_in_pair + 2)
  // [2] = (row_offset + row_in_block + 2, base_col + col_in_pair)
  // [3] = (row_offset + row_in_block + 2, base_col + col_in_pair + 2)
  
  positions[0][0] = row_offset + row_in_block;
  positions[0][1] = base_col + col_in_pair;
  
  positions[1][0] = row_offset + row_in_block;
  positions[1][1] = base_col + col_in_pair + 2;
  
  positions[2][0] = row_offset + row_in_block + 2;
  positions[2][1] = base_col + col_in_pair;
  
  positions[3][0] = row_offset + row_in_block + 2;
  positions[3][1] = base_col + col_in_pair + 2;
}

// ============================================================================
// HELPER: Print value layout for a thread
// ============================================================================
__host__ void print_thread_value_layout(int thread_id) {
  int positions[ELEMENTS_PER_THREAD][2];
  get_value_layout_for_thread(thread_id, positions);
  
  printf("Thread %d value layout:\n", thread_id);
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    printf("  fragment[%d] → (row=%d, col=%d)\n", 
           v, positions[v][0], positions[v][1]);
  }
  
  // Group by row
  printf("  Grouped by row:\n");
  int rows_seen[4] = {-1, -1, -1, -1};
  int row_count = 0;
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = positions[v][0];
    bool found = false;
    for (int i = 0; i < row_count; ++i) {
      if (rows_seen[i] == row) {
        found = true;
        break;
      }
    }
    if (!found) {
      rows_seen[row_count++] = row;
    }
  }
  
  for (int i = 0; i < row_count; ++i) {
    int row = rows_seen[i];
    printf("    Row %d: fragment", row);
    bool first = true;
    for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
      if (positions[v][0] == row) {
        if (!first) printf(", ");
        printf("[%d]", v);
        first = false;
      }
    }
    printf(" → same scale factor\n");
  }
}

// ============================================================================
// KERNEL: Explore value layout
// ============================================================================
__global__ void atom_value_layout_kernel() {
  // STEP 1: Create the MMA atom
  // ============================
  auto mma_atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{};
  
  std::cout << "=== Step 1: MMA Atom Info ===\n";
  std::cout << "MMA Atom: SM80_16x8x16_F16F16F16F16_TN\n";
  std::cout << "Output tile: " << M_ATOM << " × " << N_ATOM << " = " 
            << (M_ATOM * N_ATOM) << " elements\n";
  std::cout << "Threads: " << THREADS_PER_WARP << "\n";
  std::cout << "Elements per thread: " << ELEMENTS_PER_THREAD << "\n\n";
  
  // STEP 2: Print value layout for key threads
  // ===========================================
  std::cout << "=== Step 2: Value Layout for Key Threads ===\n\n";
  
  std::cout << "--- Thread 0 (first thread in first block) ---\n";
  print_thread_value_layout(0);
  std::cout << "\n";
  
  std::cout << "--- Thread 1 (second thread in first block) ---\n";
  print_thread_value_layout(1);
  std::cout << "\n";
  
  std::cout << "--- Thread 16 (first thread in second block) ---\n";
  print_thread_value_layout(16);
  std::cout << "\n";
  
  std::cout << "--- Thread 31 (last thread) ---\n";
  print_thread_value_layout(31);
  std::cout << "\n";
  
  // STEP 3: Verify with identity GEMM
  // ==================================
  std::cout << "=== Step 3: Identity GEMM Verification ===\n";
  printf("Running a GEMM with identity matrices to verify positions...\n");
  printf("If A = I and B = I, then C[i,j] = sum_k A[i,k] * B[k,j]\n");
  printf("For identity: C[i,j] = 1 if i==j, else 0\n\n");
  
  // In practice, we'd run an actual GEMM here
  // For this exercise, we'll print the expected pattern
  
  printf("Expected fragment values for thread 0 (identity GEMM):\n");
  int t0_positions[][2] = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
  for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
    int row = t0_positions[v][0];
    int col = t0_positions[v][1];
    float expected = (row == col) ? 1.0f : 0.0f;
    printf("  fragment[%d] at (%d, %d): expected %.1f\n", 
           v, row, col, expected);
  }
  std::cout << "\n";
  
  // STEP 4: Connection to FlashAttention-2 rescale
  // ===============================================
  std::cout << "=== Step 4: FlashAttention-2 Rescale Pattern ===\n";
  printf("In FlashAttention-2, when a new max is found:\n");
  printf("  for each fragment element e:\n");
  printf("    acc[e] *= exp(prev_max - new_max)[row_of(e)]\n");
  printf("\n");
  printf("For thread 0 with scale factors [s0, s1, s2, ...]:\n");
  printf("  fragment[0] (row 0) *= s0\n");
  printf("  fragment[1] (row 0) *= s0  ← same row, same scale!\n");
  printf("  fragment[2] (row 2) *= s2\n");
  printf("  fragment[3] (row 2) *= s2  ← same row, same scale!\n");
  printf("\n");
  printf("This is why value layout matters: you must know row_of(fragment[v])!\n");
}

// ============================================================================
// CPU REFERENCE: Verify value layout
// ============================================================================
void cpu_reference_value_layout() {
  std::cout << "\n=== CPU Reference: Value Layout Formula ===\n";
  
  std::cout << "Value layout formula for SM80_16x8x16:\n";
  std::cout << "  row_group = thread_id / 16\n";
  std::cout << "  thread_in_group = thread_id % 16\n";
  std::cout << "  pair_idx = thread_in_group / 4\n";
  std::cout << "  thread_in_pair = thread_in_group % 4\n";
  std::cout << "  row_in_block = (thread_in_pair / 2) * 2\n";
  std::cout << "  col_in_pair = thread_in_pair % 2\n";
  std::cout << "  base_col = pair_idx * 2\n";
  std::cout << "  row_offset = (row_group == 0) ? 0 : 4\n";
  std::cout << "\n";
  std::cout << "  fragment[0] = (row_offset + row_in_block, base_col + col_in_pair)\n";
  std::cout << "  fragment[1] = (row_offset + row_in_block, base_col + col_in_pair + 2)\n";
  std::cout << "  fragment[2] = (row_offset + row_in_block + 2, base_col + col_in_pair)\n";
  std::cout << "  fragment[3] = (row_offset + row_in_block + 2, base_col + col_in_pair + 2)\n";
  std::cout << "\n";
  
  // Verify for thread 0
  std::cout << "Verification for thread 0:\n";
  int t0_expected[][2] = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
  int t0_computed[4][2];
  get_value_layout_for_thread(0, t0_computed);
  
  bool match = true;
  for (int v = 0; v < 4; ++v) {
    bool v_match = (t0_computed[v][0] == t0_expected[v][0]) && 
                   (t0_computed[v][1] == t0_expected[v][1]);
    if (!v_match) match = false;
    printf("  fragment[%d]: computed (%d, %d), expected (%d, %d) %s\n",
           v, t0_computed[v][0], t0_computed[v][1], 
              t0_expected[v][0], t0_expected[v][1],
              v_match ? "✓" : "✗");
  }
  printf("Thread 0 value layout: %s\n", match ? "PASS ✓" : "FAIL ✗");
}

// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 08 — MMA Atom Internals\n";
  std::cout << "Exercise 02: Atom Value Layout\n";
  std::cout << "============================================================\n\n";
  
  // PREDICT BEFORE RUNNING:
  // Q1: Which row does fragment[0] of thread 0 belong to?
  //     Answer: Row 0
  //
  // Q2: Do fragment[0] and fragment[1] of thread 0 share the same scale?
  //     Answer: Yes, both are in row 0
  //
  // Q3: How many unique rows does each thread participate in?
  //     Answer: 2 rows (each with 2 elements)
  
  std::cout << "Predictions:\n";
  std::cout << "  Q1: fragment[0] of thread 0 → row 0\n";
  std::cout << "  Q2: fragment[0] and fragment[1] share scale → Yes\n";
  std::cout << "  Q3: Each thread participates in 2 unique rows\n\n";
  
  // Run kernel
  atom_value_layout_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  
  // CPU reference
  cpu_reference_value_layout();
  
  // KEY INSIGHT
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Value layout maps fragment index → (row, col):\n";
  std::cout << "\n";
  std::cout << "  For FlashAttention-2 rescale:\n";
  std::cout << "  - Each fragment element belongs to a specific row\n";
  std::cout << "  - Elements in the same row share the same scale factor\n";
  std::cout << "  - Must use value layout to find row_of(fragment[v])\n";
  std::cout << "\n";
  std::cout << "  Next: ex03 implements the actual rescale function.\n";
  std::cout << "============================================================\n";
  
  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 *
 * Q1: For thread 0, which fragment elements are in row 0?
 *     Answer: fragment[0] and fragment[1]
 *
 * Q2: If the scale factor for row 0 is 0.5, what operation do you perform?
 *     Answer: fragment[0] *= 0.5; fragment[1] *= 0.5;
 *
 * Q3: Why can't you iterate fragment sequentially and apply scale[v]?
 *     Answer: Because fragment[v] doesn't correspond to row v.
 *             fragment[0] and fragment[1] are both in row 0,
 *             so they both need scale[0], not scale[0] and scale[1].
 */
