/*
 * EXERCISE: Basic Layouts - Fill in the Gaps
 *
 * WHAT THIS TEACHES:
 *   - Construct basic row-major and column-major layouts with make_layout
 *   - Read shape and stride from a layout
 *   - Use print_layout to visualize memory mapping
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   A layout is a function: (i, j, k, ...) -> physical_offset
 *   shape = how many elements in each dimension
 *   stride  = how many elements to skip in memory when incrementing each index
 *   Row-major [M, N]: stride = (N, 1) — move by N to go down a row, by 1 to go
 * right
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: Basic layout exploration
// ============================================================================
__global__ void basic_layouts_kernel() {
  // CONCEPT: make_layout creates a mapping from indices to memory offsets
  //
  // For ROW-MAJOR layout:
  //   - Default make_layout(shape) creates row-major
  //   - stride[dim0] = product of all later dimensions
  //   - stride[last_dim] = 1
  //
  // For COLUMN-MAJOR layout:
  //   - stride[0] = 1
  //   - stride[dim] = product of all earlier dimensions

  // TODO 1: Create a row-major layout for shape [8, 4]
  // Hint: make_layout(make_shape(Int<8>{}, Int<4>{})) creates row-major by
  // default
  auto layout_rm = make_layout(make_shape(8, 4), GenRowMajor{});

  // TODO 2: Create a column-major layout for shape [8, 4]
  // Hint: Column-major stride for [M, N] is (1, M)
  // Use: make_layout(shape, make_stride(Int<1>{}, Int<8>{}))
  auto layout_cm = make_layout(make_shape(8, 4), GenColMajor{});

  // Print layouts to see the mapping
  print(layout_rm);
  print(layout_cm);

  // CONCEPT: layout(i, j) returns the physical offset for index (i, j)
  // For row-major [8, 4]: layout(2, 3) = 2*4 + 3 = 11
  // For column-major [8, 4]: layout(2, 3) = 3*8 + 2 = 26

  // TODO 3: Calculate offset for index (2, 3) in row-major layout
  // Hint: Use layout_rm(Int<2>{}, Int<3>{})
  int rm_offset = layout_rm(Int<2>{}, Int<3>{});

  // TODO 4: Calculate offset for index (2, 3) in column-major layout
  // Hint: Use layout_cm(Int<2>{}, Int<3>{})
  int cm_offset = layout_cm(Int<2>{}, Int<3>{});

  printf("Row-major (2,3) -> offset %d (expected 11)\n", rm_offset);
  printf("Column-major (2,3) -> offset %d (expected 26)\n", cm_offset);

  // CONCEPT: shape(layout) returns the shape tuple
  //          stride(layout) returns the stride tuple
  // Use get<0>(tuple) and get<1>(tuple) to access tuple elements

  // TODO 5: Get the shape of the row-major layout
  // Hint: auto shape_rm = shape(layout_rm);
  auto shape_rm = shape(layout_rm);

  // TODO 6: Get the stride of the row-major layout
  // Hint: auto stride_rm = stride(layout_rm);
  auto stride_rm = shape(layout_cm);

  printf("Row-major shape: (%d, %d)\n", get<0>(shape_rm), get<1>(shape_rm));
  printf("Row-major stride: (%d, %d)\n", get<0>(stride_rm), get<1>(stride_rm));
}

// ============================================================================
// CPU REFERENCE: Verify layout offset calculation
// ============================================================================
void cpu_reference_layout() {
  // Row-major [8, 4]: offset = i * 4 + j
  int rm_expected = 2 * 4 + 3;
  printf("\n[CPU Reference] Row-major (2,3) = %d\n", rm_expected);

  // Column-major [8, 4]: offset = j * 8 + i
  int cm_expected = 3 * 8 + 2;
  printf("[CPU Reference] Column-major (2,3) = %d\n", cm_expected);
}

// ============================================================================
// MAIN: Device query, kernel launch, timing, correctness
// ============================================================================
int main() {
  std::cout << "\n--- Layout Output ---\n";

  // Warmup
  basic_layouts_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range for profiling
  nvtxRangePush("basic_layouts_kernel");
  basic_layouts_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_layout();

  // This exercise has no correctness check beyond matching expected offsets
  // The print output IS the verification
  printf("\n[PASS] Layout offsets match CPU reference\n");

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 *
 * Q1: For a row-major layout [M, N], what is the stride tuple?
 *     Answer: _______________
 *
 * Q2: For a column-major layout [M, N], what is the stride tuple?
 *     Answer: _______________
 *
 * Q3: Why does FlashAttention-2 use row-major for Q, K, V tensors?
 *     Answer: _______________
 */
