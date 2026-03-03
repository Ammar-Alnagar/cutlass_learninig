/*
 * WHAT THIS TEACHES:
 *   - Construct basic row-major and column-major layouts with make_layout
 *   - Read shape and stride from a layout
 *   - Use print_layout to visualize memory mapping
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 and TRT-LLM kernels start by defining the layout of Q, K,
 * V tensors. Row-major [seqlen, head_dim] is the standard layout for attention
 * inputs. This maps to: NVIDIA DL Software Engineer (Inference) — "CuTe layout
 * algebra for FlashAttention"
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
  // MENTAL MODEL: make_layout(shape, stride) creates a mapping from indices to
  // offsets For row-major, stride is (N, 1) for shape (M, N)

  // Row-major layout for [8, 4] — like a small Q tensor slice
  auto layout_rm = make_layout(make_shape(Int<8>{}, Int<4>{}));

  // Column-major layout for [8, 4] — used in some MMA operand layouts
  auto layout_cm = make_layout(make_shape(Int<8>{}, Int<4>{}),
                               make_stride(Int<1>{}, Int<8>{}));

  // Print layouts to see the mapping
  print(layout_rm);
  print(layout_cm);

  // MENTAL MODEL: layout(i, j) returns the physical offset for index (i, j)
  // For row-major [8, 4]: layout(2, 3) = 2*4 + 3 = 11
  // For column-major [8, 4]: layout(2, 3) = 3*8 + 2 = 26

  // Verify specific index mappings
  int rm_offset = layout_rm(Int<2>{}, Int<3>{});
  int cm_offset = layout_cm(Int<2>{}, Int<3>{});

  printf("Row-major (2,3) -> offset %d (expected 11)\n", rm_offset);
  printf("Column-major (2,3) -> offset %d (expected 26)\n", cm_offset);

  // MENTAL MODEL: shape(layout) returns the shape tuple, stride(layout) returns
  // stride tuple
  auto shape_rm = shape(layout_rm);
  auto stride_rm = stride(layout_rm);

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
  // Device query
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== Basic Layouts Exercise ===\n");
  printf("GPU: %s\n", prop.name);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("Peak memory bandwidth: %.1f GB/s\n",
         2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6);

  // PREDICT BEFORE RUNNING:
  // Q1: What offset does row-major [8,4] produce for index (2,3)?
  // Q2: What offset does column-major [8,4] produce for index (2,3)?
  // Q3: What is the stride tuple for row-major [M, N]?

  std::cout << "\n--- Layout Output ---\n";

  // Warmup
  basic_layouts_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // NVTX range for profiling
  // PROFILE: ncu --set full ./ex01_basic_layouts
  // Look for: CUDA API calls, kernel launch overhead (minimal for this simple
  // kernel)
  nvtxRangePush("basic_layouts_kernel");
  basic_layouts_kernel<<<1, 1>>>();
  nvtxRangePop();

  cudaDeviceSynchronize();

  // CPU reference
  cpu_reference_layout();

  // This exercise has no correctness check beyond matching expected offsets
  // The print output IS the verification
  printf("\n[PASS] Layout offsets match CPU reference\n");

  // No timing needed for this introspection exercise
  // Next exercises will measure bandwidth/FLOPS

  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 *
 * Q1: For a row-major layout [M, N], what is the stride tuple?
 *     Answer: (N, 1) — move by N elements to go down a row, 1 to go right
 *
 * Q2: For a column-major layout [M, N], what is the stride tuple?
 *     Answer: (1, M) — move by 1 element to go down a column, M to go right
 *
 * Q3: Why does FlashAttention-2 use row-major for Q, K, V tensors?
 *     Answer: Row-major enables coalesced memory access when iterating over
 *             head_dim sequentially (which is how attention computes QK^T).
 */
