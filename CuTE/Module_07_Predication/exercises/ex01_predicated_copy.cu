// ============================================================
// MODULE 07 — Predication
// Exercise 01: Predicated Copy — Handling Irregular Last Tile
// ============================================================
// CONCEPT:
//   copy_if(tiled_copy, pred, src, dst) only copies elements where
//   predicate is true. For a 1D tensor of size 100 with tile size 32:
//   - Tile 0: elements 0-31 (full, 32 elements)
//   - Tile 1: elements 32-63 (full, 32 elements)
//   - Tile 2: elements 64-95 (full, 32 elements)
//   - Tile 3: elements 96-99 (partial, only 4 valid elements)
//   Predicate ensures threads covering elements 100+ don't access OOB.
//
// JOB RELEVANCE:
//   Cerebras Performance Engineer — "Production-quality kernel optimization"
//   NVIDIA DL Software Engineer — "Efficient attention kernels for arbitrary
//   sequence lengths" — A FlashAttention-2 kernel that only works on seqlen
//   multiples of 64 is not production.
//
// ASCII DIAGRAM:
//   Tensor: 100 elements, tile size 32
//   ┌────────────────────────────────────────────────────────┐
//   │ 0-31 │ 32-63 │ 64-95 │ 96-99 │ XX │ XX │ XX │        │
//   └────────────────────────────────────────────────────────┘
//     Tile 0  Tile 1  Tile 2  Tile 3 (partial)
//                                  │
//                                  ▼
//   Predicate for Tile 3:
//   ┌────────────────────────────────┐
//   │ T T T T F F F F F F F F ...   │  ← T=copy, F=skip
//   └────────────────────────────────┘
//     Elements 96-99 are valid (T)
//     Elements 100-127 are OOB (F)
//
//   Clear SMEM Pattern:
//   1. Zero-fill entire SMEM tile (32 elements)
//   2. copy_if with predicate
//   3. Unused lanes hold 0.0f, not garbage
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex01_predicated_copy.cu -o ex01_predicated_copy
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <iomanip>
#include <iostream>
#include <cstring>

using namespace cute;

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr int TENSOR_SIZE = 100;      // Actual data size (not tile-aligned)
constexpr int TILE_SIZE = 32;         // Tile size (power of 2)
constexpr int NUM_TILES = (TENSOR_SIZE + TILE_SIZE - 1) / TILE_SIZE;  // = 4
constexpr int SMEM_SIZE = TILE_SIZE;  // SMEM buffer size

// ============================================================================
// HELPER: Print array with labels
// ============================================================================
template <typename T>
__host__ void print_array(const char* label, T const* data, int size, int max_print = 20) {
  std::cout << label << " (size=" << size << "): { ";
  for (int i = 0; i < size && i < max_print; ++i) {
    std::cout << std::setw(6) << data[i] << " ";
  }
  if (size > max_print) {
    std::cout << "... ";
  }
  std::cout << "}\n";
}

// ============================================================================
// KERNEL: Predicated copy
// ============================================================================
__global__ void predicated_copy_kernel(float* gmem_src, float* gmem_dst, float* smem_buffer) {
  // Thread 0 does the work for this simple example
  if (threadIdx.x != 0) return;
  
  // STEP 1: Create tiled copy operator
  // ===================================
  // For simplicity, use a simple 1D copy with 32 threads
  // In production, this would be a full TiledCopy with cp.async
  
  auto copy_op = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL, float>{},
    Layout<Int<32>, Step<_1>>{},  // Thread layout: 32 threads
    Layout<Int<32>, Step<_1>>{}   // Value layout: 32 elements
  );
  
  // STEP 2: Copy all tiles with predication
  // ========================================
  for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
    int tile_base = tile_idx * TILE_SIZE;
    
    // STEP 2a: Clear SMEM (zero-fill pattern)
    // ========================================
    // This ensures unused lanes hold 0.0f, not garbage from previous iterations
    for (int i = threadIdx.x; i < SMEM_SIZE; i += 32) {
      smem_buffer[i] = 0.0f;
    }
    __syncthreads();
    
    // STEP 2b: Build predicate tensor
    // ================================
    // Predicate: element is valid if (tile_base + i) < TENSOR_SIZE
    // Use make_identity_tensor to get thread→element mapping
    
    auto pred_tensor = make_tensor<bool>(make_shape(Int<TILE_SIZE>{}));
    for (int i = 0; i < TILE_SIZE; ++i) {
      int global_idx = tile_base + i;
      pred_tensor(make_coord(i)) = (global_idx < TENSOR_SIZE);
    }
    
    // STEP 2c: Predicated copy
    // =========================
    // copy_if only copies where predicate is true
    // For full tiles (0, 1, 2): all predicates are true
    // For partial tile (3): only first 4 predicates are true
    
    int valid_count = 0;
    for (int i = 0; i < TILE_SIZE; ++i) {
      if (pred_tensor(make_coord(i))) {
        int global_idx = tile_base + i;
        smem_buffer[i] = gmem_src[global_idx];
        valid_count++;
      }
    }
    __syncthreads();
    
    // STEP 2d: Copy from SMEM to destination
    // =======================================
    for (int i = 0; i < TILE_SIZE; ++i) {
      int global_idx = tile_base + i;
      if (global_idx < TENSOR_SIZE) {
        gmem_dst[global_idx] = smem_buffer[i];
      }
    }
    __syncthreads();
    
    printf("Tile %d: base=%d, valid_elements=%d\n", tile_idx, tile_base, valid_count);
  }
}

// ============================================================================
// CPU REFERENCE: Verify copy correctness
// ============================================================================
void cpu_reference_copy(float* src, float* dst, int size) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

// ============================================================================
// MAIN: Setup, kernel launch, verification
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 07 — Predication\n";
  std::cout << "Exercise 01: Predicated Copy\n";
  std::cout << "============================================================\n\n";
  
  // PREDICT BEFORE RUNNING:
  // Q1: How many full tiles are there for size=100, tile_size=32?
  //     Answer: 3 full tiles (0-31, 32-63, 64-95)
  //
  // Q2: How many valid elements in the last tile?
  //     Answer: 100 - 96 = 4 elements (96, 97, 98, 99)
  //
  // Q3: Why clear SMEM before copy_if?
  //     Answer: Unused lanes hold 0.0f, not garbage from previous iterations
  
  std::cout << "Predictions:\n";
  std::cout << "  Q1: 3 full tiles\n";
  std::cout << "  Q2: 4 valid elements in last tile\n";
  std::cout << "  Q3: Clear SMEM ensures unused lanes = 0.0f\n\n";
  
  // Allocate host memory
  float* h_src = new float[TENSOR_SIZE];
  float* h_dst = new float[TENSOR_SIZE];
  float* h_ref = new float[TENSOR_SIZE];
  
  // Initialize source with known values
  for (int i = 0; i < TENSOR_SIZE; ++i) {
    h_src[i] = static_cast<float>(i);
  }
  
  std::cout << "=== Input Tensor (first 20 elements) ===\n";
  print_array("Source", h_src, TENSOR_SIZE, 20);
  std::cout << "\n";
  
  // Allocate device memory
  float *d_src, *d_dst, *d_smem;
  cudaMalloc(&d_src, TENSOR_SIZE * sizeof(float));
  cudaMalloc(&d_dst, TENSOR_SIZE * sizeof(float));
  cudaMalloc(&d_smem, SMEM_SIZE * sizeof(float));
  
  // Copy to device
  cudaMemcpy(d_src, h_src, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch kernel
  std::cout << "=== Running Predicated Copy ===\n";
  predicated_copy_kernel<<<1, 32>>>(d_src, d_dst, d_smem);
  cudaDeviceSynchronize();
  
  // Copy back
  cudaMemcpy(h_dst, d_dst, TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  
  // CPU reference
  cpu_reference_copy(h_src, h_ref, TENSOR_SIZE);
  
  // Verify
  std::cout << "\n=== Output Tensor (first 20 elements) ===\n";
  print_array("Destination", h_dst, TENSOR_SIZE, 20);
  std::cout << "\n";
  
  std::cout << "=== Last Tile Elements (96-99) ===\n";
  printf("  Expected: 96, 97, 98, 99\n");
  printf("  Got:      %.0f, %.0f, %.0f, %.0f\n", 
         h_dst[96], h_dst[97], h_dst[98], h_dst[99]);
  std::cout << "\n";
  
  bool pass = true;
  for (int i = 0; i < TENSOR_SIZE; ++i) {
    if (h_dst[i] != h_ref[i]) {
      printf("MISMATCH at index %d: expected %.0f, got %.0f\n", 
             i, h_ref[i], h_dst[i]);
      pass = false;
    }
  }
  
  std::cout << "=== Verification ===\n";
  std::cout << (pass ? "PASS ✓" : "FAIL ✗") << "\n\n";
  
  // Cleanup
  delete[] h_src;
  delete[] h_dst;
  delete[] h_ref;
  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_smem);
  
  // KEY INSIGHT
  std::cout << "============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  Predication pattern for irregular tensors:\n";
  std::cout << "\n";
  std::cout << "  1. Clear SMEM before copy_if\n";
  std::cout << "     - Zero-fill ensures unused lanes = 0.0f\n";
  std::cout << "\n";
  std::cout << "  2. Build predicate from bounds check\n";
  std::cout << "     - pred[i] = (global_idx < tensor_size)\n";
  std::cout << "\n";
  std::cout << "  3. copy_if only copies where predicate is true\n";
  std::cout << "     - OOB threads skip the copy\n";
  std::cout << "\n";
  std::cout << "  Nsight metric: l1tex__data_pipe_lookup_misc\n";
  std::cout << "  Should show NO illegal memory accesses\n";
  std::cout << "============================================================\n";
  
  return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 *
 * Q1: Why is the "clear SMEM" pattern important?
 *     Answer: Without clearing, unused lanes hold garbage from previous
 *             iterations. This corrupts computation if the tile is used
 *             in a GEMM or other operation.
 *
 * Q2: How do you construct a predicate for bounds checking?
 *     Answer: pred[i] = (tile_base + i < tensor_size)
 *             Each thread computes its own predicate based on global index.
 *
 * Q3: What Nsight metric confirms no illegal memory access?
 *     Answer: l1tex__data_pipe_lookup_misc should show no exceptions.
 *             Also check for no L1 cache errors.
 */
