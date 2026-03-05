// ============================================================
// MODULE 07 — Predication
// Exercise 01: Predicated Copy — Handling Irregular Last Tile
// FILL-IN VERSION — Complete the TODO sections
// ============================================================
// CONCEPT:
//   copy_if(tiled_copy, pred, src, dst) only copies where predicate is true.
//   For tensor size 100 with tile size 32, last tile has only 4 valid elements.
//
// BUILD:
//   nvcc -std=c++17 -arch=sm_80 -I/path/to/cutlass/include \
//        ex01_predicated_copy_FILL_IN.cu -o ex01_predicated_copy_FILL_IN
// ============================================================

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <iostream>

using namespace cute;

constexpr int TENSOR_SIZE = 100;
constexpr int TILE_SIZE = 32;
constexpr int NUM_TILES = (TENSOR_SIZE + TILE_SIZE - 1) / TILE_SIZE;
constexpr int SMEM_SIZE = TILE_SIZE;

// ============================================================================
// TODO: Add helper to print array values
// ============================================================================


// ============================================================================
// KERNEL: Predicated copy
// ============================================================================
__global__ void predicated_copy_kernel(float* gmem_src, float* gmem_dst, float* smem_buffer) {
  if (threadIdx.x != 0) return;
  
  for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
    int tile_base = tile_idx * TILE_SIZE;
    
    // STEP 1: Clear SMEM (zero-fill pattern)
    // ======================================
    // TODO: Zero-fill the entire SMEM buffer
    for (int i = 0; i < SMEM_SIZE; ++i) {
      /* TODO: smem_buffer[i] = 0.0f */;
    }
    __syncthreads();
    
    // STEP 2: Build predicate tensor
    // ===============================
    // TODO: Create predicate tensor with shape (TILE_SIZE,)
    auto pred_tensor = /* TODO: make_tensor<bool>(make_shape(Int<TILE_SIZE>{})) */;
    
    // TODO: Set predicate based on bounds check
    for (int i = 0; i < TILE_SIZE; ++i) {
      int global_idx = tile_base + i;
      pred_tensor(make_coord(i)) = /* TODO: (global_idx < TENSOR_SIZE) */;
    }
    
    // STEP 3: Predicated copy
    // ========================
    int valid_count = 0;
    for (int i = 0; i < TILE_SIZE; ++i) {
      if (/* TODO: pred_tensor(make_coord(i)) */) {
        int global_idx = tile_base + i;
        // TODO: Copy from gmem_src to smem_buffer
        smem_buffer[i] = /* TODO: gmem_src[global_idx] */;
        valid_count++;
      }
    }
    __syncthreads();
    
    // STEP 4: Copy from SMEM to destination
    // ======================================
    for (int i = 0; i < TILE_SIZE; ++i) {
      int global_idx = tile_base + i;
      if (global_idx < TENSOR_SIZE) {
        // TODO: Copy from smem_buffer to gmem_dst
        gmem_dst[global_idx] = /* TODO: smem_buffer[i] */;
      }
    }
    __syncthreads();
    
    printf("Tile %d: base=%d, valid_elements=%d\n", tile_idx, tile_base, valid_count);
  }
}

// ============================================================================
// TODO: Add CPU reference copy function
// Hint: Simple memcpy for all elements
// ============================================================================


// ============================================================================
// MAIN: Entry point
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "Module 07 — Predication\n";
  std::cout << "Exercise 01: Predicated Copy (FILL-IN)\n";
  std::cout << "============================================================\n\n";
  
  // PREDICTIONS:
  // Q1: How many full tiles for size=100, tile_size=32?
  //     Your answer: ___
  //
  // Q2: How many valid elements in last tile?
  //     Your answer: ___
  
  std::cout << "Your predictions:\n";
  std::cout << "  Q1: ___\n";
  std::cout << "  Q2: ___\n\n";
  
  // Allocate and initialize
  float* h_src = new float[TENSOR_SIZE];
  float* h_dst = new float[TENSOR_SIZE];
  float* h_ref = new float[TENSOR_SIZE];
  
  for (int i = 0; i < TENSOR_SIZE; ++i) {
    h_src[i] = static_cast<float>(i);
  }
  
  // TODO: Print source tensor (first 20 elements)
  
  // Allocate device memory
  float *d_src, *d_dst, *d_smem;
  cudaMalloc(&d_src, TENSOR_SIZE * sizeof(float));
  cudaMalloc(&d_dst, TENSOR_SIZE * sizeof(float));
  cudaMalloc(&d_smem, SMEM_SIZE * sizeof(float));
  
  cudaMemcpy(d_src, h_src, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch kernel
  predicated_copy_kernel<<<1, 32>>>(d_src, d_dst, d_smem);
  cudaDeviceSynchronize();
  
  // Copy back
  cudaMemcpy(h_dst, d_dst, TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  
  // TODO: Call CPU reference
  
  // TODO: Verify results match
  
  std::cout << "\n============================================================\n";
  std::cout << "KEY INSIGHT:\n";
  std::cout << "  1. Clear SMEM before copy_if (unused lanes = 0.0f)\n";
  std::cout << "  2. Build predicate from bounds check\n";
  std::cout << "  3. copy_if only copies where predicate is true\n";
  std::cout << "============================================================\n";
  
  delete[] h_src;
  delete[] h_dst;
  delete[] h_ref;
  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_smem);
  
  return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 *
 * Q1: Why clear SMEM before copy_if?
 *     Your answer: ___
 *
 * Q2: How do you construct a predicate for bounds checking?
 *     Your answer: ___
 */
