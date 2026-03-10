/*
 * EXERCISE: MMA Atom Setup - Fill in the Gaps
 *
 * WHAT THIS TEACHES:
 *   - Understand MMA atoms for Tensor Core operations
 *   - Configure MMA for sm_89 (Ada Lovelace) architecture
 *   - See the basic building block of GEMM in CuTe
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   MMA Atom = the smallest unit of Tensor Core computation
 *   sm_89 (Ada) supports: mma.sync.aligned.m16n8k16.f32.f16.f16.f32
 *   This computes: D = A * B + C where A,B are f16, C,D are f32
 *   Each warp executes multiple atoms to cover the full tile
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: MMA Atom Setup
// ============================================================================
__global__ void mma_atom_kernel(float *gmem_C) {
  // MENTAL MODEL: FlashAttention-2 uses small GEMMs repeatedly
  // QK^T: [Br, head_dim] @ [head_dim, Bc] = [Br, Bc]
  // Typical: Br=64, Bc=64, head_dim=128

  // For this exercise, small example: [16, 32] @ [32, 16] = [16, 16]
  // This fits in one warp and demonstrates the MMA atom

  // CONCEPT: MMA Atom for sm_80+ (Ampere, Ada, Hopper)
  // The op selector encodes: instruction shape + element types + layout tags
  // SM80_16x8x16_F32F16F16F32_TN means:
  //   - m16n8k16 tile
  //   - F32 accumulator (C/D), F16 inputs (A/B)
  //   - T = A row-major (transposed), N = B column-major (non-transposed)

  // TODO 1: Define MMA_Op using the CUTLASS op selector for m16n8k16
  //         with FP16 inputs (A, B) and FP32 accumulator (C/D), TN layout
  // Hint:   using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
  using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;

  // TODO 2: Wrap the op selector in MMA_Atom to get the atom type
  // Hint:   using Atom = MMA_Atom<MMA_Op>;
  using Atom = MMA_Atom<MMA_Op>;

  printf("=== MMA Atom Configuration ===\n");
  printf("MMA shape: M=16, N=8, K=16\n");
  printf("A type: FP16, B type: FP16, C/D type: FP32\n");
  printf("Instruction: mma.sync.aligned.m16n8k16.f32.f16.f16.f32\n\n");

  // MENTAL MODEL: TiledMMA wraps the atom with thread/warp layout
  // For a [16,16,32] GEMM, we need multiple atoms
  // One warp (32 threads) can execute this with proper tiling

  // MENTAL MODEL: Create layouts for A, B, C
  // A: [16, 32] row-major  -> strides (32, 1)
  // B: [32, 16] col-major  -> strides ( 1, 32)
  // C: [16, 16] row-major  -> strides (16, 1)

  // TODO 3: Create A layout [16, 32] row-major
  // Hint: make_layout(make_shape(Int<16>{}, Int<32>{}))
  //       Row-major is the default (strides derived from shape: (32, 1))
  auto A_layout = make_layout(make_shape(Int<16>{}, Int<32>{}), GenRowMajor{});

  // TODO 4: Create B layout [32, 16] column-major
  // Hint: stride tuple for column-major is (1, 32) — inner dim stride is 1
  auto B_layout = make_layout(make_shape(Int<32>{}, Int<16>{}), GenColMajor{});

  // TODO 5: Create C layout [16, 16] row-major
  // Hint: same pattern as A_layout above
  auto C_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});

  printf("A layout [16, 32] row-major:\n");
  print(A_layout);
  printf("\n");

  printf("B layout [32, 16] column-major:\n");
  print(B_layout);
  printf("\n");

  printf("C layout [16, 16] row-major:\n");
  print(C_layout);
  printf("\n");

  // MENTAL MODEL: Write output to gmem for verification
  auto C_gmem_ptr = make_gmem_ptr<float>(gmem_C);
  auto C_gmem = make_tensor(C_gmem_ptr, C_layout);

  // Initialize C with zeros (in real code, this is the accumulator)
  // NOTE: use runtime (i, j) indexing — Int<N>{} requires compile-time N
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      C_gmem(i, j) = 0.0f;
    }
  }

  if (threadIdx.x == 0) {
    printf("=== MMA Atom Ready ===\n");
    printf("To execute: gemm(tiled_mma, A_frag, B_frag, C_frag)\n");
    printf("See ex02_tiled_gemm.cu for full GEMM execution\n");
  }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== MMA Atom Setup Exercise ===\n");
  printf("GPU: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);

  if (prop.major >= 7) {
    printf("Tensor Cores: Available (sm_%d%d+)\n", prop.major, prop.minor);
    printf("Peak Tensor TFLOPS (FP16): ~%d TFLOPS\n",
           prop.multiProcessorCount * 128);
  } else {
    printf("WARNING: Tensor Cores require sm_70+ (Volta+)\n");
  }
  printf("\n");

  constexpr int C_SIZE = 16 * 16;
  float *d_C;
  cudaMalloc(&d_C, C_SIZE * sizeof(float));

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  mma_atom_kernel<<<1, 32>>>(d_C);
  cudaDeviceSynchronize();

  // Timed run
  nvtxRangePush("mma_atom_kernel");
  mma_atom_kernel<<<1, 32>>>(d_C);
  nvtxRangePop();
  cudaDeviceSynchronize();

  // Verify C was zeroed
  float h_C[C_SIZE];
  cudaMemcpy(h_C, d_C, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  bool pass = true;
  for (int i = 0; i < C_SIZE; i++) {
    if (h_C[i] != 0.0f) {
      pass = false;
      break;
    }
  }

  printf("\n[%s] MMA atom setup verified\n", pass ? "PASS" : "FAIL");

  cudaFree(d_C);
  return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 *
 * Q1: What is the MMA instruction shape for sm_80+ FP16 GEMM?
 *     Answer: SM80_16x8x16_F32F16F16F32_TN
 *
 * Q2: What are the element types for A, B, and C in FlashAttention-2?
 *     Answer: _______________
 *
 * Q3: How many threads are in a warp, and how many warps per SM on Ada?
 *     Answer: _______________
 */
