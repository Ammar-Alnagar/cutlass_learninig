/*
 * WHAT THIS TEACHES:
 *   - Construct TiledMMA with make_tiled_mma
 *   - Execute GEMM with gemm() call
 *   - Understand the full QK^T pattern from FlashAttention-2
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 computes QK^T for attention scores using TiledMMA.
 *   This is the exact pattern: gemm(tiled_mma, Q_frag, K_frag, C_frag)
 *   This maps to: NVIDIA DL Software Engineer — "FlashAttention-2 GEMM with
 * CuTe"
 *
 * MENTAL MODEL:
 *   make_tiled_mma(MMA_Atom, warp_layout) creates the GEMM operator
 *   gemm(tiled_mma, A, B, C) executes C = A * B + C
 *   A, B, C are fragment tensors (register memory, not gmem/smem)
 *   All threads in the warp cooperate to execute the GEMM
 */

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

// ============================================================================
// KERNEL: Tiled GEMM for QK^T
// ============================================================================
__global__ void tiled_gemm_kernel(half *gmem_A, half *gmem_B, float *gmem_C) {
  // MENTAL MODEL: Small GEMM for demonstration: [16, 32] @ [32, 16] = [16, 16]
  // FlashAttention-2 uses larger: [64, 128] @ [128, 64] = [64, 64]
  constexpr int M = 16;
  constexpr int N = 16;
  constexpr int K = 32;

  // MENTAL MODEL: Input tensors in gmem
  auto A_gmem = make_tensor(make_gmem_ptr<half>(gmem_A),
                            make_layout(make_shape(Int<M>{}, Int<K>{})));
  auto B_gmem = make_tensor(make_gmem_ptr<half>(gmem_B),
                            make_layout(make_shape(Int<K>{}, Int<N>{})));
  auto C_gmem = make_tensor(make_gmem_ptr<float>(gmem_C),
                            make_layout(make_shape(Int<M>{}, Int<N>{})));

  // MENTAL MODEL: Shared memory for A and B tiles
  __shared__ float smem_A[M * K];
  __shared__ float smem_B[K * N];

  auto A_smem = make_tensor(make_smem_ptr<float>(smem_A),
                            make_layout(make_shape(Int<M>{}, Int<K>{})));
  auto B_smem = make_tensor(make_smem_ptr<float>(smem_B),
                            make_layout(make_shape(Int<K>{}, Int<N>{})));

  // MENTAL MODEL: Copy from gmem to smem (simplified, not vectorized)
  for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
    A_smem(i / K, i % K) = static_cast<float>(__half2float(gmem_A[i]));
  }
  for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
    B_smem(i / N, i % N) = static_cast<float>(__half2float(gmem_B[i]));
  }
  __syncthreads();

  // MENTAL MODEL: TiledMMA construction
  // MMA_Atom for FP16 input, FP32 output (FlashAttention-2 pattern)
  using MMA_Atom =
      MMA_Atom<MMA_Traits<Shape<Int<16>, Int<8>, Int<16>>, // m16n8k16
                          Element_t<float>, Element_t<float>,
                          Element_t<float>> // Using FP32 for simplicity
               >;

  // Warp layout: 1 warp (32 threads) for this small example
  auto warp_layout = make_layout(Int<32>{});

  auto tiled_mma = make_tiled_mma(MMA_Atom{}, warp_layout);

  // MENTAL MODEL: Partition fragments for each thread
  // A_frag: thread's portion of A
  // B_frag: thread's portion of B
  // C_frag: thread's portion of accumulator
  auto A_frag = partition_fragment_A(tiled_mma, make_shape(Int<M>{}, Int<K>{}));
  auto B_frag = partition_fragment_B(tiled_mma, make_shape(Int<K>{}, Int<N>{}));
  auto C_frag = partition_fragment_C(tiled_mma, make_shape(Int<M>{}, Int<N>{}));

  // MENTAL MODEL: Load fragments from smem into registers
  // In real code, this is done with TiledCopy
  // Here we do a simple load for demonstration
  for (int i = 0; i < size(A_frag); i++) {
    A_frag(i) = A_smem(i);
  }
  for (int i = 0; i < size(B_frag); i++) {
    B_frag(i) = B_smem(i);
  }
  for (int i = 0; i < size(C_frag); i++) {
    C_frag(i) = 0.0f; // Zero accumulator
  }

  // MENTAL MODEL: Execute GEMM: C = A * B + C
  gemm(tiled_mma, A_frag, B_frag, C_frag);

  // MENTAL MODEL: Store results back to gmem
  for (int i = 0; i < size(C_frag); i++) {
    C_gmem(i) = C_frag(i);
  }

  // Thread 0 prints summary
  if (threadIdx.x == 0) {
    printf("=== Tiled GEMM Complete ===\n");
    printf("GEMM: [%d, %d] @ [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("C[0] = %.2f (expected: dot product of A[0,:] and B[:,0])\n",
           float(C_gmem(0, 0)));
  }
}

// ============================================================================
// CPU REFERENCE: Verify GEMM result
// ============================================================================
void cpu_reference_gemm(float *A, float *B, float *C, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("=== Tiled GEMM Exercise ===\n");
  printf("GPU: %s\n", prop.name);
  printf("Peak Tensor TFLOPS (FP16): ~%d TFLOPS\n\n",
         prop.multiProcessorCount * 128);

  constexpr int M = 16, N = 16, K = 32;

  // Allocate and initialize
  half *d_A;
  half *d_B;
  float *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Initialize with simple values for verification
  std::vector<float> h_A(M * K), h_B(K * N);
  for (int i = 0; i < M * K; i++)
    h_A[i] = 1.0f; // All ones
  for (int i = 0; i < K * N; i++)
    h_B[i] = 1.0f; // All ones

  // Convert to half
  std::vector<half> h_A_half(M * K), h_B_half(K * N);
  for (int i = 0; i < M * K; i++)
    h_A_half[i] = __float2half(h_A[i]);
  for (int i = 0; i < K * N; i++)
    h_B_half[i] = __float2half(h_B[i]);

  cudaMemcpy(d_A, h_A_half.data(), M * K * sizeof(half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B_half.data(), K * N * sizeof(half),
             cudaMemcpyHostToDevice);

  // PREDICT BEFORE RUNNING:
  // Q1: With A=all(1) and B=all(1), what is C[0,0]?
  // Q2: How many multiply-add operations for [16,32]@[32,16]?
  // Q3: Why do we zero the accumulator before gemm()?

  std::cout << "--- Kernel Output ---\n";

  // Warmup
  tiled_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
  cudaDeviceSynchronize();

  // NVTX range
  // PROFILE: ncu --metrics smsp__inst_executed_op_tensor.sum \
  //              ./ex02_tiled_gemm
  // Look for: Tensor Core instructions (mma.sync)
  nvtxRangePush("tiled_gemm_kernel");
  tiled_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
  nvtxRangePop();

  cudaDeviceSynchronize();

  // Verify
  float h_C[M * N];
  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // CPU reference
  std::vector<float> h_C_ref(M * N);
  cpu_reference_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

  bool pass = true;
  for (int i = 0; i < M * N; i++) {
    // Expected: C[i] = sum of 32 ones = 32
    if (fabs(h_C[i] - 32.0f) > 0.1f) {
      pass = false;
      printf("Mismatch at index %d: expected 32.0, got %.2f\n", i, h_C[i]);
      break;
    }
  }

  printf("\n[%s] Tiled GEMM verified\n", pass ? "PASS" : "FAIL");

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; i++) {
    tiled_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    tiled_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start);
  cudaEventElapsedTime(&elapsed_ms, stop);
  elapsed_ms /= 100.0f;

  // Calculate TFLOPS
  float flops = 2.0f * M * N * K; // multiply + add
  float tflops = flops / elapsed_ms / 1e9;

  printf("\n[Timing] Average kernel time: %.3f ms\n", elapsed_ms);
  printf("[Performance] Achieved: %.2f TFLOPS\n", tflops);
  printf(
      "Note: Small GEMM - overhead dominated. See Projects for real TFLOPS.\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 *
 * Q1: What are the arguments to gemm(tiled_mma, A, B, C)?
 *     Answer: tiled_mma (the operator), A_frag, B_frag, C_frag (register
 * tensors)
 *
 * Q2: What does gemm() compute?
 *     Answer: C = A * B + C (matrix multiply-accumulate)
 *
 * Q3: For [64, 128] @ [128, 64], how many FLOPs?
 *     Answer: 2 * 64 * 128 * 64 = 1,048,576 FLOPs (~1 MFLOP)
 */
