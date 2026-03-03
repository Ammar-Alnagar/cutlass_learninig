/*
 * EXERCISE: Tiled GEMM for QK^T - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Construct TiledMMA with make_tiled_mma
 *   - Execute GEMM with gemm() call
 *   - Understand the full QK^T pattern from FlashAttention-2
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   make_tiled_mma(MMA_Atom, warp_layout) creates the GEMM operator
 *   gemm(tiled_mma, A, B, C) executes C = A * B + C
 *   A, B, C are fragment tensors (register memory, not gmem/smem)
 *   All threads in the warp cooperate to execute the GEMM
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Tiled GEMM for QK^T
// ============================================================================
__global__ void tiled_gemm_kernel(half* gmem_A, half* gmem_B, float* gmem_C) {
    // MENTAL MODEL: Small GEMM for demonstration: [16, 32] @ [32, 16] = [16, 16]
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

    // Copy from gmem to smem (simplified)
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        A_smem(i / K, i % K) = static_cast<float>(__half2float(gmem_A[i]));
    }
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        B_smem(i / N, i % N) = static_cast<float>(__half2float(gmem_B[i]));
    }
    __syncthreads();

    // CONCEPT: TiledMMA construction
    // MMA_Atom for FP16 input, FP32 output (FlashAttention-2 pattern)
    
    // TODO 1: Define MMA_Atom for m16n8k16 with FP32 elements (simplified)
    // Hint: using MMA_Atom = MMA_Atom<MMA_Traits<Shape<Int<16>, Int<8>, Int<16>>, Element_t<float>, Element_t<float>, Element_t<float>>>;
    using MMA_Atom = /* YOUR CODE HERE */;

    // TODO 2: Create warp layout for 1 warp (32 threads)
    // Hint: auto warp_layout = make_layout(Int<32>{});
    auto warp_layout = /* YOUR CODE HERE */;

    // TODO 3: Create TiledMMA operator
    // Hint: auto tiled_mma = make_tiled_mma(MMA_Atom{}, warp_layout);
    auto tiled_mma = /* YOUR CODE HERE */;

    // MENTAL MODEL: Partition fragments for each thread
    // A_frag: thread's portion of A
    // B_frag: thread's portion of B
    // C_frag: thread's portion of accumulator
    
    // TODO 4: Partition A fragment
    // Hint: auto A_frag = partition_fragment_A(tiled_mma, make_shape(Int<M>{}, Int<K>{}));
    auto A_frag = /* YOUR CODE HERE */;

    // TODO 5: Partition B fragment
    auto B_frag = partition_fragment_B(tiled_mma, make_shape(Int<K>{}, Int<N>{}));

    // TODO 6: Partition C fragment
    auto C_frag = /* YOUR CODE HERE */;

    // MENTAL MODEL: Load fragments from smem into registers
    for (int i = 0; i < size(A_frag); i++) {
        A_frag(i) = A_smem(i);
    }
    for (int i = 0; i < size(B_frag); i++) {
        B_frag(i) = B_smem(i);
    }
    for (int i = 0; i < size(C_frag); i++) {
        C_frag(i) = 0.0f;  // Zero accumulator
    }

    // CONCEPT: Execute GEMM: C = A * B + C
    
    // TODO 7: Execute the GEMM operation
    // Hint: gemm(tiled_mma, A_frag, B_frag, C_frag);
    /* YOUR CODE HERE */;

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
void cpu_reference_gemm(float* A, float* B, float* C, int M, int N, int K) {
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
    half* d_A;
    half* d_B;
    float* d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Initialize with simple values for verification
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;  // All ones
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;  // All ones

    // Convert to half
    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; i++) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_half[i] = __float2half(h_B[i]);

    cudaMemcpy(d_A, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

    std::cout << "--- Kernel Output ---\n";

    // Warmup
    tiled_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // NVTX range
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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 * 
 * Q1: What are the arguments to gemm(tiled_mma, A, B, C)?
 *     Answer: _______________
 * 
 * Q2: What does gemm() compute?
 *     Answer: _______________
 * 
 * Q3: For [64, 128] @ [128, 64], how many FLOPs?
 *     Answer: _______________
 */
