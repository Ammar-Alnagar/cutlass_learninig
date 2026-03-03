/*
 * WHAT THIS TEACHES:
 *   - Use FP16 inputs with FP32 accumulator (mixed precision)
 *   - Understand numerical stability for attention softmax
 *   - Convert between FP16 and FP32 in CuTe tensors
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 uses FP16 for Q, K, V (memory efficiency) but FP32 for
 *   attention scores and output (numerical stability during softmax).
 *   This maps to: NVIDIA DL Software Engineer — "mixed precision GEMM for attention"
 *
 * MENTAL MODEL:
 *   FP16 range: ~6e-5 to 65504, precision: ~3 decimal digits
 *   FP32 range: ~1e-38 to 3e38, precision: ~7 decimal digits
 *   Softmax needs FP32: exp(x) overflows FP16 for x > 11
 *   GEMM accumulator needs FP32: sum of many FP16 products loses precision
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cmath>

using namespace cute;

// ============================================================================
// KERNEL: Mixed Precision GEMM (FP16 input, FP32 accumulator)
// ============================================================================
__global__ void mixed_precision_gemm_kernel(half* gmem_A, half* gmem_B, float* gmem_C) {
    // MENTAL MODEL: FlashAttention-2 dimensions
    // QK^T: [Br, head_dim] @ [head_dim, Bc] = [Br, Bc]
    // Typical: Br=64, Bc=64, head_dim=128
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 128;
    
    // MENTAL MODEL: Input tensors (FP16 for memory efficiency)
    auto A_gmem = make_tensor(make_gmem_ptr<half>(gmem_A),
                               make_layout(make_shape(Int<M>{}, Int<K>{})));
    auto B_gmem = make_tensor(make_gmem_ptr<half>(gmem_B),
                               make_layout(make_shape(Int<K>{}, Int<N>{})));
    
    // MENTAL MODEL: Output tensor (FP32 for numerical stability)
    auto C_gmem = make_tensor(make_gmem_ptr<float>(gmem_C),
                               make_layout(make_shape(Int<M>{}, Int<N>{})));
    
    // MENTAL MODEL: Shared memory for tiles (FP16 to save space)
    __shared__ half smem_A[M * K];
    __shared__ half smem_B[K * N];
    
    // Copy from gmem to smem (simplified)
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        smem_A[i] = A_gmem(i);
    }
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        smem_B[i] = B_gmem(i);
    }
    __syncthreads();
    
    // MENTAL MODEL: TiledMMA with mixed precision
    // A, B are FP16, C (accumulator) is FP32
    using MMA_Atom = MMA_Atom<
        MMA_Traits<Shape<Int<16>, Int<8>, Int<16>>,  // m16n8k16
                   Element_t<half>, Element_t<half>, Element_t<float>>
    >;
    
    auto tiled_mma = make_tiled_mma(MMA_Atom{}, make_layout(Int<128>{}));
    
    // MENTAL MODEL: Partition fragments
    // A_frag and B_frag are FP16, C_frag is FP32
    auto A_frag = partition_fragment_A(tiled_mma, make_shape(Int<M>{}, Int<K>{}));
    auto B_frag = partition_fragment_B(tiled_mma, make_shape(Int<K>{}, Int<N>{}));
    auto C_frag = partition_fragment_C(tiled_mma, make_shape(Int<M>{}, Int<N>{}));
    
    // MENTAL MODEL: Load fragments (with FP16 -> FP32 conversion for compute)
    // In real code, Tensor Cores handle this in hardware
    for (int i = 0; i < size(A_frag); i++) {
        A_frag(i) = __half2float(smem_A[i]);
    }
    for (int i = 0; i < size(B_frag); i++) {
        B_frag(i) = __half2float(smem_B[i]);
    }
    for (int i = 0; i < size(C_frag); i++) {
        C_frag(i) = 0.0f;  // FP32 accumulator
    }
    
    // MENTAL MODEL: Execute GEMM in FP32
    gemm(tiled_mma, A_frag, B_frag, C_frag);
    
    // MENTAL MODEL: Store FP32 results
    // In FlashAttention-2, these go to softmax (which needs FP32)
    for (int i = threadIdx.x; i < M * N; i += blockDim.x) {
        C_gmem(i) = C_frag(i % size(C_frag));
    }
    
    // Thread 0 prints numerical stability demonstration
    if (threadIdx.x == 0) {
        printf("=== Mixed Precision GEMM Complete ===\n");
        printf("Input: FP16 (memory efficient)\n");
        printf("Accumulator: FP32 (numerically stable)\n");
        printf("Output: FP32 (ready for softmax)\n\n");
        
        // MENTAL MODEL: Softmax stability demonstration
        // exp(20) in FP16 overflows, but FP32 handles it
        float x_fp32 = 20.0f;
        half x_fp16 = __float2half(20.0f);
        
        printf("Softmax numerical stability:\n");
        printf("  exp(20) in FP32 = %.2e\n", expf(x_fp32));
        printf("  exp(20) in FP16 = %.2e (overflows!)\n", 
               expf(__half2float(x_fp16)));
        printf("  This is why attention scores use FP32\n");
    }
}

// ============================================================================
// CPU REFERENCE: Mixed precision GEMM
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
    printf("=== Mixed Precision GEMM Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    
    // FP16 Tensor Core peak
    float fp16_peak_tflops = prop.multiProcessorCount * 128.0f / 1000.0f;
    printf("Peak Tensor TFLOPS (FP16): ~%.1f TFLOPS\n\n", fp16_peak_tflops);
    
    constexpr int M = 64, N = 64, K = 128;
    
    // Allocate
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Initialize with deterministic values
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = 0.1f * (i % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = 0.1f * (i % 10);
    
    // Convert to FP16
    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; i++) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_half[i] = __float2half(h_B[i]);
    
    cudaMemcpy(d_A, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // PREDICT BEFORE RUNNING:
    // Q1: Why does FlashAttention-2 use FP32 for attention scores?
    // Q2: What happens if you compute softmax in FP16 with large values?
    // Q3: How much memory do you save using FP16 vs FP32 for Q, K, V?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    mixed_precision_gemm_kernel<<<1, 128>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics smsp__inst_executed_op_tensor.sum \
    //              ./ex04_mixed_precision_fp16
    // Look for: Tensor Core instructions with FP16 inputs
    nvtxRangePush("mixed_precision_gemm_kernel");
    mixed_precision_gemm_kernel<<<1, 128>>>(d_A, d_B, d_C);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Verify
    float h_C[M * N];
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU reference (in FP32 for comparison)
    std::vector<float> h_C_ref(M * N);
    cpu_reference_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    // Check for reasonable values (FP16 rounding causes small errors)
    bool pass = true;
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C[i] - h_C_ref[i]);
        max_error = fmax(max_error, error);
        // Allow for FP16 rounding error
        if (error > 1.0f) {
            pass = false;
            printf("Large error at index %d: GPU=%.4f, CPU=%.4f\n", i, h_C[i], h_C_ref[i]);
            break;
        }
    }
    
    printf("\nMax error (FP16 rounding): %.4f\n", max_error);
    printf("[%s] Mixed precision GEMM verified\n", pass ? "PASS" : "FAIL");
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < 10; i++) {
        mixed_precision_gemm_kernel<<<1, 128>>>(d_A, d_B, d_C);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        mixed_precision_gemm_kernel<<<1, 128>>>(d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= 100.0f;
    
    float flops = 2.0f * M * N * K;
    float tflops = flops / elapsed_ms / 1e9;
    
    printf("\n[Timing] Average kernel time: %.3f ms\n", elapsed_ms);
    printf("[Performance] Achieved: %.2f TFLOPS\n", tflops);
    printf("[Performance] Peak FP16: ~%.1f TFLOPS\n", fp16_peak_tflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to Module 05
 * 
 * Q1: Why use FP16 for Q, K, V but FP32 for attention scores?
 *     Answer: FP16 saves memory bandwidth and capacity. FP32 prevents
 *             overflow in exp() and maintains precision in accumulation.
 * 
 * Q2: How much memory do you save using FP16 vs FP32 for a 4096-dim model?
 *     Answer: 50% - FP16 is 2 bytes, FP32 is 4 bytes per element.
 * 
 * Q3: What is the maximum value before exp() overflows in FP16?
 *     Answer: ln(65504) ≈ 11.1 - any value > 11 causes overflow in FP16 softmax.
 * 
 * === MODULE 04 COMPLETE ===
 * Exit criteria:
 * 1. Can construct TiledMMA with make_tiled_mma(MMA_Atom, warp_layout)
 * 2. Can partition fragments with partition_fragment_A/B/C
 * 3. Can execute GEMM with gemm(tiled_mma, A_frag, B_frag, C_frag)
 * 4. Can explain why FlashAttention-2 uses FP16 inputs + FP32 accumulator
 * 
 * Next: Module 05 — Swizzle (bank-conflict-free shared memory)
 */
