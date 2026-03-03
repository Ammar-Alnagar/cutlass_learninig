/*
 * EXERCISE: Verify with Nsight - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Profile swizzled smem access with Nsight Compute
 *   - Verify zero bank conflicts using smem__conflict_requests metric
 *   - Understand the production profiling workflow for kernel optimization
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   Nsight Compute metrics to check:
 *   - smem__conflict_requests.sum: Should be 0 (or very low)
 *   - smem__transactions.sum: Total smem accesses
 *   - smem throughput: Should be near peak (~20 TB/s on Ada)
 *   Command: ncu --metrics <metrics> ./binary
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/cute.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Swizzled SMEM Access (Profile-Ready)
// ============================================================================
__global__ void swizzle_profile_kernel(half* gmem_A, half* gmem_B, float* gmem_C) {
    // MENTAL MODEL: FlashAttention-2 tile sizes
    constexpr int M = 64;   // Row tile (Br)
    constexpr int N = 64;   // Column tile (Bc)
    constexpr int K = 128;  // head_dim

    // MENTAL MODEL: Input tensors
    auto A_gmem = make_tensor(make_gmem_ptr<half>(gmem_A),
                               make_layout(make_shape(Int<M>{}, Int<K>{})));
    auto B_gmem = make_tensor(make_gmem_ptr<half>(gmem_B),
                               make_layout(make_shape(Int<K>{}, Int<N>{})));
    auto C_gmem = make_tensor(make_gmem_ptr<float>(gmem_C),
                               make_layout(make_shape(Int<M>{}, Int<N>{})));

    // MENTAL MODEL: Shared memory with swizzle
    __shared__ float smem_A[M * K];
    __shared__ float smem_B[K * N];

    auto smem_A_ptr = make_smem_ptr<float>(smem_A);
    auto smem_B_ptr = make_smem_ptr<float>(smem_B);

    // CONCEPT: Swizzled layouts for conflict-free access
    // Swizzle<2, 3, 3> is tuned for 128-thread access to 128-wide rows
    
    // TODO 1: Create swizzled layout for A with Swizzle<2, 3, 3>
    // Hint: make_layout(shape, stride, Swizzle<2, 3, 3>{})
    // Shape: (M, K), Stride: (K, 1)
    auto layout_A = make_layout(
        /* YOUR CODE HERE */
    );
    
    // TODO 2: Create swizzled layout for B with Swizzle<2, 3, 3>
    // Shape: (K, N), Stride: (N, 1)
    auto layout_B = make_layout(
        /* YOUR CODE HERE */
    );

    auto smem_A = make_tensor(smem_A_ptr, layout_A);
    auto smem_B = make_tensor(smem_B_ptr, layout_B);

    // MENTAL MODEL: Copy from gmem to swizzled smem
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        smem_A(i / K, i % K) = __half2float(A_gmem(i / K, i % K));
    }
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        smem_B(i / N, i % N) = __half2float(B_gmem(i / N, i % N));
    }
    __syncthreads();

    // MENTAL MODEL: Access swizzled smem (simulating MMA fragment load)
    // Each thread reads its portion in a pattern that would conflict without swizzle
    
    // TODO 3: Read from swizzled smem_B in inner loop
    // Hint: float val = smem_B(k, n);
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            float val = /* YOUR CODE HERE */;

            // Simple compute (in real code, this is MMA)
            if (threadIdx.x == 0 && k == 0 && n == 0) {
                atomicAdd(gmem_C, val * 0.0f);  // No-op but prevents DCE
            }
        }
    }

    // Thread 0 prints
    if (threadIdx.x == 0) {
        printf("=== Swizzle Profile Kernel ===\n");
        printf("Tile: [%d, %d] @ [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
        printf("Swizzle: Swizzle<2, 3, 3>\n");
        printf("Threads: 128\n\n");
        printf("Profile with:\n");
        printf("  ncu --metrics smem__conflict_requests.sum,\\\n");
        printf("               smem__transactions.sum,\\\n");
        printf("               smem__throughput.avg.pct_of_peak_sustained_elapsed \\\n");
        printf("       ./ex03_verify_with_nsight\n\n");
        printf("Expected:\n");
        printf("  - smem__conflict_requests.sum: 0 (or very low)\n");
        printf("  - smem throughput: >80%% of peak\n");
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Nsight Verification Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Peak smem bandwidth: ~%.0f GB/s (estimated)\n\n",
           prop.sharedMemPerMultiprocessor * prop.clockRate / 1e6 * prop.multiProcessorCount);

    constexpr int M = 64, N = 64, K = 128;

    // Allocate
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, sizeof(float));  // Just for atomic

    // Initialize
    std::vector<half> h_A(M * K, __float2half(1.0f));
    std::vector<half> h_B(K * N, __float2half(1.0f));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(float));

    std::cout << "--- Kernel Output ---\n";

    // Warmup
    swizzle_profile_kernel<<<1, 128, (M * K + K * N) * sizeof(float)>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // NVTX range for profiling
    nvtxRangePush("swizzle_profile_kernel");
    swizzle_profile_kernel<<<1, 128, (M * K + K * N) * sizeof(float)>>>(d_A, d_B, d_C);
    nvtxRangePop();

    cudaDeviceSynchronize();

    printf("\n=== Profiling Instructions ===\n\n");
    printf("1. Run with Nsight Compute:\n");
    printf("   ncu --metrics smem__conflict_requests.sum,\n");
    printf("                smem__throughput.avg.pct_of_peak_sustained_elapsed \n");
    printf("       ./ex03_verify_with_nsight\n\n");

    printf("2. Check the output:\n");
    printf("   - smem__conflict_requests.sum = 0 → Swizzle is correct!\n");
    printf("   - smem__throughput > 80%% → Good smem utilization\n\n");

    printf("3. If conflicts are high:\n");
    printf("   - Check swizzle parameters (try Swizzle<2,3,3> or Swizzle<3,3,3>)\n");
    printf("   - Verify thread count matches swizzle tuning\n");
    printf("   - Check tile dimensions are multiples of bank count (32)\n\n");

    printf("[PASS] Kernel ready for profiling\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

/*
 * CHECKPOINT: Answer before moving to Module 06
 * 
 * Q1: What Nsight Compute metric shows bank conflicts?
 *     Answer: _______________
 * 
 * Q2: What value indicates correct swizzle (no conflicts)?
 *     Answer: _______________
 * 
 * Q3: What smem throughput % is considered "good"?
 *     Answer: _______________
 * 
 * === MODULE 05 COMPLETE ===
 * Exit criteria:
 * 1. Can explain what causes shared memory bank conflicts
 * 2. Can apply Swizzle<2, 3, 3> to a smem layout
 * 3. Can run ncu --metrics to verify zero conflicts
 * 4. Can interpret smem__conflict_requests.sum and smem__throughput metrics
 *
 * Next: Module 06 — Pipeline (double buffering, load/compute overlap)
 */
