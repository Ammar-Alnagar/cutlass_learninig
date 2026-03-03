/*
 * EXERCISE: Fragment Layout and Partitioning - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Use partition_fragment_A/B/C to distribute MMA work across warps
 *   - Understand fragment layout for each thread
 *   - Map warps to output tiles (FlashAttention-2 warp specialization)
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   partition_fragment_A(tiled_mma, shape) returns thread's portion of A
 *   Fragment shape is determined by MMA atom and thread's position in warp
 *   For m16n8k16 with 32 threads: each thread holds specific elements
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/cute.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Fragment Layout and Partitioning
// ============================================================================
__global__ void fragment_layout_kernel(float* gmem_C) {
    // MENTAL MODEL: FlashAttention-2 GEMM: [64, 128] @ [128, 64] = [64, 64]
    // Partitioned across 4 warps (128 threads)
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 128;

    // MENTAL MODEL: TiledMMA for FP32 (simplified, real uses FP16)
    
    // TODO 1: Define MMA_Atom for m16n8k16 with FP32 elements
    // Hint: using MMA_Atom = MMA_Atom<MMA_Traits<Shape<Int<16>, Int<8>, Int<16>>, Element_t<float>, Element_t<float>, Element_t<float>>>;
    using MMA_Atom = /* YOUR CODE HERE */;

    // TODO 2: Create warp layout for 4 warps (128 threads)
    // Hint: auto warp_layout = make_layout(Int<128>{});
    auto warp_layout = /* YOUR CODE HERE */;

    // TODO 3: Create TiledMMA operator
    // Hint: auto tiled_mma = make_tiled_mma(MMA_Atom{}, warp_layout);
    auto tiled_mma = /* YOUR CODE HERE */;

    printf("=== Fragment Partitioning ===\n");
    printf("GEMM: [%d, %d] @ [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Threads: 128 (4 warps)\n\n");

    // CONCEPT: Partition fragments
    // Each thread gets a portion based on its threadIdx
    
    // TODO 4: Partition A fragment
    // Hint: auto A_frag = partition_fragment_A(tiled_mma, make_shape(Int<M>{}, Int<K>{}));
    auto A_frag = /* YOUR CODE HERE */;

    // TODO 5: Partition B fragment
    auto B_frag = partition_fragment_B(tiled_mma, make_shape(Int<K>{}, Int<N>{}));

    // TODO 6: Partition C fragment
    auto C_frag = /* YOUR CODE HERE */;

    // Print fragment sizes for first few threads
    if (threadIdx.x < 4) {
        printf("Thread %d fragment sizes:\n", threadIdx.x);
        
        // TODO 7: Print A_frag size
        printf("  A_frag: %d elements\n", int(size(A_frag)));
        
        // TODO 8: Print B_frag size
        printf("  B_frag: %d elements\n", int(size(B_frag)));
        
        // TODO 9: Print C_frag size
        printf("  C_frag: %d elements\n", int(size(C_frag)));
        printf("\n");
    }

    // MENTAL MODEL: FlashAttention-2 warp specialization pattern
    // Warp 0-1: Load K/V tiles from gmem to smem
    // Warp 2-3: Compute QK^T GEMM
    // Warp 4-5: Compute softmax and PV GEMM

    // Simulate warp assignment (simplified)
    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    if (threadIdx.x < 32) {  // First warp prints
        printf("=== Warp Specialization (FlashAttention-2 Pattern) ===\n");
        printf("Warp 0 (threads 0-31): Load K tile\n");
        printf("Warp 1 (threads 32-63): Load V tile\n");
        printf("Warp 2 (threads 64-95): QK^T GEMM\n");
        printf("Warp 3 (threads 96-127): PV GEMM\n\n");

        printf("Current thread: warp=%d, lane=%d\n", warp_idx, lane_idx);
    }

    // MENTAL MODEL: Initialize C fragment and store
    auto C_gmem = make_tensor(make_gmem_ptr<float>(gmem_C),
                               make_layout(make_shape(Int<M>{}, Int<N>{})));

    // Zero the accumulator
    for (int i = 0; i < size(C_frag); i++) {
        C_frag(i) = 0.0f;
    }

    // In real code: gemm(tiled_mma, A_frag, B_frag, C_frag);

    __syncthreads();

    // Store C fragment to gmem (simplified)
    if (threadIdx.x < M * N) {
        C_gmem(threadIdx.x) = C_frag(threadIdx.x % size(C_frag));
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Fragment Layout Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n\n", prop.maxThreadsPerMultiProcessor);

    constexpr int M = 64, N = 64;

    // Allocate output
    float* d_C;
    cudaMalloc(&d_C, M * N * sizeof(float));

    std::cout << "--- Kernel Output ---\n";

    // Launch with 128 threads (4 warps)
    // Warmup
    fragment_layout_kernel<<<1, 128>>>(d_C);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("fragment_layout_kernel");
    fragment_layout_kernel<<<1, 128>>>(d_C);
    nvtxRangePop();

    cudaDeviceSynchronize();

    printf("\n[PASS] Fragment layout verified\n");

    // Occupancy analysis
    printf("\n=== Occupancy Analysis ===\n");
    printf("Threads per block: 128\n");
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Theoretical blocks per SM: %d\n",
           prop.maxThreadsPerMultiProcessor / 128);

    cudaFree(d_C);

    return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex04
 * 
 * Q1: What does partition_fragment_C return?
 *     Answer: _______________
 * 
 * Q2: In FlashAttention-2, how is work divided across warps?
 *     Answer: _______________
 * 
 * Q3: For [64,64] output with 128 threads, how many elements per thread?
 *     Answer: _______________
 */
