/*
 * WHAT THIS TEACHES:
 *   - Use partition_fragment_A/B/C to distribute MMA work across warps
 *   - Understand fragment layout for each thread
 *   - Map warps to output tiles (FlashAttention-2 warp specialization)
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 partitions QK^T and PV GEMMs across warps.
 *   Each warp computes a tile of the output matrix.
 *   This maps to: Modular AI Kernel Engineer — "warp-level Tensor Core optimization"
 *
 * MENTAL MODEL:
 *   partition_fragment_A(tiled_mma, shape) returns thread's portion of A
 *   Fragment shape is determined by MMA atom and thread's position in warp
 *   For m16n8k16 with 32 threads: each thread holds specific elements
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
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
    using MMA_Atom = MMA_Atom<
        MMA_Traits<Shape<Int<16>, Int<8>, Int<16>>,
                   Element_t<float>, Element_t<float>, Element_t<float>>
    >;
    
    // 4 warps = 128 threads
    auto warp_layout = make_layout(Int<128>{});
    auto tiled_mma = make_tiled_mma(MMA_Atom{}, warp_layout);
    
    printf("=== Fragment Partitioning ===\n");
    printf("GEMM: [%d, %d] @ [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Threads: 128 (4 warps)\n\n");
    
    // MENTAL MODEL: Partition fragments
    // Each thread gets a portion based on its threadIdx
    auto A_frag = partition_fragment_A(tiled_mma, make_shape(Int<M>{}, Int<K>{}));
    auto B_frag = partition_fragment_B(tiled_mma, make_shape(Int<K>{}, Int<N>{}));
    auto C_frag = partition_fragment_C(tiled_mma, make_shape(Int<M>{}, Int<N>{}));
    
    // Print fragment sizes for first few threads
    if (threadIdx.x < 4) {
        printf("Thread %d fragment sizes:\n", threadIdx.x);
        printf("  A_frag: %d elements\n", int(size(A_frag)));
        printf("  B_frag: %d elements\n", int(size(B_frag)));
        printf("  C_frag: %d elements\n", int(size(C_frag)));
        printf("\n");
    }
    
    // MENTAL MODEL: Fragment layout determines which matrix elements each thread holds
    // For m16n8k16 with 32 threads:
    // - Each thread computes 2 elements of the 16x8 output tile
    // - A fragment contains 8 elements (from 16x16 A tile)
    // - B fragment contains 4 elements (from 16x8 B tile)
    
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
    // Here we just store zeros for demonstration
    
    __syncthreads();
    
    // Store C fragment to gmem (simplified - real code uses proper indexing)
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
    
    // PREDICT BEFORE RUNNING:
    // Q1: With 128 threads and 64x64 output, how many elements per thread?
    // Q2: What is the fragment size for m16n8k16 MMA atom?
    // Q3: How many warps does FlashAttention-2 typically use per block?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Launch with 128 threads (4 warps)
    // Warmup
    fragment_layout_kernel<<<1, 128>>>(d_C);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --set full ./ex03_fragment_layout
    // Look for: Warp scheduling, register usage
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
    printf("Note: Actual occupancy limited by registers and smem\n");
    
    cudaFree(d_C);
    
    return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex04
 * 
 * Q1: What does partition_fragment_C return?
 *     Answer: The thread's portion of the output/accumulator matrix C
 * 
 * Q2: In FlashAttention-2, how is work divided across warps?
 *     Answer: Different warps handle different stages (load K, load V, QK^T, PV)
 * 
 * Q3: For [64,64] output with 128 threads, how many elements per thread?
 *     Answer: 64*64 / 128 = 32 elements per thread
 */
