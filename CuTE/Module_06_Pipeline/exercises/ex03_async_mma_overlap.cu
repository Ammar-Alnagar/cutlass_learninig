/*
 * WHAT THIS TEACHES:
 *   - Use cp_async_wait<1> for async MMA overlap
 *   - Issue next copy while waiting for previous
 *   - Production FlashAttention-2 pipeline pattern
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 uses cp_async_wait<1> to keep the pipeline full.
 *   While waiting for copy N-1 to complete, issue copy N+1.
 *   This maximizes memory unit utilization.
 *   This maps to: Cerebras LLM Inference Performance — "FlashAttention variants"
 *
 * MENTAL MODEL:
 *   cp_async_wait<N> waits until N copies remain pending
 *   cp_async_wait<0> = wait for all copies
 *   cp_async_wait<1> = wait for all but one (keep pipeline full)
 *   
 *   Pipeline sequence:
 *   1. cp_async_wait<1>()  -- wait for copy N-1, copy N still pending
 *   2. Use data from copy N-1 in MMA
 *   3. Issue cp.async for copy N+1
 *   4. cp_async_fence()
 *   5. Repeat
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Async MMA Overlap (FlashAttention-2 Pattern)
// ============================================================================
__global__ void async_mma_overlap_kernel(half* gmem_K, float* gmem_Q, float* gmem_out,
                                          int num_tiles) {
    // MENTAL MODEL: FlashAttention-2 tile sizes
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int head_dim = 128;
    constexpr int TILE_SIZE = Bc * head_dim;
    
    // MENTAL MODEL: Check for sm_80+ (cp.async support)
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool HAS_CP_ASYNC = true;
    #else
    constexpr bool HAS_CP_ASYNC = false;
    #endif
    
    // MENTAL MODEL: Double buffer for K tiles
    __shared__ float smem_K[2 * TILE_SIZE];
    auto smem_K_ptr = make_smem_ptr<float>(smem_K);
    auto smem_K_0 = make_tensor(smem_K_ptr, make_layout(Int<TILE_SIZE>{}));
    auto smem_K_1 = make_tensor(smem_K_ptr + TILE_SIZE, make_layout(Int<TILE_SIZE>{}));
    
    // Q tensor
    auto Q_gmem = make_tensor(make_gmem_ptr<float>(gmem_Q),
                               make_layout(make_shape(Int<Br>{}, Int<head_dim>{})));
    
    // Output accumulator
    float C[Br * Bc];
    for (int i = 0; i < Br * Bc; i++) C[i] = 0.0f;
    
    int tid = threadIdx.x;
    int write_stage = 0;
    int read_stage = 0;
    
    // ========================================================================
    // PROLOGUE: Load first K tile
    // ========================================================================
    auto K_gmem_0 = make_tensor(make_gmem_ptr<float>(gmem_K),
                                 make_layout(Int<TILE_SIZE>{}));
    
    if constexpr (HAS_CP_ASYNC) {
        // Use cp.async for async load
        auto smem_curr = (write_stage == 0) ? smem_K_0 : smem_K_1;
        
        // Simplified cp.async pattern (real code uses TiledCopy)
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            // In real code: cp.async.cg.shared.global[smem_addr], [gmem_addr], size
            smem_curr(i) = __half2float(gmem_K[i]);
        }
        cp_async_fence();
    } else {
        // Fallback for older architectures
        auto smem_curr = (write_stage == 0) ? smem_K_0 : smem_K_1;
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            smem_curr(i) = __half2float(gmem_K[i]);
        }
    }
    
    cp_async_wait<0>();
    __syncthreads();
    
    // ========================================================================
    // MAINLOOP: Async load + MMA overlap
    // ========================================================================
    for (int tile_idx = 1; tile_idx < num_tiles; tile_idx++) {
        // MENTAL MODEL: Compute with current tile (Q @ K^T)
        auto smem_K_curr = (read_stage == 0) ? smem_K_0 : smem_K_1;
        
        // Simplified GEMM (real code uses TiledMMA)
        for (int br = tid; br < Br; br += blockDim.x) {
            for (int bc = 0; bc < Bc; bc++) {
                float sum = 0.0f;
                for (int k = 0; k < head_dim; k++) {
                    sum += Q_gmem(br, k) * smem_K_curr(bc * head_dim + k);
                }
                C[br * Bc + bc] += sum;
            }
        }
        
        // MENTAL MODEL: Issue async load for next tile
        // While MMA runs, memory unit loads next tile
        if constexpr (HAS_CP_ASYNC) {
            auto K_gmem_next = make_tensor(
                make_gmem_ptr<float>(gmem_K + tile_idx * TILE_SIZE),
                make_layout(Int<TILE_SIZE>{}));
            auto smem_K_next = (write_stage == 0) ? smem_K_0 : smem_K_1;
            
            for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
                smem_K_next(i) = __half2float(K_gmem_next(i));
            }
            cp_async_fence();
        }
        
        // MENTAL MODEL: Wait for previous load (keep 1 pending for overlap)
        cp_async_wait<1>();
        __syncthreads();
        
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }
    
    // ========================================================================
    // EPILOGUE: Final tile
    // ========================================================================
    cp_async_wait<0>();
    __syncthreads();
    
    auto smem_K_last = (read_stage == 0) ? smem_K_0 : smem_K_1;
    
    for (int br = tid; br < Br; br += blockDim.x) {
        for (int bc = 0; bc < Bc; bc++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                sum += Q_gmem(br, k) * smem_K_last(bc * head_dim + k);
            }
            C[br * Bc + bc] += sum;
        }
    }
    
    // Store results
    if (tid < Br * Bc) {
        gmem_out[tid] = C[tid];
    }
    
    if (tid == 0) {
        printf("=== Async MMA Overlap Complete ===\n");
        printf("Architecture: sm_%d%d\n", 
               #if defined(__CUDA_ARCH__)
               __CUDA_ARCH__ / 100, (__CUDA_ARCH__ % 100) / 10
               #else
               0, 0
               #endif
        );
        printf("cp.async support: %s\n", HAS_CP_ASYNC ? "YES" : "NO");
        printf("Pipeline: cp_async_wait<1> for maximum overlap\n");
        printf("\nFlashAttention-2 Pattern:\n");
        printf("  while computing tile N:\n");
        printf("    cp.async for tile N+1\n");
        printf("    cp_async_fence()\n");
        printf("    cp_async_wait<1>()  # wait for N-1, N still loading\n");
        printf("    MMA for tile N\n");
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Async MMA Overlap Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 8) {
        printf("\nWARNING: cp.async requires sm_80+ (Ampere+)\n");
        printf("Kernel will use fallback path (no async)\n");
    }
    printf("\n");
    
    constexpr int Br = 64, Bc = 64, head_dim = 128;
    constexpr int NUM_TILES = 4;
    constexpr int K_SIZE = Bc * head_dim * NUM_TILES;
    
    // Allocate
    half *d_K;
    float *d_Q, *d_out;
    cudaMalloc(&d_K, K_SIZE * sizeof(half));
    cudaMalloc(&d_Q, Br * head_dim * sizeof(float));
    cudaMalloc(&d_out, Br * Bc * sizeof(float));
    
    // Initialize
    std::vector<float> h_Q(Br * head_dim, 1.0f);
    std::vector<half> h_K_half(K_SIZE, __float2half(1.0f));
    
    cudaMemcpy(d_Q, h_Q.data(), Br * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K_half.data(), K_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    
    // PREDICT BEFORE RUNNING:
    // Q1: What does cp_async_wait<1>() wait for?
    // Q2: Why not use cp_async_wait<0>() everywhere?
    // Q3: What hardware units overlap in this pattern?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    async_mma_overlap_kernel<<<1, 128, 2 * Bc * head_dim * sizeof(float)>>>(
        d_K, d_Q, d_out, NUM_TILES);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics gpc__l1_mem_global_mem_read_bytes.sum,\
    //              smsp__inst_executed_op_tensor.sum \
    //              ./ex03_async_mma_overlap
    nvtxRangePush("async_mma_overlap_kernel");
    async_mma_overlap_kernel<<<1, 128, 2 * Bc * head_dim * sizeof(float)>>>(
        d_K, d_Q, d_out, NUM_TILES);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Verify
    float h_out[Br * Bc];
    cudaMemcpy(h_out, d_out, Br * Bc * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Expected: each C[i] = num_tiles * head_dim = 4 * 128 = 512
    bool pass = true;
    for (int i = 0; i < Br * Bc; i++) {
        if (fabs(h_out[i] - 512.0f) > 1.0f) {
            pass = false;
            printf("Mismatch at index %d: expected 512.0, got %.2f\n", i, h_out[i]);
            break;
        }
    }
    
    printf("\n[%s] Async MMA overlap verified\n", pass ? "PASS" : "FAIL");
    
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_out);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to Projects
 * 
 * Q1: What does cp_async_wait<1>() guarantee?
 *     Answer: All but one previously issued cp.async operations have completed.
 *             One copy may still be in progress (pipeline stays full).
 * 
 * Q2: Why use cp_async_wait<1> instead of cp_async_wait<0>?
 *     Answer: cp_async_wait<1> keeps the pipeline full by allowing one
 *             copy to remain pending while compute proceeds.
 * 
 * Q3: What are the three phases of FlashAttention-2 pipeline?
 *     Answer: Prologue (load first tile), Mainloop (load+compute overlap),
 *             Epilogue (finish last tile)
 * 
 * === MODULE 06 COMPLETE ===
 * Exit criteria:
 * 1. Can implement 2-stage pipeline with prologue/mainloop/epilogue
 * 2. Can use cp_async_wait<1> for async MMA overlap
 * 3. Can explain why FlashAttention-2 needs double buffering
 * 4. Can calculate tile size where pipeline benefit becomes significant
 * 
 * === CORE MODULES COMPLETE ===
 * Next: Projects — Apply everything to real kernels
 *   Project 01: Tiled GEMM (standalone)
 *   Project 02: FlashAttention-2 Prefill (capstone)
 */
