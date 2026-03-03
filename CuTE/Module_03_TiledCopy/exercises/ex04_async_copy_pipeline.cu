/*
 * WHAT THIS TEACHES:
 *   - Use cp.async for asynchronous gmem -> smem copy
 *   - Pipeline load and compute with cp_async_fence and cp_async_wait
 *   - Overlap memory transfer with computation (FlashAttention-2 pattern)
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 hides memory latency by loading the next K/V tile while
 *   computing QK^T with the current tile. This is the prologue/mainloop/epilogue
 *   pipeline pattern used in all high-performance kernels.
 *   This maps to: NVIDIA DL Software Engineer — "pipelined FlashAttention-2"
 *
 * MENTAL MODEL:
 *   cp.async issues async copy (returns immediately, copy happens in background)
 *   cp_async_fence commits all pending cp.async operations
 *   cp_async_wait<N> waits until N copies remain pending (N=0 means all complete)
 *   Pipeline: while computing tile i, issue loads for tile i+1
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Async copy pipeline (simplified 2-stage)
// ============================================================================
__global__ void async_pipeline_kernel(float* gmem_data, float* gmem_out) {
    // MENTAL MODEL: Small example for clarity
    // Real FlashAttention-2 uses larger tiles and more stages
    constexpr int TILE_SIZE = 256;  // elements per tile
    constexpr int NUM_TILES = 4;    // number of tiles to process
    constexpr int TOTAL_SIZE = TILE_SIZE * NUM_TILES;
    
    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    auto gmem_out_ptr = make_gmem_ptr<float>(gmem_out);
    
    // MENTAL MODEL: Shared memory for double buffering
    // Two buffers: while computing tile 0, load tile 1 into buffer 1
    __shared__ float smem_buffer[2 * TILE_SIZE];
    auto smem_ptr = make_smem_ptr<float>(smem_buffer);
    
    // Two views into the buffer (ping-pong)
    auto smem_0 = make_tensor(smem_ptr, make_layout(Int<TILE_SIZE>{}));
    auto smem_1 = make_tensor(smem_ptr + TILE_SIZE, make_layout(Int<TILE_SIZE>{}));
    
    // MENTAL MODEL: TiledCopy for async copy
    // cp.async requires special atom: Copy_Atom<SM80_CP_ASYNC, T>
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC, float>;
    auto tiled_copy = make_tiled_copy_C<CopyAtom>(make_layout(Int<128>{}));
    
    // MENTAL MODEL: Pipeline state
    // write_stage: which buffer to load into next (0 or 1)
    // read_stage: which buffer to read from next
    
    int write_stage = 0;
    int read_stage = 0;
    
    // MENTAL MODEL: Prologue - load first tile
    auto gmem_tile = local_tile(
        make_tensor(gmem_ptr, make_layout(Int<TOTAL_SIZE>{})),
        make_layout(Int<TILE_SIZE>{}),
        make_coord(0));
    
    if (write_stage == 0) {
        copy(tiled_copy, gmem_tile, smem_0);
    } else {
        copy(tiled_copy, gmem_tile, smem_1);
    }
    cp_async_fence();  // Commit the async copy
    
    // MENTAL MODEL: Main loop - process remaining tiles
    for (int tile_idx = 1; tile_idx < NUM_TILES; tile_idx++) {
        // Wait for previous load to complete
        cp_async_wait<0>();  // Wait until 0 copies pending
        __syncthreads();
        
        // MENTAL MODEL: Process current tile (read_stage)
        // In real code, this would be QK^T GEMM
        // Here we just simulate compute with a delay
        auto smem_current = (read_stage == 0) ? smem_0 : smem_1;
        
        // Simulate compute (each thread processes its elements)
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            float val = smem_current(i);
            val = val * 2.0f + 1.0f;  // Simple "compute"
            smem_current(i) = val;
        }
        
        // Issue load for next tile (write_stage)
        gmem_tile = local_tile(
            make_tensor(gmem_ptr, make_layout(Int<TOTAL_SIZE>{})),
            make_layout(Int<TILE_SIZE>{}),
            make_coord(tile_idx));
        
        auto smem_next = (write_stage == 0) ? smem_0 : smem_1;
        copy(tiled_copy, gmem_tile, smem_next);
        cp_async_fence();
        
        // Ping-pong buffer switching
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }
    
    // MENTAL MODEL: Epilogue - wait for last load and process
    cp_async_wait<0>();
    __syncthreads();
    
    auto smem_current = (read_stage == 0) ? smem_0 : smem_1;
    for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
        float val = smem_current(i);
        val = val * 2.0f + 1.0f;
        smem_current(i) = val;
    }
    
    // MENTAL MODEL: Write results back to gmem
    for (int tile_idx = 0; tile_idx < NUM_TILES; tile_idx++) {
        auto smem_src = (tile_idx % 2 == 0) ? smem_0 : smem_1;
        auto gmem_dst = local_tile(
            make_tensor(gmem_out_ptr, make_layout(Int<TOTAL_SIZE>{})),
            make_layout(Int<TILE_SIZE>{}),
            make_coord(tile_idx));
        
        if (threadIdx.x == 0 && tile_idx == 0) {
            // Copy first tile for verification
            for (int i = 0; i < TILE_SIZE; i++) {
                gmem_dst(i) = smem_src(i);
            }
        }
    }
}

// ============================================================================
// CPU REFERENCE: Simulate pipeline computation
// ============================================================================
void cpu_reference_pipeline(float* input, float* output, int tile_size, int num_tiles) {
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 0; i < tile_size; i++) {
            int idx = t * tile_size + i;
            output[idx] = input[idx] * 2.0f + 1.0f;
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
    printf("=== Async Copy Pipeline Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    
    // Check for cp.async support (sm_80+)
    if (prop.major < 8) {
        printf("ERROR: cp.async requires sm_80 or higher (Ampere+)\n");
        return 1;
    }
    
    constexpr int TILE_SIZE = 256;
    constexpr int NUM_TILES = 4;
    constexpr int TOTAL_SIZE = TILE_SIZE * NUM_TILES;
    constexpr size_t TOTAL_BYTES = TOTAL_SIZE * sizeof(float);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, TOTAL_BYTES);
    cudaMalloc(&d_out, TOTAL_BYTES);
    
    // Initialize input
    std::vector<float> h_in(TOTAL_SIZE);
    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_in[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_in, h_in.data(), TOTAL_BYTES, cudaMemcpyHostToDevice);
    
    // PREDICT BEFORE RUNNING:
    // Q1: What does cp_async_fence() do?
    // Q2: What does cp_async_wait<0>() guarantee?
    // Q3: Why do we need __syncthreads() after cp_async_wait?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics gpc__l1_mem_global_mem_read_bytes.sum,\
    //              nssubunit__cycles_active.avg.pct_of_peak_sustained_elapsed \
    //              ./ex04_async_copy_pipeline
    // Look for:
    //   - Overlap between load and compute (if visible in timeline)
    //   - High L1 read throughput
    nvtxRangePush("async_pipeline_kernel");
    async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Verify
    std::vector<float> h_out(TOTAL_SIZE);
    cudaMemcpy(h_out.data(), d_out, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    
    std::vector<float> h_expected(TOTAL_SIZE);
    cpu_reference_pipeline(h_in.data(), h_expected.data(), TILE_SIZE, NUM_TILES);
    
    bool pass = true;
    for (int i = 0; i < TOTAL_SIZE; i++) {
        if (h_out[i] != h_expected[i]) {
            pass = false;
            printf("Mismatch at index %d: expected %.1f, got %.1f\n", 
                   i, h_expected[i], h_out[i]);
            break;
        }
    }
    
    printf("\n[%s] Async pipeline verified\n", pass ? "PASS" : "FAIL");
    
    // Timing comparison: pipelined vs. non-pipelined (simulated)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= 100.0f;
    
    printf("\n[Timing] Average kernel time: %.3f ms\n", elapsed_ms);
    printf("Note: Pipeline benefit visible when compute >> load time\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to Module 04
 * 
 * Q1: What is the purpose of cp_async_fence()?
 *     Answer: Commits all pending cp.async operations. Without fence,
 *             async copies never execute.
 * 
 * Q2: What does cp_async_wait<0>() guarantee?
 *     Answer: All previously issued cp.async operations have completed.
 *             It's safe to use the data in smem.
 * 
 * Q3: In FlashAttention-2, what is loaded during the "prologue"?
 *     Answer: The first K/V tile is loaded before the main loop starts.
 *             This ensures the pipeline is full from iteration 0.
 * 
 * === MODULE 03 COMPLETE ===
 * Exit criteria:
 * 1. Can construct TiledCopy with make_tiled_copy(Copy_Atom, thread_layout, smem_layout)
 * 2. Can use 128-bit vectorized loads (float4) for >80% bandwidth efficiency
 * 3. Can copy gmem -> smem with proper shared memory sizing
 * 4. Can pipeline with cp.async, cp_async_fence, cp_async_wait
 * 
 * Next: Module 04 — TiledMMA (warp-level GEMM with Tensor Cores)
 */
