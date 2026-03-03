/*
 * EXERCISE: Double Buffer Pipeline - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Implement 2-stage pipeline with double buffering
 *   - Use ping-pong buffers for load/compute overlap
 *   - Structure kernel as prologue/mainloop/epilogue
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   Double buffering = 2 smem buffers (buffer[0], buffer[1])
 *   Stage 0: Load tile 0 → buffer[0], Compute tile 0
 *   Stage 1: Load tile 1 → buffer[1], Compute tile 1 (while loading tile 2)
 *   Key: Load and compute happen simultaneously (different hardware units)
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Double-Buffered Pipeline
// ============================================================================
__global__ void double_buffer_kernel(float* gmem_tiles, float* gmem_out, int num_tiles) {
    // MENTAL MODEL: Each tile is 256 floats
    constexpr int TILE_SIZE = 256;

    // MENTAL MODEL: Double buffer in smem (2 tiles)
    __shared__ float smem_buffer[2 * TILE_SIZE];
    auto smem_ptr = make_smem_ptr<float>(smem_buffer);
    
    // TODO 1: Create two views into the buffer (ping-pong)
    // Hint: auto smem_0 = make_tensor(smem_ptr, make_layout(Int<TILE_SIZE>{}));
    //       auto smem_1 = make_tensor(smem_ptr + TILE_SIZE, make_layout(Int<TILE_SIZE>{}));
    auto smem_0 = /* YOUR CODE HERE */;
    auto smem_1 = /* YOUR CODE HERE */;

    // MENTAL MODEL: Input tensor (all tiles in gmem)
    auto gmem_all = make_tensor(make_gmem_ptr<float>(gmem_tiles),
                                 make_layout(Int<num_tiles * TILE_SIZE>{}));

    // MENTAL MODEL: Output tensor
    auto out_all = make_tensor(make_gmem_ptr<float>(gmem_out),
                                make_layout(Int<num_tiles * TILE_SIZE>{}));

    int tid = threadIdx.x;

    // MENTAL MODEL: Pipeline state
    int write_stage = 0;  // Which buffer to write next
    int read_stage = 0;   // Which buffer to read next

    // ========================================================================
    // PROLOGUE: Load first tile
    // ========================================================================
    
    // TODO 2: Get first tile from gmem using local_tile
    // Hint: auto gmem_tile_0 = local_tile(gmem_all, make_layout(Int<TILE_SIZE>{}), make_coord(0));
    auto gmem_tile_0 = /* YOUR CODE HERE */;

    // TODO 3: Copy tile 0 to buffer[0] (smem_0)
    // Each thread copies a portion
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        /* YOUR CODE HERE */;
    }

    __syncthreads();

    // ========================================================================
    // MAINLOOP: Load next tile while computing current tile
    // ========================================================================
    for (int tile_idx = 1; tile_idx < num_tiles; tile_idx++) {
        // MENTAL MODEL: Compute current tile (read_stage)
        
        // TODO 4: Select current buffer based on read_stage
        // Hint: auto smem_current = (read_stage == 0) ? smem_0 : smem_1;
        auto smem_current = /* YOUR CODE HERE */;

        // Simulate compute (in real code, this is QK^T GEMM)
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            float val = smem_current(i);
            val = val * 2.0f + 1.0f;  // "Compute" operation
            smem_current(i) = val;
        }

        __syncthreads();

        // MENTAL MODEL: Load next tile (write_stage)
        auto gmem_tile_next = local_tile(gmem_all, make_layout(Int<TILE_SIZE>{}),
                                          make_coord(tile_idx));
        
        // TODO 5: Select next buffer based on write_stage
        auto smem_next = (write_stage == 0) ? smem_0 : smem_1;

        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            smem_next(i) = gmem_tile_next(i);
        }

        __syncthreads();

        // TODO 6: Ping-pong buffer switch
        // Hint: write_stage = 1 - write_stage; read_stage = 1 - read_stage;
        /* YOUR CODE HERE */;
    }

    // ========================================================================
    // EPILOGUE: Process last tile
    // ========================================================================
    
    // TODO 7: Select last buffer based on read_stage
    auto smem_last = (read_stage == 0) ? smem_0 : smem_1;

    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        float val = smem_last(i);
        val = val * 2.0f + 1.0f;
        smem_last(i) = val;
    }

    __syncthreads();

    // MENTAL MODEL: Write all results back to gmem
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        auto smem_src = (tile_idx % 2 == 0) ? smem_0 : smem_1;
        auto gmem_dst = local_tile(out_all, make_layout(Int<TILE_SIZE>{}),
                                    make_coord(tile_idx));

        // Only first tile for verification (simplified)
        if (tile_idx == 0) {
            for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
                gmem_dst(i) = smem_src(i);
            }
        }
    }

    if (tid == 0) {
        printf("=== Double-Buffer Pipeline Complete ===\n");
        printf("Tiles processed: %d\n", num_tiles);
        printf("Pipeline stages: 2 (double buffer)\n");
        printf("Pattern: Prologue → Mainloop (load+compute) → Epilogue\n");
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
    printf("=== Double Buffer Pipeline Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n\n", prop.multiProcessorCount);

    constexpr int TILE_SIZE = 256;
    constexpr int NUM_TILES = 8;
    constexpr int TOTAL_SIZE = TILE_SIZE * NUM_TILES;
    constexpr size_t TOTAL_BYTES = TOTAL_SIZE * sizeof(float);

    // Allocate
    float *d_in, *d_out;
    cudaMalloc(&d_in, TOTAL_BYTES);
    cudaMalloc(&d_out, TOTAL_BYTES);

    // Initialize
    std::vector<float> h_in(TOTAL_SIZE);
    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_in[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_in, h_in.data(), TOTAL_BYTES, cudaMemcpyHostToDevice);

    std::cout << "--- Kernel Output ---\n";

    // Warmup
    double_buffer_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out, NUM_TILES);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("double_buffer_kernel");
    double_buffer_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out, NUM_TILES);
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

    printf("\n[%s] Double buffer pipeline verified\n", pass ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What are the three phases of a pipelined kernel?
 *     Answer: _______________
 * 
 * Q2: How many smem buffers does 2-stage pipeline need?
 *     Answer: _______________
 * 
 * Q3: Why does double buffering hide latency?
 *     Answer: _______________
 */
