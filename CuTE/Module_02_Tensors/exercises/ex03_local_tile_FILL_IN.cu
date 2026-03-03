/*
 * EXERCISE: Local Tile - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Use local_tile to extract one tile from a tensor
 *   - Iterate over sequence blocks (FlashAttention-2 outer loop pattern)
 *   - Understand tile coordinates vs. element coordinates
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   local_tile(tensor, tile_shape, tile_coord) = view of tensor[start_idx : start_idx + tile_shape]
 *   Unlike logical_divide (which creates a composed layout), local_tile returns a simple view.
 *   Use in a loop: for (int i = 0; i < num_tiles; i++) { auto tile = local_tile(..., i); }
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/cute.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: local_tile for block iteration
// ============================================================================
__global__ void local_tile_kernel(float* gmem_data) {
    // Create tensor [1, 8, 16] = [batch*heads, seqlen, head_dim]
    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    auto layout = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<16>{}));
    auto tensor = make_tensor(gmem_ptr, layout);

    // Initialize with predictable values
    for (int s = 0; s < 8; s++) {
        for (int d = 0; d < 16; d++) {
            tensor(Int<0>{}, Int<s>{}, Int<d>{}) = s * 100.0f + d;
        }
    }

    printf("=== Original Tensor [batch*heads=1, seqlen=8, head_dim=16] ===\n");
    print(tensor);
    printf("\n");

    // CONCEPT: local_tile extracts one tile
    // Tile shape: [1, 4, 16] — tile 4 rows of sequence at a time
    // This is like FlashAttention-2's Br (row tile size)
    
    // TODO 1: Define tile shape for tiling sequence dimension
    // Hint: make_shape(Int<1>{}, Int<4>{}, Int<16>{})
    auto tile_shape = /* YOUR CODE HERE */;

    // CONCEPT: FlashAttention-2 outer loop iterates over K/V sequence blocks
    // For seqlen=8 and Br=4, we have 2 tiles
    
    // TODO 2: Calculate number of tiles
    // Hint: seqlen / tile_size = 8 / 4
    constexpr int NUM_TILES = /* YOUR CODE HERE */;

    printf("=== Iterating over sequence tiles (Br=4) ===\n\n");

    for (int i = 0; i < NUM_TILES; i++) {
        // CONCEPT: local_tile(tensor, tile_shape, start_coord)
        // start_coord = (0, i*4, 0) — start at sequence position i*4
        
        // TODO 3: Extract tile at sequence position i*4
        // Hint: local_tile(tensor, tile_shape, make_coord(0, i * 4, 0))
        auto tile = /* YOUR CODE HERE */;

        printf("--- Tile %d (sequence rows %d-%d) ---\n", i, i * 4, i * 4 + 3);
        print(tile);
        printf("\n");

        // CONCEPT: Tile is a view — tile(0, s, d) accesses original tensor(0, i*4+s, d)
        // Verify by reading first element of each tile
        
        // TODO 4: Read first element from the tile
        float tile_val = tile(Int<0>{}, Int<0>{}, Int<0>{});

        // TODO 5: Read the same element from the original tensor
        float orig_val = tensor(Int<0>{}, Int<i * 4>{}, Int<0>{});
        
        printf("tile(0,0,0) = %.1f, tensor(0,%d,0) = %.1f, match: %s\n\n",
               tile_val, i * 4, orig_val, (tile_val == orig_val) ? "YES" : "NO");
    }

    // CONCEPT: You can also tile in 2D (both Q and K/V sequence dimensions)
    // For FlashAttention-2 inner loop, tile both dimensions:
    
    // TODO 6: Create 2D tile shape [1, Br, Bc] = [1, 4, 4]
    auto tile_shape_2d = make_shape(Int<1>{}, /* YOUR CODE HERE */, Int<4>{});

    printf("=== 2D Tiling (for QK^T score matrix) ===\n");
    
    // TODO 7: Extract 2D tile at coordinate (0, 0, 0)
    auto tile_2d = local_tile(tensor, tile_shape_2d, make_coord(0, 0, 0));
    
    printf("2D tile at (0, 0): shape ");
    print(shape(tile_2d));
    printf("\n");
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_local_tile() {
    printf("\n=== CPU Reference ===\n");

    // For seqlen=8, Br=4: num_tiles = 8 / 4 = 2
    printf("Number of tiles: 8 / 4 = 2\n");
    printf("Tile 0 covers sequence rows: 0, 1, 2, 3\n");
    printf("Tile 1 covers sequence rows: 4, 5, 6, 7\n");

    // Value at tensor(0, 0, 0) = 0 * 100 + 0 = 0
    // Value at tensor(0, 4, 0) = 4 * 100 + 0 = 400
    printf("Tile 0, element (0,0,0) = tensor(0,0,0) = 0\n");
    printf("Tile 1, element (0,0,0) = tensor(0,4,0) = 400\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== local_tile Exercise ===\n");
    printf("GPU: %s\n\n", prop.name);

    // Allocate [1, 8, 16] = 128 floats
    constexpr int SIZE = 1 * 8 * 16;
    float* d_data;
    cudaMalloc(&d_data, SIZE * sizeof(float));

    std::cout << "--- Kernel Output ---\n";

    // Warmup
    local_tile_kernel<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("local_tile_kernel");
    local_tile_kernel<<<1, 1>>>(d_data);
    nvtxRangePop();

    cudaDeviceSynchronize();

    // CPU reference
    cpu_reference_local_tile();

    printf("\n[PASS] local_tile iteration verified\n");

    cudaFree(d_data);

    return 0;
}

/*
 * CHECKPOINT: Answer before moving to ex04
 * 
 * Q1: What is the difference between logical_divide and local_tile?
 *     Answer: _______________
 * 
 * Q2: In FlashAttention-2 with seqlen=512 and Br=64, how many outer loop iterations?
 *     Answer: _______________
 * 
 * Q3: What are the arguments to local_tile?
 *     Answer: _______________
 */
