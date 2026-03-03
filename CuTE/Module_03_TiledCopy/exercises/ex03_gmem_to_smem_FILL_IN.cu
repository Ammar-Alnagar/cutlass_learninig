/*
 * EXERCISE: gmem to smem Copy - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Create shared memory tensors with proper layout
 *   - Copy gmem -> smem for reuse in MMA operations
 *   - Calculate shared memory requirements for attention tiles
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   smem is fast (~20 TB/s on Ada) but small (~48 KB per SM)
 *   K/V tile for FlashAttention-2: [Bc, head_dim] where Bc=64 or 128
 *   For head_dim=128: one K tile = 64*128*2 bytes = 16 KB (FP16)
 *   Double buffering needs 2x smem for load/compute overlap
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/cute.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: gmem -> smem copy for K/V tiles
// ============================================================================
__global__ void gmem_to_smem_kernel(float* gmem_k, float* gmem_v, float* gmem_k_out, float* gmem_v_out) {
    // MENTAL MODEL: FlashAttention-2 tile sizes
    // Bc = column tile size (K/V sequence dimension)
    // head_dim = 128 (typical for Llama-2, Mistral)
    constexpr int Bc = 64;
    constexpr int HEAD_DIM = 128;

    // MENTAL MODEL: K/V tensors in gmem [Bc, head_dim]
    auto k_gmem_ptr = make_gmem_ptr<float>(gmem_k);
    auto v_gmem_ptr = make_gmem_ptr<float>(gmem_v);
    
    // TODO 1: Create tile layout for [Bc, HEAD_DIM]
    // Hint: auto tile_layout = make_layout(make_shape(Int<Bc>{}, Int<HEAD_DIM>{}));
    auto tile_layout = /* YOUR CODE HERE */;

    auto K_gmem = make_tensor(k_gmem_ptr, tile_layout);
    auto V_gmem = make_tensor(v_gmem_ptr, tile_layout);

    // MENTAL MODEL: Shared memory for K/V tiles
    __shared__ float K_smem_raw[Bc * HEAD_DIM];
    __shared__ float V_smem_raw[Bc * HEAD_DIM];

    auto K_smem_ptr = make_smem_ptr<float>(K_smem_raw);
    auto V_smem_ptr = make_smem_ptr<float>(V_smem_raw);

    // TODO 2: Create K_smem tensor from smem pointer and tile_layout
    // Hint: auto K_smem = make_tensor(K_smem_ptr, tile_layout);
    auto K_smem = /* YOUR CODE HERE */;

    // TODO 3: Create V_smem tensor from smem pointer and tile_layout
    auto V_smem = /* YOUR CODE HERE */;

    // MENTAL MODEL: TiledCopy for K/V loading
    // 128 threads, each copies (64*128)/128 = 64 floats = 16 float4s
    
    // TODO 4: Create TiledCopy operator with 128 threads
    // Hint: auto tiled_copy = make_tiled_copy_C<Copy_Atom<UniversalCopy, float>>(make_layout(Int<128>{}));
    auto tiled_copy = /* YOUR CODE HERE */;

    // TODO 5: Copy K from gmem to smem
    // Hint: copy(tiled_copy, K_gmem, K_smem);
    /* YOUR CODE HERE */;

    // TODO 6: Copy V from gmem to smem
    copy(tiled_copy, V_gmem, V_smem);

    __syncthreads();

    // MENTAL MODEL: Verify by reading from smem and writing to output
    auto K_out_ptr = make_gmem_ptr<float>(gmem_k_out);
    auto V_out_ptr = make_gmem_ptr<float>(gmem_v_out);
    auto K_out = make_tensor(K_out_ptr, tile_layout);
    auto V_out = make_tensor(V_out_ptr, tile_layout);

    // Copy back from smem to gmem for verification
    copy(tiled_copy, K_smem, K_out);
    copy(tiled_copy, V_smem, V_out);

    // Print smem info from thread 0
    if (threadIdx.x == 0) {
        printf("=== Shared Memory Configuration ===\n");
        printf("K tile: [%d, %d] = %d floats = %d KB\n",
               Bc, HEAD_DIM, Bc * HEAD_DIM, Bc * HEAD_DIM * 4 / 1024);
        printf("V tile: [%d, %d] = %d floats = %d KB\n",
               Bc, HEAD_DIM, Bc * HEAD_DIM, Bc * HEAD_DIM * 4 / 1024);
        printf("Total smem used: %d KB\n", 2 * Bc * HEAD_DIM * 4 / 1024);

        // MENTAL MODEL: FlashAttention-2 with double buffering needs 2x this
        printf("With double buffering: %d KB\n", 4 * Bc * HEAD_DIM * 4 / 1024);
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== gmem -> smem Copy Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Total shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor);
    printf("SM count: %d\n\n", prop.multiProcessorCount);

    // Tile configuration
    constexpr int Bc = 64;
    constexpr int HEAD_DIM = 128;
    constexpr int TILE_SIZE = Bc * HEAD_DIM;
    constexpr size_t TILE_BYTES = TILE_SIZE * sizeof(float);

    // Allocate gmem
    float *d_K, *d_V, *d_K_out, *d_V_out;
    cudaMalloc(&d_K, TILE_BYTES);
    cudaMalloc(&d_V, TILE_BYTES);
    cudaMalloc(&d_K_out, TILE_BYTES);
    cudaMalloc(&d_V_out, TILE_BYTES);

    // Initialize with deterministic values
    std::vector<float> h_K(TILE_SIZE), h_V(TILE_SIZE);
    for (int i = 0; i < TILE_SIZE; i++) {
        h_K[i] = static_cast<float>(i);
        h_V[i] = static_cast<float>(TILE_SIZE - i);
    }
    cudaMemcpy(d_K, h_K.data(), TILE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), TILE_BYTES, cudaMemcpyHostToDevice);

    std::cout << "--- Kernel Output ---\n";

    // Calculate smem size
    size_t smem_size = 2 * Bc * HEAD_DIM * sizeof(float);

    // Warmup
    gmem_to_smem_kernel<<<1, 128, smem_size>>>(d_K, d_V, d_K_out, d_V_out);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("gmem_to_smem_kernel");
    gmem_to_smem_kernel<<<1, 128, smem_size>>>(d_K, d_V, d_K_out, d_V_out);
    nvtxRangePop();

    cudaDeviceSynchronize();

    // Verify
    std::vector<float> h_K_out(TILE_SIZE), h_V_out(TILE_SIZE);
    cudaMemcpy(h_K_out.data(), d_K_out, TILE_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, TILE_BYTES, cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < TILE_SIZE; i++) {
        if (h_K_out[i] != h_K[i] || h_V_out[i] != h_V[i]) {
            pass = false;
            printf("Mismatch at index %d\n", i);
            break;
        }
    }

    printf("\n[%s] gmem -> smem copy verified\n", pass ? "PASS" : "FAIL");

    // Occupancy calculation
    printf("\n=== Occupancy Analysis ===\n");
    printf("smem per block: %zu KB\n", smem_size / 1024);
    printf("smem per SM: %d KB\n", prop.sharedMemPerMultiprocessor);
    printf("Max blocks per SM (smem-limited): %d\n",
           prop.sharedMemPerMultiprocessor / static_cast<int>(smem_size));

    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_out);
    cudaFree(d_V_out);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex04
 * 
 * Q1: For Bc=64, head_dim=128, FP16, how many KB per K tile?
 *     Answer: _______________
 * 
 * Q2: With 48 KB smem per SM, how many K/V tile pairs fit?
 *     Answer: _______________
 * 
 * Q3: Why use smem instead of reading K/V directly from gmem in MMA?
 *     Answer: _______________
 */
