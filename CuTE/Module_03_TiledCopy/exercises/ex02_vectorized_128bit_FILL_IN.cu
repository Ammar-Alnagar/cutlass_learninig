/*
 * EXERCISE: Vectorized 128-bit Copy - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Use 128-bit vectorized loads (float4) for maximum bandwidth
 *   - Configure TiledCopy for coalesced access patterns
 *   - Measure achieved memory bandwidth vs. roofline
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   float4 load = 1 instruction fetches 16 bytes (128 bits)
 *   Without vectorization: 4 instructions to load 4 floats
 *   Vectorization requires aligned addresses (16-byte boundary)
 *   TiledCopy handles alignment automatically with proper layout
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/cute.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <iomanip>

using namespace cute;

// ============================================================================
// KERNEL: Vectorized 128-bit TiledCopy
// ============================================================================
__global__ void vectorized_copy_kernel(float* gmem_data, float* gmem_out) {
    // MENTAL MODEL: Larger tensor for meaningful bandwidth measurement
    // [1024, 256] = 262,144 floats = 1 MB
    constexpr int M = 1024;
    constexpr int N = 256;

    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    auto gmem_layout = make_layout(make_shape(Int<M>{}, Int<N>{}));
    auto gmem_tensor = make_tensor(gmem_ptr, gmem_layout);

    // Output tensor for verification
    auto gmem_out_ptr = make_gmem_ptr<float>(gmem_out);
    auto gmem_out_tensor = make_tensor(gmem_out_ptr, gmem_layout);

    // MENTAL MODEL: Shared memory for the tile
    __shared__ float smem_static[M * N];
    auto smem_ptr = make_smem_ptr<float>(smem_static);
    auto smem_tensor = make_tensor(smem_ptr, gmem_layout);

    // CONCEPT: TiledCopy with float4 (128-bit) vectorized loads
    // Each thread copies 4 floats = 1 float4 = 128 bits
    // Thread layout: 256 threads
    
    // TODO 1: Define Copy Atom for 128-bit transfer
    // Hint: using CopyAtom = Copy_Atom<UniversalCopy, float>;
    using CopyAtom = /* YOUR CODE HERE */;

    // TODO 2: Create thread layout for 256 threads
    // Hint: auto thread_layout = make_layout(Int<256>{});
    auto thread_layout = /* YOUR CODE HERE */;

    // TODO 3: Make tiled copy operator
    // Hint: auto tiled_copy = make_tiled_copy_C<CopyAtom>(thread_layout);
    auto tiled_copy = /* YOUR CODE HERE */;

    // MENTAL MODEL: Each thread copies a contiguous chunk
    // Thread t copies elements [t * elements_per_thread, (t+1) * elements_per_thread)
    // elements_per_thread = (M * N) / 256 = 1024

    // TODO 4: Execute copy from gmem to smem
    // Hint: copy(tiled_copy, gmem_tensor, smem_tensor);
    /* YOUR CODE HERE */;

    __syncthreads();

    // TODO 5: Copy back to output for verification
    // Hint: copy(tiled_copy, smem_tensor, gmem_out_tensor);
    /* YOUR CODE HERE */;
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_copy(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i];
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Vectorized 128-bit Copy Exercise ===\n");
    printf("GPU: %s\n", prop.name);

    // Calculate peak bandwidth
    float peak_bw = 2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6;
    printf("Peak memory bandwidth: %.1f GB/s\n\n", peak_bw);

    // Allocate tensors: [1024, 256] = 262,144 floats = 1 MB
    constexpr int M = 1024;
    constexpr int N = 256;
    constexpr int SIZE = M * N;
    constexpr size_t BYTES = SIZE * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, BYTES);
    cudaMalloc(&d_out, BYTES);

    // Initialize with deterministic values
    std::vector<float> h_data(SIZE);
    for (int i = 0; i < SIZE; i++) {
        h_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_in, h_data.data(), BYTES, cudaMemcpyHostToDevice);

    // Warmup
    vectorized_copy_kernel<<<1, 256, M * N * sizeof(float)>>>(d_in, d_out);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("vectorized_copy_kernel");

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int NUM_ITER = 100;

    // Warmup runs
    for (int i = 0; i < 10; i++) {
        vectorized_copy_kernel<<<1, 256, M * N * sizeof(float)>>>(d_in, d_out);
    }
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; i++) {
        vectorized_copy_kernel<<<1, 256, M * N * sizeof(float)>>>(d_in, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    nvtxRangePop();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= NUM_ITER;

    // Calculate bandwidth
    float bandwidth = BYTES / elapsed_ms / 1e6;
    float efficiency = 100.0f * bandwidth / peak_bw;

    printf("\n=== Results ===\n");
    printf("Transfer size: %.2f MB\n", BYTES / 1e6);
    printf("Average time: %.3f ms\n", elapsed_ms);
    printf("Achieved bandwidth: %.1f GB/s\n", bandwidth);
    printf("Peak bandwidth: %.1f GB/s\n", peak_bw);
    printf("Efficiency: %.1f%%\n", efficiency);

    // Verify correctness
    std::vector<float> h_out(SIZE);
    cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_out[i] != static_cast<float>(i)) {
            pass = false;
            printf("Mismatch at index %d: expected %f, got %f\n", i, static_cast<float>(i), h_out[i]);
            break;
        }
    }

    printf("\n[%s] Vectorized copy verified\n", pass ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 * 
 * Q1: Why does float4 require 16-byte alignment?
 *     Answer: _______________
 * 
 * Q2: If efficiency is only 50%, what are likely causes?
 *     Answer: _______________
 * 
 * Q3: How many float4 loads does each thread perform in this exercise?
 *     Answer: _______________
 */
