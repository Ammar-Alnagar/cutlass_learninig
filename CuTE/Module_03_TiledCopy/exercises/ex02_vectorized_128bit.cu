/*
 * WHAT THIS TEACHES:
 *   - Use 128-bit vectorized loads (float4) for maximum bandwidth
 *   - Configure TiledCopy for coalesced access patterns
 *   - Measure achieved memory bandwidth vs. roofline
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 achieves 90%+ of peak memory bandwidth using vectorized loads.
 *   Each thread loads 128 bits (4 floats, 8 halfs) per instruction.
 *   This maps to: Modular AI Kernel Engineer — "high-performance attention kernels"
 *
 * MENTAL MODEL:
 *   float4 load = 1 instruction fetches 16 bytes (128 bits)
 *   Without vectorization: 4 instructions to load 4 floats
 *   Vectorization requires aligned addresses (16-byte boundary)
 *   TiledCopy handles alignment automatically with proper layout
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
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
    
    // MENTAL MODEL: TiledCopy with float4 (128-bit) vectorized loads
    // Each thread copies 4 floats = 1 float4 = 128 bits
    // Thread layout: 256 threads (each copies 1024*256/256 = 1024 floats = 256 float4s)
    
    // Copy atom for 128-bit transfer
    using CopyAtom = Copy_Atom<UniversalCopy, float>;
    
    // Thread layout: 256 threads
    auto thread_layout = make_layout(Int<256>{});
    
    // Make tiled copy operator
    auto tiled_copy = make_tiled_copy_C<CopyAtom>(thread_layout);
    
    // MENTAL MODEL: Each thread copies a contiguous chunk
    // Thread t copies elements [t * elements_per_thread, (t+1) * elements_per_thread)
    // elements_per_thread = (M * N) / 256 = 1024
    
    // Execute copy
    copy(tiled_copy, gmem_tensor, smem_tensor);
    
    __syncthreads();
    
    // Copy back to output for verification
    copy(tiled_copy, smem_tensor, gmem_out_tensor);
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
    
    // PREDICT BEFORE RUNNING:
    // Q1: With 256 threads copying 1 MB, how many bytes per thread?
    // Q2: What bandwidth do you expect (% of peak)?
    // Q3: Why must addresses be 16-byte aligned for float4?
    
    // Warmup
    vectorized_copy_kernel<<<1, 256, M * N * sizeof(float)>>>(d_in, d_out);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    //              l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    //              ./ex02_vectorized_128bit
    // Look for: 
    //   - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum should be ~1 MB per iteration
    //   - l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum should be ~65,536 (262K / 4 floats per request)
    //   - High l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum / requests = vectorized
    
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
    
    // Roofline analysis
    printf("\n=== Roofline Analysis ===\n");
    printf("This kernel is MEMORY-BOUND (bandwidth-limited)\n");
    printf("Operations: 1 load + 1 store per element\n");
    printf("Arithmetic intensity: ~0.5 FLOPs/byte (very low)\n");
    printf("Expected: Achieved BW close to peak BW\n");
    
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
 *     Answer: float4 is 128 bits = 16 bytes. GPU memory transactions must be
 *             aligned to the transaction size for atomicity.
 * 
 * Q2: If efficiency is only 50%, what are likely causes?
 *     Answer: Uncoalesced access (threads not accessing consecutive addresses),
 *             or not using vectorized loads (loading 1 float at a time instead of 4).
 * 
 * Q3: How many float4 loads does each thread perform in this exercise?
 *     Answer: (1024 * 256) / 256 threads / 4 floats per load = 256 float4 loads per thread
 */
