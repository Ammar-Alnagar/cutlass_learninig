/*
 * WHAT THIS TEACHES:
 *   - Complete tiled GEMM combining all CuTe concepts
 *   - Production-quality kernel structure
 *   - Performance benchmarking and roofline analysis
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   This is the core pattern in CUTLASS, TRT-LLM, and FlashAttention-2.
 *   GEMM is 90% of LLM inference compute (GQA, MLP, attention scores).
 *   This maps to: NVIDIA DL Software Engineer — "CuTe GEMM kernels"
 *
 * MENTAL MODEL:
 *   Tiled GEMM structure:
 *   1. Load A and B tiles from gmem to swizzled smem (TiledCopy)
 *   2. Execute MMA for C += A @ B (TiledMMA)
 *   3. Pipeline: load next tiles while computing current (double buffer)
 *   4. Repeat until all tiles processed
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace cute;

// ============================================================================
// Configuration
// ============================================================================
constexpr int BM = 64;    // Row tile size (Br)
constexpr int BN = 64;    // Column tile size (Bc)
constexpr int BK = 128;   // K tile size (head_dim)
constexpr int THREADS = 128;  // Threads per block

// ============================================================================
// KERNEL: Tiled GEMM with Pipeline
// ============================================================================
__global__ void tiled_gemm_kernel(half* A, half* B, float* C,
                                   int M, int N, int K,
                                   int tiles_K) {
    // MENTAL MODEL: Thread and block indices
    int bx = blockIdx.x;  // Column tile index
    int by = blockIdx.y;  // Row tile index
    int tid = threadIdx.x;
    
    // MENTAL MODEL: Output tile position
    int row_start = by * BM;
    int col_start = bx * BN;
    
    // MENTAL MODEL: Shared memory for A and B tiles with swizzle
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];
    
    auto smem_A_ptr = make_smem_ptr<float>(smem_A);
    auto smem_B_ptr = make_smem_ptr<float>(smem_B);
    
    // Swizzled layouts for bank-conflict-free access
    auto layout_A = make_layout(
        make_shape(Int<BM>{}, Int<BK>{}),
        make_stride(Int<BK>{}, Int<1>{}),
        Swizzle<2, 3, 3>{}
    );
    auto layout_B = make_layout(
        make_shape(Int<BK>{}, Int<BN>{}),
        make_stride(Int<BN>{}, Int<1>{}),
        Swizzle<2, 3, 3>{}
    );
    
    auto smem_A_tile = make_tensor(smem_A_ptr, layout_A);
    auto smem_B_tile = make_tensor(smem_B_ptr, layout_B);
    
    // MENTAL MODEL: Output accumulator in registers
    float accum[BM * BN];
    for (int i = 0; i < BM * BN; i++) {
        accum[i] = 0.0f;
    }
    
    // MENTAL MODEL: Double buffer state
    int write_stage = 0;
    int read_stage = 0;
    
    // ========================================================================
    // PROLOGUE: Load first A and B tiles
    // ========================================================================
    if (write_stage == 0) {
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int col = i % BK;
            int global_row = row_start + row;
            int global_col = col;
            if (global_row < M && global_col < K) {
                smem_A_tile(row, col) = __half2float(A[global_row * K + global_col]);
            } else {
                smem_A_tile(row, col) = 0.0f;
            }
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int global_row = col;
            int global_col = col_start + col;
            if (global_row < K && global_col < N) {
                smem_B_tile(row, col) = __half2float(B[global_row * N + global_col]);
            } else {
                smem_B_tile(row, col) = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // ========================================================================
    // MAINLOOP: Pipeline load and compute
    // ========================================================================
    for (int tile_k = 1; tile_k < tiles_K; tile_k++) {
        // MENTAL MODEL: Compute with current tile (simplified MMA)
        auto smem_A_curr = (read_stage == 0) ? smem_A_tile : 
                           make_tensor(smem_A_ptr + BM * BK, layout_A);
        auto smem_B_curr = (read_stage == 0) ? smem_B_tile :
                           make_tensor(smem_B_ptr + BK * BN, layout_B);
        
        // Simplified GEMM (real code uses TiledMMA)
        for (int m = tid; m < BM; m += THREADS) {
            for (int n = 0; n < BN; n++) {
                float sum = 0.0f;
                for (int k = 0; k < BK; k++) {
                    sum += smem_A_curr(m, k) * smem_B_curr(k, n);
                }
                accum[m * BN + n] += sum;
            }
        }
        
        __syncthreads();
        
        // MENTAL MODEL: Load next tiles
        int next_k_start = tile_k * BK;
        
        auto smem_A_next = (write_stage == 0) ? smem_A_tile :
                           make_tensor(smem_A_ptr + BM * BK, layout_A);
        auto smem_B_next = (write_stage == 0) ? smem_B_tile :
                           make_tensor(smem_B_ptr + BK * BN, layout_B);
        
        // Load A tile
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int col = i % BK;
            int global_row = row_start + row;
            int global_col = next_k_start + col;
            if (global_row < M && global_col < K) {
                smem_A_next(row, col) = __half2float(A[global_row * K + global_col]);
            } else {
                smem_A_next(row, col) = 0.0f;
            }
        }
        
        // Load B tile
        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int global_row = next_k_start + row;
            int global_col = col_start + col;
            if (global_row < K && global_col < N) {
                smem_B_next(row, col) = __half2float(B[global_row * N + global_col]);
            } else {
                smem_B_next(row, col) = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Ping-pong buffer switch
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }
    
    // ========================================================================
    // EPILOGUE: Final tile compute
    // ========================================================================
    auto smem_A_last = (read_stage == 0) ? smem_A_tile :
                       make_tensor(smem_A_ptr + BM * BK, layout_A);
    auto smem_B_last = (read_stage == 0) ? smem_B_tile :
                       make_tensor(smem_B_ptr + BK * BN, layout_B);
    
    for (int m = tid; m < BM; m += THREADS) {
        for (int n = 0; n < BN; n++) {
            float sum = 0.0f;
            for (int k = 0; k < BK; k++) {
                sum += smem_A_last(m, k) * smem_B_last(k, n);
            }
            accum[m * BN + n] += sum;
        }
    }
    
    // MENTAL MODEL: Store results to global memory
    for (int m = tid; m < BM; m += THREADS) {
        for (int n = 0; n < BN; n++) {
            int global_row = row_start + m;
            int global_col = col_start + n;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[m * BN + n];
            }
        }
    }
}

// ============================================================================
// CPU REFERENCE: GEMM for verification
// ============================================================================
void cpu_reference_gemm(float* A, float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// ============================================================================
// MAIN: Benchmark and verify
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("=== Tiled GEMM Project ===\n");
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Peak Tensor TFLOPS (FP16): ~%.1f TFLOPS\n\n",
           prop.multiProcessorCount * 128.0f / 1000.0f);
    
    // Problem size
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    
    constexpr int tiles_K = (K + BK - 1) / BK;
    constexpr int tiles_M = (M + BM - 1) / BM;
    constexpr int tiles_N = (N + BN - 1) / BN;
    
    printf("Problem: [%d, %d] @ [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Tile size: [%d, %d] x [%d, %d]\n", BM, BK, BK, BN);
    printf("Tiles: %d x %d x %d = %d total blocks\n\n", 
           tiles_M, tiles_N, tiles_K, tiles_M * tiles_N);
    
    // Allocate
    size_t bytes_A = M * K * sizeof(half);
    size_t bytes_B = K * N * sizeof(half);
    size_t bytes_C = M * N * sizeof(float);
    
    half *d_A, *d_B;
    float *d_C, *h_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    h_C = new float[M * N];
    
    // Initialize with deterministic values
    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; i++) h_A_half[i] = __float2half(0.01f * (i % 100));
    for (int i = 0; i < K * N; i++) h_B_half[i] = __float2half(0.01f * (i % 100));
    
    cudaMemcpy(d_A, h_A_half.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half.data(), bytes_B, cudaMemcpyHostToDevice);
    
    // PREDICT BEFORE RUNNING:
    // Q1: What TFLOPS do you expect (% of peak)?
    // Q2: How many thread blocks will launch?
    // Q3: Is this kernel compute-bound or memory-bound?
    
    // Warmup
    dim3 grid(tiles_N, tiles_M);
    tiled_gemm_kernel<<<grid, THREADS, (BM * BK + BK * BN) * sizeof(float)>>>(
        d_A, d_B, d_C, M, N, K, tiles_K);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics smsp__inst_executed_op_tensor.sum,\
    //              smsp__throughput.avg.pct_of_peak_sustained_elapsed,\
    //              smem__conflict_requests.sum \
    //              ./gemm
    nvtxRangePush("tiled_gemm");
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    constexpr int NUM_WARMUP = 10;
    constexpr int NUM_TIMED = 100;
    
    for (int i = 0; i < NUM_WARMUP; i++) {
        tiled_gemm_kernel<<<grid, THREADS, (BM * BK + BK * BN) * sizeof(float)>>>(
            d_A, d_B, d_C, M, N, K, tiles_K);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_TIMED; i++) {
        tiled_gemm_kernel<<<grid, THREADS, (BM * BK + BK * BN) * sizeof(float)>>>(
            d_A, d_B, d_C, M, N, K, tiles_K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    nvtxRangePop();
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= NUM_TIMED;
    
    // Calculate performance
    float flops = 2.0f * M * N * K;  // multiply + add
    float tflops = flops / elapsed_ms / 1e9;
    float peak_tflops = prop.multiProcessorCount * 128.0f / 1000.0f;
    float efficiency = 100.0f * tflops / peak_tflops;
    
    printf("=== Results ===\n");
    printf("Average time: %.3f ms\n", elapsed_ms);
    printf("Achieved TFLOPS: %.2f\n", tflops);
    printf("Peak TFLOPS: %.1f\n", peak_tflops);
    printf("Efficiency: %.1f%%\n\n", efficiency);
    
    // Verify
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    
    // CPU reference (small sample for speed)
    std::vector<float> h_A(M * K), h_B(K * N), h_C_ref(M * N);
    for (int i = 0; i < M * K; i++) h_A[i] = __half2float(h_A_half[i]);
    for (int i = 0; i < K * N; i++) h_B[i] = __half2float(h_B_half[i]);
    
    cpu_reference_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    bool pass = true;
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C[i] - h_C_ref[i]);
        float rel_error = error / (fabs(h_C_ref[i]) + 1e-6f);
        max_error = fmax(max_error, rel_error);
        if (rel_error > 0.01f) {  // 1% tolerance
            pass = false;
            printf("Mismatch at index %d: GPU=%.4f, CPU=%.4f (rel error=%.2f%%)\n",
                   i, h_C[i], h_C_ref[i], rel_error * 100);
            break;
        }
    }
    
    printf("Max relative error: %.2f%%\n", max_error * 100);
    printf("[%s] Tiled GEMM verified\n\n", pass ? "PASS" : "FAIL");
    
    // Roofline analysis
    printf("=== Roofline Analysis ===\n");
    float bytes_moved = (M * K + K * N) * 2 + M * N * 4;  // A, B (FP16), C (FP32)
    float arithmetic_intensity = flops / bytes_moved;
    float peak_bw = 2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6;
    float compute_bound_tflops = peak_tflops;
    float memory_bound_tflops = peak_bw * arithmetic_intensity / 1e9;
    
    printf("Arithmetic intensity: %.1f FLOPs/byte\n", arithmetic_intensity);
    printf("Peak memory bandwidth: %.1f GB/s\n", peak_bw);
    printf("Compute-bound peak: %.1f TFLOPS\n", compute_bound_tflops);
    printf("Memory-bound peak: %.1f TFLOPS\n", memory_bound_tflops);
    printf("Kernel is: %s\n", 
           (compute_bound_tflops < memory_bound_tflops) ? "COMPUTE-BOUND" : "MEMORY-BOUND");
    printf("Expected efficiency: %.1f%% of %s peak\n\n",
           100.0f * tflops / fmin(compute_bound_tflops, memory_bound_tflops),
           (compute_bound_tflops < memory_bound_tflops) ? "compute" : "memory");
    
    // Occupancy info
    printf("=== Occupancy ===\n");
    printf("Threads per block: %d\n", THREADS);
    printf("smem per block: %zu KB\n", (BM * BK + BK * BN) * sizeof(float) / 1024);
    printf("smem per SM: %d KB\n", prop.sharedMemPerMultiprocessor);
    printf("Max blocks per SM (smem-limited): %d\n",
           prop.sharedMemPerMultiprocessor / static_cast<int>((BM * BK + BK * BN) * sizeof(float)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_C;
    
    return pass ? 0 : 1;
}

/*
 * === PROJECT 01 COMPLETE ===
 * 
 * Exit criteria:
 * 1. Achieved >70% of peak Tensor TFLOPS? [Check efficiency above]
 * 2. Zero bank conflicts verified with Nsight? [Run ncu --metrics smem__conflict_requests.sum]
 * 3. Can explain prologue/mainloop/epilogue? [Review kernel structure]
 * 4. Can predict tile size impact? [Try BM=128 and compare]
 * 
 * Next: Project 02 — FlashAttention-2 Prefill (capstone)
 */
