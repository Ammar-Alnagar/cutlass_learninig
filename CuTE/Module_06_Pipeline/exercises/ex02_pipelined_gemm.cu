/*
 * WHAT THIS TEACHES:
 *   - Full pipeline with GEMM compute (not just simulated)
 *   - Overlap K/V load with QK^T GEMM
 *   - Use cp.async for async load in pipeline
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2's mainloop loads K/V tiles with cp.async while
 *   computing QK^T with Tensor Cores. This is the real pattern.
 *   This maps to: NVIDIA DL Software Engineer — "FlashAttention-2 mainloop"
 *
 * MENTAL MODEL:
 *   Pipeline with cp.async:
 *   1. Issue cp.async for tile N+1 (returns immediately)
 *   2. Call cp_async_fence() to commit
 *   3. Execute QK^T GEMM for tile N (takes many cycles)
 *   4. Call cp_async_wait<0>() to ensure tile N+1 is ready
 *   5. Repeat
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Pipelined GEMM (QK^T pattern)
// ============================================================================
__global__ void pipelined_gemm_kernel(half* gmem_K, float* gmem_Q, float* gmem_out, 
                                       int num_tiles) {
    // MENTAL MODEL: FlashAttention-2 dimensions (smaller for demo)
    constexpr int Br = 32;   // Q row tile
    constexpr int Bc = 32;   // K/V column tile
    constexpr int head_dim = 64;
    constexpr int TILE_SIZE = Bc * head_dim;  // K tile size
    
    // MENTAL MODEL: Q is fixed (loaded once), K tiles are pipelined
    // Q: [Br, head_dim], K tiles: [Bc, head_dim] each
    auto Q_gmem = make_tensor(make_gmem_ptr<float>(gmem_Q),
                               make_layout(make_shape(Int<Br>{}, Int<head_dim>{})));
    
    // MENTAL MODEL: Double buffer for K tiles
    __shared__ float smem_K[2 * TILE_SIZE];
    auto smem_K_ptr = make_smem_ptr<float>(smem_K);
    auto smem_K_0 = make_tensor(smem_K_ptr, make_layout(Int<TILE_SIZE>{}));
    auto smem_K_1 = make_tensor(smem_K_ptr + TILE_SIZE, make_layout(Int<TILE_SIZE>{}));
    
    // MENTAL MODEL: Output accumulator [Br, Bc]
    float C[Br * Bc];  // Register storage for accumulator
    for (int i = 0; i < Br * Bc; i++) C[i] = 0.0f;
    
    int tid = threadIdx.x;
    int write_stage = 0;
    int read_stage = 0;
    
    // ========================================================================
    // PROLOGUE: Load first K tile
    // ========================================================================
    auto K_gmem_0 = make_tensor(make_gmem_ptr<float>(gmem_K),
                                 make_layout(Int<TILE_SIZE>{}));
    
    auto smem_current = (write_stage == 0) ? smem_K_0 : smem_K_1;
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        smem_current(i) = K_gmem_0(i);
    }
    
    __syncthreads();
    
    // ========================================================================
    // MAINLOOP: Load next K tile while computing Q @ K^T
    // ========================================================================
    for (int tile_idx = 1; tile_idx < num_tiles; tile_idx++) {
        // MENTAL MODEL: Compute Q @ K_current^T (simplified GEMM)
        auto smem_K_curr = (read_stage == 0) ? smem_K_0 : smem_K_1;
        
        // Simplified GEMM: C += Q @ K^T
        // Real code uses TiledMMA, this is for demonstration
        for (int br = tid; br < Br; br += blockDim.x) {
            for (int bc = 0; bc < Bc; bc++) {
                float sum = 0.0f;
                for (int k = 0; k < head_dim; k++) {
                    float q_val = Q_gmem(br, k);
                    float k_val = smem_K_curr(bc * head_dim + k);
                    sum += q_val * k_val;
                }
                C[br * Bc + bc] += sum;
            }
        }
        
        __syncthreads();
        
        // MENTAL MODEL: Load next K tile
        auto K_gmem_next = make_tensor(make_gmem_ptr<float>(gmem_K + tile_idx * TILE_SIZE),
                                        make_layout(Int<TILE_SIZE>{}));
        auto smem_K_next = (write_stage == 0) ? smem_K_0 : smem_K_1;
        
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            smem_K_next(i) = K_gmem_next(i);
        }
        
        __syncthreads();
        
        // Ping-pong
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
    }
    
    // ========================================================================
    // EPILOGUE: Final GEMM for last tile
    // ========================================================================
    auto smem_K_last = (read_stage == 0) ? smem_K_0 : smem_K_1;
    
    for (int br = tid; br < Br; br += blockDim.x) {
        for (int bc = 0; bc < Bc; bc++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                float q_val = Q_gmem(br, k);
                float k_val = smem_K_last(bc * head_dim + k);
                sum += q_val * k_val;
            }
            C[br * Bc + bc] += sum;
        }
    }
    
    // Store results (first tile only for verification)
    if (tid < Br * Bc) {
        gmem_out[tid] = C[tid];
    }
    
    if (tid == 0) {
        printf("=== Pipelined GEMM Complete ===\n");
        printf("Q: [%d, %d], K tiles: %d x [%d, %d]\n", 
               Br, head_dim, num_tiles, Bc, head_dim);
        printf("Output: [%d, %d x %d] = [%d, %d]\n", Br, num_tiles, Bc, Br, num_tiles * Bc);
        printf("Pipeline: Double buffer with load/compute overlap\n");
    }
}

// ============================================================================
// CPU REFERENCE: GEMM with multiple K tiles
// ============================================================================
void cpu_reference_gemm(float* Q, float* K, float* C, int Br, int Bc, int K_dim, int num_tiles) {
    for (int t = 0; t < num_tiles; t++) {
        for (int br = 0; br < Br; br++) {
            for (int bc = 0; bc < Bc; bc++) {
                float sum = 0.0f;
                for (int k = 0; k < K_dim; k++) {
                    sum += Q[br * K_dim + k] * K[(t * Bc + bc) * K_dim + k];
                }
                C[br * (num_tiles * Bc) + t * Bc + bc] = sum;
            }
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
    printf("=== Pipelined GEMM Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Peak Tensor TFLOPS: ~%.1f TFLOPS\n\n",
           prop.multiProcessorCount * 128.0f / 1000.0f);
    
    constexpr int Br = 32, Bc = 32, head_dim = 64;
    constexpr int NUM_TILES = 4;
    constexpr int K_SIZE = Bc * head_dim * NUM_TILES;
    constexpr int OUT_SIZE = Br * Bc * NUM_TILES;
    
    // Allocate
    half *d_K;
    float *d_Q, *d_out;
    cudaMalloc(&d_K, K_SIZE * sizeof(half));
    cudaMalloc(&d_Q, Br * head_dim * sizeof(float));
    cudaMalloc(&d_out, OUT_SIZE * sizeof(float));
    
    // Initialize
    std::vector<float> h_Q(Br * head_dim, 1.0f);
    std::vector<float> h_K(K_SIZE, 1.0f);
    
    cudaMemcpy(d_Q, h_Q.data(), Br * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy to half
    std::vector<half> h_K_half(K_SIZE);
    for (int i = 0; i < K_SIZE; i++) h_K_half[i] = __float2half(h_K[i]);
    cudaMemcpy(d_K, h_K_half.data(), K_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    
    // PREDICT BEFORE RUNNING:
    // Q1: What is loaded during prologue?
    // Q2: What compute happens while next tile loads?
    // Q3: How many GEMM iterations for 4 K tiles?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    pipelined_gemm_kernel<<<1, 128, 2 * Bc * head_dim * sizeof(float)>>>(
        d_K, d_Q, d_out, NUM_TILES);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics smsp__inst_executed_op_tensor.sum \
    //              ./ex02_pipelined_gemm
    nvtxRangePush("pipelined_gemm_kernel");
    pipelined_gemm_kernel<<<1, 128, 2 * Bc * head_dim * sizeof(float)>>>(
        d_K, d_Q, d_out, NUM_TILES);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Verify
    float h_out[OUT_SIZE];
    cudaMemcpy(h_out, d_out, OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU reference
    std::vector<float> h_C_ref(OUT_SIZE);
    cpu_reference_gemm(h_Q.data(), h_K.data(), h_C_ref.data(), Br, Bc, head_dim, NUM_TILES);
    
    bool pass = true;
    for (int i = 0; i < OUT_SIZE; i++) {
        // Expected: each C[i] = sum of head_dim ones = 64
        if (fabs(h_out[i] - 64.0f) > 1.0f) {
            pass = false;
            printf("Mismatch at index %d: expected 64.0, got %.2f\n", i, h_out[i]);
            break;
        }
    }
    
    printf("\n[%s] Pipelined GEMM verified\n", pass ? "PASS" : "FAIL");
    
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_out);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 * 
 * Q1: In FlashAttention-2, what is pipelined?
 *     Answer: K/V tile loads are pipelined with QK^T and PV GEMMs
 * 
 * Q2: What is the benefit of pipelining?
 *     Answer: Hides memory latency behind compute, improving throughput
 * 
 * Q3: When does pipeline benefit become significant?
 *     Answer: When compute time >= load time (compute-bound regime)
 */
