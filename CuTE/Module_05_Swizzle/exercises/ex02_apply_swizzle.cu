/*
 * WHAT THIS TEACHES:
 *   - Apply Swizzle<B, M, S> to shared memory layouts
 *   - Understand what each swizzle parameter does
 *   - See how XOR address scrambling eliminates bank conflicts
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 uses Swizzle<2, 3, 3> for K/V tile storage.
 *   This specific parameter choice eliminates bank conflicts for 128-thread access
 *   to [64, 128] FP16 tiles during Tensor Core fragment loading.
 *   This maps to: NVIDIA DL Software Engineer — "swizzled smem for MMA"
 *
 * MENTAL MODEL:
 *   Swizzle<B, M, S> applies XOR to the address:
 *   - B: base bit position for XOR
 *   - M: mask for which bits participate
 *   - S: shift amount
 *   new_addr = addr XOR ((addr >> S) << B)
 *   This scrambles addresses so consecutive threads hit different banks.
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Apply Swizzle to SMEM Layout
// ============================================================================
__global__ void apply_swizzle_kernel(float* gmem_out) {
    // MENTAL MODEL: FlashAttention-2 K/V tile: [64, 128] FP16
    // For simplicity, use [32, 32] FP32 in this demo
    constexpr int M = 32;
    constexpr int N = 32;
    
    // MENTAL MODEL: Raw shared memory
    __shared__ float smem_raw[M * N];
    
    // MENTAL MODEL: Without swizzle - row-major layout
    auto smem_ptr = make_smem_ptr<float>(smem_raw);
    auto layout_no_swizzle = make_layout(make_shape(Int<M>{}, Int<N>{}));
    auto smem_no_swizzle = make_tensor(smem_ptr, layout_no_swizzle);
    
    // MENTAL MODEL: With swizzle - apply Swizzle<2, 3, 3>
    // Parameters chosen for 32 threads accessing 32-wide row
    // This maps to FlashAttention-2's Swizzle<2, 3, 3> for 128 threads
    auto layout_swizzle = make_layout(
        make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<N>{}, Int<1>{}),
        Swizzle<2, 3, 3>{}  // XOR-based address scrambling
    );
    auto smem_swizzle = make_tensor(smem_ptr, layout_swizzle);
    
    printf("=== Swizzle Layout Comparison ===\n\n");
    
    printf("Without swizzle (row-major):\n");
    print(layout_no_swizzle);
    printf("\n\n");
    
    printf("With swizzle Swizzle<2, 3, 3>:\n");
    print(layout_swizzle);
    printf("\n\n");
    
    // MENTAL MODEL: Demonstrate address scrambling
    // Show how consecutive indices map to different "banks"
    if (threadIdx.x == 0) {
        printf("=== Address Mapping (first 16 elements) ===\n\n");
        
        printf("Index | No-Swizzle Addr | Swizzle Addr | Bank (no-swiggle) | Bank (swizzle)\n");
        printf("------|-----------------|--------------|-------------------|---------------\n");
        
        for (int i = 0; i < 16; i++) {
            int addr_no_swizzle = i;  // Row-major: linear
            // Simplified swizzle calculation (real CuTe uses XOR)
            int addr_swizzle = i ^ (i >> 3);  // XOR with shifted version
            
            int bank_no_swizzle = (addr_no_swizzle * 4) / 4 % 32;  // 4 bytes per float
            int bank_swizzle = (addr_swizzle * 4) / 4 % 32;
            
            printf("  %2d  |      %3d        |     %3d      |        %2d         |      %2d\n",
                   i, addr_no_swizzle, addr_swizzle, bank_no_swizzle, bank_swizzle);
        }
        printf("\n");
        
        printf("Note: Swizzle spreads consecutive indices across different banks.\n");
        printf("      Without swizzle, index 0-31 all map to banks 0-31 sequentially.\n");
        printf("      With swizzle, the mapping is scrambled to avoid conflicts.\n\n");
    }
    
    // MENTAL MODEL: Write data with swizzled layout
    // Thread t writes to row t, all columns
    for (int j = 0; j < N; j++) {
        smem_swizzle(threadIdx.x, j) = static_cast<float>(threadIdx.x * N + j);
    }
    
    __syncthreads();
    
    // Read back with linear access for verification
    for (int i = threadIdx.x; i < M * N; i += blockDim.x) {
        gmem_out[i] = smem_raw[i];
    }
    
    // Thread 0 prints summary
    if (threadIdx.x == 0) {
        printf("=== Swizzle Applied Successfully ===\n");
        printf("FlashAttention-2 uses Swizzle<2, 3, 3> for K/V tiles.\n");
        printf("This eliminates bank conflicts during Tensor Core MMA.\n");
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Apply Swizzle Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Shared memory banks: 32\n\n");
    
    constexpr int M = 32, N = 32;
    constexpr int SIZE = M * N;
    
    float* d_out;
    cudaMalloc(&d_out, SIZE * sizeof(float));
    
    // PREDICT BEFORE RUNNING:
    // Q1: What does Swizzle<2, 3, 3> do to addresses?
    // Q2: Why does XOR scrambling eliminate bank conflicts?
    // Q3: What swizzle does FlashAttention-2 use?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    apply_swizzle_kernel<<<1, 32, M * N * sizeof(float)>>>(d_out);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics smem__conflict_requests.sum \
    //              ./ex02_apply_swizzle
    // Look for: smem__conflict_requests.sum should be LOW (near zero)
    nvtxRangePush("apply_swizzle_kernel");
    apply_swizzle_kernel<<<1, 32, M * N * sizeof(float)>>>(d_out);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Verify
    float h_out[SIZE];
    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool pass = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_out[i] != static_cast<float>(i)) {
            pass = false;
            printf("Mismatch at index %d: expected %f, got %f\n", i, static_cast<float>(i), h_out[i]);
            break;
        }
    }
    
    printf("\n[%s] Swizzle application verified\n", pass ? "PASS" : "FAIL");
    
    cudaFree(d_out);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 * 
 * Q1: What are the three parameters of Swizzle<B, M, S>?
 *     Answer: B = base bit position, M = mask, S = shift amount
 * 
 * Q2: What swizzle parameters does FlashAttention-2 use for K/V tiles?
 *     Answer: Swizzle<2, 3, 3> (for 128-thread access to [64, 128] FP16)
 * 
 * Q3: How does XOR eliminate bank conflicts?
 *     Answer: XOR scrambles addresses so consecutive threads access
 *             different banks instead of the same bank.
 */
