/*
 * WHAT THIS TEACHES:
 *   - Understand MMA atoms for Tensor Core operations
 *   - Configure MMA for sm_89 (Ada Lovelace) architecture
 *   - See the basic building block of GEMM in CuTe
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 QK^T and PV computations are GEMMs executed via Tensor Cores.
 *   MMA atoms define the instruction-level operation (e.g., mma.sync.aligned.m16n8k16)
 *   This maps to: NVIDIA DL Software Engineer — "Tensor Core optimization with CuTe"
 *
 * MENTAL MODEL:
 *   MMA Atom = the smallest unit of Tensor Core computation
 *   sm_89 (Ada) supports: mma.sync.aligned.m16n8k16.f32.f16.f16.f32
 *   This computes: D = A * B + C where A,B are f16, C,D are f32
 *   Each warp executes multiple atoms to cover the full tile
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: MMA Atom Setup
// ============================================================================
__global__ void mma_atom_kernel(float* gmem_C) {
    // MENTAL MODEL: FlashAttention-2 uses small GEMMs repeatedly
    // QK^T: [Br, head_dim] @ [head_dim, Bc] = [Br, Bc]
    // Typical: Br=64, Bc=64, head_dim=128
    
    // For this exercise, small example: [16, 32] @ [32, 16] = [16, 16]
    // This fits in one warp and demonstrates the MMA atom
    
    // MENTAL MODEL: MMA Atom for sm_80+ (Ampere, Ada, Hopper)
    // MMA_Trait<Shape<M,N,K>, Element_A, Element_B, Element_C>
    // - Shape<16,8,16> = m16n8k16 (16x8x16 matrix multiply)
    // - Element_A = half (FP16)
    // - Element_B = half (FP16)
    // - Element_C = float (FP32 accumulator)
    
    using MMA_Traits = MMA_Traits<
        Shape<Int<16>, Int<8>, Int<16>>,  // M, N, K dimensions
        Element_t<half>,                   // A element type
        Element_t<half>,                   // B element type
        Element_t<float>                   // C/D element type (accumulator)
    >;
    
    using MMA_Atom = MMA_Atom<MMA_Traits>;
    
    printf("=== MMA Atom Configuration ===\n");
    printf("MMA shape: M=16, N=8, K=16\n");
    printf("A type: FP16, B type: FP16, C/D type: FP32\n");
    printf("Instruction: mma.sync.aligned.m16n8k16.f32.f16.f16.f32\n\n");
    
    // MENTAL MODEL: TiledMMA wraps the atom with thread/warp layout
    // For a [16,16,32] GEMM, we need multiple atoms
    // One warp (32 threads) can execute this with proper tiling
    
    // MENTAL MODEL: Create tensors for A, B, C in registers (rmem)
    // In real code, these come from smem after TiledCopy
    // Here we use small example for demonstration
    
    // A: [16, 32] row-major
    auto A_layout = make_layout(make_shape(Int<16>{}, Int<32>{}));
    // B: [32, 16] column-major (for Tensor Core efficiency)
    auto B_layout = make_layout(make_shape(Int<32>{}, Int<16>{}),
                                 make_stride(Int<1>{}, Int<32>{}));
    // C: [16, 16] row-major (output)
    auto C_layout = make_layout(make_shape(Int<16>{}, Int<16>{}));
    
    printf("A layout [16, 32] row-major:\n");
    print(A_layout);
    printf("\n");
    
    printf("B layout [32, 16] column-major:\n");
    print(B_layout);
    printf("\n");
    
    printf("C layout [16, 16] row-major:\n");
    print(C_layout);
    printf("\n");
    
    // MENTAL MODEL: Partition fragments for this warp
    // Each thread gets a piece of A, B, and C
    // The fragment layout is determined by the MMA atom
    
    // For demonstration, just print the expected fragment sizes
    // Real code uses partition_fragment_A/B/C
    
    printf("=== Fragment Layout (per thread in warp) ===\n");
    printf("Note: Actual fragment layout depends on MMA atom and warp layout\n");
    printf("For m16n8k16 with 32 threads:\n");
    printf("  Each thread holds ~8 elements of A, ~4 elements of B, ~8 elements of C\n");
    printf("  (exact layout determined by CuTe's partition functions)\n\n");
    
    // MENTAL MODEL: Write output to gmem for verification
    auto C_gmem_ptr = make_gmem_ptr<float>(gmem_C);
    auto C_gmem = make_tensor(C_gmem_ptr, C_layout);
    
    // Initialize C with zeros (in real code, this is the accumulator)
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            C_gmem(Int<i>{}, Int<j>{}) = 0.0f;
        }
    }
    
    // Thread 0 prints summary
    if (threadIdx.x == 0) {
        printf("=== MMA Atom Ready ===\n");
        printf("To execute: gemm(tiled_mma, A_frag, B_frag, C_frag)\n");
        printf("See ex02_tiled_gemm.cu for full GEMM execution\n");
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== MMA Atom Setup Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Tensor Core info
    if (prop.major >= 7) {
        printf("Tensor Cores: Available (sm_%d%d+)\n", prop.major, prop.minor);
        printf("Peak Tensor TFLOPS (FP16): ~%d TFLOPS\n", 
               prop.multiProcessorCount * 128);  // Approximate for Ada
    } else {
        printf("WARNING: Tensor Cores require sm_70+ (Volta+)\n");
    }
    printf("\n");
    
    // Allocate output tensor [16, 16]
    constexpr int C_SIZE = 16 * 16;
    float* d_C;
    cudaMalloc(&d_C, C_SIZE * sizeof(float));
    
    // PREDICT BEFORE RUNNING:
    // Q1: What is the MMA instruction for sm_89 FP16 GEMM?
    // Q2: Why is B column-major in Tensor Core GEMM?
    // Q3: What is the accumulator type and why?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    mma_atom_kernel<<<1, 32>>>(d_C);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --set full ./ex01_mma_atom_setup
    // Look for: Tensor Core instruction issuance (smsp__inst_executed_op_tensor)
    nvtxRangePush("mma_atom_kernel");
    mma_atom_kernel<<<1, 32>>>(d_C);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Verify C was initialized
    float h_C[C_SIZE];
    cudaMemcpy(h_C, d_C, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool pass = true;
    for (int i = 0; i < C_SIZE; i++) {
        if (h_C[i] != 0.0f) {
            pass = false;
            break;
        }
    }
    
    printf("\n[%s] MMA atom setup verified\n", pass ? "PASS" : "FAIL");
    
    cudaFree(d_C);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What is the MMA instruction shape for sm_80+ FP16 GEMM?
 *     Answer: m16n8k16 — 16x8x16 matrix multiply per Tensor Core instruction
 * 
 * Q2: What are the element types for A, B, and C in FlashAttention-2?
 *     Answer: A=FP16, B=FP16, C=FP32 (accumulator for numerical stability)
 * 
 * Q3: How many threads are in a warp, and how many warps per SM on Ada?
 *     Answer: 32 threads/warp, Ada has ~128 warps per SM (depends on occupancy)
 */
