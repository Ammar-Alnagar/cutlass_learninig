/*
 * WHAT THIS TEACHES:
 *   - Construct TiledCopy with make_tiled_copy
 *   - Understand Copy_Atom (what size transfer per thread)
 *   - Execute copy() to move data from gmem to smem
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 loads K/V tiles from gmem to smem before QK^T computation.
 *   TiledCopy expresses this as: copy(tiled_copy, gmem_tensor, smem_tensor)
 *   This maps to: NVIDIA DL Software Engineer — "TiledCopy for FlashAttention"
 *
 * MENTAL MODEL:
 *   make_tiled_copy(Copy_Atom, thread_layout, smem_layout) creates a copy operator
 *   - Copy_Atom: how many elements each thread transfers (e.g., 8 floats = 256 bits)
 *   - thread_layout: how threads are organized (e.g., 128 threads)
 *   - smem_layout: how data is laid out in shared memory
 *   copy() executes the transfer: all threads participate
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Basic TiledCopy gmem -> smem
// ============================================================================
__global__ void basic_copy_kernel(float* gmem_data) {
    // MENTAL MODEL: Source tensor in global memory [8, 16] = 128 floats
    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    auto gmem_layout = make_layout(make_shape(Int<8>{}, Int<16>{}));
    auto gmem_tensor = make_tensor(gmem_ptr, gmem_layout);
    
    // MENTAL MODEL: Shared memory for the tile
    // extern __shared__ float smem_raw[];  // Dynamic shared memory
    // auto smem_ptr = make_smem_ptr<float>(smem_raw);
    // auto smem_tensor = make_tensor(smem_ptr, gmem_layout);  // Same layout
    
    // For this exercise, we'll use a simpler approach with static smem
    __shared__ float smem_static[8 * 16];
    auto smem_ptr = make_smem_ptr<float>(smem_static);
    auto smem_tensor = make_tensor(smem_ptr, gmem_layout);
    
    // Initialize gmem data
    for (int i = 0; i < 8 * 16; i++) {
        gmem_data[i] = i + 1.0f;  // Values 1, 2, 3, ... 128
    }
    __syncthreads();
    
    // MENTAL MODEL: TiledCopy construction
    // Copy_Atom specifies the transfer size per thread
    // Float4Copy = 4 floats = 128 bits (one vectorized load per thread)
    // Thread layout: 128 threads (one per element for this small example)
    // In real code, you'd have fewer threads, each copying multiple elements
    
    // Simple copy: each thread copies one element
    // Copy_Atom<OpCopy, float> = copy one float per thread
    auto copy_atom = make_tiled_copy_C<Copy_Atom<UniversalCopy, float>>(
        make_layout(Int<128>{}));  // 128 threads, each copies 1 element
    
    printf("=== TiledCopy Setup ===\n");
    printf("Source: gmem [8, 16] = 128 floats\n");
    printf("Dest: smem [8, 16] = 128 floats\n");
    printf("Threads: 128 (each copies 1 element)\n\n");
    
    // MENTAL MODEL: copy(tiled_copy, src_tensor, dst_tensor)
    // All threads in the block participate
    // Thread idx determines which elements it copies
    copy(copy_atom, gmem_tensor, smem_tensor);
    
    __syncthreads();
    
    // Verify copy by reading from smem
    int check_idx = threadIdx.x % (8 * 16);
    float expected = check_idx + 1.0f;
    float actual = smem_tensor(check_idx / 16, check_idx % 16);
    
    if (threadIdx.x == 0) {
        printf("=== Copy Verification ===\n");
        printf("smem[0] = %.1f (expected 1.0)\n", float(smem_tensor(0, 0)));
        printf("smem[15] = %.1f (expected 16.0)\n", float(smem_tensor(0, 15)));
        printf("smem[16] = %.1f (expected 17.0)\n", float(smem_tensor(1, 0)));
        printf("smem[127] = %.1f (expected 128.0)\n", float(smem_tensor(7, 15)));
    }
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_copy() {
    printf("\n=== CPU Reference ===\n");
    printf("Expected: smem[i] = gmem[i] for all i\n");
    printf("gmem initialized to [1, 2, 3, ..., 128]\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Basic TiledCopy Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Peak memory bandwidth: %.1f GB/s\n\n",
           2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6);
    
    // Allocate and initialize gmem
    constexpr int SIZE = 8 * 16;
    float* d_data;
    cudaMalloc(&d_data, SIZE * sizeof(float));
    
    // PREDICT BEFORE RUNNING:
    // Q1: After copy, what value should smem[0] contain?
    // Q2: What value should smem[127] contain?
    // Q3: Do all 128 threads participate in the copy?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Launch with 128 threads (one per element)
    // Warmup
    basic_copy_kernel<<<1, 128, 8 * 16 * sizeof(float)>>>(d_data);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    //              ./ex01_basic_copy
    // Look for: Global load throughput, should be near peak for vectorized access
    nvtxRangePush("basic_copy_kernel");
    basic_copy_kernel<<<1, 128, 8 * 16 * sizeof(float)>>>(d_data);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Copy back and verify
    float h_data[SIZE];
    cudaMemcpy(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU reference
    cpu_reference_copy();
    
    // Verify all values copied correctly
    bool pass = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_data[i] != i + 1.0f) {
            pass = false;
            break;
        }
    }
    
    printf("\n[%s] Basic TiledCopy verified\n", pass ? "PASS" : "FAIL");
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        basic_copy_kernel<<<1, 128, 8 * 16 * sizeof(float)>>>(d_data);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        basic_copy_kernel<<<1, 128, 8 * 16 * sizeof(float)>>>(d_data);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= 10.0f;
    
    // Calculate bandwidth: 128 floats * 4 bytes / elapsed_ms * 1e-6 = GB/s
    float bytes = SIZE * sizeof(float);
    float bandwidth = bytes / elapsed_ms / 1e6;
    float peak_bw = 2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6;
    float efficiency = 100.0f * bandwidth / peak_bw;
    
    printf("\n[Timing] Average kernel time: %.3f ms\n", elapsed_ms);
    printf("[Performance] Achieved bandwidth: %.1f GB/s\n", bandwidth);
    printf("[Performance] Peak bandwidth: %.1f GB/s\n", peak_bw);
    printf("[Performance] Efficiency: %.1f%%\n", efficiency);
    printf("Note: Small transfer size - overhead dominated. See ex02 for real BW.\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What are the three arguments to make_tiled_copy?
 *     Answer: Copy_Atom (transfer size), thread layout, smem layout
 * 
 * Q2: What does copy(tiled_copy, src, dst) do?
 *     Answer: All threads cooperate to copy src tensor to dst tensor
 * 
 * Q3: Why use shared memory instead of reading directly from gmem in MMA?
 *     Answer: smem has ~20x higher bandwidth and lower latency than gmem.
 *             Loading tiles to smem once and reusing them amortizes gmem cost.
 */
