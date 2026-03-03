/*
 * EXERCISE: Basic TiledCopy - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Construct TiledCopy with make_tiled_copy
 *   - Understand Copy_Atom (what size transfer per thread)
 *   - Execute copy() to move data from gmem to smem
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   make_tiled_copy(Copy_Atom, thread_layout, smem_layout) creates a copy operator
 *   - Copy_Atom: how many elements each thread transfers (e.g., 4 floats = 128 bits)
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
    __shared__ float smem_static[8 * 16];
    auto smem_ptr = make_smem_ptr<float>(smem_static);
    
    // TODO 1: Create smem tensor with same layout as gmem
    // Hint: make_tensor(smem_ptr, gmem_layout)
    auto smem_tensor = /* YOUR CODE HERE */;

    // Initialize gmem data
    for (int i = 0; i < 8 * 16; i++) {
        gmem_data[i] = i + 1.0f;  // Values 1, 2, 3, ... 128
    }
    __syncthreads();

    // CONCEPT: TiledCopy construction
    // Copy_Atom specifies the transfer size per thread
    // UniversalCopy<float> = copy one float per thread
    
    // TODO 2: Create TiledCopy with 128 threads
    // Hint: make_tiled_copy_C<Copy_Atom<UniversalCopy, float>>(make_layout(Int<128>{}))
    auto copy_atom = /* YOUR CODE HERE */;

    printf("=== TiledCopy Setup ===\n");
    printf("Source: gmem [8, 16] = 128 floats\n");
    printf("Dest: smem [8, 16] = 128 floats\n");
    printf("Threads: 128 (each copies 1 element)\n\n");

    // CONCEPT: copy(tiled_copy, src_tensor, dst_tensor)
    // All threads in the block participate
    
    // TODO 3: Execute the copy from gmem to smem
    // Hint: copy(copy_atom, gmem_tensor, smem_tensor);
    /* YOUR CODE HERE */;

    __syncthreads();

    // Verify copy by reading from smem
    int check_idx = threadIdx.x % (8 * 16);
    float expected = check_idx + 1.0f;
    float actual = smem_tensor(check_idx / 16, check_idx % 16);

    if (threadIdx.x == 0) {
        printf("=== Copy Verification ===\n");
        
        // TODO 4: Print smem[0] value
        printf("smem[0] = %.1f (expected 1.0)\n", float(smem_tensor(0, 0)));
        
        // TODO 5: Print smem[15] value
        printf("smem[15] = %.1f (expected 16.0)\n", float(smem_tensor(0, 15)));
        
        // TODO 6: Print smem[16] value
        printf("smem[16] = %.1f (expected 17.0)\n", float(smem_tensor(1, 0)));
        
        // TODO 7: Print smem[127] value
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

    std::cout << "--- Kernel Output ---\n";

    // Launch with 128 threads (one per element)
    // Warmup
    basic_copy_kernel<<<1, 128, 8 * 16 * sizeof(float)>>>(d_data);
    cudaDeviceSynchronize();

    // NVTX range
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

    cudaFree(d_data);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What are the three arguments to make_tiled_copy?
 *     Answer: _______________
 * 
 * Q2: What does copy(tiled_copy, src, dst) do?
 *     Answer: _______________
 * 
 * Q3: Why use shared memory instead of reading directly from gmem in MMA?
 *     Answer: _______________
 */
