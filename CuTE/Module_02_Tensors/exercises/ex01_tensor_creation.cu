/*
 * WHAT THIS TEACHES:
 *   - Create tensors with make_tensor from device pointers
 *   - Use make_gmem_ptr and make_smem_ptr for typed pointers
 *   - Understand tensor structure: layout + data pointer
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 starts by wrapping Q, K, V device pointers in CuTe tensors.
 *   This enables clean indexing like Q(head, seq, dim) instead of manual offset math.
 *   This maps to: NVIDIA DL Software Engineer — "CuTe kernels for FlashAttention"
 *
 * MENTAL MODEL:
 *   make_tensor(ptr, layout) = a view over memory with structured indexing
 *   - ptr: device pointer (from make_gmem_ptr or make_smem_ptr)
 *   - layout: how indices map to offsets
 *   Tensor(i, j, k) = ptr[layout(i, j, k)]
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <iomanip>

using namespace cute;

// ============================================================================
// KERNEL: Tensor creation and access
// ============================================================================
__global__ void tensor_creation_kernel(float* gmem_data) {
    // MENTAL MODEL: Wrap raw pointer in CuTe pointer type
    // make_gmem_ptr<T>(raw_ptr) creates a typed CuTe pointer for global memory
    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    
    // MENTAL MODEL: Create layout for [4, 8, 16] = [heads, seqlen, head_dim]
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}));
    
    // MENTAL MODEL: make_tensor combines pointer + layout
    // Now you can index with tensor(head, seq, dim) instead of manual offset
    auto tensor = make_tensor(gmem_ptr, layout);
    
    // Print tensor structure
    print(tensor);
    printf("\n");
    
    // MENTAL MODEL: Tensor access uses the layout to compute offset
    // tensor(1, 2, 3) = gmem_ptr[layout(1, 2, 3)]
    // For row-major [4, 8, 16]: layout(1, 2, 3) = 1*8*16 + 2*16 + 3 = 128 + 32 + 3 = 163
    
    // Write a value using tensor indexing
    tensor(Int<1>{}, Int<2>{}, Int<3>{}) = 3.14f;
    
    // Read it back
    float val = tensor(Int<1>{}, Int<2>{}, Int<3>{});
    printf("tensor(1, 2, 3) = %.2f\n", val);
    
    // MENTAL MODEL: The underlying pointer and layout are accessible
    printf("Tensor data pointer: %p\n", tensor.data());
    printf("Tensor layout: ");
    print(tensor.layout());
    printf("\n");
    
    // MENTAL MODEL: Shared memory tensors use make_smem_ptr
    // extern __shared__ float smem[];  // Dynamic shared memory
    // auto smem_ptr = make_smem_ptr<float>(smem);
    // auto smem_tensor = make_tensor(smem_ptr, layout);
    
    // For this exercise, we'll just show the type
    printf("gmem_ptr type: CuTe global memory pointer\n");
    printf("smem_ptr type: CuTe shared memory pointer (same interface)\n");
}

// ============================================================================
// CPU REFERENCE: Verify tensor offset calculation
// ============================================================================
void cpu_reference_tensor() {
    printf("\n=== CPU Reference ===\n");
    
    // Row-major [4, 8, 16]: stride = (128, 16, 1)
    // offset(1, 2, 3) = 1*128 + 2*16 + 3*1 = 128 + 32 + 3 = 163
    int offset = 1 * 4 * 8 + 2 * 16 + 3;  // Wait, that's wrong...
    // Correct: stride = (8*16, 16, 1) = (128, 16, 1)
    offset = 1 * 8 * 16 + 2 * 16 + 3;
    printf("offset(1, 2, 3) = 1*8*16 + 2*16 + 3 = %d\n", offset);
}

// ============================================================================
// MAIN: Allocate, launch, verify
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Tensor Creation Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Peak memory bandwidth: %.1f GB/s\n\n",
           2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6);
    
    // Allocate device memory: [4, 8, 16] = 512 floats
    constexpr int SIZE = 4 * 8 * 16;
    float* d_data;
    cudaMalloc(&d_data, SIZE * sizeof(float));
    cudaMemset(d_data, 0, SIZE * sizeof(float));
    
    // PREDICT BEFORE RUNNING:
    // Q1: What is the offset for tensor(1, 2, 3) in row-major [4, 8, 16]?
    // Q2: What does make_gmem_ptr do that a raw pointer doesn't?
    // Q3: Can you use the same tensor interface for smem and gmem?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    tensor_creation_kernel<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --set full ./ex01_tensor_creation
    nvtxRangePush("tensor_creation_kernel");
    tensor_creation_kernel<<<1, 1>>>(d_data);
    nvtxRangePop();
    
    cudaDeviceSynchronize();
    
    // Copy back and verify
    float h_data[SIZE];
    cudaMemcpy(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify the value was written at correct offset
    int expected_offset = 1 * 8 * 16 + 2 * 16 + 3;  // = 163
    bool pass = (h_data[expected_offset] == 3.14f);
    
    // CPU reference
    cpu_reference_tensor();
    
    printf("\n[%s] Tensor creation verified\n", pass ? "PASS" : "FAIL");
    
    // Timing (trivial for this exercise, but included for consistency)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        tensor_creation_kernel<<<1, 1>>>(d_data);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        tensor_creation_kernel<<<1, 1>>>(d_data);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= 10.0f;
    
    printf("\n[Timing] Average kernel time: %.3f ms\n", elapsed_ms);
    printf("Note: This is a trivial kernel - timing is launch overhead dominated\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What are the two arguments to make_tensor?
 *     Answer: A CuTe pointer (from make_gmem_ptr/make_smem_ptr) and a layout
 * 
 * Q2: For row-major [H, S, D], what is the offset of tensor(h, s, d)?
 *     Answer: h * S * D + s * D + d
 * 
 * Q3: Does make_tensor allocate memory?
 *     Answer: No! It creates a view over existing memory. The pointer must
 *             already be allocated (via cudaMalloc or similar).
 */
