/*
 * EXERCISE: Tensor Creation - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Create tensors with make_tensor from device pointers
 *   - Use make_gmem_ptr and make_smem_ptr for typed pointers
 *   - Understand tensor structure: layout + data pointer
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
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
    // CONCEPT: Wrap raw pointer in CuTe pointer type
    // make_gmem_ptr<T>(raw_ptr) creates a typed CuTe pointer for global memory
    
    // TODO 1: Create a CuTe global memory pointer from raw float pointer
    // Hint: make_gmem_ptr<float>(gmem_data)
    auto gmem_ptr = /* YOUR CODE HERE */;

    // CONCEPT: Create layout for [4, 8, 16] = [heads, seqlen, head_dim]
    // This is a typical QKV tensor shape for a single batch
    
    // TODO 2: Create a row-major layout for shape [4, 8, 16]
    // Hint: make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}))
    auto layout = /* YOUR CODE HERE */;

    // CONCEPT: make_tensor combines pointer + layout
    // Now you can index with tensor(head, seq, dim) instead of manual offset
    
    // TODO 3: Create tensor from pointer and layout
    // Hint: make_tensor(ptr, layout)
    auto tensor = /* YOUR CODE HERE */;

    // Print tensor structure
    print(tensor);
    printf("\n");

    // CONCEPT: Tensor access uses the layout to compute offset
    // tensor(1, 2, 3) = gmem_ptr[layout(1, 2, 3)]
    // For row-major [4, 8, 16]: layout(1, 2, 3) = 1*8*16 + 2*16 + 3 = 163

    // TODO 4: Write a value using tensor indexing at position (1, 2, 3)
    // Hint: tensor(Int<1>{}, Int<2>{}, Int<3>{}) = 3.14f;
    /* YOUR CODE HERE */;

    // TODO 5: Read the value back from the same position
    // Hint: float val = tensor(Int<1>{}, Int<2>{}, Int<3>{});
    float val = /* YOUR CODE HERE */;
    printf("tensor(1, 2, 3) = %.2f\n", val);

    // CONCEPT: The underlying pointer and layout are accessible
    
    // TODO 6: Get the tensor's data pointer
    // Hint: tensor.data()
    printf("Tensor data pointer: %p\n", /* YOUR CODE HERE */);

    // TODO 7: Get the tensor's layout
    // Hint: tensor.layout()
    printf("Tensor layout: ");
    print(/* YOUR CODE HERE */);
    printf("\n");

    // CONCEPT: Shared memory tensors use make_smem_ptr
    // For this exercise, we'll just show the pattern
    printf("make_smem_ptr pattern:\n");
    printf("  extern __shared__ float smem[];\n");
    printf("  auto smem_ptr = make_smem_ptr<float>(smem);\n");
    printf("  auto smem_tensor = make_tensor(smem_ptr, layout);\n");
}

// ============================================================================
// CPU REFERENCE: Verify tensor offset calculation
// ============================================================================
void cpu_reference_tensor() {
    printf("\n=== CPU Reference ===\n");

    // Row-major [4, 8, 16]: stride = (128, 16, 1)
    // offset(1, 2, 3) = 1*128 + 2*16 + 3 = 163
    int offset = 1 * 8 * 16 + 2 * 16 + 3;
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

    std::cout << "--- Kernel Output ---\n";

    // Warmup
    tensor_creation_kernel<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();

    // NVTX range
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

    cudaFree(d_data);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What are the two arguments to make_tensor?
 *     Answer: _______________
 * 
 * Q2: For row-major [H, S, D], what is the offset of tensor(h, s, d)?
 *     Answer: _______________
 * 
 * Q3: Does make_tensor allocate memory?
 *     Answer: _______________
 */
