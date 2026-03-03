/*
 * EXERCISE: Tensor Slicing Views - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Slice tensors using underscore (_) for "keep all" dimensions
 *   - Create views without copying data
 *   - Access single heads, sequence positions, or head_dim slices
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   tensor(_, i, _) = slice keeping all dims except fixing dim 0 to index i
 *   The result is a view with one fewer dimension (the sliced dimension is removed)
 *   Underscore is CuTe's "colon" operator: tensor(_, i, _) = tensor[:, i, :]
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Tensor slicing with underscore
// ============================================================================
__global__ void tensor_slicing_kernel(float* gmem_data) {
    // Create base tensor [4, 8, 16] = [heads, seqlen, head_dim]
    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}, Int<16>{}));
    auto tensor = make_tensor(gmem_ptr, layout);

    printf("=== Original Tensor [heads=4, seqlen=8, head_dim=16] ===\n");
    print(tensor);
    printf("\n");

    // Initialize all data first for predictable output
    for (int h = 0; h < 4; h++) {
        for (int s = 0; s < 8; s++) {
            for (int d = 0; d < 16; d++) {
                tensor(Int<h>{}, Int<s>{}, Int<d>{}) = h * 100.0f + s * 10.0f + d;
            }
        }
    }

    // CONCEPT: Slice to get a single head
    // tensor(head_idx, _, _) keeps all seq and dim, fixes head
    // Result shape: [seqlen, head_dim] — head dimension is removed
    
    // TODO 1: Slice head 2 from the tensor
    // Hint: Use tensor(Int<2>{}, _, _) — underscore keeps all other dims
    auto head2 = /* YOUR CODE HERE */;

    printf("=== Sliced Tensor: head 2 ===\n");
    print(head2);
    printf("\n");

    // CONCEPT: head2(s, d) accesses the same memory as tensor(2, s, d)
    // Verify by reading the same value both ways
    
    // TODO 2: Read value at head2(3, 5) using the slice
    // Hint: head2(Int<3>{}, Int<5>{})
    float via_slice = /* YOUR CODE HERE */;

    // TODO 3: Read the same value using the original tensor
    // Hint: tensor(Int<2>{}, Int<3>{}, Int<5>{})
    float via_original = /* YOUR CODE HERE */;

    printf("head2(3, 5) = %.1f\n", via_slice);
    printf("tensor(2, 3, 5) = %.1f\n", via_original);
    printf("Match: %s\n", (via_slice == via_original) ? "YES" : "NO");

    // CONCEPT: Slice along sequence dimension
    // Get all heads at sequence position 4
    
    // TODO 4: Create slice for sequence position 4 (all heads, all dims)
    // Hint: tensor(_, Int<4>{}, _) — underscore keeps heads and head_dim
    auto seq4 = /* YOUR CODE HERE */;

    printf("\n=== Sliced Tensor: sequence position 4 ===\n");
    print(seq4);
    printf("\n");
    printf("seq4 shape: [%d, %d] = [heads, head_dim]\n",
           int(size<0>(seq4)), int(size<1>(seq4)));

    // CONCEPT: Slice a single element (all dimensions fixed)
    
    // TODO 5: Get single element at tensor(1, 2, 3)
    float elem = /* YOUR CODE HERE */;
    printf("\nSingle element tensor(1, 2, 3) = %.1f\n", elem);

    // CONCEPT: Write through a slice modifies the underlying data
    
    // TODO 6: Write 999.0f to head2(0, 0)
    // Hint: head2(Int<0>{}, Int<0>{}) = 999.0f;
    /* YOUR CODE HERE */;

    // TODO 7: Verify by reading tensor(2, 0, 0)
    float check = tensor(Int<2>{}, Int<0>{}, Int<0>{});
    printf("\nAfter head2(0,0) = 999:\n");
    printf("tensor(2, 0, 0) = %.1f (should be 999)\n", check);
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_slicing() {
    printf("\n=== CPU Reference ===\n");

    // Row-major [4, 8, 16]: stride = (128, 16, 1)
    // tensor(2, 3, 5) offset = 2*128 + 3*16 + 5 = 256 + 48 + 5 = 309
    int offset = 2 * 8 * 16 + 3 * 16 + 5;
    float expected = 2 * 100.0f + 3 * 10.0f + 5;  // = 235
    printf("tensor(2, 3, 5) offset = %d, value = %.1f\n", offset, expected);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Tensor Slicing Exercise ===\n");
    printf("GPU: %s\n\n", prop.name);

    // Allocate [4, 8, 16] = 512 floats
    constexpr int SIZE = 4 * 8 * 16;
    float* d_data;
    cudaMalloc(&d_data, SIZE * sizeof(float));

    std::cout << "--- Kernel Output ---\n";

    // Warmup
    tensor_slicing_kernel<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("tensor_slicing_kernel");
    tensor_slicing_kernel<<<1, 1>>>(d_data);
    nvtxRangePop();

    cudaDeviceSynchronize();

    // CPU reference
    cpu_reference_slicing();

    // Verify through host readback
    float h_data[SIZE];
    cudaMemcpy(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that slice write modified underlying data
    int offset_2_0_0 = 2 * 8 * 16 + 0 * 16 + 0;
    bool pass = (h_data[offset_2_0_0] == 999.0f);

    printf("\n[%s] Tensor slicing verified\n", pass ? "PASS" : "FAIL");

    cudaFree(d_data);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 * 
 * Q1: What does underscore (_) mean in tensor(_, i, _)?
 *     Answer: _______________
 * 
 * Q2: After auto head2 = tensor(2, _, _), what is the shape of head2?
 *     Answer: _______________
 * 
 * Q3: If you write to head2(0, 0), does tensor(2, 0, 0) change?
 *     Answer: _______________
 */
