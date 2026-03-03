/*
 * WHAT THIS TEACHES:
 *   - Slice tensors using underscore (_) for "keep all" dimensions
 *   - Create views without copying data
 *   - Access single heads, sequence positions, or head_dim slices
 *
 * WHY THIS MATTERS FOR LLM INFERENCE:
 *   FlashAttention-2 processes one head at a time: Q_head = Q(_, head_idx, _, _)
 *   Slicing creates a view — zero copy overhead.
 *   This maps to: Modular AI Kernel Engineer — "high-performance attention kernels"
 *
 * MENTAL MODEL:
 *   tensor(_, i, _) = slice keeping all dims except fixing dim 1 to index i
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
    
    // MENTAL MODEL: Slice to get a single head
    // tensor(_, head_idx, _) keeps all seq and dim, fixes head
    // Result shape: [seqlen, head_dim] — head dimension is removed
    
    // Initialize all data first for predictable output
    for (int h = 0; h < 4; h++) {
        for (int s = 0; s < 8; s++) {
            for (int d = 0; d < 16; d++) {
                tensor(Int<h>{}, Int<s>{}, Int<d>{}) = h * 100.0f + s * 10.0f + d;
            }
        }
    }
    
    // Slice head 2
    auto head2 = tensor(Int<2>{}, _, _);
    
    printf("=== Sliced Tensor: head 2 ===\n");
    print(head2);
    printf("\n");
    
    // MENTAL MODEL: head2(s, d) accesses the same memory as tensor(2, s, d)
    // Verify by reading the same value both ways
    float via_slice = head2(Int<3>{}, Int<5>{});
    float via_original = tensor(Int<2>{}, Int<3>{}, Int<5>{});
    
    printf("head2(3, 5) = %.1f\n", via_slice);
    printf("tensor(2, 3, 5) = %.1f\n", via_original);
    printf("Match: %s\n", (via_slice == via_original) ? "YES" : "NO");
    
    // MENTAL MODEL: Slice along sequence dimension
    // Get all heads at sequence position 4
    auto seq4 = tensor(_, Int<4>{}, _);
    
    printf("\n=== Sliced Tensor: sequence position 4 ===\n");
    print(seq4);
    printf("\n");
    printf("seq4 shape: [%d, %d] = [heads, head_dim]\n", 
           int(size<0>(seq4)), int(size<1>(seq4)));
    
    // MENTAL MODEL: Slice a single element (all dimensions fixed)
    float elem = tensor(Int<1>{}, Int<2>{}, Int<3>{});
    printf("\nSingle element tensor(1, 2, 3) = %.1f\n", elem);
    
    // MENTAL MODEL: Slice multiple dimensions
    // Get heads 0-1 (first 2 heads), all seq, all dim
    // This requires using make_coord for partial slicing
    // For now, show the pattern for single-index slicing
    
    // MENTAL MODEL: Write through a slice modifies the underlying data
    head2(Int<0>{}, Int<0>{}) = 999.0f;
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
    
    // PREDICT BEFORE RUNNING:
    // Q1: What is the shape of tensor(_, head_idx, _) when tensor is [4,8,16]?
    // Q2: Does slicing copy data?
    // Q3: What is tensor(2, 3, 5) value if initialized as h*100 + s*10 + d?
    
    std::cout << "--- Kernel Output ---\n";
    
    // Warmup
    tensor_slicing_kernel<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();
    
    // NVTX range
    // PROFILE: ncu --set full ./ex02_slicing_views
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
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < 10; i++) {
        tensor_slicing_kernel<<<1, 1>>>(d_data);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        tensor_slicing_kernel<<<1, 1>>>(d_data);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start);
    cudaEventElapsedTime(&elapsed_ms, stop);
    elapsed_ms /= 10.0f;
    
    printf("[Timing] Average kernel time: %.3f ms\n", elapsed_ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex03
 * 
 * Q1: What does underscore (_) mean in tensor(_, i, _)?
 *     Answer: Keep all elements in that dimension (like Python's :)
 * 
 * Q2: After auto head2 = tensor(2, _, _), what is the shape of head2?
 *     Answer: [seqlen, head_dim] — the head dimension is removed
 * 
 * Q3: If you write to head2(0, 0), does tensor(2, 0, 0) change?
 *     Answer: Yes! Slicing creates a view, not a copy. They share memory.
 */
