/*
 * EXERCISE: Local Partition - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Use local_partition to distribute tensor elements across threads
 *   - Understand thread-local views for parallel processing
 *   - Map warps to tensor tiles (FlashAttention-2 warp specialization pattern)
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   local_partition(tensor, num_threads, thread_id, dim) =
 *     tensor slice where 'dim' is divided among 'num_threads'
 *   Each thread gets elements where dim % num_threads == thread_id
 *   This is like MPI scatter but within a GPU thread block
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// Number of threads for partitioning
constexpr int NUM_THREADS = 4;

// ============================================================================
// KERNEL: local_partition for thread distribution
// ============================================================================
__global__ void local_partition_kernel(float* gmem_data) {
    // Create tensor [8, 16] = [seqlen, head_dim]
    auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
    auto layout = make_layout(make_shape(Int<8>{}, Int<16>{}));
    auto tensor = make_tensor(gmem_ptr, layout);

    // Initialize with predictable values
    for (int s = 0; s < 8; s++) {
        for (int d = 0; d < 16; d++) {
            tensor(Int<s>{}, Int<d>{}) = s * 100.0f + d;
        }
    }

    printf("=== Original Tensor [seqlen=8, head_dim=16] ===\n");
    print(tensor);
    printf("\n");

    // CONCEPT: local_partition divides a dimension across threads
    // Partition dim 1 (head_dim) across 4 threads
    // Each thread gets 16/4 = 4 elements along head_dim

    int tid = threadIdx.x;

    // CONCEPT: local_partition(tensor, num_threads, thread_id, partition_dim)
    // - partition_dim = 1 means we split the head_dim dimension
    // - Thread 0 gets dims 0-3, Thread 1 gets 4-7, etc.
    
    // TODO 1: Partition the tensor across NUM_THREADS threads along dim 1
    // Hint: local_partition(tensor, Int<NUM_THREADS>{}, tid, Int<1>{})
    auto local = /* YOUR CODE HERE */;

    printf("=== Thread %d: local_partition along head_dim ===\n", tid);
    print(local);
    printf("\n");

    // CONCEPT: Each thread sees a smaller tensor
    // Original: [8, 16], Local: [8, 4] (16/4 = 4 elements per thread in dim 1)
    
    // TODO 2: Get the local shape dimension 0 (should be 8)
    int local_dim0 = int(size<0>(local));
    
    // TODO 3: Get the local shape dimension 1 (should be 4)
    int local_dim1 = int(size<1>(local));
    
    printf("Thread %d local shape: [%d, %d]\n", tid, local_dim0, local_dim1);

    // CONCEPT: Access pattern — local(s, d) maps to original tensor(s, d + tid*4)
    // Thread 1, local(2, 1) = original(2, 1 + 1*4) = original(2, 5)
    
    // TODO 4: Read value from local tensor at (2, 1)
    float local_val = local(Int<2>{}, Int<1>{});

    // TODO 5: Read the same value from original tensor
    // Formula: tensor(2, 1 + tid * 4)
    float orig_val = tensor(Int<2>{}, Int<1 + tid * 4>{});
    
    printf("Thread %d: local(2,1)=%.1f, tensor(2,%d)=%.1f, match: %s\n",
           tid, local_val, 1 + tid * 4, orig_val,
           (local_val == orig_val) ? "YES" : "NO");

    __syncthreads();

    // CONCEPT: FlashAttention-2 warp partitioning pattern
    // Warp 0: loads K tile from gmem to smem
    // Warp 1: loads V tile from gmem to smem
    // Warp 2: computes QK^T GEMM
    // Warp 3: computes PV GEMM (after softmax)

    // Simulate warp-level partitioning (simplified — real code uses warp_idx)
    int warp_idx = tid / 4;  // 4 threads per warp for demo
    int lane_idx = tid % 4;

    if (tid < 4) {  // Only first warp prints
        printf("\n=== Warp %d, Lane %d ===\n", warp_idx, lane_idx);
    }
}

// ============================================================================
// CPU REFERENCE
// ============================================================================
void cpu_reference_partition() {
    printf("\n=== CPU Reference ===\n");

    // local_partition along dim 1 with 4 threads
    // Thread t gets elements where dim1 % 4 == t
    // Thread 0: dims 0, 4, 8, 12
    // Thread 1: dims 1, 5, 9, 13
    // Thread 2: dims 2, 6, 10, 14
    // Thread 3: dims 3, 7, 11, 15

    printf("Thread 0 local(2, 0) = tensor(2, 0) = 200\n");
    printf("Thread 1 local(2, 0) = tensor(2, 1) = 201\n");
    printf("Thread 2 local(2, 0) = tensor(2, 2) = 202\n");
    printf("Thread 3 local(2, 0) = tensor(2, 3) = 203\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== local_partition Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);

    // Allocate [8, 16] = 128 floats
    constexpr int SIZE = 8 * 16;
    float* d_data;
    cudaMalloc(&d_data, SIZE * sizeof(float));

    std::cout << "--- Kernel Output ---\n";

    // Launch with 4 threads (one for each partition)
    // Warmup
    local_partition_kernel<<<1, NUM_THREADS>>>(d_data);
    cudaDeviceSynchronize();

    // NVTX range
    nvtxRangePush("local_partition_kernel");
    local_partition_kernel<<<1, NUM_THREADS>>>(d_data);
    nvtxRangePop();

    cudaDeviceSynchronize();

    // CPU reference
    cpu_reference_partition();

    printf("\n[PASS] local_partition verified\n");

    cudaFree(d_data);

    return 0;
}

/*
 * CHECKPOINT: Answer before moving to Module 03
 * 
 * Q1: What is the difference between local_tile and local_partition?
 *     Answer: _______________
 * 
 * Q2: In FlashAttention-2, how might you use local_partition?
 *     Answer: _______________
 * 
 * Q3: If you partition [128, 64] along dim 0 across 32 threads, how many
 *     rows does each thread get?
 *     Answer: _______________
 * 
 * === MODULE 02 COMPLETE ===
 * Exit criteria:
 * 1. Can create tensor with make_tensor(gmem_ptr, layout)
 * 2. Can slice tensor with underscore: tensor(_, head_idx, _, _)
 * 3. Can iterate over tiles with local_tile in a loop
 * 4. Can partition work across threads with local_partition
 *
 * Next: Module 03 — TiledCopy (gmem->smem, vectorized loads, cp.async)
 */
