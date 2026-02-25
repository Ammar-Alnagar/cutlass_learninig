/**
 * Exercise 07: Tensor Memory Spaces
 * 
 * Objective: Understand how CuTe tensors work with different CUDA memory spaces
 * 
 * Tasks:
 * 1. Create tensors in global memory (gmem)
 * 2. Create tensors in shared memory (smem)
 * 3. Understand pointer wrappers for different memory spaces
 * 4. Practice memory space conversions
 * 
 * Key Concepts:
 * - Global Memory: Large, slow, persistent across kernel launches
 * - Shared Memory: Small, fast, shared within thread block
 * - Register Memory: Fastest, private to each thread
 * - Pointer Wrappers: make_gmem_ptr, make_smem_ptr, make_rmem_ptr
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 07: Tensor Memory Spaces ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a global memory tensor
    std::cout << "Task 1 - Global Memory Tensor:" << std::endl;
    float gmem_data[32];
    for (int i = 0; i < 32; ++i) {
        gmem_data[i] = static_cast<float>(i);
    }

    auto gmem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_data), gmem_layout);

    std::cout << "Global memory tensor created" << std::endl;
    std::cout << "Layout: " << gmem_tensor.layout() << std::endl;
    std::cout << "Sample access: tensor(0, 0) = " << gmem_tensor(0, 0) << std::endl;
    std::cout << "Note: In real kernels, global memory is on the GPU" << std::endl;
    std::cout << std::endl;

    // TASK 2: Create a shared memory tensor (simulated on host)
    std::cout << "Task 2 - Shared Memory Tensor (Simulated):" << std::endl;
    float smem_data[32];
    for (int i = 0; i < 32; ++i) {
        smem_data[i] = static_cast<float>(i * 2);
    }

    auto smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);

    std::cout << "Shared memory tensor created" << std::endl;
    std::cout << "Layout: " << smem_tensor.layout() << std::endl;
    std::cout << "Sample access: tensor(0, 0) = " << smem_tensor(0, 0) << std::endl;
    std::cout << "Note: In real kernels, use 'extern __shared__ float data[];'" << std::endl;
    std::cout << std::endl;

    // TASK 3: Create a register memory tensor (local to thread)
    std::cout << "Task 3 - Register Memory Tensor:" << endl;
    float rmem_data[16];
    for (int i = 0; i < 16; ++i) {
        rmem_data[i] = static_cast<float>(i * 3);
    }

    auto rmem_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    auto rmem_tensor = make_tensor(make_rmem_ptr(rmem_data), rmem_layout);

    std::cout << "Register memory tensor created" << std::endl;
    std::cout << "Layout: " << rmem_tensor.layout() << std::endl;
    std::cout << "Sample access: tensor(0, 0) = " << rmem_tensor(0, 0) << std::endl;
    std::cout << "Note: Register memory is automatic for local variables" << std::endl;
    std::cout << std::endl;

    // TASK 4: Compare memory space characteristics
    std::cout << "Task 4 - Memory Space Comparison:" << std::endl;
    std::cout << std::endl;
    std::cout << "| Property      | Global    | Shared    | Register  |" << std::endl;
    std::cout << "|---------------|-----------|-----------|-----------|" << std::endl;
    std::cout << "| Size          | GBs       | KBs/MBs   | KBs       |" << std::endl;
    std::cout << "| Latency       | High      | Low       | Lowest    |" << std::endl;
    std::cout << "| Bandwidth     | Low       | High      | Highest   |" << std::endl;
    std::cout << "| Scope         | All       | Block     | Thread    |" << std::endl;
    std::cout << "| Persistence   | Kernel    | Block     | Thread    |" << std::endl;
    std::cout << std::endl;

    // TASK 5: Simulate data movement between memory spaces
    std::cout << "Task 5 - Simulated Data Movement:" << std::endl;
    std::cout << "Typical kernel data flow: Global -> Shared -> Register -> Compute" << std::endl;
    std::cout << std::endl;

    // Simulate loading from global to shared
    std::cout << "Step 1: Load from Global to Shared" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            smem_tensor(i, j) = gmem_tensor(i, j);
        }
    }
    std::cout << "  Loaded 4x8 tile to shared memory" << std::endl;

    // Simulate loading from shared to register
    std::cout << "Step 2: Load from Shared to Register" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            rmem_tensor(i, j) = smem_tensor(i, j);
        }
    }
    std::cout << "  Loaded 4x4 tile to registers" << std::endl;

    // Verify data
    std::cout << "Step 3: Verify data in registers" << std::endl;
    std::cout << "  Register tensor contents:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("  %5.1f ", rmem_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 6: Understand pointer wrapper functions
    std::cout << "Task 6 - Pointer Wrapper Functions:" << std::endl;
    std::cout << "make_gmem_ptr(ptr) - Wraps global/device memory pointer" << std::endl;
    std::cout << "make_smem_ptr(ptr) - Wraps shared memory pointer" << std::endl;
    std::cout << "make_rmem_ptr(ptr) - Wraps register/local memory pointer" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Memory space selection
    std::cout << "=== Challenge: Choose the Right Memory Space ===" << std::endl;
    std::cout << "Scenario 1: Input/output matrices for GEMM" << std::endl;
    std::cout << "  Answer: Global memory (large, persistent)" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 2: Tile of data shared by thread block" << std::endl;
    std::cout << "  Answer: Shared memory (fast, block-wide sharing)" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 3: Accumulator for matrix multiplication" << std::endl;
    std::cout << "  Answer: Register memory (fastest, thread-private)" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 4: Lookup table used by all threads" << std::endl;
    std::cout << "  Answer: Constant memory or shared memory" << std::endl;
    std::cout << std::endl;

    // CUDA KERNEL PATTERN
    std::cout << "=== Typical CUDA Kernel Pattern ===" << std::endl;
    std::cout << R"(
__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Declare shared memory for tiles
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[TILE_SIZE * TILE_SIZE];
    
    // Create tensors
    auto A_gmem = make_tensor(make_gmem_ptr(A), ...);
    auto B_gmem = make_tensor(make_gmem_ptr(B), ...);
    auto C_gmem = make_tensor(make_gmem_ptr(C), ...);
    auto As_smem = make_tensor(make_smem_ptr(As), ...);
    auto Bs_smem = make_tensor(make_smem_ptr(Bs), ...);
    
    // Load tiles from global to shared
    // Perform computation using registers
    // Store results back to global
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Different memory spaces have different characteristics" << std::endl;
    std::cout << "2. Use pointer wrappers to specify memory space" << std::endl;
    std::cout << "3. Efficient kernels move data: gmem -> smem -> rmem" << std::endl;
    std::cout << "4. Choose memory space based on access pattern and scope" << std::endl;

    return 0;
}
