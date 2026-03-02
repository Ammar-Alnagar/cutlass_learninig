/**
 * Exercise 07: Tensor Memory Spaces
 *
 * Objective: Understand how CuTe tensors work with different CUDA memory spaces
 *
 * Instructions:
 * - Complete each TODO section by creating tensors with different memory spaces
 * - Understand the pointer wrappers: make_gmem_ptr, make_smem_ptr, make_rmem_ptr
 * - Simulate data movement between memory spaces
 *
 * Key Concepts:
 * - Global Memory (gmem): Large, slow, accessible by all threads
 * - Shared Memory (smem): Small, fast, shared within a thread block
 * - Register Memory (rmem): Fastest, private to each thread
 * - Pointer wrappers tell CuTe which memory space a pointer refers to
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 07: Tensor Memory Spaces ===" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 1: Create a global memory tensor
  // ========================================================================
  // Goal: Wrap a raw pointer with make_gmem_ptr() to create a gmem tensor
  std::cout << "Task 1 - Global Memory Tensor:" << std::endl;
  
  float gmem_data[32];
  for (int i = 0; i < 32; ++i) {
    gmem_data[i] = static_cast<float>(i);
  }

  auto gmem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});

  // TODO: Create a tensor using make_gmem_ptr() to wrap gmem_data
  // Hint: make_tensor(make_gmem_ptr(gmem_data), gmem_layout)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Global memory tensor created" << std::endl;
  std::cout << "Layout: ";
  // TODO: Print the tensor's layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Sample access - tensor(0, 0) = ";
  // TODO: Print gmem_tensor(0, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 0)" << std::endl;

  std::cout << "Sample access - tensor(3, 7) = ";
  // TODO: Print gmem_tensor(3, 7)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 31)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 2: Create a shared memory tensor (simulated on host)
  // ========================================================================
  // Goal: Wrap a pointer with make_smem_ptr() to create a smem tensor
  // Note: On host, this is just a simulation. In kernels, use extern __shared__
  std::cout << "Task 2 - Shared Memory Tensor (Simulated):" << std::endl;
  
  float smem_data[32];
  for (int i = 0; i < 32; ++i) {
    smem_data[i] = static_cast<float>(i * 10);
  }

  auto smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});

  // TODO: Create a tensor using make_smem_ptr() to wrap smem_data
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Shared memory tensor created" << std::endl;
  std::cout << "Layout: ";
  // TODO: Print the tensor's layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Sample access - tensor(0, 0) = ";
  // TODO: Print smem_tensor(0, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 0)" << std::endl;

  std::cout << "Sample access - tensor(2, 5) = ";
  // TODO: Print smem_tensor(2, 5)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 210)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 3: Create a register memory tensor
  // ========================================================================
  // Goal: Wrap a pointer with make_rmem_ptr() to create a rmem tensor
  // Note: Register memory is for thread-local data (accumulators, etc.)
  std::cout << "Task 3 - Register Memory Tensor:" << std::endl;
  
  float rmem_data[16];
  for (int i = 0; i < 16; ++i) {
    rmem_data[i] = static_cast<float>(i * 100);
  }

  auto rmem_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});

  // TODO: Create a tensor using make_rmem_ptr() to wrap rmem_data
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE

  std::cout << "Register memory tensor created" << std::endl;
  std::cout << "Layout: ";
  // TODO: Print the tensor's layout
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Sample access - tensor(0, 0) = ";
  // TODO: Print rmem_tensor(0, 0)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 0)" << std::endl;

  std::cout << "Sample access - tensor(3, 3) = ";
  // TODO: Print rmem_tensor(3, 3)
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 1500)" << std::endl;
  std::cout << std::endl;


  // ========================================================================
  // TASK 4: Compare memory space characteristics
  // ========================================================================
  // Goal: Fill in the comparison table with your understanding
  std::cout << "Task 4 - Memory Space Comparison:" << std::endl;
  std::cout << std::endl;
  std::cout << "| Property      | Global    | Shared    | Register  |" << std::endl;
  std::cout << "|---------------|-----------|-----------|-----------|" << std::endl;
  
  // TODO: Fill in the size comparison (approximate orders of magnitude)
  // Hint: Global = GBs, Shared = KBs/MBs, Register = KBs per thread
  std::cout << "| Size          | ___       | ___       | ___       |" << std::endl;
  
  // TODO: Fill in the latency comparison
  // Hint: Global = High (~400-800 cycles), Shared = Low, Register = Lowest
  std::cout << "| Latency       | ___       | ___       | ___       |" << std::endl;
  
  // TODO: Fill in the scope comparison
  // Hint: Global = All threads, Shared = Block, Register = Thread
  std::cout << "| Scope         | ___       | ___       | ___       |" << std::endl;
  
  std::cout << std::endl;


  // ========================================================================
  // TASK 5: Simulate data movement: Global -> Shared -> Register
  // ========================================================================
  // Goal: Copy data from gmem tensor to smem tensor to rmem tensor
  // This simulates the pattern used in CUDA kernels
  std::cout << "Task 5 - Simulated Data Movement:" << std::endl;
  std::cout << "Pattern: Global -> Shared -> Register -> Compute" << std::endl;
  std::cout << std::endl;

  // Create fresh tensors for data movement
  float source_data[32];
  for (int i = 0; i < 32; ++i) {
    source_data[i] = static_cast<float>(i + 100);
  }
  auto source_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
  auto source_tensor = make_tensor(make_gmem_ptr(source_data), source_layout);

  float stage1_data[32] = {0};
  auto stage1_tensor = make_tensor(make_smem_ptr(stage1_data), source_layout);

  float stage2_data[16] = {0};
  auto stage2_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
  auto stage2_tensor = make_tensor(make_rmem_ptr(stage2_data), stage2_layout);

  std::cout << "Step 1: Load from Global to Shared (4x8 tile)" << std::endl;
  // TODO: Copy all elements from source_tensor to stage1_tensor
  // Use nested loops to copy element by element
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  Copy complete" << std::endl;

  // Verify the copy
  std::cout << "  Verify stage1(2, 3) = ";
  // TODO: Print stage1_tensor(2, 3) - should be 119
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 119)" << std::endl;
  std::cout << std::endl;

  std::cout << "Step 2: Load from Shared to Register (4x4 tile)" << std::endl;
  // TODO: Copy the top-left 4x4 from stage1_tensor to stage2_tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  Copy complete" << std::endl;

  // Verify the copy
  std::cout << "  Verify stage2(1, 2) = ";
  // TODO: Print stage2_tensor(1, 2) - should be 110
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << "  (expected: 110)" << std::endl;
  std::cout << std::endl;

  std::cout << "Step 3: Print register tensor contents" << std::endl;
  // TODO: Print all 16 elements of stage2_tensor
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // TASK 6: Understand pointer wrapper functions
  // ========================================================================
  // Goal: Match each wrapper with its purpose
  std::cout << "Task 6 - Pointer Wrapper Functions:" << std::endl;
  std::cout << "Match each function with its purpose:" << std::endl;
  std::cout << std::endl;
  
  std::cout << "1. make_gmem_ptr(ptr)" << std::endl;
  std::cout << "   Purpose: ";
  // TODO: Write a brief description
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "2. make_smem_ptr(ptr)" << std::endl;
  std::cout << "   Purpose: ";
  // TODO: Write a brief description
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "3. make_rmem_ptr(ptr)" << std::endl;
  std::cout << "   Purpose: ";
  // TODO: Write a brief description
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // CHALLENGE: Choose the right memory space
  // ========================================================================
  std::cout << "=== Challenge: Choose the Right Memory Space ===" << std::endl;
  std::cout << "For each scenario, write which memory space you would use:" << std::endl;
  std::cout << "(gmem, smem, or rmem)" << std::endl;
  std::cout << std::endl;

  std::cout << "Scenario 1: Input matrix A for GEMM (large, read by all blocks)" << std::endl;
  std::cout << "  Answer: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Scenario 2: A tile of matrix A shared by threads in a block" << std::endl;
  std::cout << "  Answer: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Scenario 3: Accumulator for a thread's partial results" << std::endl;
  std::cout << "  Answer: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Scenario 4: Output matrix C for GEMM" << std::endl;
  std::cout << "  Answer: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;

  std::cout << "Scenario 5: A small lookup table used by all threads" << std::endl;
  std::cout << "  Answer: ";
  // START YOUR CODE HERE
  
  
  // END YOUR CODE HERE
  std::cout << std::endl;


  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. What are the three main CUDA memory spaces?" << std::endl;
  std::cout << "2. Which pointer wrapper is used for each memory space?" << std::endl;
  std::cout << "3. What is the typical data flow in a CUDA kernel?" << std::endl;
  std::cout << "4. Why is shared memory faster than global memory?" << std::endl;

  return 0;
}
