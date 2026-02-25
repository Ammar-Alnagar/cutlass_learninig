/**
 * Exercise 02: Vectorized Loads
 * 
 * Objective: Learn to use vectorized memory operations (128-bit loads)
 *            for improved memory bandwidth utilization
 * 
 * Tasks:
 * 1. Understand vectorized load concepts
 * 2. See how 128-bit loads work (4 floats)
 * 3. Calculate bandwidth improvement
 * 4. Practice with aligned data
 * 
 * Key Concepts:
 * - Vectorized Load: Loading multiple elements in one instruction
 * - 128-bit Load: Loading 4 floats (32-bit each) at once
 * - Alignment: Data must be properly aligned for vectorized access
 * - Bandwidth: 4x improvement over scalar loads
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 02: Vectorized Loads ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Understand vectorized load size
    std::cout << "Task 1 - Vectorized Load Sizes:" << std::endl;
    std::cout << "Scalar load (32-bit): 1 float per instruction" << std::endl;
    std::cout << "Vectorized load (64-bit): 2 floats per instruction" << std::endl;
    std::cout << "Vectorized load (128-bit): 4 floats per instruction" << std::endl;
    std::cout << "Vectorized load (256-bit): 8 floats per instruction" << std::endl;
    std::cout << std::endl;

    // TASK 2: Create aligned data for vectorized access
    std::cout << "Task 2 - Aligned Data Layout:" << std::endl;
    
    // Data aligned for 128-bit loads (multiple of 4 floats)
    alignas(16) float data[32];
    for (int i = 0; i < 32; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Layout that supports vectorized access
    // Each row has 8 elements = 2 vectorized loads of 4 elements each
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    auto tensor = make_tensor(make_gmem_ptr(data), layout);

    std::cout << "Tensor layout: " << layout << std::endl;
    std::cout << "Each row has 8 elements (2 vectorized loads of 4)" << std::endl;
    std::cout << std::endl;

    // TASK 3: Visualize vectorized load groups
    std::cout << "Task 3 - Vectorized Load Groups:" << std::endl;
    std::cout << "Data organized for 128-bit loads (4 floats per load):" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 4; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < 8; j += 4) {
            std::cout << "[";
            for (int k = 0; k < 4; ++k) {
                std::cout << tensor(i, j + k);
                if (k < 3) std::cout << ", ";
            }
            std::cout << "] ";
        }
        std::cout << " <- 2 vectorized loads" << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Calculate bandwidth improvement
    std::cout << "Task 4 - Bandwidth Calculation:" << std::endl;
    int total_elements = 32;
    int scalar_loads = total_elements;  // 1 element per load
    int vectorized_loads = total_elements / 4;  // 4 elements per load
    
    std::cout << "Total elements: " << total_elements << std::endl;
    std::cout << "Scalar loads needed: " << scalar_loads << std::endl;
    std::cout << "Vectorized (128-bit) loads needed: " << vectorized_loads << std::endl;
    std::cout << "Improvement factor: " << (float)scalar_loads / vectorized_loads << "x" << std::endl;
    std::cout << std::endl;

    // TASK 5: Alignment requirements
    std::cout << "Task 5 - Alignment Requirements:" << std::endl;
    std::cout << "128-bit load requires 16-byte alignment" << std::endl;
    std::cout << "Float is 4 bytes, so address must be divisible by 4" << std::endl;
    std::cout << std::endl;
    
    std::cout << "In our array:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        bool aligned = (i % 4 == 0);
        std::cout << "  Element " << i << ": " 
                  << (aligned ? "Aligned for 128-bit load" : "Not aligned") 
                  << std::endl;
    }
    std::cout << std::endl;

    // TASK 6: Layout for vectorized access
    std::cout << "Task 6 - Layout Design for Vectorization:" << std::endl;
    std::cout << "For vectorized loads, organize data so consecutive" << std::endl;
    std::cout << "elements in the fastest-changing dimension are contiguous" << std::endl;
    std::cout << std::endl;

    // Row-major: columns are contiguous (good for row access)
    auto rm_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    std::cout << "Row-major layout: " << rm_layout << std::endl;
    std::cout << "  Row access: consecutive (GOOD for vectorization)" << std::endl;
    std::cout << "  Column access: stride of 8 (NOT ideal)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Design layout for column vectorization
    std::cout << "=== Challenge: Column Vectorization ===" << std::endl;
    std::cout << "How would you organize data for vectorized column access?" << std::endl;
    std::cout << "Answer: Use column-major layout!" << std::endl;
    
    auto cm_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenColMajor{});
    std::cout << "Column-major layout: " << cm_layout << std::endl;
    std::cout << "  Column access: consecutive (GOOD for vectorization)" << std::endl;
    std::cout << std::endl;

    // VECTORIZED COPY PATTERN
    std::cout << "=== Vectorized Copy Pattern ===" << std::endl;
    std::cout << R"(
// Conceptual vectorized copy
__global__ void vectorized_copy(float* src, float* dst, int N) {
    // Each thread copies 4 floats per iteration
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < N) {
        // Load 4 floats as one 128-bit operation
        float4 val = reinterpret_cast<float4*>(&src[idx])[0];
        // Store 4 floats as one 128-bit operation
        reinterpret_cast<float4*>(&dst[idx])[0] = val;
    }
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Vectorized loads improve bandwidth 4x" << std::endl;
    std::cout << "2. 128-bit loads require 16-byte alignment" << std::endl;
    std::cout << "3. Layout determines which access is vectorized" << std::endl;
    std::cout << "4. Match layout to access pattern for best performance" << std::endl;

    return 0;
}
