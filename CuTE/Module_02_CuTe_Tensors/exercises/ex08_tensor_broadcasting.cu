/**
 * Exercise 08: Tensor Broadcasting
 * 
 * Objective: Learn to broadcast tensors for operations like bias addition
 *            and matrix-vector multiplication
 * 
 * Tasks:
 * 1. Understand broadcasting semantics
 * 2. Create broadcasted views of tensors
 * 3. Apply broadcasting to common operations
 * 4. Practice with different broadcast patterns
 * 
 * Key Concepts:
 * - Broadcasting: Using same value for multiple output positions
 * - Dimension of Size 1: Can be "stretched" to match other sizes
 * - Efficient: No data duplication, just layout manipulation
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 08: Tensor Broadcasting ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Broadcasting a scalar to a vector
    std::cout << "Task 1 - Scalar to Vector Broadcasting:" << std::endl;
    float scalar_data[1] = {5.0f};
    auto scalar_layout = make_layout(make_shape(Int<1>{}), GenRowMajor{});
    auto scalar_tensor = make_tensor(make_gmem_ptr(scalar_data), scalar_layout);

    std::cout << "Scalar tensor: " << scalar_tensor(0) << std::endl;
    std::cout << "Broadcast to 8 elements conceptually:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << scalar_tensor(0) << " ";  // Same value for all positions
    }
    std::cout << std::endl;
    std::cout << "Note: In CuTe, use layout with stride 0 for true broadcasting" << std::endl;
    std::cout << std::endl;

    // TASK 2: Broadcasting a vector to a matrix (row-wise)
    std::cout << "Task 2 - Vector to Matrix Broadcasting (Row-wise):" << std::endl;
    float row_vector[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vector_layout = make_layout(make_shape(Int<4>{}), GenRowMajor{});
    auto vector_tensor = make_tensor(make_gmem_ptr(row_vector), vector_layout);

    std::cout << "Original vector: ";
    for (int j = 0; j < 4; ++j) {
        std::cout << vector_tensor(j) << " ";
    }
    std::cout << std::endl;

    std::cout << "Broadcast to 4x4 matrix (same row repeated):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            // Broadcasting: use vector[j] for all rows
            std::cout << vector_tensor(j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Broadcasting a vector to a matrix (column-wise)
    std::cout << "Task 3 - Vector to Matrix Broadcasting (Column-wise):" << std::endl;
    float col_vector[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    auto col_vector_tensor = make_tensor(make_gmem_ptr(col_vector), vector_layout);

    std::cout << "Original vector: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << col_vector_tensor(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "Broadcast to 4x4 matrix (same column repeated):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            // Broadcasting: use vector[i] for all columns
            std::cout << col_vector_tensor(i) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Matrix-vector addition (bias addition)
    std::cout << "Task 4 - Matrix + Vector (Bias Addition):" << std::endl;
    float matrix_data[12];
    for (int i = 0; i < 12; ++i) {
        matrix_data[i] = static_cast<float>(i);
    }
    auto matrix_layout = make_layout(make_shape(Int<3>{}, Int<4>{}), GenRowMajor{});
    auto matrix_tensor = make_tensor(make_gmem_ptr(matrix_data), matrix_layout);

    float bias[4] = {100.0f, 200.0f, 300.0f, 400.0f};
    auto bias_tensor = make_tensor(make_gmem_ptr(bias), vector_layout);

    std::cout << "Matrix (3x4):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%5.1f ", matrix_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Bias vector: ";
    for (int j = 0; j < 4; ++j) {
        std::cout << bias_tensor(j) << " ";
    }
    std::cout << std::endl;

    std::cout << "Matrix + Bias (broadcasted):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%5.1f ", matrix_tensor(i, j) + bias_tensor(j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 5: Create a broadcast layout (stride = 0)
    std::cout << "Task 5 - Broadcast Layout (Stride = 0):" << std::endl;
    std::cout << "Creating a layout where all rows access the same data:" << std::endl;
    
    // Layout with stride 0 for the first dimension = broadcasting
    auto broadcast_layout = make_layout(
        make_shape(Int<4>{}, Int<4>{}),
        make_stride(Int<0>{}, Int<1>{})  // Stride 0 for rows = broadcast
    );
    
    std::cout << "Broadcast layout: " << broadcast_layout << std::endl;
    std::cout << "All rows access the same memory!" << std::endl;
    std::cout << std::endl;

    // Demonstrate with data
    float broadcast_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto broadcast_tensor = make_tensor(make_gmem_ptr(broadcast_data), broadcast_layout);

    std::cout << "Accessing broadcast tensor:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << broadcast_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Note: All rows show the same values (broadcasting!)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Matrix-matrix multiplication with broadcasting
    std::cout << "=== Challenge: Broadcasting in GEMM ===" << std::endl;
    std::cout << "In matrix multiplication C = A × B + bias:" << std::endl;
    std::cout << "1. A is (M, K), B is (K, N), C is (M, N)" << std::endl;
    std::cout << "2. Bias can be (1, N) - broadcasted to all M rows" << std::endl;
    std::cout << "3. Or bias can be (M, 1) - broadcasted to all N columns" << std::endl;
    std::cout << std::endl;

    // APPLICATION: Batched operations
    std::cout << "=== Application: Batched Broadcasting ===" << std::endl;
    std::cout << "In batched neural network inference:" << std::endl;
    std::cout << "- Input: (batch, features)" << std::endl;
    std::cout << "- Weights: (features, output)" << std::endl;
    std::cout << "- Bias: (output,) - broadcasted to all batches" << std::endl;
    std::cout << std::endl;

    std::cout << "Example: batch=4, features=3, output=2" << std::endl;
    std::cout << "Bias shape (2,) broadcasts to (4, 2)" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Broadcasting reuses data without copying" << std::endl;
    std::cout << "2. Stride 0 creates true broadcast layouts" << std::endl;
    std::cout << "3. Common in bias addition and batched ops" << std::endl;
    std::cout << "4. Dimension of size 1 can broadcast to any size" << std::endl;

    return 0;
}
