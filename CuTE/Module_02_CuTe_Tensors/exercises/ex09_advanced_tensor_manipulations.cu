/**
 * Exercise 09: Advanced Tensor Manipulations
 *
 * Objective: Master advanced tensor operations including complex views,
 *            tensor algebra, and performance-oriented patterns
 *
 * Tasks:
 * 1. Create complex tensor views with multiple transformations
 * 2. Implement tensor algebra operations
 * 3. Work with tensor compositions
 * 4. Optimize tensor access patterns
 *
 * Key Concepts:
 * - Complex view chains
 * - Tensor algebra
 * - Composition patterns
 * - Performance optimization
 */

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include <iostream>
#include <cuda_runtime.h>

using namespace cute;

// =========================================================================
// Helper Functions
// =========================================================================

void check_cuda_error(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " 
                  << cudaGetErrorString(result) << std::endl;
    }
}

template <typename Tensor>
void print_2d_tensor(const char* name, Tensor const& tensor, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%4d ", tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// =========================================================================
// Main Exercise
// =========================================================================

int main() {
    std::cout << "=== Exercise 09: Advanced Tensor Manipulations ===" << std::endl;
    std::cout << std::endl;

    // Allocate test data
    int* h_data = new int[256];
    for (int i = 0; i < 256; ++i) {
        h_data[i] = i;
    }

    int* d_data;
    check_cuda_error(cudaMalloc(&d_data, 256 * sizeof(int)), "cudaMalloc");
    check_cuda_error(cudaMemcpy(d_data, h_data, 256 * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");

    // =========================================================================
    // TASK 1: Chained Tensor Views
    // =========================================================================
    std::cout << "--- Task 1: Chained Tensor Views ---" << std::endl;

    // Start with a 16x16 tensor
    auto base_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto base_tensor = make_tensor(make_gmem_ptr(d_data), base_layout);

    std::cout << "Base tensor (16x16):" << std::endl;
    std::cout << "  Layout: " << base_tensor.layout() << std::endl;
    std::cout << std::endl;

    // Create a transposed view
    auto transposed_layout = make_layout(
        get<1>(base_layout),
        get<0>(base_layout)
    );
    auto transposed_tensor = make_tensor(make_gmem_ptr(d_data), transposed_layout);

    std::cout << "Transposed view (16x16):" << std::endl;
    std::cout << "  Layout: " << transposed_tensor.layout() << std::endl;
    std::cout << "  Verification: base(3, 7) should equal transposed(7, 3)" << std::endl;
    std::cout << "  (Note: actual values require device access)" << std::endl;
    std::cout << std::endl;

    // Create a sub-tensor view (8x8 from center)
    auto sub_layout = make_layout(
        make_shape(Int<8>{}, Int<8>{}),
        make_stride(Int<16>{}, Int<1>{})  // Same stride as parent
    );
    auto sub_tensor = make_tensor(make_gmem_ptr(d_data + base_layout(4, 4)), sub_layout);

    std::cout << "Sub-tensor view (8x8 from center):" << std::endl;
    std::cout << "  Layout: " << sub_tensor.layout() << std::endl;
    std::cout << "  Offset from base: " << base_layout(4, 4) << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 2: Tensor Reshape Views
    // =========================================================================
    std::cout << "--- Task 2: Tensor Reshape Views ---" << std::endl;

    // 2D tensor (16x16)
    auto tensor_2d = make_tensor(make_gmem_ptr(d_data), base_layout);

    // View as 1D (256 elements)
    auto flat_layout = make_layout(Int<256>{});
    auto flat_tensor = make_tensor(make_gmem_ptr(d_data), flat_layout);

    // View as 3D (4x4x16)
    auto tensor_3d_layout = make_layout(make_shape(Int<4>{}, Int<4>{}, Int<16>{}), GenRowMajor{});
    auto tensor_3d = make_tensor(make_gmem_ptr(d_data), tensor_3d_layout);

    std::cout << "Same data, different views:" << std::endl;
    std::cout << "  2D (16x16): " << tensor_2d.layout() << std::endl;
    std::cout << "  1D (256):   " << flat_tensor.layout() << std::endl;
    std::cout << "  3D (4x4x16): " << tensor_3d.layout() << std::endl;
    std::cout << std::endl;

    // Verify equivalence (conceptually)
    std::cout << "Equivalence (offset calculations):" << std::endl;
    std::cout << "  2D(5, 10) offset:  " << base_layout(5, 10) << std::endl;
    std::cout << "  1D(90) offset:     " << flat_layout(90) << std::endl;
    std::cout << "  3D(1, 2, 10) offset: " << tensor_3d_layout(1, 2, 10) << std::endl;
    std::cout << "  All should be 90" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 3: Tensor Broadcasting Patterns
    // =========================================================================
    std::cout << "--- Task 3: Tensor Broadcasting Patterns ---" << std::endl;

    // Scalar broadcast
    int scalar_value = 42;
    auto scalar_layout = make_layout(Int<1>{});
    auto scalar_tensor = make_tensor(&scalar_value, scalar_layout);

    // Broadcast layout (8x8, all access same scalar)
    auto broadcast_layout = make_layout(
        make_shape(Int<8>{}, Int<8>{}),
        make_stride(Int<0>{}, Int<0>{})  // Both strides = 0
    );
    auto broadcast_tensor = make_tensor(&scalar_value, broadcast_layout);

    std::cout << "Scalar broadcast:" << std::endl;
    std::cout << "  Scalar tensor: " << scalar_tensor.layout() << std::endl;
    std::cout << "  Broadcast layout: " << broadcast_layout << std::endl;
    std::cout << "  All broadcast_tensor(i,j) access the same scalar" << std::endl;
    std::cout << std::endl;

    // Vector broadcast (bias pattern)
    int bias[64];
    for (int i = 0; i < 64; ++i) {
        bias[i] = i * 10;
    }

    auto bias_layout = make_layout(Int<64>{});
    auto bias_tensor = make_tensor(bias, bias_layout);

    // Broadcast to matrix (32 rows, 64 cols, row stride = 0)
    auto bias_matrix_layout = make_layout(
        make_shape(Int<32>{}, Int<64>{}),
        make_stride(Int<0>{}, Int<1>{})  // Row stride = 0
    );
    auto bias_matrix = make_tensor(bias, bias_matrix_layout);

    std::cout << "Vector broadcast (bias pattern):" << std::endl;
    std::cout << "  Bias vector: " << bias_tensor.layout() << std::endl;
    std::cout << "  Bias matrix: " << bias_matrix.layout() << std::endl;
    std::cout << "  bias_matrix(0, j) == bias_matrix(31, j) for all j" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 4: Tensor Composition
    // =========================================================================
    std::cout << "--- Task 4: Tensor Composition ---" << std::endl;

    // Array of pointers (simulating tile pointers)
    int* tile_ptrs[4];
    for (int i = 0; i < 4; ++i) {
        tile_ptrs[i] = d_data + i * 64;  // Each tile is 64 elements
    }

    // Tile tensor (2x2 grid of tiles)
    auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{});
    auto tile_tensor = make_tensor(tile_ptrs, tile_layout);

    std::cout << "Tile tensor (2x2 grid):" << std::endl;
    std::cout << "  Layout: " << tile_tensor.layout() << std::endl;
    std::cout << "  Each element is a pointer to an 8x8 tile" << std::endl;
    std::cout << std::endl;

    // Element layout within each tile
    auto element_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});

    std::cout << "Element layout (8x8 within tile):" << std::endl;
    std::cout << "  Layout: " << element_layout << std::endl;
    std::cout << std::endl;

    // Access pattern: tile_tensor(tile_row, tile_col)[elem_row, elem_col]
    std::cout << "Access pattern:" << std::endl;
    std::cout << "  tile_tensor(0, 0) -> pointer to tile 0" << std::endl;
    std::cout << "  tile_tensor(1, 1) -> pointer to tile 3" << std::endl;
    std::cout << "  Combined: Access element within selected tile" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 5: Strided Tensor Views
    // =========================================================================
    std::cout << "--- Task 5: Strided Tensor Views ---" << std::endl;

    // Original tensor (16x16)
    auto original = make_tensor(make_gmem_ptr(d_data), base_layout);

    // Strided view: every other row
    auto strided_layout = make_layout(
        make_shape(Int<8>{}, Int<16>{}),
        make_stride(Int<32>{}, Int<1>{})  // Row stride = 32 (2 rows)
    );
    auto strided_tensor = make_tensor(make_gmem_ptr(d_data), strided_layout);

    std::cout << "Strided view (every other row):" << std::endl;
    std::cout << "  Original layout: " << original.layout() << std::endl;
    std::cout << "  Strided layout:  " << strided_layout << std::endl;
    std::cout << "  strided_tensor(i, j) accesses original(2*i, j)" << std::endl;
    std::cout << std::endl;

    // Strided view: every other column
    auto col_strided_layout = make_layout(
        make_shape(Int<16>{}, Int<8>{}),
        make_stride(Int<16>{}, Int<2>{})  // Col stride = 2
    );
    auto col_strided_tensor = make_tensor(make_gmem_ptr(d_data), col_strided_layout);

    std::cout << "Column-strided view (every other column):" << std::endl;
    std::cout << "  Layout: " << col_strided_layout << std::endl;
    std::cout << "  col_strided_tensor(i, j) accesses original(i, 2*j)" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 6: Tensor Partitioning
    // =========================================================================
    std::cout << "--- Task 6: Tensor Partitioning ---" << std::endl;

    // Partition 16x16 tensor into 4x4 tiles
    auto partitioned_layout = logical_divide(base_layout, make_shape(Int<4>{}, Int<4>{}));

    std::cout << "Partitioned tensor (4x4 tiles):" << std::endl;
    std::cout << "  Layout: " << partitioned_layout << std::endl;
    std::cout << "  Number of tiles: 4x4 = 16" << std::endl;
    std::cout << "  Each tile: 4x4 elements" << std::endl;
    std::cout << std::endl;

    // Access a specific tile
    std::cout << "Tile access:" << std::endl;
    std::cout << "  Tile (0, 0) covers elements (0-3, 0-3)" << std::endl;
    std::cout << "  Tile (1, 1) covers elements (4-7, 4-7)" << std::endl;
    std::cout << "  Tile (3, 3) covers elements (12-15, 12-15)" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 7: Tensor Alignment and Vectorization
    // =========================================================================
    std::cout << "--- Task 7: Tensor Alignment and Vectorization ---" << std::endl;

    // Aligned tensor for vectorized access (128-bit = 4 floats)
    auto aligned_layout = make_layout(
        make_shape(Int<16>{}, Int<16>{}),
        make_stride(Int<16>{}, Int<1>{})  // Column stride = 1 (aligned)
    );

    std::cout << "Aligned tensor for vectorized access:" << std::endl;
    std::cout << "  Layout: " << aligned_layout << std::endl;
    std::cout << "  Column stride = 1 enables vectorized loads" << std::endl;
    std::cout << "  4 consecutive columns can be loaded as float4" << std::endl;
    std::cout << std::endl;

    // Vectorized access pattern
    std::cout << "Vectorized access pattern:" << std::endl;
    std::cout << "  Load: float4 val = reinterpret_cast<float4*>(&tensor(i, j))[0];" << std::endl;
    std::cout << "  Requires: j % 4 == 0 (16-byte alignment)" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // TASK 8: Tensor Memory Space Transfers
    // =========================================================================
    std::cout << "--- Task 8: Tensor Memory Space Transfers ---" << std::endl;

    std::cout << "Memory space transfer pattern:" << std::endl;
    std::cout << "  1. Global -> Shared (coalesced load)" << std::endl;
    std::cout << "  2. Shared -> Register (vectorized load)" << std::endl;
    std::cout << "  3. Process in registers" << std::endl;
    std::cout << "  4. Register -> Shared" << std::endl;
    std::cout << "  5. Shared -> Global (coalesced store)" << std::endl;
    std::cout << std::endl;

    std::cout << "Example kernel structure:" << std::endl;
    std::cout << "  __global__ void kernel(float* gmem_in, float* gmem_out) {" << std::endl;
    std::cout << "    extern __shared__ float smem[];" << std::endl;
    std::cout << "    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_in), ...);" << std::endl;
    std::cout << "    auto smem_tensor = make_tensor(make_smem_ptr(smem), ...);" << std::endl;
    std::cout << "    float rmem[4];" << std::endl;
    std::cout << "    auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), ...);" << std::endl;
    std::cout << "    // Transfer and process..." << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // CHALLENGE: Design an Optimal Tensor Layout
    // =========================================================================
    std::cout << "=== Challenge: Optimal Tensor Layout Design ===" << std::endl;

    std::cout << "Design a tensor layout for matrix multiplication:" << std::endl;
    std::cout << "  Matrix A: 64x64 (row-major)" << std::endl;
    std::cout << "  Matrix B: 64x64 (column-major)" << std::endl;
    std::cout << "  Matrix C: 64x64 (row-major)" << std::endl;
    std::cout << "  Thread block: 16x16 threads" << std::endl;
    std::cout << "  Each thread computes 4x4 elements" << std::endl;
    std::cout << std::endl;

    // Design layouts
    auto A_layout = make_layout(make_shape(Int<64>{}, Int<64>{}), GenRowMajor{});
    auto B_layout = make_layout(make_shape(Int<64>{}, Int<64>{}), GenColMajor{});
    auto C_layout = make_layout(make_shape(Int<64>{}, Int<64>{}), GenRowMajor{});

    std::cout << "Your designs:" << std::endl;
    std::cout << "  A layout: " << A_layout << std::endl;
    std::cout << "  B layout: " << B_layout << std::endl;
    std::cout << "  C layout: " << C_layout << std::endl;
    std::cout << std::endl;

    // Shared memory layouts (with padding)
    auto As_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<17>{}, Int<1>{}));
    auto Bs_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<17>{}, Int<1>{}));

    std::cout << "Shared memory layouts (padded):" << std::endl;
    std::cout << "  As: " << As_layout << std::endl;
    std::cout << "  Bs: " << Bs_layout << std::endl;
    std::cout << "  Padding avoids bank conflicts during MMA" << std::endl;
    std::cout << std::endl;

    // Cleanup
    delete[] h_data;
    cudaFree(d_data);

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Tensor views are zero-copy transformations" << std::endl;
    std::cout << "2. Multiple views can reference the same data" << std::endl;
    std::cout << "3. Broadcasting uses stride 0 for virtual expansion" << std::endl;
    std::cout << "4. Tensor composition enables hierarchical access" << std::endl;
    std::cout << "5. Strided views enable sub-sampling patterns" << std::endl;
    std::cout << "6. Partitioning divides tensors into tiles" << std::endl;
    std::cout << "7. Alignment enables vectorized access" << std::endl;
    std::cout << "8. Memory space transfers follow a pattern" << std::endl;

    return 0;
}
