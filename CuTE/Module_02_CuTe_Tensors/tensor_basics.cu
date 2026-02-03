#include <iostream>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cute/layout.hpp"
#include "cute/shape.hpp"
#include "cute/tensor.hpp"
#include "cute/print.hpp"

using namespace cute;

// Function to demonstrate basic tensor creation from raw pointers and layouts
void demonstrate_tensor_creation() {
    std::cout << "\n=== Tensor Creation from Raw Pointers ===" << std::endl;
    
    // Create a simple host array to represent our data
    float data[12];  // 4x3 matrix
    
    // Initialize with values to see the mapping
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Create a 4x3 row-major layout
    auto layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
    
    // Create a tensor from the raw pointer and layout
    auto tensor = make_tensor(make_gmem_ptr(data), layout);
    
    std::cout << "Tensor created with layout: " << layout << std::endl;
    std::cout << "Tensor shape: " << tensor.layout().shape() << std::endl;
    
    // Access elements using logical coordinates
    std::cout << "\nAccessing tensor elements using logical coordinates:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "tensor(" << i << "," << j << ") = " << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to demonstrate tensor slicing
void demonstrate_tensor_slicing() {
    std::cout << "\n=== Tensor Slicing Operations ===" << std::endl;
    
    // Create a larger tensor (6x4 matrix)
    float data[24];
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto big_layout = make_layout(make_shape(Int<6>{}, Int<4>{}), GenRowMajor{});
    auto big_tensor = make_tensor(make_gmem_ptr(data), big_layout);
    
    std::cout << "Original tensor shape: " << big_tensor.layout().shape() << std::endl;
    
    // Slice the tensor to get a 3x2 sub-region starting at (1,1)
    // Use make_slice to extract a sub-tensor
    auto sliced_tensor = big_tensor(_, make_range(1, 3));  // All rows, columns 1-2
    std::cout << "Sliced tensor shape: " << sliced_tensor.layout().shape() << std::endl;
    
    std::cout << "\nSliced tensor values:" << std::endl;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << sliced_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Another slicing example: extract a single row
    auto row_tensor = big_tensor(2, _);  // Row 2, all columns
    std::cout << "\nRow 2 values: ";
    for (int j = 0; j < 4; ++j) {
        std::cout << row_tensor(j) << " ";
    }
    std::cout << std::endl;
    
    // Extract a single column
    auto col_tensor = big_tensor(_, 1);  // All rows, column 1
    std::cout << "Column 1 values: ";
    for (int i = 0; i < 6; ++i) {
        std::cout << col_tensor(i) << " ";
    }
    std::cout << std::endl;
}

// Function to demonstrate tensor composition
void demonstrate_tensor_composition() {
    std::cout << "\n=== Tensor Composition and Layout Transformations ===" << std::endl;
    
    // Create a tensor with a custom layout (e.g., transposed)
    float data[12];
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Original 3x4 layout
    auto original_layout = make_layout(make_shape(Int<3>{}, Int<4>{}), GenRowMajor{});
    auto original_tensor = make_tensor(make_gmem_ptr(data), original_layout);
    
    std::cout << "Original tensor shape: " << original_tensor.layout().shape() << std::endl;
    
    // Transpose the layout (swap dimensions)
    auto transposed_layout = make_layout(original_layout.shape().get<1>(), original_layout.shape().get<0>());
    auto transposed_tensor = make_tensor(make_gmem_ptr(data), transposed_layout);
    
    std::cout << "Transposed tensor shape: " << transposed_tensor.layout().shape() << std::endl;
    
    // Show how the same underlying data can be accessed differently
    std::cout << "\nOriginal tensor (3x4):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << original_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nAccessing same data as transposed (4x3):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            // Note: This is just for demonstration - accessing as 4x3 with original data layout
            // In practice, you'd typically create a new layout for transpose
            int linear_idx = i * 3 + j;
            if (linear_idx < 12) {
                std::cout << data[linear_idx] << " ";
            }
        }
        std::cout << std::endl;
    }
}

// Function to demonstrate memory access patterns
void demonstrate_memory_access_patterns() {
    std::cout << "\n=== Memory Access Patterns with Tensors ===" << std::endl;
    
    // Create a tensor representing a 4x4 matrix
    float data[16];
    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    auto tensor = make_tensor(make_gmem_ptr(data), layout);
    
    std::cout << "4x4 tensor with row-major layout:" << std::endl;
    print(tensor.layout());
    std::cout << std::endl;
    
    // Demonstrate different access patterns
    std::cout << "Row-wise access (coalesced):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nColumn-wise access (potentially uncoalesced):" << std::endl;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "CuTe Tensors Study - Module 02" << std::endl;
    std::cout << "=============================" << std::endl;
    
    demonstrate_tensor_creation();
    demonstrate_tensor_slicing();
    demonstrate_tensor_composition();
    demonstrate_memory_access_patterns();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "1. Tensors wrap raw pointers with layouts to create indexed views" << std::endl;
    std::cout << "2. Slicing operations create sub-tensors with different shapes" << std::endl;
    std::cout << "3. Layout transformations enable different access patterns to same data" << std::endl;
    std::cout << "4. Memory access patterns affect performance significantly" << std::endl;
    
    return 0;
}