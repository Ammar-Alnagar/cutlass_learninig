/**
 * Exercise 01: Tensor Creation from Raw Pointers
 * 
 * Objective: Learn to create CuTe tensors by wrapping raw pointers with layouts
 * 
 * Tasks:
 * 1. Create a tensor from a raw pointer and layout
 * 2. Understand the relationship between tensor, layout, and data
 * 3. Access tensor elements using logical coordinates
 * 4. Practice with different data types
 * 
 * Key Functions:
 * - make_tensor(ptr, layout) - Creates a tensor from pointer and layout
 * - make_gmem_ptr(ptr) - Wraps a global memory pointer
 * - make_smem_ptr(ptr) - Wraps a shared memory pointer
 * - tensor.layout() - Get the tensor's layout
 * - tensor.shape() - Get the tensor's shape
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: Tensor Creation from Raw Pointers ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a simple 1D tensor
    float data_1d[16];
    for (int i = 0; i < 16; ++i) {
        data_1d[i] = static_cast<float>(i);
    }

    // TODO: Create a 1D layout and tensor
    // auto layout_1d = make_layout(make_shape(Int<16>{}), GenRowMajor{});
    // auto tensor_1d = make_tensor(make_gmem_ptr(data_1d), layout_1d);
    
    std::cout << "Task 1 - 1D Tensor:" << std::endl;
    std::cout << "TODO: Create and print a 1D tensor" << std::endl;
    std::cout << std::endl;

    // TASK 2: Create a 2D tensor (4x4 matrix)
    float data_2d[16];
    for (int i = 0; i < 16; ++i) {
        data_2d[i] = static_cast<float>(i);
    }

    // TODO: Create a 2D row-major layout and tensor
    // auto layout_2d = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    // auto tensor_2d = make_tensor(make_gmem_ptr(data_2d), layout_2d);
    
    std::cout << "Task 2 - 2D Tensor (4x4):" << std::endl;
    std::cout << "TODO: Create and print a 2D tensor" << std::endl;
    std::cout << std::endl;

    // TASK 3: Access tensor elements using coordinates
    std::cout << "Task 3 - Tensor Element Access:" << std::endl;
    std::cout << "Accessing elements using tensor(row, col) syntax:" << std::endl;
    
    // Example access (uncomment after creating tensor_2d)
    // std::cout << "  tensor(0, 0) = " << tensor_2d(0, 0) << std::endl;
    // std::cout << "  tensor(1, 2) = " << tensor_2d(1, 2) << std::endl;
    // std::cout << "  tensor(3, 3) = " << tensor_2d(3, 3) << std::endl;
    std::cout << "TODO: Access and print tensor elements" << std::endl;
    std::cout << std::endl;

    // TASK 4: Create a tensor with column-major layout
    float data_cm[12];
    for (int i = 0; i < 12; ++i) {
        data_cm[i] = static_cast<float>(i * 2);
    }

    // TODO: Create a column-major tensor
    // auto layout_cm = make_layout(make_shape(Int<4>{}, Int<3>{}), GenColMajor{});
    // auto tensor_cm = make_tensor(make_gmem_ptr(data_cm), layout_cm);
    
    std::cout << "Task 4 - Column-Major Tensor (4x3):" << std::endl;
    std::cout << "TODO: Create and print a column-major tensor" << std::endl;
    std::cout << std::endl;

    // TASK 5: Create a tensor with custom stride (padded)
    float data_padded[36];  // 4x9 with padding
    for (int i = 0; i < 36; ++i) {
        data_padded[i] = static_cast<float>(i);
    }

    // TODO: Create a padded tensor (4 rows, 8 columns, stride=9)
    // auto layout_padded = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<9>{}, Int<1>{}));
    // auto tensor_padded = make_tensor(make_gmem_ptr(data_padded), layout_padded);
    
    std::cout << "Task 5 - Padded Tensor (4x8 with stride=9):" << std::endl;
    std::cout << "TODO: Create and print a padded tensor" << std::endl;
    std::cout << std::endl;

    // TASK 6: Verify tensor properties
    std::cout << "=== Tensor Properties Verification ===" << std::endl;
    std::cout << "For a 4x4 row-major tensor:" << std::endl;
    std::cout << "  Shape should be (4, 4)" << std::endl;
    std::cout << "  Stride should be (4, 1)" << std::endl;
    std::cout << "  Total elements: 16" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Create a 3D tensor
    std::cout << "=== Challenge: 3D Tensor ===" << std::endl;
    float data_3d[24];
    for (int i = 0; i < 24; ++i) {
        data_3d[i] = static_cast<float>(i);
    }
    
    std::cout << "Create a 3D tensor with shape (2, 3, 4):" << std::endl;
    std::cout << "TODO: Create layout and tensor for 3D data" << std::endl;
    // auto layout_3d = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}), GenRowMajor{});
    // auto tensor_3d = make_tensor(make_gmem_ptr(data_3d), layout_3d);
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Tensors wrap raw pointers with layouts" << std::endl;
    std::cout << "2. Access elements using tensor(coord) syntax" << std::endl;
    std::cout << "3. Layout determines how coordinates map to memory" << std::endl;
    std::cout << "4. Same data can be viewed with different layouts" << std::endl;

    return 0;
}
