#include <iostream>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cute/layout.hpp"
#include "cute/shape.hpp"
#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/swizzle.hpp"
#include "cute/print.hpp"

using namespace cute;

// Function to demonstrate basic TiledCopy operations
void demonstrate_tiled_copy_basics() {
    std::cout << "\n=== Tiled Copy Basics ===" << std::endl;
    
    // Define source and destination data
    float src_data[16];
    float dst_data[16];
    
    // Initialize source data
    for (int i = 0; i < 16; ++i) {
        src_data[i] = static_cast<float>(i);
    }
    
    // Create source and destination tensors
    auto src_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    auto dst_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    auto src_tensor = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), dst_layout);
    
    std::cout << "Source tensor layout: " << src_layout << std::endl;
    std::cout << "Destination tensor layout: " << dst_layout << std::endl;
    
    // Create a TiledCopy atom for copying 2x2 tiles
    // This represents how threads will cooperatively copy data
    auto copy_op = make_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{},
        make_layout(make_shape(Int<2>{}, Int<2>{}))  // Thread tile shape
    );
    
    std::cout << "TiledCopy operation created with thread tile shape: " 
              << copy_op.get_layout() << std::endl;
    
    // Show source data before copy
    std::cout << "\nSource data before copy:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << src_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Perform the copy operation (conceptual - actual implementation would involve thread blocks)
    // In a real kernel, this would involve multiple threads cooperating
    for (int i = 0; i < 4; i += 2) {
        for (int j = 0; j < 4; j += 2) {
            // Copy 2x2 tile from source to destination
            for (int ti = 0; ti < 2; ++ti) {
                for (int tj = 0; tj < 2; ++tj) {
                    if (i + ti < 4 && j + tj < 4) {
                        dst_tensor(i + ti, j + tj) = src_tensor(i + ti, j + tj);
                    }
                }
            }
        }
    }
    
    // Show destination data after copy
    std::cout << "\nDestination data after copy:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << dst_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to demonstrate vectorized loads (128-bit operations)
void demonstrate_vectorized_loads() {
    std::cout << "\n=== Vectorized Loads (128-bit) ===" << std::endl;
    
    // Create data aligned for 128-bit operations (4 floats = 128 bits)
    alignas(16) float data[16];
    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i * 2);
    }
    
    // Create tensor with layout that supports vectorized access
    auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto tensor = make_tensor(make_gmem_ptr(data), layout);
    
    std::cout << "Tensor layout for vectorized access: " << layout << std::endl;
    
    // Show that consecutive elements in the fastest-changing dimension 
    // (column in this case) are stored contiguously for vectorization
    std::cout << "\nData layout - each row represents contiguous memory for vectorization:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < 4; ++j) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Demonstrate how a vectorized load would work conceptually
    // Each 128-bit load gets 4 consecutive floats
    std::cout << "\nConceptual 128-bit loads (each load gets 4 consecutive elements):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "Load " << i << " gets: ";
        for (int j = 0; j < 4; ++j) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to demonstrate cp.async operations for sm_89
void demonstrate_cp_async_sm89() {
    std::cout << "\n=== cp.async Operations for sm_89 ===" << std::endl;
    
    // Simulate global memory data
    float global_data[32];
    for (int i = 0; i < 32; ++i) {
        global_data[i] = static_cast<float>(i * 3);
    }
    
    // Simulate shared memory buffer
    float shared_buffer[16];
    
    // Create tensors for global and shared memory
    auto global_layout = make_layout(make_shape(Int<8>{}, Int<4>{}), GenRowMajor{});
    auto shared_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    auto global_tensor = make_tensor(make_gmem_ptr(global_data), global_layout);
    auto shared_tensor = make_tensor(make_smem_ptr(shared_buffer), shared_layout);
    
    std::cout << "Global tensor layout: " << global_layout << std::endl;
    std::cout << "Shared tensor layout: " << shared_layout << std::endl;
    
    // Conceptual cp.async operation - copying from global to shared memory asynchronously
    // In real implementation, this would use special cp.async PTX instructions
    std::cout << "\nSimulating cp.async from global to shared memory..." << std::endl;
    
    // Copy first 16 elements (4x4 tile) from global to shared
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            shared_tensor(i, j) = global_tensor(i, j);
        }
    }
    
    std::cout << "Shared memory content after cp.async simulation:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << shared_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nBenefits of cp.async:" << std::endl;
    std::cout << "- Asynchronous memory transfer" << std::endl;
    std::cout << "- Overlaps with computation" << std::endl;
    std::cout << "- Improved memory bandwidth utilization" << std::endl;
}

// Function to demonstrate coalescing strategies
void demonstrate_coalescing_strategies() {
    std::cout << "\n=== Coalescing Strategies ===" << std::endl;
    
    // Create a larger tensor to demonstrate coalescing
    float data[64];  // 8x8 matrix
    for (int i = 0; i < 64; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Row-major layout (coalesced for row access)
    auto row_major_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto row_major_tensor = make_tensor(make_gmem_ptr(data), row_major_layout);
    
    // Column-major layout (coalesced for column access)
    auto col_major_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenColMajor{});
    auto col_major_tensor = make_tensor(make_gmem_ptr(data), col_major_layout);
    
    std::cout << "Row-major layout: " << row_major_layout << std::endl;
    std::cout << "Column-major layout: " << col_major_layout << std::endl;
    
    std::cout << "\nRow-wise access with row-major layout (COALESCED):" << std::endl;
    for (int i = 0; i < 2; ++i) {  // Just showing first 2 rows
        for (int j = 0; j < 8; ++j) {
            std::cout << row_major_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nColumn-wise access with row-major layout (UNCOALESCED):" << std::endl;
    for (int j = 0; j < 2; ++j) {  // Just showing first 2 columns
        for (int i = 0; i < 8; ++i) {
            std::cout << row_major_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nCoalescing tips:" << std::endl;
    std::cout << "- Ensure consecutive threads access consecutive memory locations" << std::endl;
    std::cout << "- Match access pattern to data layout (row access for row-major)" << std::endl;
    std::cout << "- Use vectorized loads when possible" << std::endl;
}

int main() {
    std::cout << "CuTe Tiled Copy Study - Module 03" << std::endl;
    std::cout << "=================================" << std::endl;
    
    demonstrate_tiled_copy_basics();
    demonstrate_vectorized_loads();
    demonstrate_cp_async_sm89();
    demonstrate_coalescing_strategies();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "1. TiledCopy enables cooperative data movement between threads" << std::endl;
    std::cout << "2. Vectorized loads (128-bit) maximize memory bandwidth" << std::endl;
    std::cout << "3. cp.async operations enable asynchronous memory transfers" << std::endl;
    std::cout << "4. Coalescing strategies optimize memory access patterns" << std::endl;
    
    return 0;
}