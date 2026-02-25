/**
 * Exercise 06: Multi-dimensional Tensors
 * 
 * Objective: Work with tensors beyond 2D - 3D, 4D, and higher dimensions
 * 
 * Tasks:
 * 1. Create and access 3D tensors
 * 2. Understand higher-dimensional layout
 * 3. Apply multi-dimensional tensors to real problems
 * 4. Practice with tensor products
 * 
 * Key Concepts:
 * - Multi-dimensional: Tensors with 3+ dimensions
 * - Dimension Order: How dimensions are organized in memory
 * - Applications: Images (HWC), volumes (DHW), batches (NCHW)
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 06: Multi-dimensional Tensors ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Create a 3D tensor (depth=2, rows=4, cols=4)
    // Common format for multi-channel images or feature maps
    std::cout << "Task 1 - 3D Tensor (2x4x4):" << std::endl;
    float data_3d[32];
    for (int i = 0; i < 32; ++i) {
        data_3d[i] = static_cast<float>(i);
    }

    auto layout_3d = make_layout(make_shape(Int<2>{}, Int<4>{}, Int<4>{}), GenRowMajor{});
    auto tensor_3d = make_tensor(make_gmem_ptr(data_3d), layout_3d);

    std::cout << "3D Layout: " << layout_3d << std::endl;
    std::cout << "Shape: " << tensor_3d.layout().shape() << std::endl;
    std::cout << "Stride: " << tensor_3d.layout().stride() << std::endl;
    std::cout << std::endl;

    // Access each "channel" or "depth slice"
    std::cout << "Depth slice 0:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(tensor_3d(0, i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Depth slice 1:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(tensor_3d(1, i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 2: Create a 4D tensor (batch=2, channels=2, height=4, width=4)
    // NCHW format common in deep learning
    std::cout << "Task 2 - 4D Tensor (2x2x4x4) in NCHW format:" << std::endl;
    float data_4d[64];
    for (int i = 0; i < 64; ++i) {
        data_4d[i] = static_cast<float>(i);
    }

    auto layout_4d = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<4>{}, Int<4>{}), GenRowMajor{});
    auto tensor_4d = make_tensor(make_gmem_ptr(data_4d), layout_4d);

    std::cout << "4D Layout: " << layout_4d << std::endl;
    std::cout << "Shape: " << tensor_4d.layout().shape() << std::endl;
    std::cout << std::endl;

    // Access specific elements
    std::cout << "Batch 0, Channel 0:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%3d ", static_cast<int>(tensor_4d(0, 0, i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Understand stride calculation for multi-dimensional
    std::cout << "Task 3 - Stride Calculation:" << std::endl;
    std::cout << "For 3D layout (D, H, W) in row-major:" << std::endl;
    std::cout << "  stride_D = H × W" << std::endl;
    std::cout << "  stride_H = W" << std::endl;
    std::cout << "  stride_W = 1" << std::endl;
    std::cout << std::endl;
    
    std::cout << "For our 3D tensor (2, 4, 4):" << std::endl;
    std::cout << "  stride_D = 4 × 4 = 16" << std::endl;
    std::cout << "  stride_H = 4" << std::endl;
    std::cout << "  stride_W = 1" << std::endl;
    std::cout << "  Actual stride: " << layout_3d.stride() << std::endl;
    std::cout << std::endl;

    // TASK 4: Offset calculation in 3D
    std::cout << "Task 4 - 3D Offset Calculation:" << std::endl;
    std::cout << "offset(d, h, w) = d × stride_D + h × stride_H + w × stride_W" << std::endl;
    std::cout << std::endl;
    
    int d = 1, h = 2, w = 3;
    int offset = tensor_3d.layout()(d, h, w);
    int calculated = d * 16 + h * 4 + w * 1;
    std::cout << "Position (1, 2, 3):" << std::endl;
    std::cout << "  From layout: " << offset << std::endl;
    std::cout << "  Calculated: " << calculated << std::endl;
    std::cout << "  Match: " << (offset == calculated ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // TASK 5: Image format conversion (HWC to CHW)
    std::cout << "Task 5 - Image Format Concepts:" << std::endl;
    std::cout << "HWC (Height, Width, Channels) - TensorFlow format" << std::endl;
    std::cout << "CHW (Channels, Height, Width) - PyTorch format" << std::endl;
    std::cout << std::endl;
    
    // Create HWC layout (4x4x3 = RGB image)
    auto hwc_layout = make_layout(make_shape(Int<4>{}, Int<4>{}, Int<3>{}), GenRowMajor{});
    std::cout << "HWC Layout: " << hwc_layout << std::endl;
    
    // Create CHW layout (3x4x4)
    auto chw_layout = make_layout(make_shape(Int<3>{}, Int<4>{}, Int<4>{}), GenRowMajor{});
    std::cout << "CHW Layout: " << chw_layout << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate offsets in 4D
    std::cout << "=== Challenge: 4D Offset Calculation ===" << std::endl;
    std::cout << "For 4D tensor (N, C, H, W) = (2, 2, 4, 4):" << std::endl;
    std::cout << "Calculate offset for position (1, 0, 2, 3):" << std::endl;
    
    int n = 1, c = 0, h = 2, w = 3;
    // Strides: N=64, C=32, H=8, W=2 (verify this)
    std::cout << "  stride_N = C×H×W = 2×4×4 = 32" << std::endl;
    std::cout << "  stride_C = H×W = 4×4 = 16" << std::endl;
    std::cout << "  stride_H = W = 4" << std::endl;
    std::cout << "  stride_W = 1" << std::endl;
    std::cout << "  Actual strides: " << layout_4d.stride() << std::endl;
    std::cout << std::endl;
    
    int offset_4d = tensor_4d.layout()(n, c, h, w);
    int calc_4d = n * 32 + c * 16 + h * 4 + w * 1;
    std::cout << "Offset for (1, 0, 2, 3):" << std::endl;
    std::cout << "  From layout: " << offset_4d << std::endl;
    std::cout << "  Calculated: " << calc_4d << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Multi-dimensional tensors extend 2D concepts" << std::endl;
    std::cout << "2. Stride for dim i = product of all smaller dimensions" << std::endl;
    std::cout << "3. Common formats: NCHW (PyTorch), NHWC (TensorFlow)" << std::endl;
    std::cout << "4. Offset = sum(coord[i] × stride[i]) for any dimension" << std::endl;

    return 0;
}
