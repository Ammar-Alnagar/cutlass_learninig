/**
 * Exercise 05: Offset Mapping Challenge
 * 
 * Objective: Practice calculating memory offsets from logical coordinates
 *            and verify understanding of layout mapping
 * 
 * Tasks:
 * 1. Predict offsets before running the program
 * 2. Verify predictions against actual CuTe calculations
 * 3. Understand the formula: offset = sum(coord[i] * stride[i])
 * 4. Work with different layout configurations
 * 
 * Key Formula:
 * For a layout with shape S and stride D:
 *   offset(c0, c1, ..., cn) = c0*D0 + c1*D1 + ... + cn*Dn
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/util/print.hpp"

using namespace cute;

void test_offset_prediction(const char* layout_name, Layout const& layout, 
                            int coord0, int coord1, int expected_offset) {
    int actual_offset = layout(coord0, coord1);
    bool correct = (actual_offset == expected_offset);
    
    std::cout << "  " << layout_name << "(" << coord0 << "," << coord1 << ")"
              << " = " << actual_offset
              << " (expected: " << expected_offset << ") "
              << (correct ? "✓" : "✗") << std::endl;
}

int main() {
    std::cout << "=== Exercise 05: Offset Mapping Challenge ===" << std::endl;
    std::cout << std::endl;

    // Create various layouts for testing
    auto layout_rm_4x4 = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    auto layout_cm_4x4 = make_layout(make_shape(Int<4>{}, Int<4>{}), GenColMajor{});
    auto layout_rm_8x8 = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    auto layout_padded = make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<9>{}, Int<1>{}));

    std::cout << "Layouts created:" << std::endl;
    std::cout << "1. Row-Major 4x4: " << layout_rm_4x4 << std::endl;
    std::cout << "2. Column-Major 4x4: " << layout_cm_4x4 << std::endl;
    std::cout << "3. Row-Major 8x8: " << layout_rm_8x8 << std::endl;
    std::cout << "4. Padded 8x8 (stride=9): " << layout_padded << std::endl;
    std::cout << std::endl;

    // CHALLENGE 1: Row-Major 4x4 offsets
    std::cout << "=== Challenge 1: Row-Major 4x4 ===" << std::endl;
    std::cout << "Predict the offsets, then check below:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Your predictions:" << std::endl;
    std::cout << "  layout(0, 3) = ___" << std::endl;
    std::cout << "  layout(2, 1) = ___" << std::endl;
    std::cout << "  layout(3, 3) = ___" << std::endl;
    std::cout << std::endl;

    std::cout << "Actual values:" << std::endl;
    test_offset_prediction("RM4x4", layout_rm_4x4, 0, 3, 3);   // 0*4 + 3*1 = 3
    test_offset_prediction("RM4x4", layout_rm_4x4, 2, 1, 9);   // 2*4 + 1*1 = 9
    test_offset_prediction("RM4x4", layout_rm_4x4, 3, 3, 15);  // 3*4 + 3*1 = 15
    std::cout << std::endl;

    // CHALLENGE 2: Column-Major 4x4 offsets
    std::cout << "=== Challenge 2: Column-Major 4x4 ===" << std::endl;
    std::cout << "Predict the offsets, then check below:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Your predictions:" << std::endl;
    std::cout << "  layout(3, 0) = ___" << std::endl;
    std::cout << "  layout(1, 2) = ___" << std::endl;
    std::cout << "  layout(3, 3) = ___" << std::endl;
    std::cout << std::endl;

    std::cout << "Actual values:" << std::endl;
    test_offset_prediction("CM4x4", layout_cm_4x4, 3, 0, 3);   // 3*1 + 0*4 = 3
    test_offset_prediction("CM4x4", layout_cm_4x4, 1, 2, 9);   // 1*1 + 2*4 = 9
    test_offset_prediction("CM4x4", layout_cm_4x4, 3, 3, 15);  // 3*1 + 3*4 = 15
    std::cout << std::endl;

    // CHALLENGE 3: Row-Major 8x8 offsets
    std::cout << "=== Challenge 3: Row-Major 8x8 ===" << std::endl;
    std::cout << "Predict the offsets, then check below:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Your predictions:" << std::endl;
    std::cout << "  layout(0, 7) = ___" << std::endl;
    std::cout << "  layout(4, 3) = ___" << std::endl;
    std::cout << "  layout(7, 7) = ___" << std::endl;
    std::cout << std::endl;

    std::cout << "Actual values:" << std::endl;
    test_offset_prediction("RM8x8", layout_rm_8x8, 0, 7, 7);   // 0*8 + 7*1 = 7
    test_offset_prediction("RM8x8", layout_rm_8x8, 4, 3, 35);  // 4*8 + 3*1 = 35
    test_offset_prediction("RM8x8", layout_rm_8x8, 7, 7, 63);  // 7*8 + 7*1 = 63
    std::cout << std::endl;

    // CHALLENGE 4: Padded 8x8 offsets (stride = 9)
    std::cout << "=== Challenge 4: Padded 8x8 (stride=9) ===" << std::endl;
    std::cout << "Predict the offsets, then check below:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Your predictions:" << std::endl;
    std::cout << "  layout(0, 7) = ___" << std::endl;
    std::cout << "  layout(4, 3) = ___" << std::endl;
    std::cout << "  layout(7, 7) = ___" << std::endl;
    std::cout << std::endl;

    std::cout << "Actual values:" << std::endl;
    test_offset_prediction("Padded8x8", layout_padded, 0, 7, 7);   // 0*9 + 7*1 = 7
    test_offset_prediction("Padded8x8", layout_padded, 4, 3, 39);  // 4*9 + 3*1 = 39
    test_offset_prediction("Padded8x8", layout_padded, 7, 7, 70);  // 7*9 + 7*1 = 70
    std::cout << std::endl;

    // FORMULA VERIFICATION
    std::cout << "=== Formula Verification ===" << std::endl;
    std::cout << "For row-major layout with shape (R, C) and stride (C, 1):" << std::endl;
    std::cout << "  offset(row, col) = row * C + col * 1" << std::endl;
    std::cout << std::endl;
    
    std::cout << "For column-major layout with shape (R, C) and stride (1, R):" << std::endl;
    std::cout << "  offset(row, col) = row * 1 + col * R" << std::endl;
    std::cout << std::endl;

    std::cout << "For padded layout with shape (R, C) and stride (P, 1) where P > C:" << std::endl;
    std::cout << "  offset(row, col) = row * P + col * 1" << std::endl;
    std::cout << std::endl;

    // FINAL CHALLENGE: Calculate stride from observed offsets
    std::cout << "=== Final Challenge: Reverse Engineering ===" << std::endl;
    std::cout << "Given these offsets for a 4x4 layout:" << std::endl;
    std::cout << "  layout(0,1) - layout(0,0) = 1" << std::endl;
    std::cout << "  layout(1,0) - layout(0,0) = 5" << std::endl;
    std::cout << "What is the stride? Answer: (5, 1) - This is a padded layout!" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Offset = sum(coordinate[i] * stride[i])" << std::endl;
    std::cout << "2. Row-major: stride = (num_cols, 1)" << std::endl;
    std::cout << "3. Column-major: stride = (1, num_rows)" << std::endl;
    std::cout << "4. Padding changes the stride, not the shape" << std::endl;

    return 0;
}
