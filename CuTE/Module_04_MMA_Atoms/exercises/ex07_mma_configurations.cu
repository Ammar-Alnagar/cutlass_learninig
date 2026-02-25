/**
 * Exercise 07: MMA Atom Configurations
 * 
 * Objective: Learn about different MMA atom configurations
 *            for various architectures and use cases
 * 
 * Tasks:
 * 1. Understand MMA atom naming convention
 * 2. Explore configurations for sm_80
 * 3. Compare different tile sizes
 * 4. Select appropriate configuration
 * 
 * Key Concepts:
 * - MMA Configuration: M × N × K tile size
 * - Architecture: Different GPUs support different configs
 * - Data Type: FP16, BF16, INT8, FP64, etc.
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 07: MMA Atom Configurations ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: MMA atom naming convention
    std::cout << "Task 1 - MMA Atom Naming:" << std::endl;
    std::cout << "Format: SM<arch>_MxNxK_<DType>_<Layout>" << std::endl;
    std::cout << std::endl;
    std::cout << "Example: SM80_16x8x16_F32F16F16F32_TN" << std::endl;
    std::cout << "  SM80: sm_80 architecture (A100)" << std::endl;
    std::cout << "  16x8x16: M=16, N=8, K=16 tile" << std::endl;
    std::cout << "  F32F16F16F32: D=A=B=C data types" << std::endl;
    std::cout << "  TN: Transpose A, Not transpose B" << std::endl;
    std::cout << std::endl;

    // TASK 2: Common MMA configurations for sm_80
    std::cout << "Task 2 - Common sm_80 Configurations:" << std::endl;
    std::cout << std::endl;

    struct MMAConfig {
        const char* name;
        int M, N, K;
        const char* dtype;
        const char* use_case;
    };

    MMAConfig configs[] = {
        {"SM80_16x8x16_F32F16F16F32", 16, 8, 16, "FP16", "General GEMM"},
        {"SM80_16x8x32_F32F16F16F32", 16, 8, 32, "FP16", "K-parallel GEMM"},
        {"SM80_8x8x4_F32F64F64F64",   8,  8, 4,  "FP64", "Scientific computing"},
        {"SM80_16x8x32_S32S8S8S32",   16, 8, 32, "INT8", "Inference"},
        {"SM80_16x8x8_F32BF16BF16F32",16, 8, 8,  "BF16", "ML training"},
    };

    std::cout << "| Configuration              | M  | N  | K  | Type | Use Case        |" << std::endl;
    std::cout << "|----------------------------|----|----|----|------|-----------------|" << std::endl;
    for (auto& cfg : configs) {
        printf("| %-26s | %2d | %2d | %2d | %-4s | %-15s |\n",
               cfg.name, cfg.M, cfg.N, cfg.K, cfg.dtype, cfg.use_case);
    }
    std::cout << std::endl;

    // TASK 3: Architecture comparison
    std::cout << "Task 3 - Architecture Comparison:" << std::endl;
    std::cout << std::endl;

    std::cout << "| Arch | GPU    | FP16 Tensor | INT8 Tensor | FP64 Tensor |" << std::endl;
    std::cout << "|------|--------|-------------|-------------|-------------|" << std::endl;
    std::cout << "| sm_70| V100   | 16x16x16    | 8x8x16      | No          |" << std::endl;
    std::cout << "| sm_75| T4     | 16x16x16    | 8x8x16      | No          |" << std::endl;
    std::cout << "| sm_80| A100   | 16x8x16/32  | 16x8x32     | 8x8x4       |" << std::endl;
    std::cout << "| sm_86| A10    | 16x8x16/32  | 16x8x32     | 8x8x4       |" << std::endl;
    std::cout << "| sm_89| H100   | 16x8x16/32  | 16x8x32     | 8x8x4       |" << std::endl;
    std::cout << "| sm_90| H100   | 16x8x16/32  | 16x8x32     | 8x8x4       |" << std::endl;
    std::cout << std::endl;

    // TASK 4: Tile size impact
    std::cout << "Task 4 - Tile Size Impact:" << std::endl;
    std::cout << "Larger K tile:" << std::endl;
    std::cout << "  + More computation per load" << std::endl;
    std::cout << "  + Better arithmetic intensity" << std::endl;
    std::cout << "  - More register pressure" << std::endl;
    std::cout << std::endl;
    std::cout << "Smaller M/N tile:" << std::endl;
    std::cout << "  + Better for small matrices" << std::endl;
    std::cout << "  + More parallelism" << std::endl;
    std::cout << "  - Less efficient for large matrices" << std::endl;
    std::cout << std::endl;

    // TASK 5: Layout options
    std::cout << "Task 5 - Layout Options:" << std::endl;
    std::cout << "Layout specifies transpose of operands:" << std::endl;
    std::cout << "  T (Transpose): A^T × B" << std::endl;
    std::cout << "  N (Not transpose): A × B" << std::endl;
    std::cout << std::endl;
    std::cout << "Common layouts:" << std::endl;
    std::cout << "  TN: A^T × B (common in attention)" << std::endl;
    std::cout << "  NN: A × B (standard GEMM)" << std::endl;
    std::cout << "  NT: A × B^T (gradient computation)" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Select configuration
    std::cout << "=== Challenge: Select Configuration ===" << std::endl;
    std::cout << "Scenario 1: FP16 GEMM on A100" << std::endl;
    std::cout << "  Answer: SM80_16x8x16_F32F16F16F32_TN or _NN" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 2: INT8 inference on A100" << std::endl;
    std::cout << "  Answer: SM80_16x8x32_S32S8S8S32" << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario 3: FP64 scientific computing" << std::endl;
    std::cout << "  Answer: SM80_8x8x4_F32F64F64F64" << std::endl;
    std::cout << std::endl;

    // CONFIGURATION SELECTION GUIDE
    std::cout << "=== Configuration Selection Guide ===" << std::endl;
    std::cout << R"(
1. Determine your architecture (sm_XX)
2. Choose data type (FP16, BF16, INT8, FP64)
3. Select tile size based on:
   - Matrix dimensions (M, N, K)
   - Register availability
   - Desired parallelism
4. Choose layout (TN, NN, NT)
5. Test and profile for best performance
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. MMA configs specify M×N×K tile and data types" << std::endl;
    std::cout << "2. Different architectures support different configs" << std::endl;
    std::cout << "3. Choose config based on use case" << std::endl;
    std::cout << "4. Layout specifies operand transpose" << std::endl;

    return 0;
}
