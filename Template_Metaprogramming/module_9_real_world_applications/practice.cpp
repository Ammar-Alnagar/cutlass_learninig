#include <iostream>
#include <vector>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <memory>

// Module 9: Real-world Applications and Case Studies Practice
// Hands-on tutorial for practical CUTLASS applications
// TODO: Implement each exercise to learn real-world CUTLASS applications

/*
 * EXERCISE 1: DEEP LEARNING FRAMEWORK INTEGRATION
 * Integrating CUTLASS with deep learning frameworks
 */
// Simulated PyTorch integration example
class PyTorchIntegrationSimulator {
public:
    // Simulate CUTLASS GEMM operation
    static std::vector<std::vector<float>> cutlass_gemm_fp16(
        const std::vector<std::vector<float>>& A,
        const std::vector<std::vector<float>>& B,
        float alpha = 1.0f,
        float beta = 0.0f) {

        int M = A.size();
        int K = A[0].size();
        int N = B[0].size();

        std::cout << "Simulating CUTLASS GEMM: " << M << "x" << K << " * " << K << "x" << N << std::endl;

        // Create output matrix C = A * B
        std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));

        // Perform matrix multiplication
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
                C[i][j] = alpha * C[i][j] + beta * C[i][j]; // Apply alpha/beta
            }
        }

        std::cout << "GEMM completed successfully" << std::endl;
        return C;
    }

    static void demonstrate_integration() {
        std::cout << "\nDeep Learning Framework Integration:" << std::endl;
        std::cout << "CUTLASS can be integrated with popular frameworks like PyTorch and TensorFlow." << std::endl;

        // Example: Create simple matrices
        std::vector<std::vector<float>> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        std::vector<std::vector<float>> B = {{5.0f, 6.0f}, {7.0f, 8.0f}};

        std::cout << "\nInput matrices:" << std::endl;
        std::cout << "A = [[1, 2], [3, 4]]" << std::endl;
        std::cout << "B = [[5, 6], [7, 8]]" << std::endl;

        auto result = cutlass_gemm_fp16(A, B);

        std::cout << "\nResult of A * B:" << std::endl;
        std::cout << "C = [[" << result[0][0] << ", " << result[0][1] << "], ["
                  << result[1][0] << ", " << result[1][1] << "]]" << std::endl;

        std::cout << "\nIntegration benefits:" << std::endl;
        std::cout << "1. High-performance GEMM operations" << std::endl;
        std::cout << "2. Support for various data types (FP16, INT8, etc.)" << std::endl;
        std::cout << "3. Optimized for specific GPU architectures" << std::endl;
        std::cout << "4. Custom epilogue operations (bias, activation functions)" << std::endl;
    }
};

// Simulated TensorFlow integration example
class TensorFlowIntegrationSimulator {
public:
    struct Tensor {
        std::vector<float> data;
        std::vector<int> shape;

        Tensor(const std::vector<int>& s) : shape(s) {
            int total_size = 1;
            for (int dim : s) total_size *= dim;
            data.resize(total_size);
        }

        float& operator[](size_t idx) { return data[idx]; }
        const float& operator[](size_t idx) const { return data[idx]; }
    };

    static Tensor cutlass_gemm_op(
        const Tensor& A, const Tensor& B, const Tensor& C,
        float alpha = 1.0f, float beta = 0.0f) {

        int M = A.shape[0];
        int K = A.shape[1];
        int N = B.shape[1];

        std::cout << "TensorFlow CUTLASS GEMM: " << M << "x" << K << " * " << K << "x" << N << std::endl;

        Tensor output({M, N});

        // Perform GEMM: output = alpha * A * B + beta * C
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                output[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }

        return output;
    }

    static void demonstrate_integration() {
        std::cout << "\nTensorFlow Integration Example:" << std::endl;

        Tensor A({2, 3}); // 2x3 matrix
        Tensor B({3, 2}); // 3x2 matrix
        Tensor C({2, 2}); // 2x2 matrix (bias)

        // Initialize with sample values
        for (int i = 0; i < 6; ++i) A[i] = i + 1.0f;
        for (int i = 0; i < 6; ++i) B[i] = i + 7.0f;
        for (int i = 0; i < 4; ++i) C[i] = 1.0f;

        auto result = cutlass_gemm_op(A, B, C, 1.0f, 0.0f);

        std::cout << "Result shape: [" << result.shape[0] << ", " << result.shape[1] << "]" << std::endl;
        std::cout << "Sample values: [" << result[0] << ", " << result[1] << ", "
                  << result[2] << ", " << result[3] << "]" << std::endl;
    }
};

void exercise_deep_learning_framework_integration() {
    std::cout << "\n=== Exercise 1: Deep Learning Framework Integration ===" << std::endl;

    PyTorchIntegrationSimulator::demonstrate_integration();
    TensorFlowIntegrationSimulator::demonstrate_integration();

    std::cout << "\nReal-world integration considerations:" << std::endl;
    std::cout << "1. Memory layout compatibility (row-major vs column-major)" << std::endl;
    std::cout << "2. Data type conversion and quantization" << std::endl;
    std::cout << "3. Error handling and fallback mechanisms" << std::endl;
    std::cout << "4. Performance profiling and optimization" << std::endl;
}

/*
 * EXERCISE 2: QUANTIZED MATRIX MULTIPLICATION
 * Implementing quantized operations with CUTLASS
 */
class QuantizationUtils {
public:
    // Quantize float tensor to INT8
    static std::vector<int8_t> quantize_tensor(
        const std::vector<float>& input, float scale, int8_t zero_point = 0) {

        std::vector<int8_t> output(input.size());

        for (size_t i = 0; i < input.size(); ++i) {
            float scaled = input[i] / scale + zero_point;
            int8_t quantized = static_cast<int8_t>(std::round(scaled));
            // Clamp to [-128, 127]
            quantized = std::max(static_cast<int8_t>(-128),
                               std::min(static_cast<int8_t>(127), quantized));
            output[i] = quantized;
        }

        return output;
    }

    // Dequantize INT8 tensor to float
    static std::vector<float> dequantize_tensor(
        const std::vector<int8_t>& input, float scale, int8_t zero_point = 0) {

        std::vector<float> output(input.size());

        for (size_t i = 0; i < input.size(); ++i) {
            float dequantized = (input[i] - zero_point) * scale;
            output[i] = dequantized;
        }

        return output;
    }

    // Calculate optimal scale for quantization
    static float calculate_quantization_scale(const std::vector<float>& data) {
        float max_abs = 0.0f;
        for (float val : data) {
            max_abs = std::max(max_abs, std::abs(val));
        }

        // Use 127 as maximum representable value for INT8
        return max_abs / 127.0f;
    }

    static void demonstrate_quantization() {
        std::cout << "\nQuantization Example:" << std::endl;

        // Sample data
        std::vector<float> original = {1.2f, -0.8f, 3.5f, -2.1f, 0.9f, 4.7f};

        std::cout << "Original values: ";
        for (float val : original) std::cout << val << " ";
        std::cout << std::endl;

        float scale = calculate_quantization_scale(original);
        std::cout << "Calculated scale: " << scale << std::endl;

        auto quantized = quantize_tensor(original, scale);
        std::cout << "Quantized values: ";
        for (int8_t val : quantized) std::cout << static_cast<int>(val) << " ";
        std::cout << std::endl;

        auto dequantized = dequantize_tensor(quantized, scale);
        std::cout << "Dequantized values: ";
        for (float val : dequantized) std::cout << val << " ";
        std::cout << std::endl;

        // Calculate quantization error
        float mse = 0.0f;
        for (size_t i = 0; i < original.size(); ++i) {
            float error = original[i] - dequantized[i];
            mse += error * error;
        }
        mse /= original.size();
        std::cout << "Mean squared error: " << mse << std::endl;
    }
};

// Simulated quantized GEMM
class QuantizedGemmSimulator {
public:
    static std::vector<std::vector<int32_t>> quantized_gemm(
        const std::vector<std::vector<int8_t>>& A,
        const std::vector<std::vector<int8_t>>& B) {

        int M = A.size();
        int K = A[0].size();
        int N = B[0].size();

        std::vector<std::vector<int32_t>> C(M, std::vector<int32_t>(N, 0));

        // Perform INT8 GEMM with INT32 accumulator
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int32_t sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += static_cast<int32_t>(A[i][k]) * static_cast<int32_t>(B[k][j]);
                }
                C[i][j] = sum;
            }
        }

        return C;
    }

    static void demonstrate_quantized_gemm() {
        std::cout << "\nQuantized GEMM Example:" << std::endl;

        // Create sample INT8 matrices
        std::vector<std::vector<int8_t>> A = {{1, 2, 3}, {4, 5, 6}};  // 2x3
        std::vector<std::vector<int8_t>> B = {{7, 8}, {9, 10}, {11, 12}};  // 3x2

        std::cout << "INT8 Matrices:" << std::endl;
        std::cout << "A = [[1, 2, 3], [4, 5, 6]]" << std::endl;
        std::cout << "B = [[7, 8], [9, 10], [11, 12]]" << std::endl;

        auto result = quantized_gemm(A, B);

        std::cout << "INT32 Result:" << std::endl;
        std::cout << "C = [[" << result[0][0] << ", " << result[0][1] << "], ["
                  << result[1][0] << ", " << result[1][1] << "]]" << std::endl;

        std::cout << "\nQuantized GEMM benefits:" << std::endl;
        std::cout << "1. Reduced memory usage (4x reduction from FP32 to INT8)" << std::endl;
        std::cout << "2. Improved memory bandwidth utilization" << std::endl;
        std::cout << "3. Faster computation on INT8-optimized hardware" << std::endl;
        std::cout << "4. Suitable for inference in edge devices" << std::endl;
    }
};

void exercise_quantized_matrix_multiplication() {
    std::cout << "\n=== Exercise 2: Quantized Matrix Multiplication ===" << std::endl;

    QuantizationUtils::demonstrate_quantization();
    QuantizedGemmSimulator::demonstrate_quantized_gemm();

    std::cout << "\nQuantization considerations:" << std::endl;
    std::cout << "1. Calibration for optimal scale factors" << std::endl;
    std::cout << "2. Per-channel vs per-tensor quantization" << std::endl;
    std::cout << "3. Handling of outliers and dynamic ranges" << std::endl;
    std::cout << "4. Numerical accuracy trade-offs" << std::endl;
}

/*
 * EXERCISE 3: SPARSE OPERATIONS
 * Working with sparse matrix operations
 */
class SparsityPattern {
public:
    // Create 2:4 structured sparsity pattern
    static std::vector<std::vector<float>> create_2to4_sparsity(
        const std::vector<std::vector<float>>& dense_weights) {

        int rows = dense_weights.size();
        int cols = dense_weights[0].size();

        std::vector<std::vector<float>> sparse_weights(rows, std::vector<float>(cols, 0.0f));

        // Process in groups of 4 columns
        for (int row = 0; row < rows; ++row) {
            for (int col_group = 0; col_group < cols; col_group += 4) {
                // Find the 2 elements with highest magnitude in this group of 4
                std::vector<std::pair<float, int>> magnitudes;

                for (int k = 0; k < 4 && col_group + k < cols; ++k) {
                    float mag = std::abs(dense_weights[row][col_group + k]);
                    magnitudes.push_back({mag, col_group + k});
                }

                // Sort by magnitude (descending)
                std::sort(magnitudes.rbegin(), magnitudes.rend());

                // Keep only the 2 with highest magnitude
                for (int k = 0; k < 2 && k < magnitudes.size(); ++k) {
                    int col_idx = magnitudes[k].second;
                    sparse_weights[row][col_idx] = dense_weights[row][col_idx];
                }
            }
        }

        return sparse_weights;
    }

    // Calculate sparsity ratio
    static float calculate_sparsity_ratio(const std::vector<std::vector<float>>& matrix) {
        int total_elements = 0;
        int zero_elements = 0;

        for (const auto& row : matrix) {
            for (float val : row) {
                total_elements++;
                if (val == 0.0f) zero_elements++;
            }
        }

        return static_cast<float>(zero_elements) / total_elements;
    }

    static void demonstrate_sparsity() {
        std::cout << "\nSparsity Pattern Example:" << std::endl;

        // Create a sample dense matrix
        std::vector<std::vector<float>> dense = {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f}
        };

        std::cout << "Original dense matrix:" << std::endl;
        for (const auto& row : dense) {
            for (float val : row) std::cout << val << " ";
            std::cout << std::endl;
        }

        auto sparse = create_2to4_sparsity(dense);

        std::cout << "\n2:4 Structured sparse matrix:" << std::endl;
        for (const auto& row : sparse) {
            for (float val : row) std::cout << val << " ";
            std::cout << std::endl;
        }

        float original_sparsity = calculate_sparsity_ratio(dense);
        float new_sparsity = calculate_sparsity_ratio(sparse);

        std::cout << "\nSparsity ratios:" << std::endl;
        std::cout << "Original: " << original_sparsity << std::endl;
        std::cout << "After 2:4 sparsification: " << new_sparsity << std::endl;
    }
};

// Simulated sparse-dense GEMM
class SparseGemmSimulator {
public:
    static std::vector<std::vector<float>> sparse_dense_gemm(
        const std::vector<std::vector<float>>& sparse_A,
        const std::vector<std::vector<float>>& dense_B) {

        int M = sparse_A.size();
        int K = sparse_A[0].size();
        int N = dense_B[0].size();

        std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));

        // Perform sparse-dense GEMM (skip zero elements in sparse matrix)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    if (sparse_A[i][k] != 0.0f) {  // Skip zero elements
                        sum += sparse_A[i][k] * dense_B[k][j];
                    }
                }
                C[i][j] = sum;
            }
        }

        return C;
    }

    static void demonstrate_sparse_gemm() {
        std::cout << "\nSparse-Dense GEMM Example:" << std::endl;

        // Create a sparse matrix (with some zeros)
        std::vector<std::vector<float>> sparse_A = {
            {1.0f, 0.0f, 3.0f, 0.0f},
            {0.0f, 2.0f, 0.0f, 4.0f},
            {5.0f, 0.0f, 0.0f, 6.0f}
        };

        // Create a dense matrix
        std::vector<std::vector<float>> dense_B = {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f},
            {7.0f, 8.0f}
        };

        std::cout << "Sparse A matrix:" << std::endl;
        for (const auto& row : sparse_A) {
            for (float val : row) std::cout << val << " ";
            std::cout << std::endl;
        }

        std::cout << "\nDense B matrix:" << std::endl;
        for (const auto& row : dense_B) {
            for (float val : row) std::cout << val << " ";
            std::cout << std::endl;
        }

        auto result = sparse_dense_gemm(sparse_A, dense_B);

        std::cout << "\nResult C = A * B:" << std::endl;
        for (const auto& row : result) {
            for (float val : row) std::cout << val << " ";
            std::cout << std::endl;
        }

        std::cout << "\nSparse GEMM benefits:" << std::endl;
        std::cout << "1. Reduced computation (skip zero elements)" << std::endl;
        std::cout << "2. Lower memory bandwidth requirements" << std::endl;
        std::cout << "3. Specialized hardware support (structured sparsity)" << std::endl;
        std::cout << "4. Model compression without significant accuracy loss" << std::endl;
    }
};

void exercise_sparse_operations() {
    std::cout << "\n=== Exercise 3: Sparse Operations ===" << std::endl;

    SparsityPattern::demonstrate_sparsity();
    SparseGemmSimulator::demonstrate_sparse_gemm();

    std::cout << "\nSparse operation considerations:" << std::endl;
    std::cout << "1. Sparsity pattern affects performance significantly" << std::endl;
    std::cout << "2. Structured sparsity (2:4) vs unstructured sparsity" << std::endl;
    std::cout << "3. Hardware support varies across GPU generations" << std::endl;
    std::cout << "4. Training vs inference sparsity strategies" << std::endl;
}

/*
 * EXERCISE 4: MIXED PRECISION COMPUTATIONS
 * Using mixed precision for improved performance
 */
class MixedPrecisionSimulator {
public:
    // Simulate mixed precision: FP16 inputs, FP32 accumulation, FP16 output
    static std::vector<std::vector<float>> mixed_precision_gemm(
        const std::vector<std::vector<float>>& A_fp16,
        const std::vector<std::vector<float>>& B_fp16) {

        int M = A_fp16.size();
        int K = A_fp16[0].size();
        int N = B_fp16[0].size();

        std::vector<std::vector<float>> C_fp32(M, std::vector<float>(N, 0.0f));

        // Perform computation with FP32 accumulation for higher precision
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;  // FP32 accumulator
                for (int k = 0; k < K; ++k) {
                    sum += A_fp16[i][k] * B_fp16[k][j];  // Inputs are treated as FP16
                }
                C_fp32[i][j] = sum;
            }
        }

        // Convert back to FP16 (simulated)
        std::vector<std::vector<float>> C_fp16(M, std::vector<float>(N));
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                // Simulate FP16 conversion with potential precision loss
                C_fp16[i][j] = C_fp32[i][j];
            }
        }

        return C_fp16;
    }

    static void demonstrate_mixed_precision() {
        std::cout << "\nMixed Precision Example:" << std::endl;

        // Create sample matrices (representing FP16 values)
        std::vector<std::vector<float>> A = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };

        std::vector<std::vector<float>> B = {
            {0.5f, 1.5f},
            {2.5f, 3.5f},
            {4.5f, 5.5f}
        };

        std::cout << "Input matrices (simulating FP16):" << std::endl;
        std::cout << "A = [[1, 2, 3], [4, 5, 6]]" << std::endl;
        std::cout << "B = [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]" << std::endl;

        auto result = mixed_precision_gemm(A, B);

        std::cout << "\nResult (with FP32 accumulation):" << std::endl;
        std::cout << "C = [[" << result[0][0] << ", " << result[0][1] << "], ["
                  << result[1][0] << ", " << result[1][1] << "]]" << std::endl;

        std::cout << "\nMixed precision benefits:" << std::endl;
        std::cout << "1. Faster computation with FP16 inputs" << std::endl;
        std::cout << "2. Higher numerical accuracy with FP32 accumulation" << std::endl;
        std::cout << "3. Better memory bandwidth utilization" << std::endl;
        std::cout << "4. Hardware acceleration on Tensor Core-capable GPUs" << std::endl;
    }

    // Compare precision between different approaches
    static void compare_precisions() {
        std::cout << "\nPrecision Comparison:" << std::endl;

        // Simulate the same computation with different precisions
        std::vector<std::vector<float>> A = {{1.0f/3.0f, 1.0f/3.0f}};
        std::vector<std::vector<float>> B = {{3.0f}, {3.0f}};

        // FP32 computation
        std::vector<std::vector<float>> fp32_result(1, std::vector<float>(1, 0.0f));
        fp32_result[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];

        // Mixed precision (simulated)
        std::vector<std::vector<float>> mixed_result = mixed_precision_gemm(A, B);

        std::cout << "Input: [1/3, 1/3] * [3; 3]^T = [2]" << std::endl;
        std::cout << "FP32 result: " << fp32_result[0][0] << std::endl;
        std::cout << "Mixed precision result: " << mixed_result[0][0] << std::endl;
        std::cout << "Expected: 2.0" << std::endl;
    }
};

void exercise_mixed_precision_computations() {
    std::cout << "\n=== Exercise 4: Mixed Precision Computations ===" << std::endl;

    MixedPrecisionSimulator::demonstrate_mixed_precision();
    MixedPrecisionSimulator::compare_precisions();

    std::cout << "\nMixed precision considerations:" << std::endl;
    std::cout << "1. Gradient scaling for training stability" << std::endl;
    std::cout << "2. Loss scaling techniques for gradient preservation" << std::endl;
    std::cout << "3. Hardware-specific optimizations (Tensor Cores)" << std::endl;
    std::cout << "4. Accuracy vs performance trade-offs" << std::endl;
}

/*
 * EXERCISE 5: MEMORY BANDWIDTH AND OCCUPANCY OPTIMIZATION
 * Optimizing for memory and computational efficiency
 */
class MemoryOptimizationSimulator {
public:
    // Simulate coalesced memory access pattern
    static void demonstrate_coalesced_access() {
        std::cout << "\nMemory Access Pattern Example:" << std::endl;

        int num_threads = 32;  // Warp size
        int matrix_width = 1024;

        std::cout << "Simulating memory access for " << num_threads << " threads (one warp)" << std::endl;
        std::cout << "Matrix width: " << matrix_width << " elements" << std::endl;

        // Coalesced access: consecutive threads access consecutive memory
        std::cout << "\nCoalesced access pattern:" << std::endl;
        for (int i = 0; i < 8; ++i) {  // Show first 8 accesses
            int thread_id = i;
            int memory_address = thread_id;  // Consecutive addresses
            std::cout << "Thread " << thread_id << " -> Address " << memory_address << std::endl;
        }

        // Uncoalesced access: consecutive threads access strided memory
        std::cout << "\nUncoalesced access pattern:" << std::endl;
        for (int i = 0; i < 8; ++i) {  // Show first 8 accesses
            int thread_id = i;
            int stride = 64;  // Strided access
            int memory_address = thread_id * stride;
            std::cout << "Thread " << thread_id << " -> Address " << memory_address << std::endl;
        }

        std::cout << "\nCoalesced access benefits:" << std::endl;
        std::cout << "1. Better memory bandwidth utilization" << std::endl;
        std::cout << "2. Reduced memory transactions" << std::endl;
        std::cout << "3. Improved overall performance" << std::endl;
    }

    // Simulate shared memory banking
    static void demonstrate_shared_memory_banking() {
        std::cout << "\nShared Memory Banking Example:" << std::endl;

        int num_banks = 32;
        std::cout << "GPU shared memory has " << num_banks << " banks" << std::endl;

        // Example: 32 threads accessing 32 consecutive elements (no conflict)
        std::cout << "\nNo bank conflict scenario:" << std::endl;
        for (int i = 0; i < 8; ++i) {
            int thread_id = i;
            int shared_mem_address = i;  // Each thread accesses different bank
            int bank_id = shared_mem_address % num_banks;
            std::cout << "Thread " << thread_id << " -> Address " << shared_mem_address
                      << " -> Bank " << bank_id << std::endl;
        }

        // Example: 32 threads accessing same bank (conflict)
        std::cout << "\nBank conflict scenario:" << std::endl;
        for (int i = 0; i < 8; ++i) {
            int thread_id = i;
            int shared_mem_address = i * num_banks;  // All access bank 0
            int bank_id = shared_mem_address % num_banks;
            std::cout << "Thread " << thread_id << " -> Address " << shared_mem_address
                      << " -> Bank " << bank_id << " (CONFLICT!)" << std::endl;
        }

        std::cout << "\nBank conflict mitigation:" << std::endl;
        std::cout << "1. Add padding to matrix dimensions" << std::endl;
        std::cout << "2. Use different access patterns" << std::endl;
        std::cout << "3. Restructure algorithms to avoid conflicts" << std::endl;
    }
};

class OccupancyOptimizer {
public:
    // Calculate theoretical occupancy
    static float calculate_occupancy(int active_blocks, int block_size, int max_threads_per_sm) {
        int active_threads = active_blocks * block_size;
        return static_cast<float>(active_threads) / max_threads_per_sm;
    }

    // Simulate different block sizes and their impact on occupancy
    static void demonstrate_occupancy_impact() {
        std::cout << "\nOccupancy Optimization Example:" << std::endl;

        int max_threads_per_sm = 2048;  // Example for modern GPU
        int max_blocks_per_sm = 32;     // Example for modern GPU

        std::cout << "GPU limits: " << max_threads_per_sm << " threads per SM, "
                  << max_blocks_per_sm << " blocks per SM" << std::endl;

        std::vector<int> block_sizes = {64, 128, 256, 512};

        std::cout << "\nBlock size analysis:" << std::endl;
        for (int block_size : block_sizes) {
            // Calculate max active blocks considering thread limit
            int max_blocks_by_threads = max_threads_per_sm / block_size;
            int max_blocks = std::min(max_blocks_per_sm, max_blocks_by_threads);

            float occupancy = calculate_occupancy(max_blocks, block_size, max_threads_per_sm);

            std::cout << "Block size " << block_size << ": max " << max_blocks
                      << " blocks, " << occupancy * 100 << "% occupancy" << std::endl;
        }

        std::cout << "\nOccupancy optimization strategies:" << std::endl;
        std::cout << "1. Balance block size to maximize occupancy" << std::endl;
        std::cout << "2. Consider shared memory usage per block" << std::endl;
        std::cout << "3. Account for register usage per thread" << std::endl;
        std::cout << "4. Use CUDA occupancy calculator API" << std::endl;
    }
};

void exercise_memory_bandwidth_and_occupancy_optimization() {
    std::cout << "\n=== Exercise 5: Memory Bandwidth and Occupancy Optimization ===" << std::endl;

    MemoryOptimizationSimulator::demonstrate_coalesced_access();
    MemoryOptimizationSimulator::demonstrate_shared_memory_banking();
    OccupancyOptimizer::demonstrate_occupancy_impact();

    std::cout << "\nOptimization best practices:" << std::endl;
    std::cout << "1. Profile memory access patterns" << std::endl;
    std::cout << "2. Use appropriate data layouts (AOS vs SOA)" << std::endl;
    std::cout << "3. Optimize for cache line utilization" << std::endl;
    std::cout << "4. Balance occupancy with resource usage" << std::endl;
    std::cout << "5. Consider algorithmic changes for better memory access" << std::endl;
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these real-world scenarios
 */

// Challenge 1: Quantized Linear Layer
class QuantizedLinearLayer {
public:
    std::vector<float> weights_scale;
    std::vector<float> input_scale;
    std::vector<float> output_scale;

    // Simulate quantized linear layer: Y = activation(alpha * W_quant * X_quant + bias)
    std::vector<float> forward(
        const std::vector<std::vector<int8_t>>& weight_quantized,
        const std::vector<int8_t>& input_quantized,
        const std::vector<float>& bias,
        float alpha = 1.0f) {

        int output_size = weight_quantized.size();
        int input_size = weight_quantized[0].size();

        std::vector<float> output(output_size, 0.0f);

        // Perform quantized matrix-vector multiplication
        for (int i = 0; i < output_size; ++i) {
            int32_t accumulator = 0;
            for (int j = 0; j < input_size; ++j) {
                accumulator += static_cast<int32_t>(weight_quantized[i][j]) *
                              static_cast<int32_t>(input_quantized[j]);
            }

            // Apply bias and scaling
            float dequantized_result = accumulator * (weights_scale[i] * input_scale[0]);
            output[i] = alpha * dequantized_result + bias[i];

            // Apply activation function (ReLU)
            if (output[i] < 0) output[i] = 0;
        }

        return output;
    }

    void initialize_scales(int weight_size, int input_size) {
        weights_scale.resize(weight_size);
        input_scale.resize(input_size);
        output_scale.resize(weight_size);

        // Initialize with dummy scale values
        for (auto& scale : weights_scale) scale = 0.01f;
        for (auto& scale : input_scale) scale = 0.02f;
        for (auto& scale : output_scale) scale = 0.005f;
    }
};

// Challenge 2: Sparse Matrix Format Converter
class SparseFormatConverter {
public:
    struct CoordinateFormat {
        std::vector<float> values;
        std::vector<int> row_indices;
        std::vector<int> col_indices;
        int rows, cols;
    };

    struct CSRFormat {
        std::vector<float> values;
        std::vector<int> column_indices;
        std::vector<int> row_pointers;
        int rows, cols;
    };

    static CSRFormat dense_to_csr(const std::vector<std::vector<float>>& dense) {
        CSRFormat csr;
        csr.rows = dense.size();
        csr.cols = dense.empty() ? 0 : dense[0].size();

        for (int i = 0; i < csr.rows; ++i) {
            csr.row_pointers.push_back(csr.values.size()); // Start of row

            for (int j = 0; j < csr.cols; ++j) {
                if (dense[i][j] != 0.0f) {
                    csr.values.push_back(dense[i][j]);
                    csr.column_indices.push_back(j);
                }
            }
        }
        csr.row_pointers.push_back(csr.values.size()); // End of last row

        return csr;
    }

    static void print_csr_format(const CSRFormat& csr) {
        std::cout << "CSR Format:" << std::endl;
        std::cout << "Values: ";
        for (float val : csr.values) std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Column Indices: ";
        for (int idx : csr.column_indices) std::cout << idx << " ";
        std::cout << std::endl;

        std::cout << "Row Pointers: ";
        for (int ptr : csr.row_pointers) std::cout << ptr << " ";
        std::cout << std::endl;
    }
};

// Challenge 3: Performance Profiler
class PerformanceProfiler {
public:
    struct Metrics {
        float gflops;
        float bandwidth_gb_s;
        float occupancy_percent;
        float achieved_compute_peak_percent;
    };

    static Metrics profile_gemm_performance(
        int m, int n, int k,
        float execution_time_ms) {

        Metrics metrics;

        // Calculate GFLOPS (2 operations per multiply-add)
        long long total_ops = 2LL * m * n * k;
        metrics.gflops = (total_ops / 1e9) / (execution_time_ms / 1000.0f);

        // Calculate memory bandwidth (simplified)
        long long bytes_read = (m * k + k * n) * sizeof(float);
        long long bytes_written = m * n * sizeof(float);
        long long total_bytes = bytes_read + bytes_written;
        metrics.bandwidth_gb_s = (total_bytes / 1e9) / (execution_time_ms / 1000.0f);

        // Simulated occupancy (would come from CUDA occupancy calculator in real implementation)
        metrics.occupancy_percent = 85.0f; // Example value
        metrics.achieved_compute_peak_percent = 75.0f; // Example value

        return metrics;
    }

    static void print_metrics(const Metrics& metrics) {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  GFLOPS: " << metrics.gflops << std::endl;
        std::cout << "  Bandwidth: " << metrics.bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  Occupancy: " << metrics.occupancy_percent << "%" << std::endl;
        std::cout << "  Compute Utilization: " << metrics.achieved_compute_peak_percent << "%" << std::endl;
    }
};

void run_challenges() {
    std::cout << "\n=== Challenge Solutions ===" << std::endl;

    // Challenge 1: Quantized Linear Layer
    std::cout << "\nChallenge 1 - Quantized Linear Layer:" << std::endl;
    QuantizedLinearLayer layer;
    layer.initialize_scales(4, 3); // 4 outputs, 3 inputs

    // Create a simple quantized weight matrix (2x3 -> 2 outputs)
    std::vector<std::vector<int8_t>> weights = {{10, -5, 8}, {-3, 12, 6}};
    std::vector<int8_t> input = {5, -2, 7};
    std::vector<float> bias = {0.1f, 0.2f};

    auto output = layer.forward(weights, input, bias);
    std::cout << "Quantized linear layer output: ";
    for (float val : output) std::cout << val << " ";
    std::cout << std::endl;

    // Challenge 2: Sparse Matrix Format Converter
    std::cout << "\nChallenge 2 - Sparse Matrix Format Converter:" << std::endl;
    std::vector<std::vector<float>> dense_sparse = {
        {1.0f, 0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 3.0f, 4.0f},
        {5.0f, 0.0f, 0.0f, 6.0f}
    };

    auto csr_result = SparseFormatConverter::dense_to_csr(dense_sparse);
    SparseFormatConverter::print_csr_format(csr_result);

    // Challenge 3: Performance Profiler
    std::cout << "\nChallenge 3 - Performance Profiler:" << std::endl;
    auto metrics = PerformanceProfiler::profile_gemm_performance(1024, 1024, 512, 2.5f);
    PerformanceProfiler::print_metrics(metrics);
}

int main() {
    std::cout << "Module 9: Real-world Applications and Case Studies Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_deep_learning_framework_integration();
    exercise_quantized_matrix_multiplication();
    exercise_sparse_operations();
    exercise_mixed_precision_computations();
    exercise_memory_bandwidth_and_occupancy_optimization();

    // Run challenges
    run_challenges();

    std::cout << "\nSummary:" << std::endl;
    std::cout << "This module covered real-world applications of CUTLASS including:" << std::endl;
    std::cout << "- Integration with deep learning frameworks (PyTorch, TensorFlow)" << std::endl;
    std::cout << "- Quantized operations for efficient inference" << std::endl;
    std::cout << "- Sparse computations for model compression" << std::endl;
    std::cout << "- Mixed precision techniques for performance/accuracy balance" << std::endl;
    std::cout << "- Memory and occupancy optimization strategies" << std::endl;
    std::cout << "These techniques are essential for deploying high-performance ML workloads." << std::endl;

    return 0;
}