#pragma once
/**
 * CUTLASS 3.x Reference Implementation Utilities
 * 
 * cuBLAS reference for correctness verification.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>

namespace cutlass_ref {

// cuBLAS handle wrapper
class CublasHandle {
public:
    CublasHandle() {
        cublasCreate(&handle_);
    }

    ~CublasHandle() {
        cublasDestroy(handle_);
    }

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_;
};

// Reference GEMM using cuBLAS (FP32)
inline void gemm_ref_fp32(
    int M, int N, int K,
    const float* A, const float* B, float* C,
    float alpha = 1.0f, float beta = 0.0f
) {
    static CublasHandle handle;
    
    cublasSgemm(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );
}

// Reference GEMM using cuBLAS (FP16)
inline void gemm_ref_fp16(
    int M, int N, int K,
    const half* A, const half* B, half* C,
    float alpha = 1.0f, float beta = 0.0f
) {
    static CublasHandle handle;
    
    const half alpha_h = __float2half(alpha);
    const half beta_h = __float2half(beta);
    
    cublasHgemm(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        B, N,
        A, K,
        &beta_h,
        C, N
    );
}

// Reference GEMM using cuBLAS (BF16)
inline void gemm_ref_bf16(
    int M, int N, int K,
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    float alpha = 1.0f, float beta = 0.0f
) {
    static CublasHandle handle;
    
    const float alpha_f = alpha;
    const float beta_f = beta;
    
    cublasBf16Gemm(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_f,
        B, N,
        A, K,
        &beta_f,
        C, N,
        CUBLAS_COMPUTE_32F
    );
}

// Reference GEMM with Tensor Core (FP16 accumulator FP32)
inline void gemm_ref_tensor_core_fp16(
    int M, int N, int K,
    const half* A, const half* B, float* C,
    float alpha = 1.0f, float beta = 0.0f
) {
    static CublasHandle handle;
    
    cublasGemmEx(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_TENSOR_OP
    );
}

// Compute max absolute error between two tensors
template <typename T>
inline float compute_max_error(const T* computed, const T* reference, int size) {
    float max_err = 0.0f;
    
    // Copy to host for comparison
    std::vector<T> h_computed(size);
    std::vector<T> h_reference(size);
    
    cudaMemcpy(h_computed.data(), computed, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reference.data(), reference, size * sizeof(T), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i) {
        float err = fabsf(float(h_computed[i]) - float(h_reference[i]));
        if (err > max_err) max_err = err;
    }
    
    return max_err;
}

// Compute relative error (with epsilon to avoid division by zero)
template <typename T>
inline float compute_relative_error(const T* computed, const T* reference, int size) {
    float max_rel = 0.0f;
    const float eps = 1e-6f;
    
    std::vector<T> h_computed(size);
    std::vector<T> h_reference(size);
    
    cudaMemcpy(h_computed.data(), computed, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reference.data(), reference, size * sizeof(T), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i) {
        float c = float(h_computed[i]);
        float r = float(h_reference[i]);
        float rel = fabsf(c - r) / (fabsf(r) + eps);
        if (rel > max_rel) max_rel = rel;
    }
    
    return max_rel;
}

// Verify correctness against cuBLAS reference
template <typename T>
inline bool verify_gemm(
    const T* computed, const T* reference, int size,
    float tolerance = 1e-3f, bool verbose = true
) {
    float max_err = compute_max_error(computed, reference, size);
    float rel_err = compute_relative_error(computed, reference, size);
    
    bool passed = (rel_err < tolerance);
    
    if (verbose) {
        std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "  Max absolute error: " << max_err << std::endl;
        std::cout << "  Max relative error: " << rel_err << std::endl;
        std::cout << "  Tolerance: " << tolerance << std::endl;
    }
    
    return passed;
}

// Initialize matrix with random values
template <typename T>
inline void init_matrix_random(T* data, int size, float scale = 1.0f) {
    std::vector<float> h_data(size);
    for (int i = 0; i < size; ++i) {
        h_data[i] = scale * (float(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    
    if constexpr (std::is_same<T, half>::value) {
        std::vector<half> h_half(size);
        for (int i = 0; i < size; ++i) {
            h_half[i] = __float2half(h_data[i]);
        }
        cudaMemcpy(data, h_half.data(), size * sizeof(half), cudaMemcpyHostToDevice);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        std::vector<__nv_bfloat16> h_bf16(size);
        for (int i = 0; i < size; ++i) {
            h_bf16[i] = __float2bfloat16(h_data[i]);
        }
        cudaMemcpy(data, h_bf16.data(), size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(data, h_data.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    }
}

// Initialize matrix with zeros
template <typename T>
inline void init_matrix_zeros(T* data, int size) {
    cudaMemset(data, 0, size * sizeof(T));
}

} // namespace cutlass_ref
