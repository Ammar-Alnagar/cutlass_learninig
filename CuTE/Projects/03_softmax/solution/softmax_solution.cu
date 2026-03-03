/**
 * Project 03: Softmax - Reference Solution
 *
 * Implements numerically stable row-wise softmax using two-pass reduction.
 */

#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cute/tensor.hpp>

namespace cute {

constexpr int BLOCK_SIZE = 256;

__global__ void softmax_cute_kernel(float *X, float *Y, int batch,
                                    int seq_len) {
  int row = blockIdx.x;
  if (row >= batch)
    return;

  int tid = threadIdx.x;
  int stride = blockDim.x;

  // Create 1D layout for this row
  auto layout = make_layout(seq_len);

  // Create tensors for this row
  float *row_X = X + row * seq_len;
  float *row_Y = Y + row * seq_len;
  auto tensor_X = make_tensor(make_gmem_ptr(row_X), layout);
  auto tensor_Y = make_tensor(make_gmem_ptr(row_Y), layout);

  // Allocate shared memory for reductions
  __shared__ float s_max[BLOCK_SIZE];
  __shared__ float s_sum[BLOCK_SIZE];

  // Pass 1: Find maximum value in this row
  float thread_max = -std::numeric_limits<float>::infinity();
  for (int i = tid; i < seq_len; i += stride) {
    thread_max = fmaxf(thread_max, tensor_X(i));
  }

  // Reduce thread_max across all threads
  s_max[tid] = thread_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
    }
    __syncthreads();
  }

  float row_max = s_max[0];
  __syncthreads();

  // Pass 2: Compute sum of exponentials
  float thread_sum = 0.0f;
  for (int i = tid; i < seq_len; i += stride) {
    thread_sum += expf(tensor_X(i) - row_max);
  }

  // Reduce thread_sum across all threads
  s_sum[tid] = thread_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_sum[tid] += s_sum[tid + s];
    }
    __syncthreads();
  }

  float row_sum = s_sum[0];
  __syncthreads();

  // Pass 3: Normalize
  for (int i = tid; i < seq_len; i += stride) {
    tensor_Y(i) = expf(tensor_X(i) - row_max) / row_sum;
  }
}

} // namespace cute

// ============================================================================
// Host Code
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void init_matrix_random(std::vector<float> &mat, int rows, int cols,
                        float mean = 0.0f, float std_dev = 1.0f) {
  for (int i = 0; i < rows * cols; ++i) {
    mat[i] = mean + std_dev * (float)(i % 100) / 100.0f;
  }
}

void softmax_reference(const std::vector<float> &X, std::vector<float> &Y,
                       int batch, int seq_len) {
  for (int b = 0; b < batch; ++b) {
    const float *row = X.data() + b * seq_len;
    float *out = Y.data() + b * seq_len;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < seq_len; ++i) {
      max_val = fmaxf(max_val, row[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
      sum_exp += expf(row[i] - max_val);
    }

    for (int i = 0; i < seq_len; ++i) {
      out[i] = expf(row[i] - max_val) / sum_exp;
    }
  }
}

bool verify_softmax(const std::vector<float> &Y, int batch, int seq_len,
                    float tolerance = 1e-5f) {
  float max_deviation = 0.0f;

  for (int b = 0; b < batch; ++b) {
    float row_sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
      row_sum += Y[b * seq_len + i];
    }
    float deviation = std::abs(row_sum - 1.0f);
    if (deviation > max_deviation) {
      max_deviation = deviation;
    }
  }

  std::cout << "Max deviation from 1.0: " << max_deviation << std::endl;
  return max_deviation <= tolerance;
}

void print_row(const std::vector<float> &mat, int row, int seq_len,
               int max_cols = 8) {
  std::cout << "  Row " << row << ": ";
  for (int i = 0; i < std::min(seq_len, max_cols); ++i) {
    std::cout << std::setw(8) << std::fixed << std::setprecision(4)
              << mat[row * seq_len + i] << " ";
  }
  std::cout << std::endl;
}

int main() {
  const int batch = 32;
  const int seq_len = 512;

  const int block_size = cute::BLOCK_SIZE;
  const int grid_size = batch;

  std::cout << "=== Project 03: Softmax with CuTe (Solution) ===" << std::endl;
  std::cout << "Input shape: " << batch << " x " << seq_len << std::endl;
  std::cout << std::endl;

  std::vector<float> h_X(batch * seq_len), h_Y(batch * seq_len, 0.0f);
  init_matrix_random(h_X, batch, seq_len, 0.0f, 2.0f);

  std::cout << "Input sample (first row, first 8 elements):" << std::endl;
  print_row(h_X, 0, seq_len, 8);
  std::cout << std::endl;

  std::cout << "Computing CPU reference..." << std::endl;
  std::vector<float> h_Y_ref(batch * seq_len);
  softmax_reference(h_X, h_Y_ref, batch, seq_len);
  std::cout << "Reference output (first row, first 8 elements):" << std::endl;
  print_row(h_Y_ref, 0, seq_len, 8);
  std::cout << std::endl;

  float *d_X, *d_Y;
  CUDA_CHECK(cudaMalloc(&d_X, batch * seq_len * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y, batch * seq_len * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), batch * seq_len * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::cout << "Launching CuTe softmax kernel..." << std::endl;
  cute::softmax_cute_kernel<<<grid_size, block_size>>>(d_X, d_Y, batch,
                                                       seq_len);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, batch * seq_len * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::cout << "Verifying result..." << std::endl;
  std::cout << "GPU output (first row, first 8 elements):" << std::endl;
  print_row(h_Y, 0, seq_len, 8);
  std::cout << std::endl;

  if (verify_softmax(h_Y, batch, seq_len)) {
    std::cout << "\n[PASS] Softmax: All rows sum to 1.0" << std::endl;
  } else {
    std::cout << "\n[FAIL] Softmax: Row sums deviate from 1.0!" << std::endl;
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Y));

  std::cout << "\n=== Solution Complete! ===" << std::endl;

  return EXIT_SUCCESS;
}
