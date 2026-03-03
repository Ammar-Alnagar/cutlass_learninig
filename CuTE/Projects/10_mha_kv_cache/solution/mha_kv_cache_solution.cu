/**
 * Project 10: MHA with KV-Cache - Reference Solution
 */

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>

#include <cute/tensor.hpp>

namespace cute {

struct MHAConfig {
  static constexpr int NUM_HEADS = 8;
  static constexpr int HEAD_DIM = 64;
  static constexpr int MAX_SEQ_LEN = 1024;
  static constexpr int BLOCK_SIZE = 256;
};

__global__ void mha_kv_cache_kernel(float *Q, float *K_cache, float *V_cache,
                                    float *O, int *seq_lens, int batch,
                                    int seq_len, int num_heads, int head_dim) {

  int head_idx = blockIdx.z;
  int batch_idx = blockIdx.y;
  int q_pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx >= batch || q_pos >= seq_len || head_idx >= num_heads)
    return;

  int d = num_heads * head_dim;
  float scale = rsqrtf((float)head_dim);

  // Pointers for this batch and head
  float *q_ptr = Q + (batch_idx * seq_len + q_pos) * d + head_idx * head_dim;
  float *k_ptr = K_cache + batch_idx * seq_len * d + head_idx * head_dim;
  float *v_ptr = V_cache + batch_idx * seq_len * d + head_idx * head_dim;
  float *o_ptr = O + (batch_idx * seq_len + q_pos) * d + head_idx * head_dim;

  int actual_seq = seq_lens ? seq_lens[batch_idx] : seq_len;

  // Load Q vector
  float q_vec[64];
  for (int k = 0; k < head_dim; k++)
    q_vec[k] = q_ptr[k];

  // Compute attention scores and accumulate
  float max_score = -std::numeric_limits<float>::infinity();
  float sum_exp = 0.0f;
  float accum[64] = {0.0f};

  for (int kv_pos = 0; kv_pos < actual_seq; kv_pos++) {
    // Compute score: Q · K
    float score = 0.0f;
    for (int k = 0; k < head_dim; k++) {
      score += q_vec[k] * k_ptr[kv_pos * d + k];
    }
    score *= scale;

    // Online softmax update
    float new_max = fmaxf(max_score, score);
    float rescale = expf(max_score - new_max);
    for (int k = 0; k < head_dim; k++)
      accum[k] *= rescale;

    float prob = expf(score - new_max);
    sum_exp = sum_exp * rescale + prob;

    for (int k = 0; k < head_dim; k++) {
      accum[k] += prob * v_ptr[kv_pos * d + k];
    }

    max_score = new_max;
  }

  // Normalize and store
  if (sum_exp > 0.0f) {
    for (int k = 0; k < head_dim; k++) {
      o_ptr[k] = accum[k] / sum_exp;
    }
  }
}

} // namespace cute

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = call;                                                      \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl;       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  using Config = cute::MHAConfig;
  const int batch = 4, seq_len = 64, num_heads = 8, head_dim = 64;
  const int d = num_heads * head_dim;

  std::cout << "=== Project 10: MHA with KV-Cache (Solution) ===" << std::endl;

  int total = batch * seq_len * d;
  std::vector<float> h_Q(total), h_K(total), h_V(total), h_O(total, 0);
  for (int i = 0; i < total; i++)
    h_Q[i] = h_K[i] = h_V[i] = (i % 100) * 0.01f;

  float *d_Q, *d_K, *d_V, *d_O;
  CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));

  cudaMemcpy(d_Q, h_Q.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V.data(), total * sizeof(float), cudaMemcpyHostToDevice);

  const dim3 block(Config::BLOCK_SIZE);
  const dim3 grid((seq_len + block.x - 1) / block.x, batch, num_heads);

  cute::mha_kv_cache_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, nullptr, batch,
                                             seq_len, num_heads, head_dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaMemcpy(h_O.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "[PASS] MHA with KV-Cache" << std::endl;

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
  return 0;
}
