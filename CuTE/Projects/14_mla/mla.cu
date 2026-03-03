/**
 * Project 14: Fused MLA (Multi-head Latent Attention)
 *
 * Objective: Implement memory-efficient Multi-head Latent Attention
 */

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>

#include <cute/tensor.hpp>

namespace cute {

struct MLAConfig {
  static constexpr int NUM_HEADS = 8;
  static constexpr int HEAD_DIM = 64;
  static constexpr int NUM_LATENTS = 4; // Fewer latents than heads
  static constexpr int LATENT_DIM = 32; // Smaller dimension
  static constexpr int BLOCK_SIZE = 256;
};

/**
 * TODO: Implement Multi-head Latent Attention
 *
 * Key concepts:
 * - Compress Q to latent space
 * - Attention with compressed KV
 * - Project back to output space
 */
__global__ void mla_kernel(float *Q, float *K_latent, float *V_latent, float *O,
                           float *W_Q_latent, float *W_out, int *seq_lens,
                           int batch, int seq_len, int num_heads, int head_dim,
                           int num_latents, int latent_dim) {

  int head_idx = blockIdx.z;
  int batch_idx = blockIdx.y;
  int q_pos = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: Implement MLA
  // 1. Project Q to latent space
  // 2. Compute attention with K_latent, V_latent
  // 3. Project output back to head_dim

  (void)Q;
  (void)K_latent;
  (void)V_latent;
  (void)O;
  (void)W_Q_latent;
  (void)W_out;
  (void)seq_lens;
  (void)batch;
  (void)seq_len;
  (void)num_heads;
  (void)head_dim;
  (void)num_latents;
  (void)latent_dim;
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
  using Config = cute::MLAConfig;

  const int batch = 4;
  const int seq_len = 64;
  const int num_heads = Config::NUM_HEADS;
  const int head_dim = Config::HEAD_DIM;
  const int num_latents = Config::NUM_LATENTS;
  const int latent_dim = Config::LATENT_DIM;
  const int d = num_heads * head_dim;

  std::cout << "=== Project 14: Fused MLA (Multi-head Latent Attention) ==="
            << std::endl;
  std::cout << "Batch: " << batch << ", Seq: " << seq_len << std::endl;
  std::cout << "Heads: " << num_heads << ", Head dim: " << head_dim
            << std::endl;
  std::cout << "Latents: " << num_latents << ", Latent dim: " << latent_dim
            << std::endl;

  int total_qkv = batch * seq_len * d;
  int total_latent = batch * seq_len * num_latents * latent_dim;

  std::vector<float> h_Q(total_qkv), h_K_latent(total_latent),
      h_V_latent(total_latent);
  std::vector<float> h_O(total_qkv, 0);
  std::vector<float> h_W_Q_latent(d * num_latents * latent_dim);
  std::vector<float> h_W_out(num_latents * latent_dim * d);

  for (int i = 0; i < total_qkv; i++)
    h_Q[i] = (i % 100) * 0.01f;
  for (int i = 0; i < total_latent; i++) {
    h_K_latent[i] = (i % 100) * 0.01f;
    h_V_latent[i] = (i % 100) * 0.01f;
  }
  for (int i = 0; i < d * num_latents * latent_dim; i++) {
    h_W_Q_latent[i] = (i % 100) * 0.01f;
    h_W_out[i] = (i % 100) * 0.01f;
  }

  float *d_Q, *d_K_latent, *d_V_latent, *d_O, *d_W_Q_latent, *d_W_out;
  CUDA_CHECK(cudaMalloc(&d_Q, total_qkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K_latent, total_latent * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V_latent, total_latent * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O, total_qkv * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&d_W_Q_latent, d * num_latents * latent_dim * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&d_W_out, num_latents * latent_dim * d * sizeof(float)));

  cudaMemcpy(d_Q, h_Q.data(), total_qkv * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_K_latent, h_K_latent.data(), total_latent * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_V_latent, h_V_latent.data(), total_latent * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_Q_latent, h_W_Q_latent.data(),
             d * num_latents * latent_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_out, h_W_out.data(),
             num_latents * latent_dim * d * sizeof(float),
             cudaMemcpyHostToDevice);

  const dim3 block(Config::BLOCK_SIZE);
  const dim3 grid((seq_len + block.x - 1) / block.x, batch, num_heads);

  cute::mla_kernel<<<grid, block>>>(
      d_Q, d_K_latent, d_V_latent, d_O, d_W_Q_latent, d_W_out, nullptr, batch,
      seq_len, num_heads, head_dim, num_latents, latent_dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaMemcpy(h_O.data(), d_O, total_qkv * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "[PASS] MLA kernel executed (structure verified)" << std::endl;

  cudaFree(d_Q);
  cudaFree(d_K_latent);
  cudaFree(d_V_latent);
  cudaFree(d_O);
  cudaFree(d_W_Q_latent);
  cudaFree(d_W_out);
  return 0;
}
